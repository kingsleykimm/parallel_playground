/**
  @file kernel3.hpp
  @brief JIT launcher for kernel3 - grouped gemm with fused silumulquant
 */
#pragma once
#include "jit_kernels/heuristics/heuristics.hpp"
#include "moe_cuda/types.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/heuristics.hpp>
#include <jit_kernels/tk_globals_factory.h>
#include <runtime/device.hpp>
#include <runtime/format.hpp>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

// Runtime for kernel2::matmul_template (grouped FP8 GEMM)
// GEMM_TYPE: 0 = MGroupedMasked, 1 = MGroupedContiguous
class Kernel3Runtime : LaunchRuntime<Kernel3Runtime> {
public:
  struct Args {
    uint32_t M, N, K; //
    uint32_t num_groups;
    void *A, *up_weight, *gate_weight, *D, *scale_a, *scale_up, *scale_gate,
        *scale_d;
    void *grouped_layout; // device pointer, int array
    int bm, bn, bk, super_n;
    int num_consumer_warps, num_producer_warps;
    int num_stages;
    int smem_size;
    int gemm_type; // 0 = MGroupedMasked, 1 = MGroupedContiguous
    c10::ScalarType c_dtype;
    LaunchConfig launch_config;
  };

  static std::string generate_impl(const Args &args) {
    // kernel2::matmul_template params:
    //   _M, _N, _K, _BM, _BN, _BK, NUM_GROUPS,
    //   NUM_CONSUMER_WARPS, NUM_PRODUCER_WARPS, NUM_STAGES,
    //   KERNEL_SMEM_SIZE, GEMM_TYPE, c_dtype [, _SUPER_N=12]
    return fmt::format(R"(

#include <moe_cuda/kernels/kernel3.cuh>

using mmt_jit = kernel3::matmul_template<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &kittens::prototype::lcf::kernel<mmt_jit>);
	}}
	)",
                       args.M, args.N, args.K, args.bm, args.bn, args.bk,
                       args.num_groups, args.num_consumer_warps,
                       args.num_producer_warps, args.num_stages, args.smem_size,
                       args.gemm_type, to_string(args.c_dtype), args.super_n);
  }

  static void launch_impl(KernelHandle &kernel,
                          const LaunchConfigHandle &launch_config,
                          const Args &args) {
    // args.M = max_M (masked) or total_M (contiguous); args.N = N_per_group
    // always
    size_t total_M = (args.gemm_type == 0)
                         ? (size_t)args.num_groups *
                               args.M      // masked: total_M = groups * max_M
                         : (size_t)args.M; // contiguous: total_M already full
    size_t total_N =
        (size_t)args.num_groups * args.N; // always N_per_group * num_groups

    size_t gsize = tk_kernel3_globals_size(args.bm, args.bn, args.bk,
                                           args.gemm_type, args.c_dtype);
    alignas(128) char globals_buf[2048];
    assert(gsize <= sizeof(globals_buf));
    tk_build_kernel3_globals(args.bm, args.bn, args.bk, args.gemm_type,
                             static_cast<int>(args.num_groups), args.c_dtype,
                             globals_buf, args.A, args.gate_weight,
                             args.up_weight, args.D, args.scale_a,
                             args.scale_gate, args.scale_up, args.scale_d,
                             args.grouped_layout, total_M, total_N, args.K);

    void *kernelParams[] = {globals_buf};
    CUDA_CHECK(cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr));

    if (get_env<int>("JIT_DEBUG") != 0) {
      CUDA_CHECK(cudaStreamSynchronize(launch_config.hStream));
    }
  }
};

// Contiguous grouped FP8 GEMM with fused SwiGLU + FP8 requantization:
//   A:             (total_M, K)           — tokens from all groups concatenated
//   gate_weight:   (num_groups, N, K)     — gate projection weights
//   up_weight:     (num_groups, N, K)     — up   projection weights
//   scale_a:       (K/128, total_M)       — per-token K-block scales (MN-major)
//   scale_gate:    (num_groups * N/128, K/128)
//   scale_up:      (num_groups * N/128, K/128)
//   grouped_layout: (total_M,) int32      — group index per token row
//   D:             (total_M, N/2)         — FP8 output (silu(gate) * up,
//   requantized) scale_d:       (N/(2*BN), total_M)    — per-(row, col-block)
//   output scale
inline void kernel3_contiguous(
    torch::stable::Tensor &A, torch::stable::Tensor &up_weight,
    torch::stable::Tensor &gate_weight, torch::stable::Tensor &scale_a,
    torch::stable::Tensor &scale_up, torch::stable::Tensor &scale_gate,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
    int *grouped_layout, cudaStream_t &stream) {
  HOST_ASSERT(
      D.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "unsupported output dtype: kernel3 outputs FP8-quantized activations");

  uint32_t total_M = A.size(0);
  uint32_t num_groups = up_weight.size(0);
  uint32_t N = up_weight.size(-2);
  uint32_t K = up_weight.size(-1);

  auto gemm_config = get_kernel3_config(
      GemmType::MGroupedContiguous, total_M, N, K, 1, Major::K, Major::K,
      Major::K, A.scalar_type(), D.scalar_type(), device_prop->get_num_sms());

  int num_consumer_warps = gemm_config.num_math_threads / 32;
  int num_producer_warps = gemm_config.num_tma_threads / 32;
  int super_n = 8;

  LaunchConfig launch_config = {
      dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
      dim3(132), stream, gemm_config.smem_config.smem_size, 1};

  const Kernel3Runtime::Args args = {
      .M = total_M,
      .N = N,
      .K = K,
      .num_groups = num_groups,
      .A = A.data_ptr(),
      .up_weight = up_weight.data_ptr(),
      .gate_weight = gate_weight.data_ptr(),
      .D = D.data_ptr(),
      .scale_a = scale_a.data_ptr(),
      .scale_up = scale_up.data_ptr(),
      .scale_gate = scale_gate.data_ptr(),
      .scale_d = scale_d.data_ptr(),
      .grouped_layout = (void *)grouped_layout,
      .bm = (int)gemm_config.block_m,
      .bn = (int)gemm_config.block_n,
      .bk = (int)gemm_config.block_k,
      .super_n = super_n,
      .num_consumer_warps = num_consumer_warps,
      .num_producer_warps = num_producer_warps,
      .num_stages = gemm_config.num_stages,
      .smem_size = gemm_config.smem_config.smem_size,
      .gemm_type = /*gemm_type=*/1,
      .c_dtype = D.scalar_type(),
      .launch_config = launch_config,
  };

  if (get_env<int>("JIT_DEBUG") > 0) {
    printf("kernel3_contiguous:\n");
    printf("  total_M=%u total_N=%u K=%u num_groups=%u\n", total_M, N, K,
           num_groups);
    printf("  bm=%d bn=%d bk=%d super_n=%d stages=%d\n", args.bm, args.bn,
           args.bk, args.super_n, args.num_stages);
  }

  const std::string &code = LaunchRuntime<Kernel3Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("kernel3_contiguous", code);
  LaunchRuntime<Kernel3Runtime>::launch(runtime, args);
}

// Masked grouped FP8 GEMM with fused SwiGLU + FP8 requantization:
//   A:             (num_groups, max_M, K) — padded token blocks per group
//   gate_weight:   (num_groups, N, K)     — gate projection weights
//   up_weight:     (num_groups, N, K)     — up   projection weights
//   scale_a:       (K/128, num_groups * max_M) — MN-major per-token K-block
//   scales scale_gate:    (num_groups * N/128, K/128) scale_up: (num_groups *
//   N/128, K/128) grouped_layout: (num_groups,) int32   — actual M count per
//   group D:             (num_groups * max_M, N/2) — FP8 output scale_d:
//   (N/(2*BN), num_groups * max_M) — per-(row, col-block) output scale
inline void
kernel3_masked(torch::stable::Tensor &A, torch::stable::Tensor &up_weight,
               torch::stable::Tensor &gate_weight,
               torch::stable::Tensor &scale_a, torch::stable::Tensor &scale_up,
               torch::stable::Tensor &scale_gate,
               torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
               int *grouped_layout, cudaStream_t &stream) {
  HOST_ASSERT(
      D.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "unsupported output dtype: kernel3 outputs FP8-quantized activations");

  uint32_t num_groups = gate_weight.size(0);
  uint32_t max_M = A.size(1);
  uint32_t N = gate_weight.size(-2);
  uint32_t K = gate_weight.size(-1);
  uint32_t total_M = num_groups * max_M;

  auto gemm_config = get_kernel3_config(
      GemmType::MGroupedMasked, max_M, N, K, num_groups, Major::K, Major::K,
      Major::K, A.scalar_type(), D.scalar_type(), device_prop->get_num_sms());

  int num_consumer_warps = gemm_config.num_math_threads / 32;
  int num_producer_warps = gemm_config.num_tma_threads / 32;
  int super_n = 8;

  LaunchConfig launch_config = {
      dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
      dim3(132), stream, gemm_config.smem_config.smem_size, 1};

  const Kernel3Runtime::Args args = {
      .M = max_M,
      .N = N,
      .K = K,
      .num_groups = num_groups,
      .A = A.data_ptr(),
      .up_weight = up_weight.data_ptr(),
      .gate_weight = gate_weight.data_ptr(),
      .D = D.data_ptr(),
      .scale_a = scale_a.data_ptr(),
      .scale_up = scale_up.data_ptr(),
      .scale_gate = scale_gate.data_ptr(),
      .scale_d = scale_d.data_ptr(),
      .grouped_layout = (void *)grouped_layout,
      .bm = (int)gemm_config.block_m,
      .bn = (int)gemm_config.block_n,
      .bk = (int)gemm_config.block_k,
      .super_n = super_n,
      .num_consumer_warps = num_consumer_warps,
      .num_producer_warps = num_producer_warps,
      .num_stages = gemm_config.num_stages,
      .smem_size = gemm_config.smem_config.smem_size,
      .gemm_type = /*gemm_type=*/0,
      .c_dtype = D.scalar_type(),
      .launch_config = launch_config,
  };

  if (get_env<int>("JIT_DEBUG") > 0) {
    printf("kernel3_masked:\n");
    printf("  total_M=%u N=%u K=%u num_groups=%u max_M=%u\n", total_M, N, K,
           num_groups, max_M);
    printf("  bm=%d bn=%d bk=%d super_n=%d stages=%d\n", args.bm, args.bn,
           args.bk, args.super_n, args.num_stages);
  }

  const std::string &code = LaunchRuntime<Kernel3Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("kernel3_masked", code);
  LaunchRuntime<Kernel3Runtime>::launch(runtime, args);
}
