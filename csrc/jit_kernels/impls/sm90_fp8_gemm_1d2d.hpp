#pragma once
#include <torch/headeronly/core/ScalarType.h>
#include "jit_kernels/heuristics/heuristics.hpp"
#include "moe_cuda/types.h"
#include <cassert>
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/heuristics.hpp>
#include <jit_kernels/tk_globals_factory.h>
#include <runtime/device.hpp>
#include <runtime/format.hpp>
#include <torch/csrc/stable/tensor.h>

class SM90_FP8_GEMM1D2D_TK_Runtime
    : LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime> {
public:
  struct Args {
    uint32_t M, N, K;
    void *A, *B, *C, *scale_a, *scale_b;
    int bm, bn, bk, super_m;
    int num_consumer_warps, num_producer_warps;
    int num_stages;
    int smem_size;
    c10::ScalarType c_dtype;
    LaunchConfig launch_config;
  };

  static std::string generate_impl(const Args &args) {
    // JIT code uses template instantiation — no #defines needed.
    // NVRTC compiles this to produce a cubin with the kernel entry point.
    return fmt::format(R"(
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>

using mmt_jit = matmul_template<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &kittens::prototype::lcf::kernel<mmt_jit>);
}}
)",
                       args.M, args.N, args.K, args.bm, args.bn, args.bk,
                       args.num_consumer_warps, args.num_producer_warps,
                       args.num_stages, args.smem_size, to_string(args.c_dtype),
                       args.super_m);
  }

  static void launch_impl(KernelHandle &kernel,
                          const LaunchConfigHandle &launch_config,
                          const Args &args) {
    // Build globals via pre-compiled factory
    size_t gsize = tk_globals_size(args.bm, args.bn, args.bk, args.c_dtype);
    alignas(128) char globals_buf[2048];
    assert(gsize <= sizeof(globals_buf));
    // copies the tk factory struct (containing matmul_layout::globals) into
    // globals_buf
    tk_build_globals(args.bm, args.bn, args.bk, args.c_dtype, globals_buf,
                     args.A, args.B, args.C, args.scale_a, args.scale_b, args.M,
                     args.N, args.K);

    // Launch via cuLaunchKernelEx with globals as kernelParams[0] - this is the
    // only argument required for LCF
    void *kernelParams[] = {globals_buf};
    CUDA_CHECK(cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr));
  }
};

// [DEPRECATED] SM90_FP8_GEMM1D2D_Ref_Runtime and
// sm90_fp8_grouped_gemm_1d2d_contiguous_ref removed — replaced by TK
// globals factory path. See sm90_fp8_gemm_1d2d_nt below.

// persistent kernel style
inline void sm90_fp8_gemm_1d2d_nt(torch::stable::Tensor &A, torch::stable::Tensor &B,
                                  torch::stable::Tensor &scale_a, torch::stable::Tensor &scale_b,
                                  torch::stable::Tensor &D, cudaStream_t &stream) {

  HOST_ASSERT(D.scalar_type() == c10::ScalarType::BFloat16 ||
                  D.scalar_type() == c10::ScalarType::Float,
              "unsupported output dtype");

  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = B.size(1);

  // Default tile config
  // TK lcf kernel uses (NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * 32 threads
  // NUM_CONSUMER_WARPS=8, NUM_PRODUCER_WARPS=1 (default) => 9 warps => 288
  // threads MAX_SHARED_MEMORY - 1024 (same as the standalone TK kernel)

  auto gemm_config = search_configs(
      GemmType::Normal, M, N, K, 1, Major::K, Major::K, Major::K,
      A.scalar_type(), D.scalar_type(), device_prop->get_num_sms());

  int num_consumer_warps = gemm_config.num_math_threads / 32;
  int num_producer_warps = gemm_config.num_tma_threads / 32;

  int super_m = 8; // set to this for now , DG uses 8 or 16
                   // we'll do always persistent grid for now
  LaunchConfig launch_config = {
      dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
      dim3(device_prop->get_num_sms()), stream,
      gemm_config.smem_config.smem_size,
      1 // no multicast/clustering
  };

  const SM90_FP8_GEMM1D2D_TK_Runtime::Args args = {
      M,
      N,
      K,
      A.data_ptr(),
      B.data_ptr(),
      D.data_ptr(),
      scale_a.data_ptr(),
      scale_b.data_ptr(),
      (int)gemm_config.block_m,
      (int)gemm_config.block_n,
      (int)gemm_config.block_k,
      super_m,
      num_consumer_warps,
      num_producer_warps,
      gemm_config.num_stages,
      gemm_config.smem_config.smem_size,
      D.scalar_type(),
      launch_config,
  };
  if (get_env<int>("JIT_DEBUG") > 0) {
    printf("Args (primitives):\n");
    printf("  M = %u\n", M);
    printf("  N = %u\n", N);
    printf("  K = %u\n", K);
    printf("  bm = %d\n", args.bm);
    printf("  bn = %d\n", args.bn);
    printf("  bk = %d\n", args.bk);
    printf("  super_m = %d\n", args.super_m);
    printf("  num_consumer_warps = %d\n", args.num_consumer_warps);
    printf("  num_producer_warps = %d\n", args.num_producer_warps);
    printf("  num_stages = %d\n", args.num_stages);
    printf("  smem_size = %d\n", args.smem_size);
  }

  const std::string &code =
      LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("sm90_fp8_gemm_1d2d_tk", code);
  LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime>::launch(runtime, args);
}

// [DEPRECATED] sm90_fp8_grouped_gemm_1d2d_contiguous_ref removed —
// used host-side TMA descriptor creation (make_tma_*_desc). Replaced by
// TK globals factory path (kernel2 grouped GEMM).

