/**
  @file kernel5.hpp
  @brief JIT launcher for kernel5 - Fused Dispatch + FC1 of SwiGLU MLP, (Grouped GEMM)
         Uses cooperative grid launch for grid-wide sync between routing and compute/comm phases.
 */
#pragma once
#include "c10/core/ScalarType.h"
#include "jit_kernels/heuristics/heuristics.hpp"
#include "moe_cuda/types.h"
#include <runtime/format.hpp>
#include <runtime/device.hpp>
#include <jit/runtime.hpp>
#include <jit/compiler.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/tk_globals_factory.h>
#include <pyutils/parallel_tensor.cuh>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <ctime>


class Kernel5Runtime : LaunchRuntime<Kernel5Runtime> {
public:
    // kernel5 constants (mirrored from kernel5.cuh)
    static constexpr int SM_COUNT = 132;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

    struct Args {
        uint32_t M, I, H;
        uint32_t BM, BN;
        uint32_t num_consumer_warps;
        uint32_t num_producer_warps;
        uint32_t num_stages;
        uint32_t kernel_smem_size;
        uint32_t num_experts;
        uint32_t experts_per_token;
        uint32_t super_m;

        // tensor args for globals construction
        kittens::py::TKParallelTensor *in_tokens;
        kittens::py::TKParallelTensor *in_tokens_scales;
        at::Tensor *expert_x_tokens;
        at::Tensor *expert_x_tokens_scale;
        at::Tensor *comm_comp_barrier;
        at::Tensor *gate;
        at::Tensor *up;
        at::Tensor *C;
        at::Tensor *scale_gate;
        at::Tensor *scale_up;
        at::Tensor *out_scales;
        at::Tensor *indices;
        kittens::py::TKParallelTensor *global_num_routed;
        kittens::py::TKParallelTensor *expert_to_token_map;
        at::Tensor *padded_expert_counts;
        at::Tensor *src_token_idx;
        at::Tensor *src_dev_idx;
        kittens::py::TKParallelTensor *barrier;

        int num_tokens;
        int *num_recv_tokens;
        int dp_rank;
        int rank;
        int dp_size;
        int cur_dp_group;
        int num_dp_groups;
        int num_comm_sms;
        int num_comp_sms;

        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(

#include <moe_cuda/kernels/kernel5.cuh>

static void __instantiate_kernel() {{
auto ptr = reinterpret_cast<void *>(&kernel5::global_kernel5<
        {}, {}, {},
        {}, 
        {},
        {}, {},
        {},
        {},
        {}, {},
        {}
>);
}}
)",
            args.M, args.I, args.H,
            args.BM, args.BN,
            args.num_consumer_warps, args.num_producer_warps,
            args.num_stages,
            args.kernel_smem_size,
            args.num_experts, args.experts_per_token,
            args.super_m);
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& /*launch_config*/, const Args& args) {
        // Build globals via the pre-compiled factory
        size_t gsize = tk_kernel5_globals_size();
        alignas(128) char globals_buf[4096];
        assert(gsize <= sizeof(globals_buf));

        tk_build_kernel5_globals(
            globals_buf,
            *args.in_tokens, *args.in_tokens_scales,
            *args.expert_x_tokens, *args.expert_x_tokens_scale,
            *args.comm_comp_barrier, *args.gate, *args.up, *args.C,
            *args.scale_gate, *args.scale_up, *args.out_scales, *args.indices,
            *args.global_num_routed, *args.expert_to_token_map,
            *args.padded_expert_counts, *args.src_token_idx, *args.src_dev_idx,
            *args.barrier,
            args.num_tokens, args.num_recv_tokens,
            args.dp_rank, args.rank, args.dp_size, args.cur_dp_group,
            args.num_dp_groups, args.num_comm_sms, args.num_comp_sms);

        // kernel5 uses cooperative_groups::this_grid().sync(), so we need
        // cooperative launch via CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        CUlaunchAttribute attrs[1];
        attrs[0].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attrs[0].value.cooperative = 1;

        CUlaunchConfig coop_config;
        std::memset(&coop_config, 0, sizeof(coop_config));
        coop_config.blockDimX = args.launch_config.blockDim.x;
        coop_config.blockDimY = 1;
        coop_config.blockDimZ = 1;
        coop_config.gridDimX  = args.launch_config.gridDim.x;
        coop_config.gridDimY  = 1;
        coop_config.gridDimZ  = 1;
        coop_config.hStream   = (CUstream)args.launch_config.stream;
        coop_config.sharedMemBytes = args.launch_config.smem_size;
        coop_config.attrs     = attrs;
        coop_config.numAttrs  = 1;

        if (args.launch_config.smem_size > 0) {
            CUDA_CHECK(cuFuncSetAttribute(kernel,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                args.launch_config.smem_size));
        }

        void* kernelParams[] = { globals_buf };
        if (get_env<int>("JIT_DEBUG") != 0) {
            printf("Launching kernel5: grid=%u, block=%u, smem=%d, cooperative=1\n",
                   coop_config.gridDimX, coop_config.blockDimX, coop_config.sharedMemBytes);
        }
        CUDA_CHECK(cuLaunchKernelEx(&coop_config, kernel, kernelParams, nullptr));
        if (get_env<int>("JIT_DEBUG") != 0) {
            CUDA_CHECK(cudaStreamSynchronize(args.launch_config.stream));
        }
    }
};


inline void fused_dispatch_grouped_gemm_swiglu(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_tokens_scales,
    at::Tensor &expert_x_tokens, at::Tensor &expert_x_tokens_scale,
    at::Tensor &comm_comp_barrier, at::Tensor &gate, at::Tensor &up,
    at::Tensor &C, at::Tensor &scale_gate, at::Tensor &scale_up,
    at::Tensor &out_scales, at::Tensor &indices,
    kittens::py::TKParallelTensor &global_num_routed,
    kittens::py::TKParallelTensor &expert_to_token_map,
    at::Tensor &padded_expert_counts, at::Tensor &src_token_idx,
    at::Tensor &src_dev_idx, kittens::py::TKParallelTensor &barrier,
    int num_tokens, int *num_recv_tokens,
    int dp_rank, int rank, int dp_size, int cur_dp_group, int num_dp_groups,
    int num_experts, int experts_per_token,
    int num_comm_sms, int num_comp_sms,
    cudaStream_t &stream)
{
    // for persistent case
    HOST_ASSERT(num_comm_sms + num_comp_sms <= 132, "num_comm_sms + num_comp_sms must be less than or equal to SM_COUNT");
    uint32_t M = in_tokens.shape_[0];
    uint32_t H = in_tokens.shape_[1];
    uint32_t I = gate.size(-2);  // intermediate size (N dimension of gate/up weights)

    int total_sms = num_comm_sms + num_comp_sms;

    auto gemm_config = get_kernel5_config(M, I, num_experts, num_comp_sms);

    uint32_t BM = gemm_config.block_m;
    uint32_t BN = gemm_config.block_n;
    uint32_t num_consumer_warps = gemm_config.num_math_threads / 32;
    uint32_t num_producer_warps = gemm_config.num_tma_threads / 32;
    uint32_t num_stages = gemm_config.num_stages;
    uint32_t kernel_smem_size = gemm_config.smem_config.smem_size;
    uint32_t super_m = 8;

    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(total_sms),
        stream,
        Kernel5Runtime::DYNAMIC_SHARED_MEMORY,
        1  // num_multicast (not used, cooperative handles grid sync)
    };

    if (get_env<int>("JIT_DEBUG") > 0) {
        printf("fused_dispatch_grouped_gemm_swiglu:\n");
        printf("  M=%u I=%u H=%u num_experts=%d experts_per_token=%d\n",
               M, I, H, num_experts, experts_per_token);
        printf("  BM=%u BN=%u stages=%u super_m=%u\n", BM, BN, num_stages, super_m);
        printf("  consumer_warps=%u producer_warps=%u\n", num_consumer_warps, num_producer_warps);
        printf("  num_comm_sms=%d num_comp_sms=%d total_sms=%d\n",
               num_comm_sms, num_comp_sms, total_sms);
        printf("  smem_size=%u\n", Kernel5Runtime::DYNAMIC_SHARED_MEMORY);
    }

    const Kernel5Runtime::Args args = {
        .M = M, .I = I, .H = H,
        .BM = BM, .BN = BN,
        .num_consumer_warps = num_consumer_warps,
        .num_producer_warps = num_producer_warps,
        .num_stages = num_stages,
        .kernel_smem_size = kernel_smem_size,
        .num_experts = (uint32_t)num_experts,
        .experts_per_token = (uint32_t)experts_per_token,
        .super_m = super_m,
        .in_tokens = &in_tokens,
        .in_tokens_scales = &in_tokens_scales,
        .expert_x_tokens = &expert_x_tokens,
        .expert_x_tokens_scale = &expert_x_tokens_scale,
        .comm_comp_barrier = &comm_comp_barrier,
        .gate = &gate,
        .up = &up,
        .C = &C,
        .scale_gate = &scale_gate,
        .scale_up = &scale_up,
        .out_scales = &out_scales,
        .indices = &indices,
        .global_num_routed = &global_num_routed,
        .expert_to_token_map = &expert_to_token_map,
        .padded_expert_counts = &padded_expert_counts,
        .src_token_idx = &src_token_idx,
        .src_dev_idx = &src_dev_idx,
        .barrier = &barrier,
        .num_tokens = num_tokens,
        .num_recv_tokens = num_recv_tokens,
        .dp_rank = dp_rank,
        .rank = rank,
        .dp_size = dp_size,
        .cur_dp_group = cur_dp_group,
        .num_dp_groups = num_dp_groups,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = num_comp_sms,
        .launch_config = launch_config,
    };

    const std::string& code = LaunchRuntime<Kernel5Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("fused_dispatch_grouped_gemm_swiglu", code);
    LaunchRuntime<Kernel5Runtime>::launch(runtime, args);
}
