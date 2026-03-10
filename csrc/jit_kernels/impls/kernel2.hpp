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
#include <cassert>
#include <jit_kernels/heuristics/heuristics.hpp>

// Runtime for kernel2::matmul_template (grouped FP8 GEMM)
// GEMM_TYPE: 0 = MGroupedMasked, 1 = MGroupedContiguous
class SM90_FP8_GroupedGEMM_Runtime : LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime> {
public:
    struct Args {
        uint32_t M, N, K;
        uint32_t num_groups;
        void *A, *B, *C, *scale_a, *scale_b;
        void *grouped_layout;  // device pointer, int array
        int bm, bn, bk, super_n;
        int num_consumer_warps, num_producer_warps;
        int num_stages;
        int smem_size;
        int gemm_type;  // 0 = MGroupedMasked, 1 = MGroupedContiguous
        at::ScalarType c_dtype;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        // kernel2::matmul_template params:
        //   _M, _N, _K, _BM, _BN, _BK, NUM_GROUPS,
        //   NUM_CONSUMER_WARPS, NUM_PRODUCER_WARPS, NUM_STAGES,
        //   KERNEL_SMEM_SIZE, GEMM_TYPE, c_dtype [, _SUPER_N=12]
        return fmt::format(R"(
#include <moe_cuda/kernels/kernel2.cuh>

using mmt_jit = kernel2::matmul_template<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &kittens::prototype::lcf::kernel<mmt_jit>);
}}
)",
            args.M, args.N, args.K,
            args.bm, args.bn, args.bk,
            args.num_groups,
            args.num_consumer_warps, args.num_producer_warps,
            args.num_stages, args.smem_size,
            args.gemm_type,
            to_string(args.c_dtype),
            args.super_n);
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config, const Args& args) {
        // args.M = max_M (masked) or total_M (contiguous); args.N = N_per_group always
        size_t total_M = (args.gemm_type == 0)
                             ? (size_t)args.num_groups * args.M   // masked: total_M = groups * max_M
                             : (size_t)args.M;                    // contiguous: total_M already full
        size_t total_N = (size_t)args.num_groups * args.N;        // always N_per_group * num_groups

        size_t gsize = tk_grouped_globals_size(args.bm, args.bn, args.bk, args.gemm_type, args.c_dtype);
        alignas(128) char globals_buf[2048];
        assert(gsize <= sizeof(globals_buf));
        tk_build_grouped_globals(
            args.bm, args.bn, args.bk, args.gemm_type,
            static_cast<int>(args.num_groups), args.c_dtype,
            globals_buf,
            args.A, args.B, args.C, args.scale_a, args.scale_b,
            args.grouped_layout,
            total_M, total_N, args.K);

        void* kernelParams[] = { globals_buf };
        CUDA_CHECK(cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr));
    }
};


// Contiguous grouped FP8 GEMM:
//   A:          (total_M, K)           — tokens from all groups concatenated
//   B:          (num_groups, N, K)     — per-group weight matrices
//   scale_a:    (K/128, total_M)       — per-token K-block scales (MN-major)
//   scale_b:    (num_groups * N/128, K/128)
//   grouped_layout: (total_M,) int32 device array — group index per token row
//   D:          (total_M, num_groups * N)
inline void sm90_fp8_grouped_gemm_contiguous(
    at::Tensor& A, at::Tensor& B,
    at::Tensor& scale_a, at::Tensor& scale_b,
    at::Tensor& D, int* grouped_layout, cudaStream_t& stream)
{
    HOST_ASSERT(D.scalar_type() == at::ScalarType::BFloat16 || D.scalar_type() == at::ScalarType::Float,
                "unsupported output dtype");

    uint32_t total_M   = A.size(0);
    uint32_t num_groups = B.size(0);
    uint32_t N = B.size(-2);
    uint32_t K          = B.size(-1);

    auto gemm_config = search_configs(
        GemmType::MGroupedContiguous, total_M, N, K,
        1, Major::K, Major::K, Major::K,
        A.scalar_type(), D.scalar_type(),
        device_prop->get_num_sms());

    int num_consumer_warps = gemm_config.num_math_threads / 32;
    int num_producer_warps = gemm_config.num_tma_threads / 32;
    int super_n = 8;

    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(device_prop->get_num_sms()),
        stream,
        gemm_config.smem_config.smem_size,
        1
    };

    const SM90_FP8_GroupedGEMM_Runtime::Args args = {
        total_M, N, K,
        num_groups,
        A.data_ptr(), B.data_ptr(), D.data_ptr(),
        scale_a.data_ptr(), scale_b.data_ptr(),
        (void*)grouped_layout,
        (int)gemm_config.block_m, (int)gemm_config.block_n, (int)gemm_config.block_k,
        super_n,
        num_consumer_warps, num_producer_warps,
        gemm_config.num_stages,
        gemm_config.smem_config.smem_size,
        /*gemm_type=*/1,
        D.scalar_type(),
        launch_config,
    };

    if (get_env<int>("JIT_DEBUG") > 0) {
        printf("sm90_fp8_grouped_gemm_contiguous:\n");
        printf("  total_M=%u total_N=%u K=%u num_groups=%u\n", total_M, N, K, num_groups);
        printf("  bm=%d bn=%d bk=%d super_n=%d stages=%d\n",
               args.bm, args.bn, args.bk, args.super_n, args.num_stages);
    }

    const std::string& code = LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_grouped_gemm_contiguous", code);
    LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::launch(runtime, args);
}


// Masked grouped FP8 GEMM:
//   A:          (num_groups, max_M, K)
//   B:          (num_groups, N, K)
//   scale_a:    (K/128, num_groups * max_M)  (MN-major)
//   scale_b:    (num_groups * N/128, K/128)
//   grouped_layout: (num_groups,) int32 device array — actual M per group
//   D:          (num_groups, max_M,  N)
inline void sm90_fp8_grouped_gemm_masked(
    at::Tensor& A, at::Tensor& B,
    at::Tensor& scale_a, at::Tensor& scale_b,
    at::Tensor& D, int* grouped_layout, cudaStream_t& stream)
{
    HOST_ASSERT(D.scalar_type() == at::ScalarType::BFloat16 || D.scalar_type() == at::ScalarType::Float,
                "unsupported output dtype");

    uint32_t num_groups  = A.size(0);
    uint32_t max_M       = A.size(1);
    uint32_t N = B.size(-2);
    uint32_t K           = B.size(-1);
    uint32_t total_M     = num_groups * max_M;
    auto gemm_config = search_configs(
        GemmType::MGroupedMasked, max_M, N, K,
        num_groups, Major::K, Major::K, Major::K,
        A.scalar_type(), D.scalar_type(),
        device_prop->get_num_sms());

    int num_consumer_warps = gemm_config.num_math_threads / 32;
    int num_producer_warps = gemm_config.num_tma_threads / 32;
    int super_n = 12;

    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(device_prop->get_num_sms()),
        stream,
        gemm_config.smem_config.smem_size,
        1
    };

    const SM90_FP8_GroupedGEMM_Runtime::Args args = {
        max_M, N, K,
        num_groups,
        A.data_ptr(), B.data_ptr(), D.data_ptr(),
        scale_a.data_ptr(), scale_b.data_ptr(),
        (void*)grouped_layout,
        (int)gemm_config.block_m, (int)gemm_config.block_n, (int)gemm_config.block_k,
        super_n,
        num_consumer_warps, num_producer_warps,
        gemm_config.num_stages,
        gemm_config.smem_config.smem_size,
        /*gemm_type=*/0,
        D.scalar_type(),
        launch_config,
    };

    if (get_env<int>("JIT_DEBUG") > 0) {
        printf("sm90_fp8_grouped_gemm_masked:\n");
        printf("  total_M=%u N=%u K=%u num_groups=%u max_M=%u\n",
               total_M, N, K, num_groups, max_M);
        printf("  bm=%d bn=%d bk=%d super_n=%d stages=%d\n",
               args.bm, args.bn, args.bk, args.super_n, args.num_stages);
    }

    const std::string& code = LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_grouped_gemm_masked", code);
    LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::launch(runtime, args);
}
