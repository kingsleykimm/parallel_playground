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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <jit_kernels/heuristics/heuristics.hpp>

namespace {

inline void assert_cuda_device_ptr(const void* ptr, const char* name) {
    HOST_ASSERT(ptr != nullptr, fmt::format("{} pointer is null", name).c_str());

    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    HOST_ASSERT(
        err == cudaSuccess,
        fmt::format("{} is not a CUDA pointer or cudaPointerGetAttributes failed", name).c_str());

    HOST_ASSERT(
        attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged,
        fmt::format("{} must live on device or managed memory", name).c_str());

    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (attr.type == cudaMemoryTypeDevice) {
        HOST_ASSERT(
            attr.device == current_device,
            fmt::format("{} is on device {} but current device is {}",
                        name, attr.device, current_device).c_str());
    }
}

inline void assert_grouped_gemm_ptrs_device_resident(
    const void* A, const void* B, const void* C,
    const void* scale_a, const void* scale_b, const void* grouped_layout) {
    assert_cuda_device_ptr(A, "A");
    assert_cuda_device_ptr(B, "B");
    assert_cuda_device_ptr(C, "C");
    assert_cuda_device_ptr(scale_a, "scale_a");
    assert_cuda_device_ptr(scale_b, "scale_b");
    assert_cuda_device_ptr(grouped_layout, "grouped_layout");
}

inline bool grouped_kernel_trace_host_enabled() {
    return get_env<int>("KERNEL2_TRACE_HOST", 0) > 0;
}

inline void dump_hex_prefix(const char* label, const void* ptr, size_t size,
                            size_t limit = 64) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(ptr);
    const size_t n = std::min(size, limit);
    printf("%s", label);
    for (size_t i = 0; i < n; ++i) {
        printf("%02x", static_cast<unsigned int>(bytes[i]));
        if (i + 1 != n) {
            printf(" ");
        }
    }
    printf("\n");
}

inline std::string kernel2_trace_define_block() {
    return fmt::format(
        "#define KERNEL2_TRACE_STAGE {}\n"
        "#define KERNEL2_TRACE_CTA {}\n"
        "#define KERNEL2_TRACE_TASK_ITER {}\n"
        "#define KERNEL2_TRACE_TRAP_STAGE {}\n",
        get_env<int>("KERNEL2_TRACE_STAGE", -1),
        get_env<int>("KERNEL2_TRACE_CTA", -1),
        get_env<int>("KERNEL2_TRACE_TASK_ITER", -1),
        get_env<int>("KERNEL2_TRACE_TRAP_STAGE", -1));
}

} // namespace

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
{}
#include <moe_cuda/kernels/kernel2.cuh>

using mmt_jit = kernel2::matmul_template<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &kittens::prototype::lcf::kernel<mmt_jit>);
	}}
	)",
            kernel2_trace_define_block(),
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

        if (grouped_kernel_trace_host_enabled()) {
            printf("[kernel2] launch_impl\n");
            printf("  gemm_type=%d c_dtype=%s num_groups=%u\n", args.gemm_type,
                   to_string(args.c_dtype).c_str(), args.num_groups);
            printf("  M=%u N=%u K=%u total_M=%zu total_N=%zu\n",
                   args.M, args.N, args.K, total_M, total_N);
            printf("  bm=%d bn=%d bk=%d super_n=%d stages=%d\n",
                   args.bm, args.bn, args.bk, args.super_n, args.num_stages);
            printf("  consumer_warps=%d producer_warps=%d smem=%d\n",
                   args.num_consumer_warps, args.num_producer_warps,
                   args.smem_size);
            printf("  ptrs A=%p B=%p C=%p scale_a=%p scale_b=%p grouped_layout=%p\n",
                   args.A, args.B, args.C, args.scale_a, args.scale_b,
                   args.grouped_layout);
            printf("  globals_size=%zu\n", gsize);
            tk_dump_grouped_globals(args.bm, args.bn, args.bk, args.gemm_type,
                                    args.c_dtype, globals_buf);
            dump_hex_prefix("  globals_prefix=", globals_buf, gsize);
        }

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
//   D:          (total_M, N)
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
    if (grouped_kernel_trace_host_enabled()) {
        printf("[kernel2] contiguous launch request\n");
        printf("  total_M=%u N=%u K=%u num_groups=%u\n", total_M, N, K, num_groups);
        printf("  block_m=%u block_n=%u block_k=%u num_sms=%u\n",
               gemm_config.block_m, gemm_config.block_n, gemm_config.block_k,
               gemm_config.num_sms);
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

    uint32_t num_groups  = B.size(0);
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
    int super_n = 8;

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
    if (grouped_kernel_trace_host_enabled()) {
        printf("[kernel2] masked launch request\n");
        printf("  max_M=%u N=%u K=%u total_M=%u num_groups=%u\n",
               max_M, N, K, total_M, num_groups);
        printf("  block_m=%u block_n=%u block_k=%u num_sms=%u\n",
               gemm_config.block_m, gemm_config.block_n, gemm_config.block_k,
               gemm_config.num_sms);
    }

    const std::string& code = LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_grouped_gemm_masked", code);
    LaunchRuntime<SM90_FP8_GroupedGEMM_Runtime>::launch(runtime, args);
}
