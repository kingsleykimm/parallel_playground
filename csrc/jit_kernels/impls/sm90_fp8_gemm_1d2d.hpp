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

namespace {

inline std::string fp8_grouped_ref_trace_define_block() {
    return fmt::format(
        "#define KERNEL2_TRACE_REF {}\n"
        "#define KERNEL2_TRACE_REF_CTA_LIMIT {}\n"
        "#define KERNEL2_TRACE_REF_ITER_LIMIT {}\n",
        get_env<int>("KERNEL2_TRACE_REF", 0),
        get_env<int>("KERNEL2_TRACE_REF_CTA_LIMIT", 8),
        get_env<int>("KERNEL2_TRACE_REF_ITER_LIMIT", 2));
}

inline void fill_fp8_ref_swizzles(
    GemmConfig& gemm_config,
    const at::Tensor& ab_tensor,
    const at::Tensor& d_tensor) {
    gemm_config.smem_config.swizzle_a_mode =
        get_swizzle_mode(gemm_config.block_k, get_type_size(dtype_of(ab_tensor)));
    gemm_config.smem_config.swizzle_b_mode =
        get_swizzle_mode(gemm_config.block_k, get_type_size(dtype_of(ab_tensor)));
    gemm_config.smem_config.swizzle_cd_mode =
        get_swizzle_mode(gemm_config.block_n, get_type_size(dtype_of(d_tensor)));
}

} // namespace

class SM90_FP8_GEMM1D2D_TK_Runtime : LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime> {
public:
    struct Args {
        uint32_t M, N, K;
        void *A, *B, *C, *scale_a, *scale_b;
        int bm, bn, bk, super_m;
        int num_consumer_warps, num_producer_warps;
        int num_stages;
        int smem_size;
        at::ScalarType c_dtype;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        // JIT code uses template instantiation — no #defines needed.
        // NVRTC compiles this to produce a cubin with the kernel entry point.
        return fmt::format(R"(
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>

using mmt_jit = matmul_template<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &kittens::prototype::lcf::kernel<mmt_jit>);
}}
)", args.M, 
args.N,
 args.K,
  args.bm, 
  args.bn,
   args.bk, 
   args.num_consumer_warps, 
   args.num_producer_warps, args.num_stages, 
args.smem_size,
to_string(args.c_dtype),
args.super_m);
    }

    static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config, const Args& args) {
        // Build globals via pre-compiled factory
        size_t gsize = tk_globals_size(args.bm, args.bn, args.bk, args.c_dtype);
        alignas(128) char globals_buf[2048];
        assert(gsize <= sizeof(globals_buf));
        // copies the tk factory struct (containing matmul_layout::globals) into globals_buf
        tk_build_globals(args.bm, args.bn, args.bk, args.c_dtype, globals_buf,
            args.A, args.B, args.C, args.scale_a, args.scale_b,
            args.M, args.N, args.K);

        // Launch via cuLaunchKernelEx with globals as kernelParams[0] - this is the only argument required for LCF
	        void* kernelParams[] = { globals_buf };
	        CUDA_CHECK(cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr));
	    }
};

class SM90_FP8_GEMM1D2D_Ref_Runtime
    : LaunchRuntime<SM90_FP8_GEMM1D2D_Ref_Runtime> {
public:
    struct Args {
        uint32_t num_groups;
        uint32_t M, N, K;
        CUtensorMap a_tensor_map;
        CUtensorMap sfa_tensor_map;
        CUtensorMap b_tensor_map;
        CUtensorMap d_tensor_map;
        float* sfb;
        int* grouped_layout;
        GemmConfig gemm_config;
        GemmType gemm_type;
        LaunchConfig launch_config;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
{}
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>

using namespace moe_cuda::kernels::sm90_fp8_gemm_impl;

static void __instantiate_kernel() {{
    auto kernel_ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d<
        {},
        {}, {}, {},
        {}, {}, {},
        static_cast<Major>({}),
        {}, {},
        {},
        {},
        static_cast<GemmType>({}),
        {},
        {}, {}, {},
        {},
        false>);
}}
)",
            fp8_grouped_ref_trace_define_block(),
            args.num_groups,
            args.M, args.N, args.K,
            args.gemm_config.block_m, args.gemm_config.block_n,
            args.gemm_config.block_k,
            static_cast<int>(Major::K),
            args.gemm_config.num_math_threads, args.gemm_config.num_tma_threads,
            args.gemm_config.tma_multicast_a ? "true" : "false",
            args.gemm_config.num_tma_multicast,
            static_cast<int>(args.gemm_type),
            args.gemm_config.num_sms,
            args.gemm_config.smem_config.swizzle_a_mode,
            args.gemm_config.smem_config.swizzle_b_mode,
            args.gemm_config.smem_config.swizzle_cd_mode,
            args.gemm_config.num_stages);
    }

    static void launch_impl(KernelHandle& kernel,
                            const LaunchConfigHandle& launch_config,
                            const Args& args) {
        CUDA_CHECK(launch_kernel(kernel, launch_config,
                                 args.M, args.N, args.K,
                                 args.a_tensor_map, args.sfa_tensor_map,
                                 args.b_tensor_map, args.d_tensor_map,
                                 args.sfb, args.grouped_layout));
    }
};


// persistent kernel style
inline void sm90_fp8_gemm_1d2d_nt(at::Tensor& A, at::Tensor& B,
    at::Tensor& scale_a, at::Tensor& scale_b,
    at::Tensor& D, cudaStream_t& stream) {

    HOST_ASSERT(D.scalar_type() == at::ScalarType::BFloat16 || D.scalar_type() == at::ScalarType::Float, "unsupported output dtype");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = B.size(1);

    // Default tile config
    // TK lcf kernel uses (NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * 32 threads
    // NUM_CONSUMER_WARPS=8, NUM_PRODUCER_WARPS=1 (default) => 9 warps => 288 threads
    // MAX_SHARED_MEMORY - 1024 (same as the standalone TK kernel)

    auto gemm_config = search_configs(GemmType::Normal, M, N, K, 1, Major::K, Major::K, Major::K, A.scalar_type(), 
        D.scalar_type(), device_prop->get_num_sms());
    
    int num_consumer_warps = gemm_config.num_math_threads / 32;
    int num_producer_warps = gemm_config.num_tma_threads / 32;


    int super_m = 8; // set to this for now , DG uses 8 or 16
        // we'll do always persistent grid for now
    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(device_prop->get_num_sms()),
        stream,
        gemm_config.smem_config.smem_size,
        1  // no multicast/clustering
    };



    const SM90_FP8_GEMM1D2D_TK_Runtime::Args args = {
        M, N, K,
        A.data_ptr(), B.data_ptr(), D.data_ptr(),
        scale_a.data_ptr(), scale_b.data_ptr(),
        (int)gemm_config.block_m, (int)gemm_config.block_n, (int)gemm_config.block_k, super_m,
        num_consumer_warps, num_producer_warps, gemm_config.num_stages,
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

    const std::string& code = LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_gemm_1d2d_tk", code);
    LaunchRuntime<SM90_FP8_GEMM1D2D_TK_Runtime>::launch(runtime, args);
}

inline void sm90_fp8_grouped_gemm_1d2d_contiguous_ref(
    at::Tensor& A, at::Tensor& B,
    at::Tensor& scale_a, at::Tensor& scale_b,
    at::Tensor& D, int* grouped_layout, cudaStream_t& stream) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped FP8 GEMM");
    HOST_ASSERT(D.scalar_type() == at::ScalarType::BFloat16 ||
                    D.scalar_type() == at::ScalarType::Float,
                "unsupported output dtype");

    uint32_t total_M = A.size(0);
    uint32_t num_groups = B.size(0);
    uint32_t N = B.size(-2);
    uint32_t K = B.size(-1);

    GemmConfig gemm_config = search_configs(
        GemmType::MGroupedContiguous, total_M, N, K, 1, Major::K, Major::K,
        Major::K, A.scalar_type(), D.scalar_type(), device_prop->get_num_sms());
    fill_fp8_ref_swizzles(gemm_config, A, D);

    size_t non_contig_A_stride = A.stride(-2);
    size_t non_contig_B_stride = B.stride(-2);
    size_t non_contig_D_stride = D.stride(-2);

    CUtensorMap a_tensor_map = make_tma_a_desc(
        A, Major::K, 1, gemm_config.block_m, gemm_config.block_k,
        non_contig_A_stride, gemm_config.smem_config.swizzle_a_mode,
        host_align(gemm_config.block_k, 64));
    CUtensorMap sfa_tensor_map = make_tma_sf_desc(
        scale_a, Major::MN, 1, total_M, K, gemm_config.block_m,
        gemm_config.block_k);
    CUtensorMap b_tensor_map = make_tma_b_desc(
        B, Major::K, num_groups, gemm_config.block_n, gemm_config.block_k,
        non_contig_B_stride, gemm_config.smem_config.swizzle_b_mode,
        host_align(gemm_config.block_k, 64));
    CUtensorMap d_tensor_map = make_tma_d_desc(
        D, Major::K, 1, gemm_config.block_m, gemm_config.block_n,
        non_contig_D_stride, gemm_config.smem_config.swizzle_cd_mode,
        host_align(gemm_config.block_n, 64));

    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast,
    };

    const SM90_FP8_GEMM1D2D_Ref_Runtime::Args args = {
        num_groups,
        total_M, N, K,
        a_tensor_map,
        sfa_tensor_map,
        b_tensor_map,
        d_tensor_map,
        scale_b.data_ptr<float>(),
        grouped_layout,
        gemm_config,
        GemmType::MGroupedContiguous,
        launch_config,
    };

    const std::string& code =
        LaunchRuntime<SM90_FP8_GEMM1D2D_Ref_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build(
        "sm90_fp8_grouped_gemm_1d2d_ref_contiguous", code);
    LaunchRuntime<SM90_FP8_GEMM1D2D_Ref_Runtime>::launch(runtime, args);
}

inline void sm90_fp8_grouped_gemm_1d2d_masked_ref(
    at::Tensor& A, at::Tensor& B,
    at::Tensor& scale_a, at::Tensor& scale_b,
    at::Tensor& D, int* grouped_layout, cudaStream_t& stream) {
    (void)A;
    (void)B;
    (void)scale_a;
    (void)scale_b;
    (void)D;
    (void)grouped_layout;
    (void)stream;
    HOST_ASSERT(false, "masked FP8 grouped reference tracing is not implemented");
}
