#pragma once
#include <runtime/format.hpp>
#include <runtime/device.hpp>
#include <jit/runtime.hpp>
#include <jit/compiler.hpp>
#include <runtime/tensor.h>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/heuristics.hpp>

class SM90_FP8_GEMM1D2D_Runtime : LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime> {
    public:
        
        struct Args {
            uint32_t num_groups;
            uint32_t M, N, K;
            const std::string& compiled_dims;
            Major sfbMajor;
            CUtensorMap a_tensor_map;
            CUtensorMap b_tensor_map;
            CUtensorMap d_tensor_map;
            CUtensorMap sfa_tensor_map;
            float * sfb;
            int * grouped_layout;
            GemmConfig gemm_config;
            GemmType gemm_type;
            LaunchConfig launch_config;
        };

        static std::string generate_impl(const Args& args) {
            const std::string code = fmt::format(R"(
            #include <moe_cuda/kernels/sm90_fp8_gemm_1d2d.cuh>

            using namespace moe_cuda::kernels::sm90_fp8_gemm_impl;

            static void __instantiate_kernel() {{
                
            auto kernel_ptr = reinterpret_cast<void *>(&sm90_fp8_gemm_1d2d<
            {},
            {}, {}, {},
            {}, {}, {},
            static_cast<Major>({}),
            {}, {},
            {}, {},
            static_cast<GemmType>({}), {},
            {}, {}, {},
            {}, {}
            >);
            }}
            )", 
            args.num_groups,
            get_compiled_dim(args.compiled_dims, 'm', args.M), get_compiled_dim(args.compiled_dims, 'n', args.N), get_compiled_dim(args.compiled_dims, 'k', args.K),
            args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
            static_cast<int>(args.sfbMajor),
            args.gemm_config.num_math_threads, args.gemm_config.num_tma_threads,
            args.gemm_config.tma_multicast_a, args.gemm_config.num_tma_multicast,
            static_cast<int>(args.gemm_type), args.gemm_config.num_sms,
            args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
            args.gemm_config.num_stages, args.gemm_type == GemmType::Batched ? 1 : 0
            );
            return code;
        }

        static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config, const Args& args) {
            CUDA_CHECK(launch_kernel(kernel, launch_config,
                args.M, args.N, args.K,
                args.a_tensor_map, args.sfa_tensor_map, args.b_tensor_map, args.d_tensor_map,
                args.sfb, args.grouped_layout));
        }
};

// when calling any of these methods, it is assumed sfa and sfb are row-major
inline void sm90_fp8_gemm_1d2d(at::Tensor& A, at::Tensor& B, at::Tensor& sfa, at::Tensor & sfb,
at::Tensor& D, const std::string& compiled_dims, cudaStream_t& stream) {
    
    // for grouped gemms:
    // the expert weight will always have the shape (groups, N, K)
    // for MGroupedContiguous : the expert tokens are concatenated into a single tensor, in blocks of 128 (GEMM M Block size)
    // for MGroupedMasked : the expert tokens are grouped into blocked chunks, for the CUDA graph, so they are fied sizes
    // for Batched : each batch is a group
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = B.size(-1);

    GemmConfig gemm_config = search_configs(
        GemmType::Normal, 
        M, N, K, 
        1,
        major_of(A), major_of(B), major_of(D),
        dtype_of(A), dtype_of(D), 
        device_prop->get_num_sms());
    
    
    size_t non_contig_A_stride = major_of(A) == Major::K ? A.stride(-2) : A.stride(-1);
    size_t non_contig_B_stride = major_of(B) == Major::K ? B.stride(-2) : B.stride(-1);
    // Major::K is a misnomer here, but it should be interpreted as row-major
    size_t non_contig_D_stride = major_of(D) == Major::K ? D.stride(-2) : D.stride(-1);
    CUtensorMap a_tensor_map = make_tma_a_desc(A, major_of(A), 1, gemm_config.block_m, gemm_config.block_k, 
        non_contig_A_stride, gemm_config.smem_config.swizzle_a_mode, ti_align(gemm_config.block_k, 64));
    CUtensorMap b_tensor_map = make_tma_b_desc(B, major_of(B), 1, gemm_config.block_n, gemm_config.block_k,
         non_contig_B_stride, gemm_config.smem_config.swizzle_b_mode, ti_align(gemm_config.block_k, 64));
    CUtensorMap d_tensor_map = make_tma_d_desc(D, major_of(D), 1, gemm_config.block_m, gemm_config.block_n,
         non_contig_D_stride, gemm_config.smem_config.swizzle_cd_mode, ti_align(gemm_config.block_n, 64));
    CUtensorMap sfa_tensor_map = make_tma_sf_desc(sfa, major_of(sfa), 1, M, K, gemm_config.block_m, gemm_config.block_k);
    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast
    };
    
    const SM90_FP8_GEMM1D2D_Runtime::Args args = {
        1,
        M, N, K,
        compiled_dims,
        major_of(sfb),
        a_tensor_map, b_tensor_map, d_tensor_map, sfa_tensor_map,
        sfb.data_ptr<float>(), nullptr, 
        gemm_config, GemmType::Normal,
        launch_config,
    };
    // set up launch config shape
    const std::string& code = LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_gemm_1d2d_normal", code);
    LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::launch(runtime, args);
}

// for moe implementations, it is more efficient to use a permute operation to group ALL the tokens across ALL sequences under groups
inline void sm90_fp8_grouped_gemm_1d2d_contiguous(at::Tensor& A, at::Tensor& B, at::Tensor& sfa, at::Tensor& sfb,
    at::Tensor& D, std::string compiled_dims, int * grouped_layout, cudaStream_t&  stream) {
    HOST_ASSERT(grouped_layout != nullptr, "Need a group layout for grouped cases");
    uint32_t M = A.size(0); // == (batch_size * max_seq_length)
    uint32_t N = B.size(-2);
    uint32_t K = B.size(-1);

    const uint32_t aligned_k = ti_align(K, 128);

    uint32_t num_groups = B.size(0);

    GemmConfig gemm_config = search_configs(
        GemmType::MGroupedContiguous, 
        M, N, K, 
        1, // num_groups is already included
        major_of(A), major_of(B), major_of(D),
        dtype_of(A), dtype_of(D), 
        device_prop->get_num_sms());
    
    size_t non_contig_A_stride = major_of(A) == Major::K ? A.stride(0) : A.stride(1);
    // B is 3D (num_groups, N, K): stride(-2) = stride(1) = K gives the row stride,
    // NOT stride(0) = N*K which is the group stride and would cause illegal TMA addresses
    // Major::K is a misnomer here, but it should be interpreted as row-major
    size_t non_contig_B_stride = major_of(B) == Major::K ? B.stride(-2) : B.stride(-1);
    size_t non_contig_D_stride = major_of(D) == Major::K ? D.stride(-2) : D.stride(-1);
    CUtensorMap a_tensor_map = make_tma_a_desc(A, major_of(A), 1, gemm_config.block_m, gemm_config.block_k,
        non_contig_A_stride, gemm_config.smem_config.swizzle_a_mode, ti_align(gemm_config.block_k, 64));
    CUtensorMap b_tensor_map = make_tma_b_desc(B, major_of(B), num_groups, gemm_config.block_n, gemm_config.block_k,
            non_contig_B_stride, gemm_config.smem_config.swizzle_b_mode, ti_align(gemm_config.block_k, 64));
    CUtensorMap d_tensor_map = make_tma_d_desc(D, major_of(D), 1, gemm_config.block_m, gemm_config.block_n,
            non_contig_D_stride, gemm_config.smem_config.swizzle_cd_mode, ti_align(gemm_config.block_n, 64));    
    CUtensorMap sfa_tensor_map = make_tma_sf_desc(sfa, major_of(sfa), 1, M, K, gemm_config.block_m, gemm_config.block_k);
    // uint32_t num_blocks = ti_ceil_div(M, gemm_config.block_m) * ti_ceil_div(N, gemm_config.block_n);
    // num_blocks = ti_align(num_blocks, gemm_config.num_tma_multicast);
    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast
    };
    
    const SM90_FP8_GEMM1D2D_Runtime::Args args = {
        1,
        M, N, aligned_k,
        compiled_dims,
        major_of(sfb),
        a_tensor_map, b_tensor_map, d_tensor_map, sfa_tensor_map,
        sfb.data_ptr<float>(), grouped_layout, 
        gemm_config, GemmType::MGroupedContiguous,
        launch_config
    };
    // set up launch config shape
    const std::string& code = LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_grouped_gemm_1d2d_contiguous", code);
    LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::launch(runtime, args);
}
inline void sm90_fp8_grouped_gemm_1d2d_masked(at::Tensor& A, at::Tensor& B, at::Tensor& sfa, at::Tensor& sfb,
    at::Tensor& D, std::string compiled_dims, int * grouped_layout, cudaStream_t& stream) {
        // input is (num_groups, max_m, k)
        // B is (num_groups, n, k)
        // D is (num_groups, max_m, n)
        HOST_ASSERT(grouped_layout != nullptr, "Need a group layout for grouped cases");
        uint32_t num_groups = A.size(0);
        uint32_t M = A.size(-2); // == (batch_size * max_seq_length)
        uint32_t N = B.size(-2);
        uint32_t K = B.size(-1);
    
        const uint32_t & aligned_k = ti_align(K, 128);
    
        GemmConfig gemm_config = search_configs(
            GemmType::MGroupedMasked, 
            M, N, K, 
            num_groups,
            major_of(A), major_of(B), major_of(D),
            dtype_of(A), dtype_of(D), 
            device_prop->get_num_sms());
        
        
        CUtensorMap a_tensor_map = make_tma_a_desc_3d(A, major_of(A), num_groups, gemm_config.block_m, gemm_config.block_k,
            gemm_config.smem_config.swizzle_a_mode, ti_align(gemm_config.block_k, 64));
        CUtensorMap b_tensor_map = make_tma_b_desc_3d(B, major_of(B), num_groups, gemm_config.block_n, gemm_config.block_k,
                gemm_config.smem_config.swizzle_b_mode, ti_align(gemm_config.block_k, 64));
        CUtensorMap d_tensor_map = make_tma_d_desc_3d(D, major_of(D), num_groups, gemm_config.block_m, gemm_config.block_n,
             gemm_config.smem_config.swizzle_cd_mode, ti_align(gemm_config.block_n, 64));    
        CUtensorMap sfa_tensor_map = make_tma_sf_desc(sfa, major_of(sfa), num_groups, M, K, gemm_config.block_m, gemm_config.block_k);

        LaunchConfig launch_config = {
            dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
            dim3(gemm_config.num_sms),
            stream,
            gemm_config.smem_config.smem_size,
            gemm_config.num_tma_multicast
        };
        
        const SM90_FP8_GEMM1D2D_Runtime::Args args = {
            num_groups,
            M, N, aligned_k,
            compiled_dims,
            major_of(sfb),
            a_tensor_map, b_tensor_map, d_tensor_map, sfa_tensor_map,
            sfb.data_ptr<float>(), grouped_layout, 
            gemm_config, GemmType::MGroupedMasked,
            launch_config
        };
        // set up launch config shape
        const std::string& code = LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::generate(args);    
        std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_fp8_grouped_gemm_1d2d_masked", code);
        LaunchRuntime<SM90_FP8_GEMM1D2D_Runtime>::launch(runtime, args);
}

