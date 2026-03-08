#pragma once
#include <runtime/format.hpp>
#include <runtime/device.hpp>
#include <jit/runtime.hpp>
#include <jit/compiler.hpp>
#include <runtime/tensor.h>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/heuristics.hpp>

class SM90_BF16_GEMM_Runtime : LaunchRuntime<SM90_BF16_GEMM_Runtime> {
    public:
        
        struct Args {
            uint32_t num_groups;
            uint32_t M, N, K;
            const std::string& compiled_dims;
            CUtensorMap a_tensor_map;
            CUtensorMap b_tensor_map;
            CUtensorMap d_tensor_map;
            int * grouped_layout;
            bool with_accumulation;
            Major major_a;
            Major major_b;
            GemmConfig gemm_config;
            GemmType gemm_type;
            LaunchConfig launch_config;
            bool dtype_d_float;
        };

        static std::string generate_impl(const Args& args) {
            const std::string code = fmt::format(R"(
            #include <moe_cuda/kernels/sm90_bf16_gemm.cuh>

            using namespace moe_cuda::kernels::sm90_bf16_gemm_impl;

            static void __instantiate_kernel() {{
                
            auto kernel_ptr = reinterpret_cast<void *>(&sm90_bf16_gemm<
            static_cast<Major>({}), static_cast<Major>({}),
            {}, {}, {}, {},
            {}, {}, {},
            {}, {}, {},
            {},
            {}, {},
            {}, {},
            {}, static_cast<GemmType>({}), 
            {}, {}
            >);
            }}
            )", 
            static_cast<int>(args.major_a), static_cast<int>(args.major_b),
            get_compiled_dim(args.compiled_dims, 'm', args.M), get_compiled_dim(args.compiled_dims, 'n', args.N), get_compiled_dim(args.compiled_dims, 'k', args.K),
            args.num_groups,
            args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
            args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
            args.gemm_config.num_stages,
            args.gemm_config.num_tma_threads, args.gemm_config.num_math_threads,
            args.gemm_config.num_tma_multicast, args.gemm_config.tma_multicast_a,
            args.gemm_config.num_sms, static_cast<int>(args.gemm_type), 
            args.with_accumulation, args.dtype_d_float ? "float" : "__nv_bfloat16"
            );
            return code;
        }

        static void launch_impl(KernelHandle& kernel, const LaunchConfigHandle& launch_config, const Args& args) {
            CUDA_CHECK(launch_kernel(kernel, launch_config,
                args.a_tensor_map, args.b_tensor_map, args.d_tensor_map,
                args.M, args.N, args.K,
                args.grouped_layout));
        }
};

// when calling any of these methods, it is assumed sfa and sfb are row-major
inline void sm90_bf16_gemm(at::Tensor& A, at::Tensor& B, std::optional<at::Tensor>& C,
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
        non_contig_A_stride, gemm_config.smem_config.swizzle_a_mode, host_align(gemm_config.block_k, 64));
    CUtensorMap b_tensor_map = make_tma_b_desc(B, major_of(B), 1, gemm_config.block_n, gemm_config.block_k,
         non_contig_B_stride, gemm_config.smem_config.swizzle_b_mode, host_align(gemm_config.block_k, 64));
    CUtensorMap d_tensor_map = make_tma_d_desc(D, major_of(D), 1, gemm_config.block_m, gemm_config.block_n,
         non_contig_D_stride, gemm_config.smem_config.swizzle_cd_mode, host_align(gemm_config.block_n, 64));
    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast
    };
    
    const SM90_BF16_GEMM_Runtime::Args args = {
        1,
        M, N, K,
        compiled_dims,
        a_tensor_map, b_tensor_map, d_tensor_map, nullptr, false, 
        major_of(A), major_of(B),
        gemm_config, GemmType::Normal,
        launch_config, dtype_of(D) == c10::ScalarType::Float
    };
    // set up launch config shape
    const std::string& code = LaunchRuntime<SM90_BF16_GEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_bf16_gemm_normal", code);
    LaunchRuntime<SM90_BF16_GEMM_Runtime>::launch(runtime, args);
}

// for moe implementations, it is more efficient to use a permute operation to group ALL the tokens across ALL sequences under groups
inline void sm90_bf16_grouped_gemm_contiguous(at::Tensor& A, at::Tensor& B,
    at::Tensor& D,  std::string compiled_dims, int * grouped_layout, cudaStream_t&  stream) {
    HOST_ASSERT(grouped_layout != nullptr, "Need a group layout for grouped cases");
    uint32_t M = A.size(0); // == (batch_size * max_seq_length)
    uint32_t N = B.size(-2);
    uint32_t K = B.size(-1);

    uint32_t num_groups = B.size(0);

    GemmConfig gemm_config = search_configs(
        GemmType::MGroupedContiguous, 
        M, N, K, 
        1, // num_groups is already included
        major_of(A), major_of(B), major_of(D),
        dtype_of(A), dtype_of(D), 
        device_prop->get_num_sms());
    
    size_t non_contig_A_stride = major_of(A) == Major::K ? A.stride(0) : A.stride(1);
    size_t non_contig_B_stride = major_of(B) == Major::K ? B.stride(-2) : B.stride(-1);
    size_t non_contig_D_stride = major_of(D) == Major::K ? D.stride(0) : D.stride(1);
    CUtensorMap a_tensor_map = make_tma_a_desc(A, major_of(A), 1, gemm_config.block_m, gemm_config.block_k,
        non_contig_A_stride, gemm_config.smem_config.swizzle_a_mode, host_align(gemm_config.block_k, 64));
        CUtensorMap b_tensor_map = make_tma_b_desc(B, major_of(B), num_groups, gemm_config.block_n, gemm_config.block_k,
            non_contig_B_stride, gemm_config.smem_config.swizzle_b_mode, host_align(gemm_config.block_k, 64));
    CUtensorMap d_tensor_map = make_tma_d_desc(D, major_of(D), 1, gemm_config.block_m, gemm_config.block_n,
            non_contig_D_stride, gemm_config.smem_config.swizzle_cd_mode, host_align(gemm_config.block_n, 64));    
    // uint32_t num_blocks = host_ceil_div(M, gemm_config.block_m) * host_ceil_div(N, gemm_config.block_n);
    // num_blocks = host_align(num_blocks, gemm_config.num_tma_multicast);
    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast
    };
    
      
    const SM90_BF16_GEMM_Runtime::Args args = {
        1,
        M, N, K,
        compiled_dims,
        a_tensor_map, b_tensor_map, d_tensor_map, grouped_layout, false, 
        major_of(A), major_of(B),
        gemm_config, GemmType::MGroupedContiguous,
        launch_config, dtype_of(D) == c10::ScalarType::Float
    };
    // set up launch config shape
    const std::string& code = LaunchRuntime<SM90_BF16_GEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_bf16_grouped_gemm_1d2d_contiguous", code);
    LaunchRuntime<SM90_BF16_GEMM_Runtime>::launch(runtime, args);
}
inline void sm90_bf16_grouped_gemm_masked(at::Tensor& A, at::Tensor& B,
    at::Tensor& D, std::string compiled_dims, int * grouped_layout, cudaStream_t& stream) {
        // input is (num_groups, max_m, k)
        // B is (num_groups, n, k)
        // D is (num_groups, max_m, n)
        HOST_ASSERT(grouped_layout != nullptr, "Need a group layout for grouped cases");
        uint32_t num_groups = A.size(0);
        uint32_t M = A.size(-2); // == (batch_size * max_seq_length)
        uint32_t N = B.size(-2);
        uint32_t K = B.size(-1);
    
    
        GemmConfig gemm_config = search_configs(
            GemmType::MGroupedMasked, 
            M, N, K, 
            num_groups,
            major_of(A), major_of(B), major_of(D),
            dtype_of(A), dtype_of(D), 
            device_prop->get_num_sms());
        
        CUtensorMap a_tensor_map = make_tma_a_desc_3d(A, major_of(A), num_groups, gemm_config.block_m, gemm_config.block_k,
            gemm_config.smem_config.swizzle_a_mode, host_align(gemm_config.block_k, 64));
        CUtensorMap b_tensor_map = make_tma_b_desc_3d(B, major_of(B), num_groups, gemm_config.block_n, gemm_config.block_k,
                gemm_config.smem_config.swizzle_b_mode, host_align(gemm_config.block_k, 64));
        CUtensorMap d_tensor_map = make_tma_d_desc_3d(D, major_of(D), num_groups, gemm_config.block_m, gemm_config.block_n,
             gemm_config.smem_config.swizzle_cd_mode, host_align(gemm_config.block_n, 64));    

        LaunchConfig launch_config = {
            dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
            dim3(gemm_config.num_sms),
            stream,
            gemm_config.smem_config.smem_size,
            gemm_config.num_tma_multicast
        };
        
        const SM90_BF16_GEMM_Runtime::Args args = {
            num_groups,
            M, N, K,
            compiled_dims,
            a_tensor_map, b_tensor_map, d_tensor_map, grouped_layout, false, 
            major_of(A), major_of(B),
            gemm_config, GemmType::MGroupedMasked,
            launch_config, dtype_of(D) == c10::ScalarType::Float
        };
        // set up launch config shape
        const std::string& code = LaunchRuntime<SM90_BF16_GEMM_Runtime>::generate(args);    
        std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_bf16_grouped_gemm_masked", code);
        LaunchRuntime<SM90_BF16_GEMM_Runtime>::launch(runtime, args);
}

// TODO: need to finish implementing

// bmk,nk -> bmn
inline void sm90_bf16_batched_gemm(at::Tensor& A, at::Tensor& B,
    at::Tensor& D, std::string compiled_dims, cudaStream_t& stream) {
    uint32_t BS = A.size(0);
    uint32_t M = A.size(1);
    uint32_t N = B.size(0);
    uint32_t K = B.size(-1);

    GemmConfig gemm_config = search_configs(
        GemmType::Batched, 
        M, N, K, 
        1,
        major_of(A), major_of(B), major_of(D),
        dtype_of(A), dtype_of(D), 
        device_prop->get_num_sms());
    
        
        size_t non_contig_A_stride_0 = A.stride(0);
        size_t non_contig_A_stride_1 = major_of(A) == Major::K ? A.stride(-2) : A.stride(-1);
    size_t non_contig_B_stride_0 = B.stride(0);
    size_t non_contig_B_stride_1 = major_of(B) == Major::K ? B.stride(-2) : B.stride(-1);
    // Major::K is a misnomer here, but it should be interpreted as row-major
    size_t non_contig_D_stride_0 = D.stride(0);
    size_t non_contig_D_stride_1 = D.stride(-2); // output is always K-majoir
    // CUtensorMap a_tensor_map = make_tma_3d_desc(
    //     A, 
    // )

    auto [A_inner, A_outer] = get_inner_outer_dims (major_of(A), gemm_config.block_m, gemm_config.block_k);
    auto [B_inner, B_outer] = get_inner_outer_dims (major_of(B), gemm_config.block_n, gemm_config.block_k);
    CUtensorMap a_tensor_map = make_tma_3d_desc(
        A, K, M, BS, non_contig_A_stride_1, non_contig_A_stride_0, A_inner, A_outer, 1, gemm_config.smem_config.swizzle_a_mode, host_align(A_inner, 64)
    );
    CUtensorMap b_tensor_map = make_tma_3d_desc(
        B, K, N, BS, non_contig_B_stride_1, non_contig_B_stride_0, B_inner, B_outer, 1, gemm_config.smem_config.swizzle_b_mode, host_align(B_inner, 64)
    );

    CUtensorMap d_tensor_map = make_tma_3d_desc(
        D, N, M, BS, non_contig_D_stride_1, non_contig_D_stride_0, gemm_config.block_n, gemm_config.block_m, 1, gemm_config.smem_config.swizzle_cd_mode, host_align(gemm_config.block_n, 64)
    );


    LaunchConfig launch_config = {
        dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
        dim3(gemm_config.num_sms),
        stream,
        gemm_config.smem_config.smem_size,
        gemm_config.num_tma_multicast
    };
    
    const SM90_BF16_GEMM_Runtime::Args args = {
        1,
        M, N, K,
        compiled_dims,
        a_tensor_map, b_tensor_map, d_tensor_map, nullptr, false, 
        major_of(A), major_of(B),
        gemm_config, GemmType::Batched,
        launch_config, dtype_of(D) == c10::ScalarType::Float
    };
    // set up launch config shape
    const std::string& code = LaunchRuntime<SM90_BF16_GEMM_Runtime>::generate(args);
    std::shared_ptr<KernelRuntime> runtime = compiler->build("sm90_bf16_gemm_batched", code);
    LaunchRuntime<SM90_BF16_GEMM_Runtime>::launch(runtime, args);
    }


// write other nt/tt/tn/nt normal gemms here