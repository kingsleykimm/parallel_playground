/**
  @file kernel5_1.hpp
  @brief JIT launcher for kernel5_1 - Fused Dispatch + FC1 of SwiGLU MLP, (Grouped
  GEMM) Uses cooperative grid launch for grid-wide sync between routing and
  compute/comm phases.
 */
 #pragma once
 #include "c10/core/ScalarType.h"
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
 #include <jit_kernels/tk_globals_factory.h>
 #include <pyutils/parallel_tensor.cuh>
 #include <runtime/device.hpp>
 #include <runtime/format.hpp>
 
 class Kernel5_2Runtime : LaunchRuntime<Kernel5_2Runtime> {
 public:
   // kernel5_1 constants (mirrored from kernel5_1.cuh)
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
     kittens::py::TKParallelTensor *out_tokens;
     at::Tensor *expert_y_tokens;
     at::Tensor *expert_y_tokens_scale;
     at::Tensor *comm_comp_barrier;
     at::Tensor *down;
     at::Tensor *scale_down;
     at::Tensor *C;
     at::Tensor *weights;
     at::Tensor *padded_expert_counts;
     at::Tensor *src_token_idx;
     at::Tensor *src_dev_idx;
     at::Tensor *src_slot_idx;
 
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
 
   static std::string generate_impl(const Args &args) {
     return fmt::format(R"(
 
 #include <moe_cuda/kernels/kernel5_2.cuh>
 
 static void __instantiate_kernel() {{
 auto ptr = reinterpret_cast<void *>(&kernel5_2::global_kernel<
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
                        args.M, args.I, args.H, args.BM, args.BN,
                        args.num_consumer_warps, args.num_producer_warps,
                        args.num_stages, args.kernel_smem_size, args.num_experts,
                        args.experts_per_token, args.super_m);
   }
 
   static void launch_impl(KernelHandle &kernel,
                           const LaunchConfigHandle &launch_config,
                           const Args &args) {
     // Build globals via the pre-compiled factory
     size_t gsize = tk_kernel5_2_globals_size(args.H);
     alignas(128) char globals_buf[4096];
     assert(gsize <= sizeof(globals_buf));
     
     tk_build_kernel5_2_globals(
        args.H, globals_buf, *args.out_tokens, *args.expert_y_tokens,
         *args.expert_y_tokens_scale, *args.comm_comp_barrier, *args.down,
          *args.scale_down, *args.C, *args.weights, *args.padded_expert_counts,
           *args.src_token_idx, *args.src_dev_idx, *args.src_slot_idx,
            args.num_recv_tokens, args.dp_rank, args.rank, args.dp_size,
             args.cur_dp_group, args.num_dp_groups, args.num_comm_sms, args.num_comp_sms);
      
     // kernel5_1 uses cooperative_groups::this_grid().sync(), so we need
     // cooperative launch via CU_LAUNCH_ATTRIBUTE_COOPERATIVE
 
     void *kernelParams[] = {globals_buf};
     if (get_env<int>("JIT_DEBUG") != 0) {
       printf("Launching kernel5_2: grid=%u, block=%u, smem=%d, cooperative=1\n",
              launch_config.gridDimX, launch_config.blockDimX,
              launch_config.sharedMemBytes);
     }
     CUDA_CHECK(cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr));
     if (get_env<int>("JIT_DEBUG") != 0) {
       CUDA_CHECK(cudaStreamSynchronize(args.launch_config.stream));
     }
   }
 };
 
 inline void fused_grouped_gemm_combine(
    kittens::py::TKParallelTensor &out_tokens,
    at::Tensor &expert_y_tokens, at::Tensor &expert_y_tokens_scale,
    at::Tensor &down, at::Tensor &scale_down,
    at::Tensor &C, at::Tensor &weights, at::Tensor &padded_expert_counts,
    at::Tensor &src_token_idx,
    at::Tensor &src_dev_idx, at::Tensor &src_slot_idx, int num_experts, int experts_per_token,
    int *num_recv_tokens,
    int dp_rank, int rank, int dp_size, int cur_dp_group, int num_dp_groups,
    int num_comm_sms, int num_comp_sms, cudaStream_t &stream) {
   // for persistent case
   HOST_ASSERT(
       num_comm_sms + num_comp_sms <= 132,
       "num_comm_sms + num_comp_sms must be less than or equal to SM_COUNT");
   uint32_t M = expert_y_tokens.size(0);
   uint32_t I = expert_y_tokens.size(1);
   uint32_t H = C.size(-1);
 
   int total_sms = num_comm_sms + num_comp_sms;

   int num_experts_per_dev = C.size(0);
   auto gemm_config = search_configs(
    GemmType::MGroupedMasked, M, H, I, num_experts_per_dev, Major::K, Major::K, Major::K, torch::kFloat8_e4m3fn, torch::kBFloat16, total_sms
   );
 
   uint32_t BM = gemm_config.block_m;
   uint32_t BN = gemm_config.block_n;
   uint32_t num_consumer_warps = gemm_config.num_math_threads / 32;
   uint32_t num_producer_warps = gemm_config.num_tma_threads / 32;
   uint32_t num_stages = gemm_config.num_stages;
   uint32_t kernel_smem_size = gemm_config.smem_config.smem_size;
   uint32_t super_m = 1;
 
   at::Tensor comm_comp_barrier = at::zeros(std::vector<int64_t>{host_ceil_div(src_token_idx.size(0), BM)}, at::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
 
   LaunchConfig launch_config = {
       dim3(gemm_config.num_math_threads + gemm_config.num_tma_threads, 1, 1),
       dim3(total_sms), stream, (int)kernel_smem_size,
       1, // num_multicast (not used, cooperative handles grid sync),
       true // cooperative launch
   };
 
   if (get_env<int>("JIT_DEBUG") > 0) {
     printf("fused_dispatch_grouped_gemm_swiglu:\n");
     printf("  M=%u I=%u H=%u num_experts=%d experts_per_token=%d\n", M, I, H,
            num_experts, experts_per_token);
     printf("  BM=%u BN=%u stages=%u super_m=%u\n", BM, BN, num_stages, super_m);
     printf("  consumer_warps=%u producer_warps=%u\n", num_consumer_warps,
            num_producer_warps);
     printf("  num_comm_sms=%d num_comp_sms=%d total_sms=%d\n", num_comm_sms,
            num_comp_sms, total_sms);
     printf("  smem_size=%u\n", kernel_smem_size);
   }
 
   const Kernel5_2Runtime::Args args = {
       .M = M,
       .I = I,
       .H = H,
       .BM = BM,
       .BN = BN,
       .num_consumer_warps = num_consumer_warps,
       .num_producer_warps = num_producer_warps,
       .num_stages = num_stages,
       .kernel_smem_size = kernel_smem_size,
       .num_experts = (uint32_t)num_experts,
       .experts_per_token = (uint32_t)experts_per_token,
       .super_m = super_m,
       .out_tokens = &out_tokens,
       .expert_y_tokens = &expert_y_tokens,
       .expert_y_tokens_scale = &expert_y_tokens_scale,
       .comm_comp_barrier = &comm_comp_barrier,
       .down = &down,
       .scale_down = &scale_down,
       .C = &C,
       .weights = &weights,
       .padded_expert_counts = &padded_expert_counts,
       .src_token_idx = &src_token_idx,
       .src_dev_idx = &src_dev_idx,
       .src_slot_idx = &src_slot_idx,
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
 
   const std::string &code = LaunchRuntime<Kernel5_2Runtime>::generate(args);
   std::shared_ptr<KernelRuntime> runtime =
       compiler->build("fused_grouped_gemm_combine", code);
   LaunchRuntime<Kernel5_2Runtime>::launch(runtime, args);
 }
 