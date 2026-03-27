/**
   @brief: CUDA source file to link gcc defined apis with kernel calls
 */

// ================================================
// JIT launcher (gcc) APIs
// ================================================
#include <apis/moe.hpp>
#include <jit_kernels/impls/kernel5_1.hpp>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>

// ================================================
// Kernel APIs
// ================================================
#include <ATen/core/TensorBase.h>
#include <all2all/all2all_base.hpp>
#include <all2all/all2all_tk.hpp>
#include <kernels/cast.cuh>
#include <kernels/internal_api.hpp>
#include <torch/csrc/stable/ops.h>

namespace moe_cuda {
namespace kernels {

void cast_dispatch(torch::stable::Tensor &inp, torch::stable::Tensor &out,
                   std::optional<torch::stable::Tensor> &scale,
                   cudaStream_t stream) {
  cast_impl::cast_(inp, out, scale, stream);
}

void fused_silu_mul_quant(torch::stable::Tensor &gemm_out,
                          torch::stable::Tensor &swiglu_out,
                          torch::stable::Tensor &scale, cudaStream_t stream) {
  size_t num_rows = 1;
  for (int i = 0; i < gemm_out.dim() - 1; ++i) {
    num_rows *= gemm_out.size(i);
  }
  const size_t hidden_dim = gemm_out.size(-1) / 2;
  cast_impl::fused_silu_mul_quant(
      reinterpret_cast<__nv_bfloat16 *>(gemm_out.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3 *>(swiglu_out.data_ptr()),
      (float *)scale.data_ptr(), num_rows, hidden_dim, stream);
}

void fp8_grouped_gemm_swiglu(
    torch::stable::Tensor &A, torch::stable::Tensor &gate_weight,
    torch::stable::Tensor &up_weight, torch::stable::Tensor &scale_a,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
    GemmType gemm_type, int *grouped_layout, cudaStream_t &stream) {
  api::fp8_grouped_gemm_swiglu(A, gate_weight, up_weight, scale_a, scale_gate,
                               scale_up, scale_d, D, gemm_type, grouped_layout,
                               stream);
}

void fp8_grouped_gemm_swiglu_consumer_pp(
    torch::stable::Tensor &A, torch::stable::Tensor &gate_weight,
    torch::stable::Tensor &up_weight, torch::stable::Tensor &scale_a,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
    GemmType gemm_type, int *grouped_layout, cudaStream_t &stream) {
  api::fp8_grouped_gemm_swiglu_consumer_pp(A, gate_weight, up_weight, scale_a,
                                           scale_gate, scale_up, scale_d, D,
                                           gemm_type, grouped_layout, stream);
};

void fused_dispatch_grouped_gemm_swiglu(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_tokens_scales,
    torch::stable::Tensor &expert_x_tokens,
    torch::stable::Tensor &expert_x_tokens_scale, torch::stable::Tensor &gate,
    torch::stable::Tensor &up, torch::stable::Tensor &C,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &out_scales, torch::stable::Tensor &indices,
    kittens::py::TKParallelTensor &global_num_routed,
    kittens::py::TKParallelTensor &expert_to_token_map,
    torch::stable::Tensor &padded_expert_counts,
    torch::stable::Tensor &src_token_idx, torch::stable::Tensor &src_dev_idx,
    kittens::py::TKParallelTensor &barrier, int num_tokens,
    int *num_recv_tokens, int dp_rank, int rank, int dp_size, int cur_dp_group,
    int num_dp_groups, int world_size, int num_experts, int experts_per_token,
    int num_comm_sms, int num_comp_sms, cudaStream_t &stream) {
  ::fused_dispatch_grouped_gemm_swiglu(
      in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale, gate,
      up, C, scale_gate, scale_up, out_scales, indices, global_num_routed,
      expert_to_token_map, padded_expert_counts, src_token_idx, src_dev_idx,
      barrier, num_tokens, num_recv_tokens, dp_rank, rank, dp_size,
      cur_dp_group, num_dp_groups, world_size, num_experts, experts_per_token,
      num_comm_sms, num_comp_sms, stream);
}

void fp8_gemm_nt(torch::stable::Tensor &act, torch::stable::Tensor &act_scale,
                 torch::stable::Tensor &weight,
                 torch::stable::Tensor &weight_scale,
                 torch::stable::Tensor &output,
                 const std::string &compiled_dims, cudaStream_t &stream) {
  api::fp8_gemm_nt(act, act_scale, weight, weight_scale, output, compiled_dims,
                   stream);
}

void fp8_grouped_gemm_nt(torch::stable::Tensor &act,
                         torch::stable::Tensor &act_scale,
                         torch::stable::Tensor &weight,
                         torch::stable::Tensor &weight_scale,
                         torch::stable::Tensor &output, GemmType gemm_type,
                         int *grouped_layout, cudaStream_t &stream) {
  api::fp8_grouped_gemm(act, act_scale, weight, weight_scale, output, gemm_type,
                        grouped_layout, stream);
}

void bf16_gemm(torch::stable::Tensor &A, torch::stable::Tensor &B,
               std::optional<torch::stable::Tensor> &C,
               torch::stable::Tensor &D, GemmType gemm_type,
               const std::string &compiled_dims, int *grouped_layout,
               cudaStream_t &stream) {
  api::bf16_gemm(A, B, C, D, gemm_type, compiled_dims, grouped_layout, stream);
}

// this uses the All2All struct, which manages all the a2a information and
// kernel launches, and connects it with the fused swiglu matmuls
// this does not perform any fusing between the dispatch + swiglu (up) and
// combine + down, and it performs routing calculation in the host side, so
// expectedly the performance is worse
template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
void naive_moe_forward(
    All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &a2a, GemmType gemm_type,
    torch::stable::Tensor &input, torch::stable::Tensor &input_scales,
    torch::stable::Tensor &gate, torch::stable::Tensor &gate_scales,
    torch::stable::Tensor &up, torch::stable::Tensor &up_scales,
    torch::stable::Tensor &down, torch::stable::Tensor &down_scales,
    torch::stable::Tensor &indices, torch::stable::Tensor &weights,
    torch::stable::Tensor &out_tokens, torch::stable::Tensor &expert_x,
    torch::stable::Tensor &expert_x_scales, torch::stable::Tensor &inter_y,
    torch::stable::Tensor &inter_y_scales, torch::stable::Tensor &expert_y,
    cudaStream_t stream) {

  uint32_t max_recv_tokens = a2a.max_recv_tokens();
  at::TensorOptions options = at::TensorOptions().device(torch::kCUDA);
  torch::stable::Tensor out_expert_num_tokens =
      at::empty(std::vector<int64_t>{a2a.num_experts_per_rank()},
                options.dtype(torch::kUInt32));

  int H = input.size(-1);
  int I = gate.size(-2);

  // to maintain consistency with opt api
  std::optional<torch::stable::Tensor> input_scales_opt = input_scales;
  std::optional<torch::stable::Tensor> expert_x_scales_opt = expert_x_scales;
  std::optional<torch::stable::Tensor> bound_m = std::nullopt;
  a2a.dispatch(out_expert_num_tokens, expert_x, expert_x_scales_opt, input,
               input_scales_opt, indices, weights, bound_m, true, true, stream);

  int *grouped_layout = gemm_type == GemmType::MGroupedMasked
                            ? (int *)(a2a.context().worker.padded_offsets)
                            : (int *)(a2a.context().worker.tokens_per_expert);
  fp8_grouped_gemm_swiglu(expert_x, gate, up, expert_x_scales, gate_scales,
                          up_scales, inter_y_scales, inter_y, gemm_type,
                          grouped_layout, stream);
  fp8_grouped_gemm_nt(inter_y, inter_y_scales, down, down_scales, expert_y,
                      gemm_type, grouped_layout, stream);
  a2a.combine(out_tokens, indices, weights, expert_y, bound_m, true, true,
              stream);
}

} // namespace kernels

std::shared_ptr<All2AllBase>
make_all2all(uint32_t max_num_tokens, uint32_t num_experts,
             uint32_t experts_per_token, uint32_t expert_padding,
             uint32_t hidden_dim, std::optional<uint32_t> hidden_dim_scale,
             c10::ScalarType in_dtype, c10::ScalarType out_dtype,
             std::optional<c10::ScalarType> scale_dtype,
             std::optional<uint32_t> max_private_tokens_opt, int local_rank,
             ParallelConfig parallel_config, cudaStream_t stream) {
  LAUNCH_NUM_EXPERTS(num_experts, NUM_EXPERTS, {
    LAUNCH_NUM_EXPERTS_PER_TOKEN(
        experts_per_token, EXPERTS_PER_TOKEN,
        {LAUNCH_TOKEN_DIM(hidden_dim, TOKEN_DIM, {
          return std::make_shared<
              All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>>(
              max_num_tokens, num_experts, expert_padding, hidden_dim,
              hidden_dim_scale, in_dtype, out_dtype, scale_dtype,
              experts_per_token, max_private_tokens_opt, local_rank,
              parallel_config, stream);
        })});
  });
}
void naive_moe_forward_dispatch(
    All2AllBase &a2a_base, uint32_t num_experts, uint32_t experts_per_token,
    uint32_t hidden_dim, GemmType gemm_type, torch::stable::Tensor &input,
    torch::stable::Tensor &input_scales, torch::stable::Tensor &gate,
    torch::stable::Tensor &gate_scales, torch::stable::Tensor &up,
    torch::stable::Tensor &up_scales, torch::stable::Tensor &down,
    torch::stable::Tensor &down_scales, torch::stable::Tensor &indices,
    torch::stable::Tensor &weights, torch::stable::Tensor &out_tokens,
    torch::stable::Tensor &expert_x, torch::stable::Tensor &expert_x_scales,
    torch::stable::Tensor &inter_y, torch::stable::Tensor &inter_y_scales,
    torch::stable::Tensor &expert_y, cudaStream_t stream) {
  LAUNCH_NUM_EXPERTS(num_experts, NUM_EXPERTS, {
    LAUNCH_NUM_EXPERTS_PER_TOKEN(experts_per_token, EXPERTS_PER_TOKEN, {
      LAUNCH_TOKEN_DIM(hidden_dim, TOKEN_DIM, {
        auto &a2a =
            static_cast<All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &>(
                a2a_base);
        moe_cuda::kernels::naive_moe_forward(
            a2a, gemm_type, input, input_scales, gate, gate_scales, up,
            up_scales, down, down_scales, indices, weights, out_tokens,
            expert_x, expert_x_scales, inter_y, inter_y_scales, expert_y,
            stream);
      });
    });
  });
}

} // namespace moe_cuda
