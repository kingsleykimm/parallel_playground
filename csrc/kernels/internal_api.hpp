#pragma once

#include <cuda_runtime.h>
#include <moe_cuda/types.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <utility>

namespace kittens {
namespace py {
struct TKParallelTensor;
}
} // namespace kittens

namespace moe_cuda {
namespace kernels {

void cast(at::Tensor &inp, at::Tensor &out, std::optional<at::Tensor> &scale,
          cudaStream_t stream);

void cast(at::Tensor &inp, at::Tensor &out, cudaStream_t stream);

void fused_silu_mul_quant(at::Tensor &gemm_out, at::Tensor &swiglu_out,
                          at::Tensor &scale, cudaStream_t stream);

void fp8_gemm_nt(at::Tensor &act, at::Tensor &act_scale, at::Tensor &weight,
                 at::Tensor &weight_scale, at::Tensor &output,
                 GemmType gemm_type, const std::string &compiled_dims,
                 int *grouped_layout, cudaStream_t &stream);

void fp8_grouped_gemm_swiglu(at::Tensor &A, at::Tensor &gate_weight,
                             at::Tensor &up_weight, at::Tensor &scale_a,
                             at::Tensor &scale_gate, at::Tensor &scale_up,
                             at::Tensor &scale_d, at::Tensor &D,
                             GemmType gemm_type, int *grouped_layout,
                             cudaStream_t &stream);

void fp8_grouped_gemm_swiglu_sub(at::Tensor &A, at::Tensor &gate_weight,
                                at::Tensor &up_weight, at::Tensor &scale_a,
                                at::Tensor &scale_gate, at::Tensor &scale_up,
                                at::Tensor &scale_d, at::Tensor &D,
                                GemmType gemm_type, int *grouped_layout,
                                cudaStream_t &stream);

void fp8_grouped_gemm_swiglu_consumer_pp(
    at::Tensor &A, at::Tensor &gate_weight, at::Tensor &up_weight,
    at::Tensor &scale_a, at::Tensor &scale_gate, at::Tensor &scale_up,
    at::Tensor &scale_d, at::Tensor &D, GemmType gemm_type, int *grouped_layout,
    cudaStream_t &stream);

void fp8_grouped_gemm_nt(at::Tensor &act, at::Tensor &act_scale,
                         at::Tensor &weight, at::Tensor &weight_scale,
                         at::Tensor &output, GemmType gemm_type,
                         int *grouped_layout, cudaStream_t &stream);

void bf16_gemm(at::Tensor &A, at::Tensor &B, std::optional<at::Tensor> &C,
               at::Tensor &D, GemmType gemm_type,
               const std::string &compiled_dims, int *grouped_layout,
               cudaStream_t &stream);

void fused_dispatch_grouped_gemm_swiglu(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_tokens_scales,
    at::Tensor &expert_x_tokens, at::Tensor &expert_x_tokens_scale,
    at::Tensor &gate, at::Tensor &up, at::Tensor &C, at::Tensor &scale_gate,
    at::Tensor &scale_up, at::Tensor &out_scales, at::Tensor &indices,
    kittens::py::TKParallelTensor &global_num_routed,
    kittens::py::TKParallelTensor &expert_to_token_map,
    kittens::py::TKParallelTensor &expert_to_slot_map,
    at::Tensor &padded_expert_counts, at::Tensor &src_token_idx,
    at::Tensor &src_dev_idx, at::Tensor &src_slot_idx,
    kittens::py::TKParallelTensor &barrier, int num_tokens,
    int *num_recv_tokens, int dp_rank, int rank, int dp_size, int cur_dp_group,
    int num_dp_groups, int world_size, int num_experts, int experts_per_token,
    int num_comm_sms, int num_comp_sms, cudaStream_t &stream);

void fused_grouped_gemm_combine(
    kittens::py::TKParallelTensor &out_tokens, at::Tensor &expert_y_tokens,
    at::Tensor &expert_y_tokens_scale, at::Tensor &down, at::Tensor &scale_down,
    at::Tensor &C, at::Tensor &weights, at::Tensor &padded_expert_counts,
    at::Tensor &src_token_idx, at::Tensor &src_dev_idx,
    at::Tensor &src_slot_idx, int num_experts, int experts_per_token,
    int *num_recv_tokens, int dp_rank, int rank, int dp_size, int cur_dp_group,
    int num_dp_groups, int num_comm_sms, int num_comp_sms,
    cudaStream_t &stream);

} // namespace kernels
} // namespace moe_cuda
