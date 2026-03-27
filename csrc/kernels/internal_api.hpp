#pragma once

#include <cuda_runtime.h>
#include <moe_cuda/types.h>
#include <torch/csrc/stable/tensor.h>

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

void cast_dispatch(torch::stable::Tensor &inp, torch::stable::Tensor &out, std::optional<torch::stable::Tensor> &scale,
                   cudaStream_t stream);

void fused_silu_mul_quant(torch::stable::Tensor &gemm_out, torch::stable::Tensor &swiglu_out,
                          torch::stable::Tensor &scale, cudaStream_t stream);

void fp8_gemm_nt(torch::stable::Tensor &act, torch::stable::Tensor &act_scale, torch::stable::Tensor &weight,
                 torch::stable::Tensor &weight_scale, torch::stable::Tensor &output,
                 const std::string &compiled_dims, cudaStream_t &stream);

void fp8_grouped_gemm_swiglu(torch::stable::Tensor &A, torch::stable::Tensor &gate_weight,
                             torch::stable::Tensor &up_weight, torch::stable::Tensor &scale_a,
                             torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
                             torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
                             GemmType gemm_type, int *grouped_layout,
                             cudaStream_t &stream);

void fp8_grouped_gemm_swiglu_consumer_pp(
    torch::stable::Tensor &A, torch::stable::Tensor &gate_weight, torch::stable::Tensor &up_weight,
    torch::stable::Tensor &scale_a, torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D, GemmType gemm_type, int *grouped_layout,
    cudaStream_t &stream);

void fp8_grouped_gemm_nt(torch::stable::Tensor &act, torch::stable::Tensor &act_scale,
                         torch::stable::Tensor &weight, torch::stable::Tensor &weight_scale,
                         torch::stable::Tensor &output, GemmType gemm_type,
                         int *grouped_layout, cudaStream_t &stream);

void bf16_gemm(torch::stable::Tensor &A, torch::stable::Tensor &B, std::optional<torch::stable::Tensor> &C,
               torch::stable::Tensor &D, GemmType gemm_type,
               const std::string &compiled_dims, int *grouped_layout,
               cudaStream_t &stream);

void fused_dispatch_grouped_gemm_swiglu(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_tokens_scales,
    torch::stable::Tensor &expert_x_tokens, torch::stable::Tensor &expert_x_tokens_scale,
    torch::stable::Tensor &gate, torch::stable::Tensor &up,
    torch::stable::Tensor &C, torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &out_scales, torch::stable::Tensor &indices,
    kittens::py::TKParallelTensor &global_num_routed,
    kittens::py::TKParallelTensor &expert_to_token_map,
    torch::stable::Tensor &padded_expert_counts, torch::stable::Tensor &src_token_idx,
    torch::stable::Tensor &src_dev_idx, kittens::py::TKParallelTensor &barrier,
    int num_tokens, int *num_recv_tokens, int dp_rank, int rank, int dp_size,
    int cur_dp_group, int num_dp_groups, int world_size, int num_experts, int experts_per_token,
    int num_comm_sms, int num_comp_sms, cudaStream_t &stream);

} // namespace kernels
} // namespace moe_cuda
