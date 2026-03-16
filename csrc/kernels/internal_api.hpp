#pragma once

#include <cuda_runtime.h>
#include <moe_cuda/types.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <utility>

namespace moe_cuda {
namespace kernels {

void cast(
    at::Tensor& inp,
    at::Tensor& out,
    std::optional<at::Tensor>& scale,
    cudaStream_t stream);

void cast(
    at::Tensor& inp,
    at::Tensor& out,
    cudaStream_t stream);

void fused_silu_mul_quant(
    at::Tensor& gemm_out,
    at::Tensor& swiglu_out,
    at::Tensor& scale,
    cudaStream_t stream);

void fp8_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_swiglu_contiguous(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_swiglu_masked(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream);

void bf16_gemm(
    at::Tensor& A,
    at::Tensor& B,
    std::optional<at::Tensor>& C,
    at::Tensor& D,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream);

// template <typename All2AllT>
// inline void a2a_dispatch(
//     All2AllT& all2all,
//     at::Tensor& out_expert_num_tokens,
//     at::Tensor& out_expert_x,
//     std::optional<at::Tensor>& out_expert_x_scale,
//     at::Tensor& dp_x,
//     std::optional<at::Tensor>& dp_x_scale,
//     at::Tensor& indices,
//     at::Tensor& weights,
//     std::optional<at::Tensor>& bound_m,
//     bool do_send = true,
//     bool do_recv = true,
//     cudaStream_t stream = nullptr);

// template <typename All2AllT>
// inline void a2a_combine(
//     All2AllT& all2all,
//     at::Tensor& out_tokens,
//     at::Tensor& indices,
//     at::Tensor& weights,
//     at::Tensor& expert_y,
//     std::optional<at::Tensor>& bound_m,
//     bool do_send = true,
//     bool do_recv = true,
//     bool accumulate = false,
//     cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace moe_cuda
