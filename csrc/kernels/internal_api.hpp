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
    at::Tensor& act,
    at::Tensor& act_scale,
    at::Tensor& weight,
    at::Tensor& weight_scale,
    at::Tensor& output,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_swiglu(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_swiglu_consumer_pp(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream);    


void fp8_grouped_gemm_nt(
    at::Tensor& act,
    at::Tensor& act_scale,
    at::Tensor& weight,
    at::Tensor& weight_scale,
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


}  // namespace kernels
}  // namespace moe_cuda
