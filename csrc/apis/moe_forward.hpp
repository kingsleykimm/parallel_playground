#pragma once

#include <moe_cuda/types.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <utility>

namespace moe_cuda {

void init(const std::string& library_root, const std::string& cuda_home);

void bf16_gemm(
    at::Tensor& A,
    at::Tensor& B,
    std::optional<at::Tensor>& C,
    at::Tensor& D,
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

void fp8_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream);

void fp8_grouped_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream);

}  // namespace moe_cuda
