#pragma once

#include <moe_cuda/types.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <utility>
#ifdef MOE_CUDA_USE_MPI
#include <all2all/all2all.hpp>
#endif

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

#ifdef MOE_CUDA_USE_MPI
void a2a_dispatch(
    All2All& all2all,
    at::Tensor& out_expert_num_tokens,
    at::Tensor& out_expert_x,
    std::optional<at::Tensor>& out_expert_x_scale,
    at::Tensor& dp_x,
    std::optional<at::Tensor>& dp_x_scale,
    at::Tensor& indices,
    at::Tensor& weights,
    std::optional<at::Tensor>& bound_m,
    bool do_send = true,
    bool do_recv = true,
    cudaStream_t stream = nullptr);

void a2a_combine(
    All2All& all2all,
    at::Tensor& out_tokens,
    at::Tensor& indices,
    at::Tensor& weights,
    at::Tensor& expert_y,
    std::optional<at::Tensor>& bound_m,
    bool do_send = true,
    bool do_recv = true,
    bool accumulate = false,
    cudaStream_t stream = nullptr);
#endif

}  // namespace moe_cuda
