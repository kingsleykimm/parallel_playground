#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <moe_cuda/types.h>
#include <optional>
#include <runtime/parallel.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

namespace moe_cuda {

class All2AllBase {
public:
  All2AllBase() = default;
  virtual ~All2AllBase() = default;

  virtual void
  dispatch(torch::stable::Tensor &out_expert_num_tokens,
           torch::stable::Tensor &out_expert_x,
           std::optional<torch::stable::Tensor> &out_expert_x_scale,
           torch::stable::Tensor &dp_x,
           std::optional<torch::stable::Tensor> &dp_x_scale,
           torch::stable::Tensor &indices, torch::stable::Tensor &weights,
           std::optional<torch::stable::Tensor> &bound_m, bool do_send = true,
           bool do_recv = true, cudaStream_t stream = nullptr) = 0;

  virtual void
  combine(torch::stable::Tensor &out_tokens, torch::stable::Tensor &indices,
          torch::stable::Tensor &weights, torch::stable::Tensor &expert_y,
          std::optional<torch::stable::Tensor> &bound_m, bool do_send = true,
          bool do_recv = true, bool accumulate = false,
          cudaStream_t stream = nullptr) = 0;

  virtual uint32_t max_recv_tokens() const = 0;
};

std::shared_ptr<All2AllBase>
make_all2all(uint32_t max_num_tokens, uint32_t num_experts,
             uint32_t experts_per_token, uint32_t expert_padding,
             uint32_t hidden_dim, std::optional<uint32_t> hidden_dim_scale,
             c10::ScalarType in_dtype, c10::ScalarType out_dtype,
             std::optional<c10::ScalarType> scale_dtype,
             std::optional<uint32_t> max_private_tokens_opt, int local_rank,
             ParallelConfig parallel_config, cudaStream_t stream);

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
    torch::stable::Tensor &expert_y, cudaStream_t stream);

} // namespace moe_cuda
