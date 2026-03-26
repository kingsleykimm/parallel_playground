#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <moe_cuda/types.h>
#include <optional>
#include <runtime/parallel.h>
#include <torch/torch.h>

namespace moe_cuda {

class All2AllBase {
public:
  All2AllBase() = default;
  virtual ~All2AllBase() = default;

  virtual void dispatch(at::Tensor &out_expert_num_tokens,
                        at::Tensor &out_expert_x,
                        std::optional<at::Tensor> &out_expert_x_scale,
                        at::Tensor &dp_x, std::optional<at::Tensor> &dp_x_scale,
                        at::Tensor &indices, at::Tensor &weights,
                        std::optional<at::Tensor> &bound_m, bool do_send = true,
                        bool do_recv = true, cudaStream_t stream = nullptr) = 0;

  virtual void combine(at::Tensor &out_tokens, at::Tensor &indices,
                       at::Tensor &weights, at::Tensor &expert_y,
                       std::optional<at::Tensor> &bound_m, bool do_send = true,
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
    uint32_t hidden_dim, GemmType gemm_type, at::Tensor &input,
    at::Tensor &input_scales, at::Tensor &gate, at::Tensor &gate_scales,
    at::Tensor &up, at::Tensor &up_scales, at::Tensor &down,
    at::Tensor &down_scales, at::Tensor &indices, at::Tensor &weights,
    at::Tensor &out_tokens, at::Tensor &expert_x, at::Tensor &expert_x_scales,
    at::Tensor &inter_y, at::Tensor &inter_y_scales, at::Tensor &expert_y,
    cudaStream_t stream);

} // namespace moe_cuda
