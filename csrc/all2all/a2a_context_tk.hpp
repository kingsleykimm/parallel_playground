/**
 @file Context Manager for All2All Kernels
 */
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include <kernels/all2all/a2a_combine_recv_tk.cuh>
#include <kernels/all2all/a2a_combine_send_tk.cuh>
#include <kernels/all2all/a2a_dispatch_recv_tk.cuh>
#include <kernels/all2all/a2a_dispatch_send_tk.cuh>
#include <kernels/all2all/a2a_kernels.h>
#include <moe_cuda/dtype.h>
#include <runtime/device.hpp>

#include "a2a_worker_tk.hpp"
#include <torch/csrc/stable/tensor.h>

#include <torch/csrc/inductor/aoti_torch/utils.h>
inline at::Tensor &stable_to_aten(torch::stable::Tensor &t) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(t.get());
}
inline const at::Tensor &stable_to_aten(const torch::stable::Tensor &t) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(t.get());
}

class P2PDeviceWorkspace {
public:
  uint32_t *expert_offsets = nullptr;
  uint32_t *token_offset = nullptr;
  uint32_t *token_counter = nullptr;
  uint32_t *sync_counter = nullptr;

  P2PDeviceWorkspace() = default;

  P2PDeviceWorkspace(uint32_t num_experts, uint32_t max_num_tokens,
                     uint32_t num_experts_per_token, cudaStream_t stream) {
    CUDA_CHECK(cudaMallocAsync(&this->expert_offsets,
                               num_experts * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(
        &this->token_offset,
        max_num_tokens * num_experts_per_token * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&this->token_counter, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&this->sync_counter, sizeof(uint32_t), stream));
    CUDA_CHECK(
        cudaMemsetAsync(this->token_counter, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(
        cudaMemsetAsync(this->sync_counter, 0, sizeof(uint32_t), stream));
  }
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
class All2AllContext {
public:
  uint32_t num_blocks;
  uint32_t hidden_dim;
  uint32_t hidden_scale_dim;
  uint32_t in_elemsize;
  uint32_t out_elemsize;
  c10::ScalarType out_dtype;
  uint32_t scale_elemsize;
  uint32_t num_experts;
  uint32_t num_experts_per_token;
  uint32_t max_num_tokens;
  uint32_t max_private_tokens;
  uint32_t rank;
  uint32_t dp_size;
  uint32_t node_size;
  uint32_t world_size;
  uint32_t dp_group;

  kittens::py::TKParallelTensor num_routed_tensor;
  kittens::py::TKParallelTensor send_buffer;
  kittens::py::TKParallelTensor send_scale_buffer;
  kittens::py::TKParallelTensor barrier;
  kittens::py::TKParallelTensor recv_buffer;
  kittens::py::TKParallelTensor recv_scale_buffer;

  P2PDeviceWorkspace workspace;
  WorkerState worker;
  cudaStream_t stream;

  All2AllContext(uint32_t hidden_dim, std::optional<uint32_t> hidden_scale_dim,
                 size_t in_elemsize, size_t out_elemsize,
                 c10::ScalarType out_dtype,
                 std::optional<size_t> scale_elemsize, uint32_t max_num_tokens,
                 uint32_t max_recv_tokens, uint32_t max_private_tokens,
                 uint32_t num_experts, uint32_t expert_padding,
                 uint32_t num_experts_per_token, uint32_t rank,
                 uint32_t dp_size, uint32_t node_size, uint32_t world_size,
                 cudaStream_t stream)
      : num_routed_tensor(
            std::vector<int64_t>{static_cast<int64_t>(world_size / dp_size),
                                 static_cast<int64_t>(num_experts)},
            at::ScalarType::UInt32, rank, world_size, false),
        send_buffer(
            std::vector<int64_t>{
                static_cast<int64_t>(max_recv_tokens),
                static_cast<int64_t>(TOKEN_DIM *
                                     std::max(in_elemsize, out_elemsize))},
            at::ScalarType::Char, rank, world_size, false),
        send_scale_buffer(
            std::vector<int64_t>{static_cast<int64_t>(TOKEN_DIM / 128),
                                 static_cast<int64_t>(max_recv_tokens)},
            at::ScalarType::Float, rank, world_size, false),
        barrier(std::vector<int64_t>{static_cast<int64_t>(node_size * 2)},
                at::ScalarType::Int, rank, world_size, false),
        recv_buffer(
            std::vector<int64_t>{
                static_cast<int64_t>(max_recv_tokens),
                static_cast<int64_t>(TOKEN_DIM *
                                     std::max(in_elemsize, out_elemsize))},
            at::ScalarType::Char, rank, world_size, false),
        recv_scale_buffer(
            std::vector<int64_t>{static_cast<int64_t>(TOKEN_DIM / 128),
                                 static_cast<int64_t>(max_recv_tokens)},
            at::ScalarType::Float, rank, world_size, false),
        workspace(num_experts, max_num_tokens, num_experts_per_token, stream),
        worker(max_num_tokens, max_recv_tokens, max_private_tokens, hidden_dim,
               hidden_scale_dim, in_elemsize, out_elemsize, scale_elemsize,
               num_experts, num_experts_per_token, expert_padding, rank,
               dp_size, node_size, world_size,
               this->num_routed_tensor.raw_ptrs_, stream),
        stream(stream) {

    // zero init barrier here on initialization
    this->num_blocks = device_prop->get_num_sms();
    this->hidden_dim = hidden_dim;
    this->hidden_scale_dim =
        hidden_scale_dim.has_value() ? hidden_scale_dim.value() : 0;
    this->in_elemsize = in_elemsize;
    this->out_elemsize = out_elemsize;
    this->out_dtype = out_dtype;
    this->scale_elemsize =
        scale_elemsize.has_value() ? scale_elemsize.value() : 0;
    this->num_experts = num_experts;
    this->num_experts_per_token = num_experts_per_token;
    this->max_private_tokens = max_private_tokens;
    this->rank = rank;
    this->dp_size = dp_size;
    this->node_size = node_size;
    this->world_size = world_size;
    this->dp_group = rank / dp_size;
    this->max_num_tokens = max_num_tokens;
    CUDA_CHECK(cudaMemset(barrier.raw_ptrs_[barrier.local_rank_], 0,
                          node_size * 2 * sizeof(int)));
    kittens::py::TKParallelTensor::brokers_
        .at({(int)this->rank, (int)this->world_size})
        .sync(this->world_size);
  }

  void dispatch_send(torch::stable::Tensor &in_tokens,
                     torch::stable::Tensor &in_scales,
                     torch::stable::Tensor &indices,
                     torch::stable::Tensor &weights, uint32_t *sync_counter,
                     uint32_t num_tokens, cudaStream_t stream) {
    if (num_tokens > this->max_num_tokens) {
      throw std::runtime_error("Number of tokens exceeds maximum allowed");
    }
    cudaError_t status =
        a2a_kernels::fp8e4m3_a2a_dispatch_send<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                               TOKEN_DIM>(
            stable_to_aten(in_tokens), stable_to_aten(in_scales),
            stable_to_aten(indices), stable_to_aten(weights),
            this->workspace.token_offset, this->workspace.expert_offsets,
            this->recv_buffer, this->recv_scale_buffer,
            this->num_routed_tensor.data_, this->send_buffer,
            this->send_scale_buffer, this->barrier, sync_counter,
            this->max_private_tokens, this->worker.dispatch_route_done,
            this->rank, this->dp_size, stream);
    CUDA_CHECK(status);
  }

  void dispatch_recv(torch::stable::Tensor &out_tokens,
                     torch::stable::Tensor &out_scales,
                     uint32_t *out_num_tokens_ptr, cudaStream_t stream) {
    (void)out_num_tokens_ptr;
    cudaError_t status =
        a2a_kernels::fp8e4m3_a2a_dispatch_recv<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                               TOKEN_DIM>(
            this->recv_buffer, this->recv_scale_buffer, this->send_buffer,
            this->send_scale_buffer, out_tokens, out_scales, this->barrier,
            this->workspace.sync_counter, this->worker.source_rank,
            this->worker.source_dispatch_offset, this->worker.padded_index,
            this->worker.num_recv_tokens, this->rank, this->dp_size, stream);
    CUDA_CHECK(status);
  }

  void combine_send(torch::stable::Tensor &experts_out, cudaStream_t stream) {
    HOST_ASSERT(experts_out.scalar_type() == at::ScalarType::BFloat16,
                "Only Bf16 combine kernels supported for now");
    int num_recv_tokens = static_cast<int>(this->worker.num_recv_tokens);
    cudaError_t status = a2a_kernels::a2a_combine_send_tk<TOKEN_DIM>(
        stable_to_aten(experts_out), this->recv_buffer, this->barrier,
        this->worker.combine_send_offset, this->worker.source_rank,
        this->worker.padded_index, this->workspace.sync_counter,
        num_recv_tokens, this->rank, this->dp_group, this->dp_size, stream);
    CUDA_CHECK(status);
  }

  void combine_recv(torch::stable::Tensor &out_tokens,
                    torch::stable::Tensor &indices,
                    torch::stable::Tensor &weights, bool accumulate,
                    cudaStream_t stream) {
    HOST_ASSERT(out_tokens.scalar_type() == at::ScalarType::BFloat16,
                "Only Bf16 combine kernels supported for now");
    cudaError_t status =
        a2a_kernels::a2a_combine_recv_tk<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                         TOKEN_DIM>(
            this->barrier, this->recv_buffer, stable_to_aten(out_tokens),
            stable_to_aten(indices), stable_to_aten(weights),
            this->workspace.token_offset, this->workspace.expert_offsets,
            this->workspace.sync_counter, out_tokens.size(-2), accumulate,
            this->rank, this->dp_group, stream);
    CUDA_CHECK(status);
  }
};
