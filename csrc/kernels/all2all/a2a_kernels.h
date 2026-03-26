#pragma once
#include <ATen/Tensor.h>
#include <moe_cuda/dtype.h>
#include <moe_cuda/error.hpp>
#include <pyutils/parallel_tensor.cuh>
#include <runtime/utils.h>

// namespace kittens {
// namespace py {
// struct TKParallelTensor;
// }  // namespace py
// }  // namespace kittens

namespace a2a_kernels {
cudaError_t a2a_dispatch_recv(
    size_t num_blocks, size_t hidden_dim, size_t hidden_dim_scale,
    size_t x_elemsize, size_t x_scale_elemsize, size_t num_experts, size_t rank,
    size_t node_size, size_t world_size, int32_t *out_num_tokens_ptr,
    uint8_t *out_x_ptr, size_t out_x_stride, uint8_t *out_x_scale_ptr,
    size_t out_x_scale_stride_elem, size_t out_x_scale_stride_token,
    uint32_t *tokens_per_expert, uint8_t *send_buffer, uint8_t *recv_buffer,
    uint32_t *source_rank, uint32_t *source_offset, uint32_t *padded_index,
    uint32_t *num_routed, uint32_t *num_recv_tokens_ptr, uint32_t *sync_counter,
    uint32_t **sync_ptrs, uint8_t **send_ptrs, cudaStream_t stream);

cudaError_t a2a_dispatch_send(
    const uint32_t num_blocks, size_t hidden_dim, size_t hidden_dim_scale,
    size_t x_elemsize, size_t x_scale_elemsize, size_t num_experts,
    size_t num_experts_per_token, size_t num_tokens, size_t max_private_tokens,
    size_t rank, size_t dp_size, size_t node_size, size_t world_size,
    int32_t *bound_m_ptr, std::byte *x_ptr, size_t x_stride, float *x_scale_ptr,
    size_t x_scale_stride_elem, size_t x_scale_stride_token, int32_t *indices,
    size_t indices_stride, float *weights, size_t weight_stride,
    uint32_t *token_offset, uint32_t *num_routed, uint32_t *expert_offsets,
    uint8_t *send_buffer,
    uint32_t *sync_counter, // used to sync across nvlink
    uint32_t **sync_ptrs, std::byte **recv_ptrs, cudaStream_t stream);
// namespace a2a_kernels

cudaError_t
a2a_combine_send(size_t num_blocks, size_t hidden_dim,
                 size_t x_elemsize, // bf16 here, or what output activation size
                 size_t rank, size_t node_size, size_t dp_size,
                 uint8_t *expert_x_ptr, size_t expert_x_stride,
                 uint8_t *send_buffer, uint8_t *recv_buffer,
                 uint32_t *source_rank, uint32_t *combine_send_offset,
                 uint32_t *padded_index, uint32_t *num_recv_tokens_ptr,
                 uint32_t *sync_counter, uint32_t **sync_ptrs,
                 uint8_t **recv_ptrs, cudaStream_t stream);

cudaError_t a2a_combine_recv(
    size_t num_blocks, size_t hidden_dim, size_t x_elemsize,
    c10::ScalarType in_dtype, c10::ScalarType out_dtype, size_t num_experts,
    size_t num_experts_per_token, size_t rank, size_t node_size,
    size_t world_size, size_t num_tokens, int32_t *bound_m_ptr,
    int32_t *indices_ptr, size_t indices_stride, float *weights_ptr,
    size_t weights_stride, uint8_t *out_tokens_ptr, size_t out_tokens_stride,
    bool accumulate, uint8_t *recv_buffer, uint32_t *token_offset,
    uint32_t *expert_offsets, uint32_t *sync_counter, uint32_t **sync_ptrs,
    cudaStream_t stream);
template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t fp8e4m3_a2a_dispatch_send(
    at::Tensor &input_tokens_tensor, at::Tensor &input_scales_tensor,
    at::Tensor &indices_tensor, at::Tensor &weights_tensor,
    uint32_t *token_offsets_ptr, uint32_t *expert_offsets_ptr,
    kittens::py::TKParallelTensor &out_tokens,
    kittens::py::TKParallelTensor &out_scales, at::Tensor &num_routed_tensor,
    kittens::py::TKParallelTensor &send_buffer,
    kittens::py::TKParallelTensor &send_scale_buffer,
    kittens::py::TKParallelTensor &barrier, uint32_t *sync_counter_ptr,
    uint32_t max_private_tokens, uint8_t *dispatch_route_done, int local_rank,
    int dp_size, cudaStream_t stream);
template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t fp8e4m3_a2a_dispatch_recv(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_scales,
    kittens::py::TKParallelTensor &send_buffer,
    kittens::py::TKParallelTensor &send_scale_buffer,
    at::Tensor &out_tokens_tensor, at::Tensor &out_scales_tensor,
    kittens::py::TKParallelTensor &barrier, uint32_t *sync_counter,
    uint32_t *source_rank, uint32_t *source_offset, uint32_t *padded_index,
    uint32_t num_recv_tokens, int rank, int dp_size, cudaStream_t stream);

template <int TOKEN_DIM>
cudaError_t
a2a_combine_send_tk(at::Tensor &in_tokens_tensor,
                    kittens::py::TKParallelTensor &recv_buffer_tensor,
                    kittens::py::TKParallelTensor &barrier_tensor,
                    uint32_t *combine_send_offset, uint32_t *source_rank,
                    uint32_t *padded_index, uint32_t *sync_counter,
                    int num_recv_tokens, int rank, int dp_group, int dp_size,
                    cudaStream_t stream);

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t
a2a_combine_recv_tk(kittens::py::TKParallelTensor &barrier_tensor,
                    kittens::py::TKParallelTensor &recv_buffer_tensor,
                    at::Tensor &out_tokens_tensor, at::Tensor &indices_tensor,
                    at::Tensor &weights_tensor, uint32_t *token_offset,
                    uint32_t *expert_offsets, uint32_t *sync_counter,
                    int num_tokens, bool accumulate, int rank, int dp_group,
                    cudaStream_t stream);
} // namespace a2a_kernels
