#pragma once
#include <vector>
#include <optional>
#include <moe_cuda/dtype.h>
#include <runtime/device.hpp>
#include "a2a_worker.hpp"
#include <kernels/all2all/a2a_kernels.h>

class P2PDeviceWorkspace {

    public:
    
        // offset of each expert in contiguous token buffer
        uint32_t * expert_offsets;
        // offset of token within the current expert group
        uint32_t * token_offset;
        // counter for num tokens sent during combine
        uint32_t * token_counter;
        // sync across nvlink
        uint32_t * sync_counter;

        uint32_t ** sync_ptrs;
        uint8_t ** send_ptrs;
        uint8_t ** recv_ptrs;

        P2PDeviceWorkspace() {};
        
        P2PDeviceWorkspace(
            uint32_t num_experts,
            uint32_t max_num_tokens,
            uint32_t num_experts_per_token,
            uint32_t ** sync_ptrs,
            uint8_t ** send_ptrs,
            uint8_t ** recv_ptrs,
            cudaStream_t stream
        ) {
            CUDA_CHECK(cudaMallocAsync(&this->expert_offsets, num_experts * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&this->token_offset, max_num_tokens * num_experts_per_token * sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&this->token_counter, sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMallocAsync(&this->sync_counter, sizeof(uint32_t), stream));

            this->sync_ptrs = sync_ptrs;
            this->send_ptrs = send_ptrs;
            this->recv_ptrs = recv_ptrs;
        }
};


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
        uint32_t device;
        P2PDeviceWorkspace workspace;
        WorkerState worker;

        All2AllContext() {};

        All2AllContext(
            uint32_t hidden_dim,
            std::optional<uint32_t> hidden_scale_dim,
            size_t in_elemsize,
            size_t out_elemsize,
            c10::ScalarType out_dtype,
            std::optional<size_t> scale_elemsize,
            uint32_t max_num_tokens,
            uint32_t max_recv_tokens,
            uint32_t max_private_tokens,
            uint32_t num_experts,
            uint32_t expert_padding,
            uint32_t num_experts_per_token,
            uint32_t rank,
            uint32_t dp_size,
            uint32_t node_size,
            uint32_t world_size,
            uint32_t * num_routed_ptr,
            uint8_t * send_buffer_ptr,
            uint8_t * recv_buffer_ptr,
            uint32_t ** sync_ptrs,
            uint8_t ** send_ptrs,
            uint8_t ** recv_ptrs,
            uint32_t ** num_routed_ptrs,
            int device,
            cudaStream_t stream
        ) {
            this->worker = WorkerState(
                max_num_tokens,
                max_recv_tokens,
                max_private_tokens,
                hidden_dim,
                hidden_scale_dim.has_value() ? hidden_scale_dim.value() : 0,
                in_elemsize,
                out_elemsize,
                scale_elemsize.has_value() ? scale_elemsize.value() : 0,
                num_experts,
                num_experts_per_token,
                expert_padding,
                rank,
                dp_size,
                node_size,
                world_size,
                device, 
                num_routed_ptr,
                send_buffer_ptr,
                recv_buffer_ptr,
                num_routed_ptrs,
                stream
            );

            this->workspace = P2PDeviceWorkspace(
                num_experts, max_num_tokens, num_experts_per_token, sync_ptrs, send_ptrs, recv_ptrs, stream
            );

            uint32_t num_blocks = device_prop->get_num_sms();
            this->num_blocks = num_blocks;
            this->hidden_dim = hidden_dim;
            this->hidden_scale_dim = hidden_scale_dim.has_value() ? hidden_scale_dim.value() : 0;
            this->in_elemsize = in_elemsize;
            this->out_elemsize = out_elemsize;
            this->out_dtype = out_dtype;
            this->scale_elemsize = scale_elemsize.has_value() ? scale_elemsize.value() : 0;
            this->num_experts = num_experts;
            this->num_experts_per_token = num_experts_per_token;
            this->max_private_tokens = max_private_tokens;
            this->rank = rank;
            this->dp_size = dp_size;
            this->node_size = node_size;
            this->world_size = world_size;
            this->device = device;
            this->max_num_tokens = max_num_tokens;
        }

        void dispatch_send(
            uint32_t num_tokens,
            uint8_t * x_ptr,
            uint32_t x_stride,
            float * x_scale_ptr, // can be nullptr if not
            uint32_t x_scale_stride_elem, // these can be 0
            uint32_t x_scale_stride_token,
            uint32_t * indices,
            uint32_t indices_stride,
            float * weights,
            uint32_t weights_stride,
            uint32_t * bound_m_ptr,
            cudaStream_t stream
        ) {
            if (num_tokens > this->max_num_tokens) {
                throw std::runtime_error("Number of tokens exceeds maximum allowed");
            }

            CUDA_CHECK(
                a2a_kernels::a2a_dispatch_send(
                    this->num_blocks,
                    this->hidden_dim,
                    this->hidden_scale_dim,
                    this->in_elemsize,
                    this->scale_elemsize,
                    this->num_experts,
                    this->num_experts_per_token,
                    num_tokens,
                    this->max_private_tokens,
                    this->rank,
                    this->dp_size,
                    this->node_size,
                    this->world_size,
                    reinterpret_cast<int32_t*>(bound_m_ptr),
                    reinterpret_cast<std::byte*>(x_ptr),
                    x_stride,
                    x_scale_ptr,
                    x_scale_stride_elem,
                    x_scale_stride_token,
                    reinterpret_cast<int32_t*>(indices),
                    indices_stride,
                    weights,
                    weights_stride,
                    this->workspace.token_offset,
                    this->worker.num_routed_ptr,
                    this->workspace.expert_offsets,
                    this->worker.send_buffer_ptr,
                    this->workspace.sync_counter,
                    this->workspace.sync_ptrs,
                    reinterpret_cast<std::byte**>(this->workspace.recv_ptrs),
                    stream
                )
            );
        }

        void dispatch_recv(
            uint32_t * out_num_tokens_ptr,
            uint8_t * out_x_ptr,
            uint32_t out_x_stride,
            uint8_t * out_x_scale_ptr,
            uint32_t out_x_scale_stride_elem,
            uint32_t out_x_scale_stride_token,
            cudaStream_t stream
        ) {
            CUDA_CHECK(
                a2a_kernels::a2a_dispatch_recv(
                    this->num_blocks,
                    this->hidden_dim,
                    this->hidden_scale_dim,
                    this->in_elemsize,
                    this->scale_elemsize,
                    this->num_experts,
                    this->rank,
                    this->node_size,
                    this->world_size,
                    reinterpret_cast<int32_t*>(out_num_tokens_ptr),
                    out_x_ptr,
                    out_x_stride,
                    out_x_scale_ptr,
                    out_x_scale_stride_elem,
                    out_x_scale_stride_token,
                    this->worker.tokens_per_expert,
                    this->worker.send_buffer_ptr,
                    this->worker.recv_buffer_ptr,
                    this->worker.source_rank,
                    this->worker.source_dispatch_offset,
                    this->worker.padded_index,
                    this->worker.num_routed_ptr,
                    this->worker.num_recv_tokens,
                    this->workspace.sync_counter,
                    this->workspace.sync_ptrs,
                    this->workspace.send_ptrs,
                    stream
                )
            );
        }

        void combine_send(
            void * expert_y_ptr, // Y = XW output of groupped gemm kenrel
            uint32_t expert_y_stride,
            cudaStream_t stream
        ) {
            CUDA_CHECK(a2a_kernels::a2a_combine_send(
                this->num_blocks,
                this->hidden_dim,
                this->out_elemsize,
                this->rank,
                this->node_size,
                this->dp_size,
                (uint8_t * )expert_y_ptr,
                expert_y_stride,
                this->worker.send_buffer_ptr,
                this->worker.recv_buffer_ptr,
                this->worker.source_rank,
                this->worker.combine_send_offset,
                this->worker.padded_index,
                this->worker.num_recv_tokens, 
                this->workspace.sync_counter,
                this->workspace.sync_ptrs,
                this->workspace.recv_ptrs,
                stream
            ));
        }

        void combine_recv (
            uint32_t num_tokens, //
            uint32_t num_recv_tokens,
            c10::ScalarType expert_dtype,
            uint8_t * out_tokens_ptr,
            size_t out_tokens_stride,
            uint32_t * indices_ptr,
            size_t indices_stride,
            float * weights_ptr,
            size_t weights_stride,
            uint32_t * bound_m_ptr,
            bool accumulate,
            cudaStream_t stream
        ) {
            CUDA_CHECK(a2a_kernels::a2a_combine_recv(this->num_blocks,
                this->hidden_dim,
                this->out_elemsize,
                expert_dtype,
                this->out_dtype,
                this->num_experts,
                this->num_experts_per_token,
                this->rank,
                this->node_size,
                this->world_size,
                num_tokens,
                reinterpret_cast<int32_t*>(bound_m_ptr),
                reinterpret_cast<int32_t*>(indices_ptr),
                indices_stride,
                weights_ptr,
                weights_stride,
                out_tokens_ptr,
                out_tokens_stride,
                accumulate,
                this->worker.recv_buffer_ptr,
                this->workspace.token_offset,
                this->workspace.expert_offsets,
                this->workspace.sync_counter,
                this->workspace.sync_ptrs,
                stream
            ));
        }


};