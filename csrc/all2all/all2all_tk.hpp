/**
 @file All2All handler for token quantized MoE kernels, shared across all layers
*/

#pragma once
#include <chrono>
#include <optional>
#include <utility>
#include <atomic>
#include <cuda.h>
#include "utils.h"
#include <runtime/utils.h>
#include <moe_cuda/dtype.h>
#include <runtime/parallel.h>
#include <moe_cuda/error.hpp>
#include <runtime/cumem.h>
#include <jit/utils/lazy_init.hpp>
#include <jit/utils/common.hpp>
#include <runtime/tensor.h>
#include "a2a_context_tk.hpp"
#include <kittens.cuh>
#include <pyutils/parallel_tensor.cuh>

namespace moe_cuda {

    
// System page size for padding
constexpr int PAGE_SIZE = 4096; // 4KB
constexpr int NUM_DEVICES = 4; // i'm only on a H100 x 4 node for now


// entrypoint - api methods should call this method. this is only instantiated once per proces at the start
template<int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
class All2All {

    public:
        struct RoutingDebugState {
            at::Tensor tokens_per_expert;
            at::Tensor source_rank;
            at::Tensor source_dispatch_offset;
            at::Tensor combine_send_offset;
            at::Tensor padded_index;
            uint32_t num_recv_tokens = 0;
        };

        All2All (
            uint32_t max_num_tokens,
            uint32_t num_experts,
            uint32_t expert_padding,
            uint32_t hidden_dim,
            std::optional<uint32_t> hidden_dim_scale,
            c10::ScalarType in_dtype,
            c10::ScalarType out_dtype,
            std::optional<c10::ScalarType> scale_dtype,
            uint32_t num_experts_per_token,
            std::optional<uint32_t> max_private_tokens_opt,
            int local_rank,
            ParallelConfig parallel_config,
            cudaStream_t stream
        ) {
            HOST_ASSERT(parallel_config.world_size == NUM_DEVICES,
                        "TK all2all currently expects a 4-GPU single-node run");
            HOST_ASSERT(parallel_config.node_size == NUM_DEVICES,
                        "TK all2all currently expects node_size == 4");
            HOST_ASSERT(get_num_experts(num_experts) == NUM_EXPERTS,
                        "Template NUM_EXPERTS must match runtime num_experts");
            HOST_ASSERT(get_num_experts_per_token(num_experts_per_token) ==
                            EXPERTS_PER_TOKEN,
                        "Template EXPERTS_PER_TOKEN must match runtime top-k");
            HOST_ASSERT(get_token_dim(hidden_dim) == TOKEN_DIM,
                        "Template TOKEN_DIM must match runtime hidden_dim");
            this->initialized = true;
            this->hidden_dim = hidden_dim;
            this->hidden_dim_scale = hidden_dim_scale;
            this->num_experts = get_num_experts(num_experts);
            this->num_experts_per_token = num_experts_per_token;
            this->in_dtype = in_dtype;
            this->out_dtype = out_dtype;
            this->scale_dtype = scale_dtype;
            this->local_rank = local_rank;
            this->rank = local_rank;
            this->dp_group = local_rank / parallel_config.dp_size;
            this->node_group = local_rank / parallel_config.node_size;
            this->dp_size = parallel_config.dp_size;
            this->world_size = parallel_config.world_size;
            this->parallel_config = parallel_config;
            uint32_t num_dp_groups = this->world_size / this->dp_size;
            this->num_local_experts = host_ceil_div(this->num_experts, this->world_size);

            // recv buffer size
            uint32_t avg_tokens_per_expert = host_ceil_div(
                max_num_tokens * num_experts_per_token, num_experts
            ) * 1.2;
            uint32_t max_private_tokens;
            if (!max_private_tokens_opt.has_value()) {
                max_private_tokens = avg_tokens_per_expert * this->num_local_experts;
            }
            else {
                max_private_tokens = max_private_tokens_opt.value();
            }
            HOST_ASSERT(max_private_tokens > 0, "max_private_tokens must be greater than 0");

            uint32_t num_tokens = max_num_tokens * num_dp_groups;
            uint32_t max_recv_tokens = max_private_tokens * num_dp_groups; // private size for direct NVlink copies

            // max_recv_tokens is bounded between [this->num_local_experts * expert_padding, num_tokens * num_experts_per_token]
            // the actual value is the 
            max_recv_tokens += host_align(std::max(
                std::min(num_tokens * num_experts_per_token 
                    + this->num_local_experts * (expert_padding - 1), 
              this->num_local_experts * num_tokens), // upper bound
               this->num_local_experts * expert_padding), // lower bound
               expert_padding);

            // siz send buffer sizes + token_dim (combing optional scale factors)
            uint32_t token_dim_dispatch = host_align(hidden_dim * get_type_size(in_dtype), 16) + 16;
            if (hidden_dim_scale.has_value() && scale_dtype.has_value()) {
                
                token_dim_dispatch += host_align(hidden_dim_scale.value() * get_type_size(scale_dtype.value()), 16);

                HOST_ASSERT(scale_dtype.value() == c10::ScalarType::Float, "Only float scales supported");
            }

            // combine token dim is just the hidden dim, since the outputs of MoE kernels are unquantized

            uint32_t token_dim_combine = host_align(hidden_dim * get_type_size(in_dtype), 16);
            uint32_t token_dim = std::max(token_dim_combine, token_dim_dispatch);
            (void)token_dim;

            this->context.emplace(
                hidden_dim,
                hidden_dim_scale,
                get_type_size(in_dtype),
                get_type_size(out_dtype),
                out_dtype,
                scale_dtype.has_value()
                    ? std::optional<size_t>(get_type_size(scale_dtype.value()))
                    : std::nullopt,
                max_num_tokens,
                max_recv_tokens,
                max_private_tokens,
                num_experts,
                expert_padding,
                num_experts_per_token,
                this->rank,
                this->dp_size,
                parallel_config.node_size,
                parallel_config.world_size,
                stream);
        }

        RoutingDebugState debug_routing_state(cudaStream_t stream = nullptr) const {
            HOST_ASSERT(initialized, "All2All handler is not initialized");
            HOST_ASSERT(this->context.has_value(), "All2All context is not initialized");

            auto effective_stream = stream != nullptr ? stream : this->context->stream;
            auto opts =
                at::TensorOptions()
                    .dtype(c10::ScalarType::UInt32)
                    .device(c10::Device(c10::DeviceType::CUDA, this->local_rank));

            RoutingDebugState state;
            state.num_recv_tokens = this->context->worker.num_recv_tokens;

            c10::IntArrayRef size = {static_cast<int64_t>(this->num_local_experts)};
            state.tokens_per_expert = at::empty(size, opts);

            // cudaPointerAttributes attr;
            // CUDA_CHECK(cudaPointerGetAttributes(&attr, this->context->worker.tokens_per_expert));
            // printf("tokens_per_expert is in %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");

            
            CUDA_CHECK(cudaMemcpyAsync(
                (uint32_t *)state.tokens_per_expert.data_ptr(),
                this->context->worker.tokens_per_expert,
                this->num_local_experts * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice,
                effective_stream));

            state.source_rank =
                at::empty({static_cast<int64_t>(state.num_recv_tokens)}, opts);
            state.source_dispatch_offset =
                at::empty({static_cast<int64_t>(state.num_recv_tokens)}, opts);
            state.combine_send_offset =
                at::empty({static_cast<int64_t>(state.num_recv_tokens)}, opts);
            state.padded_index =
                at::empty({static_cast<int64_t>(state.num_recv_tokens)}, opts);

            if (state.num_recv_tokens > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.source_rank.data_ptr(),
                    this->context->worker.source_rank,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.source_dispatch_offset.data_ptr(),
                    this->context->worker.source_dispatch_offset,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.combine_send_offset.data_ptr(),
                    this->context->worker.combine_send_offset,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.padded_index.data_ptr(),
                    this->context->worker.padded_index,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
            }

            return state;
        }

        at::Tensor debug_num_routed() const {
            HOST_ASSERT(initialized, "All2All handler is not initialized");
            HOST_ASSERT(this->context.has_value(), "All2All context is not initialized");
            const auto& hw = this->context->worker.host_num_routed;
            auto t = at::empty(
                {static_cast<int64_t>(hw.size())},
                at::TensorOptions().dtype(c10::ScalarType::UInt32).device(at::kCPU));
            std::memcpy(t.data_ptr<uint32_t>(), hw.data(), hw.size() * sizeof(uint32_t));
            return t;
        }


        void dispatch(
            at::Tensor& out_expert_num_tokens, // output counts per expert
            at::Tensor& out_expert_x, // inputs into the grouped gemm kenrel, with optional scale factors (to be quantized)
            std::optional<at::Tensor>& out_expert_x_scale,
            at::Tensor& dp_x, // input tokens to be dispatched
            std::optional<at::Tensor>& dp_x_scale, // input scale factors to be dispatched, we allow optionality in where the quantize kernel is inserted
            at::Tensor& indices,
            at::Tensor &weights,
            std::optional<at::Tensor>& bound_m,
            bool do_send = true,
            bool do_recv = true,
            cudaStream_t stream = nullptr
        ) {
            HOST_ASSERT(initialized, "All2All handler is not initialized");
            HOST_ASSERT (do_send || do_recv, "do_send and do_recv must be true");    

            HOST_ASSERT(out_expert_num_tokens.size(0) == this->num_local_experts, "Shape check failed");
            HOST_ASSERT(dtype_of(out_expert_num_tokens) == c10::ScalarType::UInt32, "Dtype check failed");
            uint32_t *out_expert_num_tokens_ptr = out_expert_num_tokens.data_ptr<uint32_t>();

            uint32_t num_expert_tokens = out_expert_x.size(0);
            HOST_ASSERT(out_expert_x.dim() == 2, "Expected 2D tensor");
            HOST_ASSERT(out_expert_x.stride(1) == 1, "Expected stride of 1");
            HOST_ASSERT(dtype_of(out_expert_x) == this->in_dtype, "Dtype check failed");
            uint8_t * out_x_ptr = (uint8_t *) out_expert_x.data_ptr();
            uint32_t out_x_stride = get_type_size(this->in_dtype) * out_expert_x.stride(0); // in bytes
            
            float * out_x_scale_ptr = nullptr;
            uint32_t out_x_scale_stride_elem = 0;
            uint32_t out_x_scale_stride_token = 0;
            if (out_expert_x_scale.has_value()) {
                HOST_ASSERT(out_expert_x_scale->dim() == 2, "Expert x scale ndimensions = 2");
                HOST_ASSERT(this->scale_dtype.has_value(), "scale_dtype must be set when out_expert_x_scale is provided");
                HOST_ASSERT(dtype_of(*out_expert_x_scale) == this->scale_dtype.value(), "Scale dtypes do not match");
                out_x_scale_ptr = out_expert_x_scale->data_ptr<float>();
                out_x_scale_stride_elem = out_expert_x_scale->stride(1);
                out_x_scale_stride_token = out_expert_x_scale->stride(0);
            }

            HOST_ASSERT(dp_x.dim() == 2, "input tokens ndim == 2");
            HOST_ASSERT(dp_x.stride(1) == 1, "contiguous check");
            HOST_ASSERT(dtype_of(dp_x) == this->in_dtype, "input dtype check");

            uint8_t * x_ptr = (uint8_t *) dp_x.data_ptr();
            uint32_t x_stride = dp_x.stride(0);

            float * x_scale_ptr = nullptr;
            uint32_t x_scale_stride_token = 0;
            uint32_t x_scale_stride_elem = 0;

            if (dp_x_scale.has_value()) {
                HOST_ASSERT(dp_x_scale->dim() == 2, "token x scales check");
                HOST_ASSERT(this->scale_dtype.has_value(), "scale_dtype must be set when dp_x_scale is provided");
                HOST_ASSERT(dtype_of(*dp_x_scale) == this->scale_dtype.value(), "Dtype check");
                x_scale_ptr = dp_x_scale->data_ptr<float>();
                x_scale_stride_token = dp_x_scale->stride(0);
                x_scale_stride_elem = dp_x_scale->stride(1);
            }

            
            // weight and indices checks
            HOST_ASSERT(indices.dim() == 2 && indices.size(0) == dp_x.size(0) && indices.size(1) == this->num_experts_per_token, "indices shape check");
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::UInt32, "dtype check");

            uint32_t * indices_ptr = indices.data_ptr<uint32_t>();
            uint32_t indices_stride = indices.stride(0);

            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == dp_x.size(0) && weights.size(1) == this->num_experts_per_token, "Weight ndimension check");
            HOST_ASSERT(dtype_of(weights) == c10::ScalarType::Float, "dtype check");

            float * weights_ptr = weights.data_ptr<float>();
            uint32_t weights_stride = weights.stride(0);

            uint32_t * bound_m_ptr = nullptr;
            if (bound_m.has_value()) {
                HOST_ASSERT(bound_m->numel() == 1, "only one m bound");
                HOST_ASSERT(dtype_of(*bound_m) == c10::ScalarType::UInt32, "bound_m dtype check");
                bound_m_ptr = bound_m->data_ptr<uint32_t>();
            }
            
            if (do_send) {
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    printf("launching dispatch send from all2all_tk.hpp\n");
                }

                this->context->dispatch_send(
                    dp_x,
                    dp_x_scale.value(),
                    indices,
                    weights,
                    this->context->workspace.sync_counter,
                    dp_x.size(0),
                    stream
                );
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
            // wait on dispatch_route_done flag, then scatter num_routed through route_write_op


            std::atomic_ref<uint8_t> dispatch_route_ptr(*this->context->worker.dispatch_route_done);
            while (!dispatch_route_ptr.load()) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
            dispatch_route_ptr.store(0); // reset the flag
            this->context->worker.route_write_op();
            this->context->worker.process_routing_info();
            CUDA_CHECK(cudaMemcpyAsync(
                out_expert_num_tokens_ptr,
                this->context->worker.tokens_per_expert,
                this->num_local_experts * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice,
                stream));

            if (do_recv) {
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    printf("launching dispatch recv from all2all_tk.hpp\n");
                }
                this->context->dispatch_recv(
                    out_expert_x,
                    out_expert_x_scale.value(),
                    out_expert_num_tokens_ptr,
                    stream
                );
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
        }

        void combine(
            at::Tensor& out_tokens, // (activations for this device)
            at::Tensor& indices,
            at::Tensor& weights,
            at::Tensor & expert_y,
            std::optional<at::Tensor>& bound_m,
            bool do_send = true,
            bool do_recv = true,
            bool accumulate = false,
            cudaStream_t stream = nullptr
        ) {
            HOST_ASSERT(initialized, "All2All handler is not initialized");
            HOST_ASSERT(do_send || do_recv, "needed");

            uint32_t num_tokens = indices.size(0);
            uint32_t num_recv_tokens = expert_y.size(0);
            HOST_ASSERT(out_tokens.dim() == 2, "input tokens ndim == 2");
            HOST_ASSERT(out_tokens.stride(1) == 1, "contiguous check");
            HOST_ASSERT(dtype_of(out_tokens) == this->out_dtype, "input dtype check");

            uint8_t * out_tokens_ptr = (uint8_t *) out_tokens.data_ptr();
            uint32_t out_tokens_stride = out_tokens.stride(0);

            HOST_ASSERT(indices.dim() == 2 && indices.size(0) == num_tokens && indices.size(1) == this->num_experts_per_token, "indices shape check");
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::UInt32, "dtype check");

            uint32_t * indices_ptr = indices.data_ptr<uint32_t>();
            uint32_t indices_stride = indices.stride(0);

            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == num_tokens && weights.size(1) == this->num_experts_per_token, "Weight ndimension check");
            HOST_ASSERT(dtype_of(weights) == c10::ScalarType::Float, "dtype check");

            float * weights_ptr = weights.data_ptr<float>();
            uint32_t weights_stride = weights.stride(0);

            HOST_ASSERT(expert_y.dim() == 2, "outputs should be 2 dimensional (even for batched)");
            uint8_t * expert_y_ptr = (uint8_t *) expert_y.data_ptr();
            uint32_t expert_y_stride = expert_y.stride(0) * get_type_size(dtype_of(expert_y));

            uint32_t * bound_m_ptr = nullptr;
            if (bound_m.has_value()) {
                HOST_ASSERT(bound_m->numel() == 1, "only one m bound");
                HOST_ASSERT(dtype_of(*bound_m) == c10::ScalarType::UInt32, "bound_m dtype check");
                bound_m_ptr = bound_m->data_ptr<uint32_t>();
            }

            if (do_send) {
                this->context->combine_send(
                    expert_y_ptr, // Y = XW output of groupped gemm kenrel
                    expert_y_stride,
                    stream
                );
            }

            if (do_recv) {
                this->context->combine_recv(
                    num_tokens,// 
                    num_recv_tokens,
                    dtype_of(expert_y),
                    out_tokens_ptr,
                    out_tokens_stride,
                    indices_ptr,
                    indices_stride,
                    weights_ptr,
                    weights_stride,
                    bound_m_ptr,
                    accumulate,
                    stream );
            }

        }

        // Static wrapper methods for kernel_api entry points
        // static void a2a_dispatch(
        //     All2All& all2all,
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
        //     cudaStream_t stream = nullptr
        // ) {
        //     all2all.dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale,
        //                      dp_x, dp_x_scale, indices, weights, bound_m,
        //                      do_send, do_recv, stream);
        // }

        // static void a2a_combine(
        //     All2All& all2all,
        //     at::Tensor& out_tokens,
        //     at::Tensor& indices,
        //     at::Tensor& weights,
        //     at::Tensor& expert_y,
        //     std::optional<at::Tensor>& bound_m,
        //     bool do_send = true,
        //     bool do_recv = true,
        //     bool accumulate = false,
        //     cudaStream_t stream = nullptr
        // ) {
        //     all2all.combine(out_tokens, indices, weights, expert_y, bound_m,
        //                     do_send, do_recv, accumulate, stream);
        // }

    private:
        uint32_t num_experts;
        uint32_t hidden_dim;
        std::optional<uint32_t> hidden_dim_scale;
        c10::ScalarType in_dtype;
        c10::ScalarType out_dtype;
        std::optional<c10::ScalarType> scale_dtype;
        uint32_t num_experts_per_token;
        std::optional<uint32_t> max_private_tokens;
        int local_rank;
        uint32_t num_local_experts;
        uint32_t rank;
        int world_size;
        int dp_size;
        int  dp_group;
        int node_group;
        ParallelConfig parallel_config;
        bool initialized = false;
        std::optional<All2AllContext<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>> context;
};





}  // namespace moe_cuda
