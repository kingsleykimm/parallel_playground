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
#include "all2all_base.hpp"

namespace moe_cuda {


// System page size for padding
constexpr int PAGE_SIZE = 4096; // 4KB
constexpr int NUM_DEVICES = 4; // i'm only on a H100 x 4 node for now

// entrypoint - api methods should call this method. this is only instantiated once per proces at the start
template<int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
class All2All : public All2AllBase {

    public:
        struct RoutingDebugState {
            at::Tensor tokens_per_expert;
            at::Tensor source_rank;
            at::Tensor source_dispatch_offset;
            at::Tensor combine_send_offset;
            at::Tensor padded_index;
            uint32_t num_recv_tokens = 0;
        };

        All2All  (
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
        )  {
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
            this->world_size_ = parallel_config.world_size;
            this->initialized_ = true;
            this->hidden_dim_ = hidden_dim;
            this->hidden_dim_scale_ = hidden_dim_scale;
            this->num_experts_ = num_experts;
            this->num_experts_per_rank_ = num_experts / this->world_size_;
            this->num_experts_per_token_ = num_experts_per_token;
            this->in_dtype_ = in_dtype;
            this->out_dtype_ = out_dtype;
            this->scale_dtype_ = scale_dtype;
            this->local_rank_ = local_rank;
            this->rank_ = local_rank;
            this->dp_group_ = local_rank / parallel_config.dp_size;
            this->node_group_ = local_rank / parallel_config.node_size;
            this->dp_size_ = parallel_config.dp_size;
            this->parallel_config_ = parallel_config;
            uint32_t num_dp_groups = this->world_size_ / this->dp_size_;
            this->num_local_experts_ = host_ceil_div(this->num_experts_, this->world_size_);

            // recv buffer size
            uint32_t avg_tokens_per_expert = host_ceil_div(
                max_num_tokens * num_experts_per_token, num_experts
            ) * 1.2;
            uint32_t max_private_tokens;
            if (!max_private_tokens_opt.has_value()) {
                max_private_tokens = avg_tokens_per_expert * this->num_local_experts_;
            }
            else {
                max_private_tokens = max_private_tokens_opt.value();
            }
            HOST_ASSERT(max_private_tokens > 0, "max_private_tokens must be greater than 0");

            uint32_t num_tokens = max_num_tokens * num_dp_groups;
            // after the recv buffer size, the is bounded between [this->num_local_experts_ * expert_padding, num_tokens * num_experts_per_token]
            // the actual value is the
            this->max_recv_tokens_ = host_align(std::max(
                std::min(num_tokens * num_experts_per_token
                    + this->num_local_experts_ * (expert_padding - 1),
              this->num_local_experts_ * num_tokens), // upper bound
               this->num_local_experts_ * expert_padding), // lower bound
               expert_padding);
            if (hidden_dim_scale.has_value() && scale_dtype.has_value()) {

                HOST_ASSERT(scale_dtype.value() == c10::ScalarType::Float, "Only float scales supported");
            }


            this->context_.emplace(
                hidden_dim,
                hidden_dim_scale,
                get_type_size(in_dtype),
                get_type_size(out_dtype),
                out_dtype,
                scale_dtype.has_value()
                    ? std::optional<size_t>(get_type_size(scale_dtype.value()))
                    : std::nullopt,
                max_num_tokens,
                max_recv_tokens_,
                max_private_tokens,
                num_experts,
                expert_padding,
                num_experts_per_token,
                this->rank_,
                this->dp_size_,
                parallel_config.node_size,
                parallel_config.world_size,
                stream);
        }


        // for testing
        RoutingDebugState debug_routing_state(cudaStream_t stream = nullptr) const {
            HOST_ASSERT(initialized_, "All2All handler is not initialized");
            HOST_ASSERT(this->context_.has_value(), "All2All context is not initialized");

            auto effective_stream = stream != nullptr ? stream : this->context_->stream;
            auto opts =
                at::TensorOptions()
                    .dtype(c10::ScalarType::UInt32)
                    .device(c10::Device(c10::DeviceType::CUDA, this->local_rank_));

            RoutingDebugState state;
            state.num_recv_tokens = this->context_->worker.num_recv_tokens;

            c10::IntArrayRef size = {static_cast<int64_t>(this->num_local_experts_)};
            state.tokens_per_expert = at::empty(size, opts);

            // cudaPointerAttributes attr;
            // CUDA_CHECK(cudaPointerGetAttributes(&attr, this->context->worker.tokens_per_expert));
            // printf("tokens_per_expert is in %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");

            
            CUDA_CHECK(cudaMemcpyAsync(
                (uint32_t *)state.tokens_per_expert.data_ptr(),
                this->context_->worker.tokens_per_expert,
                this->num_local_experts_ * sizeof(uint32_t),
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
                    this->context_->worker.source_rank,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.source_dispatch_offset.data_ptr(),
                    this->context_->worker.source_dispatch_offset,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.combine_send_offset.data_ptr(),
                    this->context_->worker.combine_send_offset,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    (uint32_t * )state.padded_index.data_ptr(),
                    this->context_->worker.padded_index,
                    state.num_recv_tokens * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice,
                    effective_stream));
            }

            return state;
        }

        at::Tensor debug_num_routed() const {
            HOST_ASSERT(initialized_, "All2All handler is not initialized");
            HOST_ASSERT(this->context_.has_value(), "All2All context is not initialized");
            const auto& hw = this->context_->worker.host_num_routed;
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
        ) override {
            HOST_ASSERT(initialized_, "All2All handler is not initialized");
            HOST_ASSERT (do_send || do_recv, "do_send and do_recv must be true");

            HOST_ASSERT(out_expert_num_tokens.size(0) == this->num_local_experts_, "Shape check failed");
            HOST_ASSERT(dtype_of(out_expert_num_tokens) == c10::ScalarType::UInt32, "Dtype check failed");
            uint32_t *out_expert_num_tokens_ptr = out_expert_num_tokens.data_ptr<uint32_t>();

            uint32_t num_expert_tokens = out_expert_x.size(0);
            HOST_ASSERT(out_expert_x.dim() == 2, "Expected 2D tensor");
            HOST_ASSERT(out_expert_x.stride(1) == 1, "Expected stride of 1");
            HOST_ASSERT(dtype_of(out_expert_x) == this->in_dtype_, "Dtype check failed");
            uint8_t * out_x_ptr = (uint8_t *) out_expert_x.data_ptr();
            uint32_t out_x_stride = get_type_size(this->in_dtype_) * out_expert_x.stride(0); // in bytes
            
            float * out_x_scale_ptr = nullptr;
            uint32_t out_x_scale_stride_elem = 0;
            uint32_t out_x_scale_stride_token = 0;
            if (out_expert_x_scale.has_value()) {
                HOST_ASSERT(out_expert_x_scale->dim() == 2, "Expert x scale ndimensions = 2");
                HOST_ASSERT(this->scale_dtype_.has_value(), "scale_dtype must be set when out_expert_x_scale is provided");
                HOST_ASSERT(dtype_of(*out_expert_x_scale) == this->scale_dtype_.value(), "Scale dtypes do not match");
                out_x_scale_ptr = out_expert_x_scale->data_ptr<float>();
                out_x_scale_stride_elem = out_expert_x_scale->stride(1);
                out_x_scale_stride_token = out_expert_x_scale->stride(0);
            }

            HOST_ASSERT(dp_x.dim() == 2, "input tokens ndim == 2");
            HOST_ASSERT(dp_x.stride(1) == 1, "contiguous check");
            HOST_ASSERT(dtype_of(dp_x) == this->in_dtype_, "input dtype check");

            uint8_t * x_ptr = (uint8_t *) dp_x.data_ptr();
            uint32_t x_stride = dp_x.stride(0);

            float * x_scale_ptr = nullptr;
            uint32_t x_scale_stride_token = 0;
            uint32_t x_scale_stride_elem = 0;

            if (dp_x_scale.has_value()) {
                HOST_ASSERT(dp_x_scale->dim() == 2, "token x scales check");
                HOST_ASSERT(this->scale_dtype_.has_value(), "scale_dtype must be set when dp_x_scale is provided");
                HOST_ASSERT(dtype_of(*dp_x_scale) == this->scale_dtype_.value(), "Dtype check");
                x_scale_ptr = dp_x_scale->data_ptr<float>();
                x_scale_stride_token = dp_x_scale->stride(0);
                x_scale_stride_elem = dp_x_scale->stride(1);
            }

            
            // weight and indices checks
            HOST_ASSERT(indices.dim() == 2 && indices.size(0) == dp_x.size(0) && indices.size(1) == this->num_experts_per_token_, "indices shape check");
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::Int, "dtype check");


            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == dp_x.size(0) && weights.size(1) == this->num_experts_per_token_, "Weight ndimension check");
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

                this->context_->dispatch_send(
                    dp_x,
                    dp_x_scale.value(),
                    indices,
                    weights,
                    this->context_->workspace.sync_counter,
                    dp_x.size(0),
                    stream
                );
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
            // wait on dispatch_route_done flag, then scatter num_routed through route_write_op


            std::atomic_ref<uint8_t> dispatch_route_ptr(*this->context_->worker.dispatch_route_done);
            while (!dispatch_route_ptr.load()) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
            dispatch_route_ptr.store(0); // reset the flag
            this->context_->worker.route_write_op();
            this->context_->worker.process_routing_info();
            CUDA_CHECK(cudaMemcpyAsync(
                out_expert_num_tokens_ptr,
                this->context_->worker.tokens_per_expert,
                this->num_local_experts_ * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice,
                stream));

            if (do_recv) {
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    printf("launching dispatch recv from all2all_tk.hpp\n");
                }
                this->context_->dispatch_recv(
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
        ) override {
            HOST_ASSERT(initialized_, "All2All handler is not initialized");
            HOST_ASSERT(do_send || do_recv, "needed");

            uint32_t num_tokens = indices.size(0);
            uint32_t num_recv_tokens = expert_y.size(0);
            HOST_ASSERT(out_tokens.dim() == 2, "input tokens ndim == 2");
            HOST_ASSERT(out_tokens.stride(1) == 1, "contiguous check");
            HOST_ASSERT(dtype_of(out_tokens) == this->out_dtype_, "input dtype check");

            uint8_t * out_tokens_ptr = (uint8_t *) out_tokens.data_ptr();

            HOST_ASSERT(indices.dim() == 2 && indices.size(0) == num_tokens && indices.size(1) == this->num_experts_per_token_, "indices shape check");
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::Int, "dtype check");

            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == num_tokens && weights.size(1) == this->num_experts_per_token_, "Weight ndimension check");
            HOST_ASSERT(dtype_of(weights) == c10::ScalarType::Float, "dtype check");

            float * weights_ptr = weights.data_ptr<float>();
            uint32_t weights_stride = weights.stride(0);

            HOST_ASSERT(expert_y.dim() == 2, "outputs should be 2 dimensional (even for batched)");

            uint32_t * bound_m_ptr = nullptr;
            if (bound_m.has_value()) {
                HOST_ASSERT(bound_m->numel() == 1, "only one m bound");
                HOST_ASSERT(dtype_of(*bound_m) == c10::ScalarType::UInt32, "bound_m dtype check");
                bound_m_ptr = bound_m->data_ptr<uint32_t>();
            }

            if (do_send) {
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    printf("launching combine send from all2all_tk.hpp\n");
                }
                this->context_->combine_send(
                    expert_y, // Y = XW output of groupped gemm kenrel
                    stream
                );
            }

            if (do_recv) {
                if (get_env<int>("A2A_DEBUG") > 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    printf("launching combine recv from all2all_tk.hpp\n");
                }
                this->context_->combine_recv(
                    out_tokens, indices, weights, accumulate, stream
                );
            }

        }



        uint32_t max_recv_tokens() const { return max_recv_tokens_; }
        uint32_t num_experts_per_rank() const { return num_experts_per_rank_; }
        All2AllContext<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>& context() {
            HOST_ASSERT(context_.has_value(), "All2All context is not initialized");
            return context_.value();
        }
        const All2AllContext<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>& context() const {
            HOST_ASSERT(context_.has_value(), "All2All context is not initialized");
            return context_.value();
        }

    private:
        uint32_t num_experts_;
        uint32_t hidden_dim_;
        std::optional<uint32_t> hidden_dim_scale_;
        c10::ScalarType in_dtype_;
        c10::ScalarType out_dtype_;
        std::optional<c10::ScalarType> scale_dtype_;
        uint32_t num_experts_per_rank_;
        uint32_t num_experts_per_token_;
        std::optional<uint32_t> max_private_tokens_;
        int local_rank_;
        uint32_t num_local_experts_;
        uint32_t rank_;
        int world_size_;
        int dp_size_;
        int dp_group_;
        int node_group_;
        ParallelConfig parallel_config_;
        bool initialized_ = false;
        std::optional<All2AllContext<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>> context_;
        uint32_t max_recv_tokens_;
};







}  // namespace moe_cuda
