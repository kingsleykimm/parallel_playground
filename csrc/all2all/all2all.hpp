#pragma once
#include <optional>
#include <cuda.h>
#include <moe_cuda/kernels/common/common.hpp>
#include <runtime/utils.h>
#include <moe_cuda/dtype.h>
#include <runtime/parallel.h>
#include <moe_cuda/error.hpp>
#include <runtime/cumem.h>
#include <jit/utils/lazy_init.hpp>
#include <runtime/tensor.h>
#ifdef MOE_CUDA_USE_MPI
#include <mpi.h>
#include "a2a_context.hpp"
#endif

namespace moe_cuda {

#ifdef MOE_CUDA_USE_MPI
// System page size for padding
constexpr int PAGE_SIZE = 4096; // 4KB

struct NVLinkLoad { // load to send through MPI
    CUMemExportData sync_fd;
    CUMemExportData recv_fd;
    CUMemExportData send_fd;
    CUMemExportData num_routed_fd;
};

struct NVLRankMappings {
    CUMemMapping sync_mapping;
    CUMemMapping recv_mapping;
    CUMemMapping send_mapping;
    CUMemMapping num_routed_mapping;
};

// entrypoint - api methods should call this method
class All2All {

    public:
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
            std::optional<ParallelGroup> dp_group,
            ParallelGroup node_group,
            int device,
            ParallelGroup global_group,
            cudaStream_t stream
        ) {
            this->initialized = true;
            this->hidden_dim = hidden_dim;
            this->hidden_dim_scale = hidden_dim_scale;
            this->num_experts = num_experts;
            this->num_experts_per_token = num_experts_per_token;
            this->in_dtype = in_dtype;
            this->out_dtype = out_dtype;
            this->scale_dtype = scale_dtype;
            this->device = device;
            this->dp_group = dp_group;
            this->node_group = node_group;
            if (dp_group.has_value()) {
                this->dp_size = dp_group->size;
            }
            else {
                this->dp_size = 1;
            }

            // global group tells global parallel information
            this->rank = global_group.rank;
            this->world_size = global_group.size;
            uint32_t num_dp_groups = this->world_size / this->dp_size;

            this->num_local_experts = ti_ceil_div(this->num_experts, this->world_size);

            // recv buffer size
            uint32_t avg_tokens_per_expert = ti_ceil_div(
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

            // first min is to ensure there are at most num_tokens * num_local_experts, the maximum possible upper bound where all tokens get mapped to this rank
            // the next max is to ensure at least num_local_experts * expert_padding is on each rank - this is the lower bound of expert tokens mapped to this device
            max_recv_tokens += ti_align(std::max(
                std::min(num_tokens * num_experts_per_token 
                    + this->num_local_experts * (expert_padding - 1), 
              this->num_local_experts * num_tokens),
               this->num_local_experts * expert_padding), 
               expert_padding);

            // num_routed initialization
            // TODO: zero initialization here?
            this->_num_routed_handle = CUMemAllocHandle(num_dp_groups * num_experts * sizeof(uint32_t), device, CUMemHandleKind::FileDescriptor);
            this->_num_routed_mapping = _num_routed_handle.map(this->device);
            
            // siz send buffer sizes + token_dim (combing optional scale factors)
            uint32_t token_dim_dispatch = ti_align(hidden_dim * get_type_size(in_dtype), 16) + 16;
            if (hidden_dim_scale.has_value() && scale_dtype.has_value()) {
                
                token_dim_dispatch += ti_align(hidden_dim_scale.value() * get_type_size(scale_dtype.value()), 16);

                HOST_ASSERT(scale_dtype.value() == c10::ScalarType::Float, "Only float scales supported");
            }

            // combine token dim is just the hidden dim, since the outputs of MoE kernels are unquantized

            uint32_t token_dim_combine = ti_align(hidden_dim * get_type_size(in_dtype), 16);
            uint32_t token_dim = std::max(token_dim_combine, token_dim_dispatch);


            
            uint32_t send_buffer_bytes = ti_align(max_recv_tokens * token_dim, PAGE_SIZE);
            this->_send_buffer_handle = CUMemAllocHandle (send_buffer_bytes, device, CUMemHandleKind::FileDescriptor);
            this->_send_buffer_mapping = _send_buffer_handle.map(this->device);

            this->_recv_buffer_handle = CUMemAllocHandle (send_buffer_bytes, device, CUMemHandleKind::FileDescriptor);
            this->_recv_buffer_mapping = _recv_buffer_handle.map(this->device);
            
            std::vector<uint32_t *> sync_ptrs;
            std::vector<uint8_t *> send_ptrs;
            std::vector<uint8_t *> recv_ptrs;
            std::vector<uint32_t *> num_routed_ptrs;
            if (node_group.size > 1) { // nvlink buffers
                printf("Setting up NVLink for node group size (%d)", node_group.size);

                this->_sync_buffer_handle = CUMemAllocHandle(node_group.size * 2 * sizeof(uint32_t), this->device, CUMemHandleKind::FileDescriptor);
                CUMemMapping sync_mapping = _sync_buffer_handle.map(this->device);
                CUDA_CHECK(cudaMemset(sync_mapping.data_ptr(), 0, sizeof(uint32_t) * node_group.size * 2));

                {
                    auto local_handle = NVLinkLoad{
                    this->_sync_buffer_handle.export_handle(),
                    this->_recv_buffer_handle.export_handle(),
                    this->_send_buffer_handle.export_handle(),
                    this->_num_routed_handle.export_handle()
                    };

                    std::vector<NVLinkLoad> handles (node_group.size);
                    MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE, handles.data(), sizeof(local_handle) * node_group.size, MPI_BYTE, node_group.comm);

                    for (size_t i = 0; i < handles.size(); i++) {
                        auto handle = handles[i];

                        if (i == this->node_group.rank) {
                            this->nvl_rank_mappings.push_back(NVLRankMappings {
                                std::move(sync_mapping), std::move(this->_recv_buffer_mapping), std::move(this->_send_buffer_mapping), std::move(this->_num_routed_mapping)
                            });
                        }
                        else {
                            this->nvl_rank_mappings.push_back(NVLRankMappings {
                                CUMemImportHandle::from_export(handle.sync_fd).map(this->device),
                                CUMemImportHandle::from_export(handle.recv_fd).map(this->device),
                                CUMemImportHandle::from_export(handle.send_fd).map(this->device),
                                CUMemImportHandle::from_export(handle.num_routed_fd).map(this->device)
                            });
                        }
                    }
                    MPI_Barrier(node_group.comm);
                }

                for (uint32_t i = 0; i < node_group.size; i++) {
                    recv_ptrs.push_back((uint8_t * )this->nvl_rank_mappings[i].recv_mapping.data_ptr());
                    send_ptrs.push_back((uint8_t * )this->nvl_rank_mappings[i].recv_mapping.data_ptr());
                    sync_ptrs.push_back((uint32_t * )this->nvl_rank_mappings[i].recv_mapping.data_ptr());
                    num_routed_ptrs.push_back((uint32_t * )this->nvl_rank_mappings[i].num_routed_mapping.data_ptr());
                }
            }
            this->context = All2AllContext(
                hidden_dim,
                hidden_dim_scale,
                get_type_size(in_dtype),
                get_type_size(out_dtype),
                out_dtype,
                scale_dtype.has_value() ? get_type_size(scale_dtype.value()) : 0,
                max_num_tokens,
                max_recv_tokens,
                max_private_tokens,
                num_experts,
                expert_padding,
                num_experts_per_token,
                rank,
                dp_size,
                node_group.size,
                world_size,
                (uint32_t * )this->_num_routed_mapping.data_ptr(),
                (uint8_t * )this->_send_buffer_mapping.data_ptr(),
                (uint8_t * )this->_recv_buffer_mapping.data_ptr(),
                sync_ptrs.data(),
                send_ptrs.data(),
                recv_ptrs.data(),
                num_routed_ptrs.data(),
                device,
                stream
            );
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

            HOST_ASSERT(out_expert_num_tokens.size(0) == num_local_experts, "Shape check failed");
            HOST_ASSERT(dtype_of(out_expert_num_tokens) == c10::ScalarType::Int, "Dtype check failed");
            uint32_t *out_expert_num_tokens_ptr = out_expert_num_tokens.data_ptr<uint32_t>();

            uint32_t num_expert_tokens = out_expert_x.size(0);
            HOST_ASSERT(out_expert_x.dim() == 2, "Expected 2D tensor");
            HOST_ASSERT(out_expert_x.stride(1) == 1, "Expected stride of 1");
            HOST_ASSERT(dtype_of(out_expert_x) == this->in_dtype, "Dtype check failed");
            uint8_t * out_x_ptr = out_expert_x.data_ptr<uint8_t>();
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

            uint8_t * x_ptr = dp_x.data_ptr<uint8_t>();
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
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::Int, "dtype check");

            uint32_t * indices_ptr = indices.data_ptr<uint32_t>();
            uint32_t indices_stride = indices.stride(0);

            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == dp_x.size(0) && weights.size(1) == this->num_experts_per_token, "Weight ndimension check");
            HOST_ASSERT(dtype_of(weights) == c10::ScalarType::Float, "dtype check");

            float * weights_ptr = weights.data_ptr<float>();
            uint32_t weights_stride = weights.stride(0);

            uint32_t * bound_m_ptr = nullptr;
            if (bound_m.has_value()) {
                HOST_ASSERT(bound_m->numel() == 1, "only one m bound");
                bound_m_ptr = bound_m->data_ptr<uint32_t>();
            }
            
            if (do_send) {
                this->context.dispatch_send(
                    dp_x.size(0),
                    x_ptr,
                    x_stride,
                    x_scale_ptr, // can be nullptr if not
                    x_scale_stride_elem, // these can be 0
                     x_scale_stride_token,
                      indices_ptr,
                    indices_stride,
                     weights_ptr,
                     weights_stride,
                    bound_m_ptr,
                     stream
                );
            }

            this->context.worker.process_routing_info(); // although there are cuda async functions in here, they are all done on the same stream per device

            if (do_recv) {
                this->context.dispatch_recv(
                    out_expert_num_tokens_ptr,
                    out_x_ptr,
                    out_x_stride,
                    reinterpret_cast<uint8_t*>(out_x_scale_ptr),
                    out_x_scale_stride_elem,
                    out_x_scale_stride_token,
                    stream
                );
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
            HOST_ASSERT(dtype_of(out_tokens) == this->in_dtype, "input dtype check");

            uint8_t * out_tokens_ptr = out_tokens.data_ptr<uint8_t>();
            uint32_t out_tokens_stride = out_tokens.stride(0);

            HOST_ASSERT(indices.dim() == 2 && indices.size(0) == num_tokens && indices.size(1) == this->num_experts_per_token, "indices shape check");
            HOST_ASSERT(dtype_of(indices) == c10::ScalarType::Int, "dtype check");

            uint32_t * indices_ptr = indices.data_ptr<uint32_t>();
            uint32_t indices_stride = indices.stride(0);

            HOST_ASSERT(weights.dim() == 2 && weights.size(0) == num_tokens && weights.size(1) == this->num_experts_per_token, "Weight ndimension check");
            HOST_ASSERT(dtype_of(weights) == c10::ScalarType::Float, "dtype check");

            float * weights_ptr = weights.data_ptr<float>();
            uint32_t weights_stride = weights.stride(0);

            HOST_ASSERT(expert_y.dim() == 2, "outputs should be 2 dimensional (even for batched)");
            uint8_t * expert_y_ptr = expert_y.data_ptr<uint8_t>();
            uint32_t expert_y_stride = expert_y.stride(0) * get_type_size(dtype_of(expert_y));

            uint32_t * bound_m_ptr = nullptr;
            if (bound_m.has_value()) {
                HOST_ASSERT(bound_m->numel() == 1, "only one m bound");
                bound_m_ptr = bound_m->data_ptr<uint32_t>();
            }

            if (do_send) {
                this->context.combine_send(
                    expert_y_ptr, // Y = XW output of groupped gemm kenrel
                    expert_y_stride,
                    stream
                );
            }

            if (do_recv) {
                this->context.combine_recv(
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
        int device;
        uint32_t num_local_experts;
        uint32_t rank;
        int world_size;
        int dp_size;
        std::optional<ParallelGroup> dp_group;
        ParallelGroup node_group;
        bool initialized;
        CUMemAllocHandle _send_buffer_handle, _recv_buffer_handle, _sync_buffer_handle, _num_routed_handle;
        CUMemMapping _send_buffer_mapping, _recv_buffer_mapping, _num_routed_mapping;

        std::vector<NVLRankMappings> nvl_rank_mappings; // this is used to persist the mappings we receive from other devices after importing
        All2AllContext context;
};

#else

class All2All {
    public:
        template <typename... Args>
        explicit All2All(Args&&...) {
            HOST_ERROR("All2All is unavailable: build with MPI to enable all2all support");
        }

        void dispatch(
            at::Tensor&,
            at::Tensor&,
            std::optional<at::Tensor>&,
            at::Tensor&,
            std::optional<at::Tensor>&,
            at::Tensor&,
            at::Tensor&,
            std::optional<at::Tensor>&,
            bool = true,
            bool = true,
            cudaStream_t = nullptr) {
            HOST_ERROR("All2All dispatch is unavailable: build with MPI to enable all2all support");
        }

        void combine(
            at::Tensor&,
            at::Tensor&,
            at::Tensor&,
            at::Tensor&,
            std::optional<at::Tensor>&,
            bool = true,
            bool = true,
            bool = false,
            cudaStream_t = nullptr) {
            HOST_ERROR("All2All combine is unavailable: build with MPI to enable all2all support");
        }
};

#endif

}  // namespace moe_cuda
