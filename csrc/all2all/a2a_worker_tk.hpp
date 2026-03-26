#pragma once
#include "runtime/utils.h"
#include <cstdint>
#include <driver_types.h>
#include <functional>
#include <kittens.cuh>
#include <moe_cuda/error.hpp>
#include <optional>
#include <pyutils/parallel_tensor.cuh>
#include <vector>

/*
Worker state to perform auxiliary functilns while kernels are running, separated
from kernel frontends
*/

class WorkerState {
public:
  uint32_t max_num_tokens;
  uint32_t max_recv_tokens;
  uint32_t max_private_tokens;
  uint32_t hidden_dim;
  std::optional<uint32_t> hidden_dim_scale;
  uint32_t in_elemsize;
  uint32_t out_elemsize;
  std::optional<uint32_t> scale_elemsize;
  uint32_t num_experts;
  uint32_t num_experts_per_token;
  uint32_t expert_padding;
  uint32_t rank;
  uint32_t dp_rank;
  uint32_t dp_group;
  uint32_t dp_size;
  uint32_t node_size;
  uint32_t world_size;
  std::vector<void *> num_routed_ptrs;
  std::vector<uint32_t> host_num_routed;

  std::vector<uint32_t> tokens_per_expert_host;
  uint32_t *tokens_per_expert;
  uint32_t *source_rank;
  uint32_t *source_dispatch_offset;
  uint32_t *combine_send_offset;
  uint32_t *padded_index;   // grouped layout for contiguous
  uint32_t *padded_offsets; // grouped layout for masked
  uint32_t num_recv_tokens;
  uint8_t *dispatch_route_done;

  uint32_t num_dp_groups;

  cudaStream_t stream;

  std::function<void()> route_write_op;

  WorkerState() {};

  WorkerState(uint32_t max_num_tokens, uint32_t max_recv_tokens,
              uint32_t max_private_tokens, uint32_t hidden_dim,
              std::optional<uint32_t> hidden_dim_scale, uint32_t in_elemsize,
              uint32_t out_elemsize, std::optional<uint32_t> scale_elemsize,
              uint32_t num_experts, uint32_t num_experts_per_token,
              uint32_t expert_padding, uint32_t rank, uint32_t dp_size,
              uint32_t node_size, uint32_t world_size,
              std::vector<void *> num_routed_ptrs,

              cudaStream_t stream) {
    this->max_num_tokens = max_num_tokens;
    this->max_recv_tokens = max_recv_tokens;
    this->max_private_tokens = max_private_tokens;
    this->hidden_dim = hidden_dim;
    this->hidden_dim_scale = hidden_dim_scale;
    this->in_elemsize = in_elemsize;
    this->out_elemsize = out_elemsize;
    this->scale_elemsize = scale_elemsize;
    this->num_experts = num_experts;
    this->num_experts_per_token = num_experts_per_token;
    this->expert_padding = expert_padding;
    this->rank = rank;
    this->dp_rank = rank % dp_size;
    this->dp_group = rank / dp_size;
    this->dp_size = dp_size;
    this->node_size = node_size;
    this->world_size = world_size;
    this->num_routed_ptrs = num_routed_ptrs;
    this->stream = stream;

    this->num_dp_groups = world_size / dp_size;
    // initialize buffers on devie that are needed by kernels

    // worker one alloc
    uint32_t num_local_experts = (num_experts + world_size - 1) / world_size;

    CUDA_CHECK(cudaSetDevice(rank));
    size_t worker_malloc_size = 0;
    worker_malloc_size +=
        num_local_experts * sizeof(uint32_t); // tokens_per_expert
    worker_malloc_size += max_recv_tokens * sizeof(uint32_t); // source_rank
    worker_malloc_size +=
        max_recv_tokens * sizeof(uint32_t); // source_dispatch_offset
    worker_malloc_size +=
        max_recv_tokens * sizeof(uint32_t); // combine_send_offset
    worker_malloc_size += max_recv_tokens * sizeof(uint32_t); // padded_index
    worker_malloc_size += max_recv_tokens * sizeof(uint32_t); // padded_offsets

    uint32_t *worker_base_ptr;
    CUDA_CHECK(cudaMallocAsync(&worker_base_ptr, worker_malloc_size, stream));

    this->tokens_per_expert = worker_base_ptr;
    this->source_rank = worker_base_ptr + num_local_experts;
    this->source_dispatch_offset =
        worker_base_ptr + num_local_experts + max_recv_tokens;
    this->combine_send_offset =
        worker_base_ptr + num_local_experts + max_recv_tokens * 2;
    this->padded_index =
        worker_base_ptr + num_local_experts + max_recv_tokens * 3;
    this->padded_offsets =
        worker_base_ptr + num_local_experts + max_recv_tokens * 4;

    CUDA_CHECK(cudaHostAlloc(&this->dispatch_route_done, sizeof(uint8_t),
                             cudaHostAllocMapped));
    this->host_num_routed =
        std::vector<uint32_t>(num_dp_groups * num_experts, 0);
    this->route_write_op = [&]() {
      // Wait for all ranks to have written their local num_routed slice to
      // device memory (each rank's GPU kernel signals this via
      // dispatch_route_done + __threadfence_system)
      kittens::py::TKParallelTensor::brokers_
          .at({(int)this->rank, (int)this->world_size})
          .sync(this->world_size);
      // Pull each dp_group's slice directly from the source rank's device
      // memory. NVLink reads are coherent (source DRAM is authoritative after
      // __threadfence_system); unlike D2D pushed writes, there is no
      // posted-write ordering ambiguity.
      for (uint32_t g = 0; g < this->num_dp_groups; g++) {
        uint32_t source_rank = g * this->dp_size + this->dp_rank;
        CUDA_CHECK(cudaMemcpy(
            this->host_num_routed.data() + g * this->num_experts,
            (uint8_t *)this->num_routed_ptrs[source_rank] +
                g * this->num_experts * sizeof(uint32_t),
            this->num_experts * sizeof(uint32_t), cudaMemcpyDefault));
      }

      if (get_env("A2A_DEBUG", 0)) {
        if (this->rank == 2) {
          for (int i = 0; i < this->host_num_routed.size(); i++) {
            printf("num_routed[%d] = %d\n", i, this->host_num_routed[i]);
          }
        }
      }
    };
  }

  uint32_t get_num_routed(uint32_t dp_group, uint32_t expert) {
    HOST_ASSERT(dp_group < this->world_size / this->dp_size,
                "Dp group is out of bounds");
    HOST_ASSERT(expert < this->num_experts, "Expert argument is out of bounds");

    return this->host_num_routed[dp_group * this->num_experts + expert];
  }

  // this entire method sets up all the different padding metadata to ensure
  // that we can keep the receiving buffers of each rank contiguous
  void process_routing_info() {
    uint32_t num_dp_groups = this->world_size / this->dp_size;
    uint32_t experts_per_rank =
        (this->num_experts + this->world_size - 1) / this->world_size;

    uint32_t first_local_expert = this->rank * experts_per_rank;
    uint32_t last_local_expert =
        std::min(first_local_expert + experts_per_rank, this->num_experts);
    uint32_t num_local_experts = last_local_expert - first_local_expert;
    uint32_t rank_node = this->rank / this->node_size;
    uint32_t groups_per_node = this->node_size / this->dp_size;
    uint32_t num_nodes = this->world_size / this->node_size;
    HOST_ASSERT(this->world_size == this->node_size,
                "TK all2all test path is single-node only");
    HOST_ASSERT(num_nodes == 1, "Inter-node routing is not supported");
    std::vector<uint32_t> tokens_from_group(num_dp_groups, 0);

    // =========== OFFSETS FOR GROUPS =====================
    // per group offset into receiving buffer, tokens that come into this rank,
    // per group
    std::vector<uint32_t> src_group_offset(num_dp_groups, 0);

    // for each dp_group, token offset
    std::vector<uint32_t> dst_group_offset(num_dp_groups, 0);

    // local number, padded to experts_per_rank
    this->tokens_per_expert_host = std::vector<uint32_t>(experts_per_rank, 0);

    // =========== ABSOLUTE OFFSETS FROM CURRENT RANK=====================
    // per rank offset of tokens FROM current rank
    std::vector<uint32_t> dispatch_from_cur_offset(this->world_size, 0);
    // per expert offset (in tokens) FROM current rank
    std::vector<uint32_t> dispatch_from_cur_expert_offset(this->num_experts, 0);
    std::vector<uint32_t> tokens_to_rank(this->world_size, 0);

    this->num_recv_tokens = 0; // tracker of receiving tokens and offsets
    {
      uint32_t rank_offset = 0;
      for (uint32_t dp_group = 0; dp_group < num_dp_groups; dp_group++) {
        uint32_t num_tokens = 0;

        // process outgoing tokens
        for (uint32_t i = 0; i < this->dp_size; i++) {
          uint32_t rank = dp_group * this->dp_size + i;
          dispatch_from_cur_offset[rank] = rank_offset;

          uint32_t first_expert = rank * experts_per_rank;
          uint32_t last_expert =
              std::min(first_expert + experts_per_rank, this->num_experts);

          // for current rank's experts, we accumulate
          for (uint32_t expert = first_expert; expert < last_expert; expert++) {
            uint32_t n = this->get_num_routed(
                this->dp_group,
                expert); // this tells the number route FROM (dp_group, expert)
            dispatch_from_cur_expert_offset[expert] =
                rank_offset; // cumulative sum for offsets
            tokens_to_rank[rank] += n;
            rank_offset += n;
          }
        }

        // calculate group offset, matches position calculation in
        // a2a_dispatch_send_tk.cuh
        uint32_t offset = 0;
        for (uint32_t expert = 0; expert < first_local_expert; expert++) {
          offset += this->get_num_routed(dp_group, expert);
        }
        dst_group_offset[dp_group] = offset;

        // for each dp_group, we get the number of tokens FROM that dp_group
        // coming into current rank
        for (uint32_t expert = first_local_expert; expert < last_local_expert;
             expert++) {
          uint32_t n = this->get_num_routed(dp_group, expert);
          num_tokens += n;
          this->tokens_per_expert_host[expert - first_local_expert] += n;
        }

        tokens_from_group[dp_group] += num_tokens;
        src_group_offset[dp_group] =
            num_recv_tokens; // cumulative sum of tokens_from_group
        num_recv_tokens += num_tokens;
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        this->tokens_per_expert, this->tokens_per_expert_host.data(),
        sizeof(uint32_t) * num_local_experts, cudaMemcpyHostToDevice, stream));

    // =========== PADDING CALCS FOR GROUPED GEMM =====================
    std::vector<uint32_t> padded_offsets;
    padded_offsets.reserve(num_local_experts);
    uint32_t base_expert_offset = 0;

    for (auto count : this->tokens_per_expert_host) {
      uint32_t padded_expert_count =
          ((count + this->expert_padding - 1) / this->expert_padding) *
          this->expert_padding;
      padded_offsets.push_back(base_expert_offset);
      base_expert_offset += padded_expert_count;
    }

    // where in the current recv or peer's send buffer to read from, used in
    // dispatch_recv
    std::vector<uint32_t> source_dispatch_offset =
        std::vector<uint32_t>(num_recv_tokens, 0);
    // ? don't know yet
    std::vector<uint32_t> combine_send_offset =
        std::vector<uint32_t>(num_recv_tokens, 0);
    std::vector<uint32_t> source_rank =
        std::vector<uint32_t>(num_recv_tokens, 0);
    std::vector<uint32_t> padded_index =
        std::vector<uint32_t>(num_recv_tokens, -1);

    // uint32_t base_offset = this->max_private_tokens * num_dp_groups;
    uint32_t last = 0;

    // sum of tokens coming from each group
    std::vector<uint32_t> src_dispatch_count =
        std::vector<uint32_t>(num_dp_groups, 0);
    std::vector<uint32_t> src_combine_count =
        std::vector<uint32_t>(num_dp_groups, 0);
    std::vector<uint32_t> expert_count =
        std::vector<uint32_t>(num_local_experts, 0);

    auto route_group = [&](uint32_t peer_group) -> uint32_t {
      uint32_t num_routed = 0;
      for (uint32_t expert = first_local_expert; expert < last_local_expert;
           expert++) {
        uint32_t private_offset = this->max_private_tokens * peer_group;
        // number of tokens routed from current peer_group
        uint32_t routed = this->get_num_routed(peer_group, expert);
        num_routed += routed;

        uint32_t local_expert = expert - first_local_expert;
        uint32_t src_offset = src_group_offset[peer_group];
        uint32_t dst_offset = dst_group_offset[peer_group];

        uint32_t peer_rank =
            (peer_group * this->dp_size + this->dp_rank) % this->world_size;

        // per token assigned to expert
        for (uint32_t i = 0; i < routed; i++) {
          if (peer_rank == this->rank) {
            // if self copy, then we use the per expert count from our own rank
            uint32_t local_offset = dispatch_from_cur_expert_offset[expert];
            dispatch_from_cur_expert_offset[expert] +=
                1; // aadded one token here to the already previous accumulated
                   // for this->dp_group
            source_dispatch_offset[last] = local_offset;
            combine_send_offset[last] = local_offset;
          } else {
            uint32_t index_on_rank =
                src_dispatch_count[peer_group]; // starts at 0 offset
            src_dispatch_count[peer_group] += 1;

            // if we can take it from nvlink
            if (index_on_rank < this->max_private_tokens) {
              // group offset + current index on rank
              source_dispatch_offset[last] = private_offset + index_on_rank;
            } else if (peer_rank / this->node_size == rank_node) { // same node
              // add 31-bit flag for NV Link, this is also an offset into the
              // peer group's send buffer dst_offset corresponds here to the
              // position calculated in dispatch_send
              source_dispatch_offset[last] =
                  (dst_offset + index_on_rank) | (1u << 31);
            } else {
              // not supported
              HOST_ERROR("RDMA Inter-node comm is not supported");
            }
          }

          if (peer_rank != this->rank || this->dp_size > 1) {
            // offset for current group
            auto combine_index = src_combine_count[peer_group];
            src_combine_count[peer_group] += 1;
            if (peer_rank / this->node_size == rank_node) {
              combine_send_offset[last] =
                  dst_offset +
                  combine_index; // dst offset, since we're writing back into
                                 // the source ranks recv buffers, each local
                                 // expert token
            } else {
              HOST_ERROR("RDMA Inter-node comm is not supported");
            }
          }

          source_rank[last] = peer_rank; // per token source rank
          padded_index[last] =
              padded_offsets[local_expert] + expert_count[local_expert];

          expert_count[local_expert] += 1;
          last += 1;
        }
      }
      return num_routed;
    };
    // Single-node ordering follows pplx-garden: non-self local groups first,
    // then the current local group last.
    for (uint32_t local_group = 1; local_group < groups_per_node;
         local_group++) {
      route_group(rank_node * groups_per_node +
                  (this->dp_group + local_group) % groups_per_node);
    }
    route_group(this->dp_group);
    CUDA_CHECK(cudaMemcpyAsync(this->padded_index, padded_index.data(),
                               sizeof(uint32_t) * num_recv_tokens,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(this->padded_offsets, padded_offsets.data(),
                               sizeof(uint32_t) * num_local_experts,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(this->source_rank, source_rank.data(),
                               sizeof(uint32_t) * num_recv_tokens,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        this->source_dispatch_offset, source_dispatch_offset.data(),
        sizeof(uint32_t) * num_recv_tokens, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        this->combine_send_offset, combine_send_offset.data(),
        sizeof(uint32_t) * num_recv_tokens, cudaMemcpyHostToDevice, stream));
  }
};
