#pragma once
#include <cstdint>
#include <driver_types.h>
#include <functional>
#include <moe_cuda/error.hpp>
#include <optional>
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
  uint32_t device;

  uint32_t *tokens_per_expert;
  uint32_t *source_rank;
  uint32_t *source_dispatch_offset;
  uint32_t *combine_send_offset;
  uint32_t *padded_index;
  uint32_t *num_recv_tokens;

  uint32_t *local_num_routed;
  cudaStream_t stream;

  // std::function<void()> route_write_op;

  WorkerState() {};

  WorkerState(uint32_t max_num_tokens, uint32_t max_recv_tokens,
              uint32_t max_private_tokens, uint32_t hidden_dim,
              std::optional<uint32_t> hidden_dim_scale, uint32_t in_elemsize,
              uint32_t out_elemsize, std::optional<uint32_t> scale_elemsize,
              uint32_t num_experts, uint32_t num_experts_per_token,
              uint32_t expert_padding, uint32_t rank, uint32_t dp_size,
              uint32_t node_size, uint32_t world_size, uint32_t device,
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
    this->device = device;
    this->stream = stream;

    // initialize buffers on devie that are needed by kernels

    // worker one alloc
    uint32_t num_local_experts = (num_experts + world_size - 1) / world_size;

    CUDA_CHECK(cudaSetDevice(device));
    size_t worker_malloc_size = 0;
    worker_malloc_size +=
        num_local_experts * sizeof(uint32_t); // tokens_per_expert
    worker_malloc_size += max_recv_tokens * sizeof(uint32_t); // source_rank
    worker_malloc_size +=
        max_recv_tokens * sizeof(uint32_t); // source_dispatch_offset
    worker_malloc_size +=
        max_recv_tokens * sizeof(uint32_t); // combine_send_offset
    worker_malloc_size += max_recv_tokens * sizeof(uint32_t); // padded_index
    worker_malloc_size += 3 * sizeof(uint32_t); // num_recv_tokens (0, 0, 0)

    uint32_t *worker_base_ptr;
    CUDA_CHECK(cudaMallocAsync(&worker_base_ptr, worker_malloc_size, stream));

    this->tokens_per_expert = worker_base_ptr;
    this->source_rank = worker_base_ptr + num_local_experts;
    this->source_dispatch_offset =
        worker_base_ptr + num_local_experts + max_recv_tokens;
    this->combine_send_offset =
        worker_base_ptr + num_local_experts + max_recv_tokens + max_recv_tokens;
    this->padded_index = worker_base_ptr + num_local_experts + max_recv_tokens +
                         max_recv_tokens + max_recv_tokens;
    this->num_recv_tokens = worker_base_ptr + num_local_experts +
                            max_recv_tokens + max_recv_tokens +
                            max_recv_tokens + 3;
  }

  uint32_t get_num_routed(uint32_t dp_group, uint32_t expert) {
    HOST_ASSERT(dp_group < this->world_size / this->dp_size,
                "Dp group is out of bounds");
    HOST_ASSERT(expert < num_experts, "Expert argument is out of bounds");

    return *(this->num_routed_ptr + dp_group * num_experts + expert);
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
    std::vector<uint32_t> tokens_from_group(num_dp_groups, 0);

    // per group offset into receiving buffer, tokens that come into this rank
    std::vector<uint32_t> src_group_offset(num_dp_groups, 0);

    // offset for the SENDER's send buffer, per dp group
    std::vector<uint32_t> dst_group_offset(num_dp_groups, 0);
    // local number, padded to experts_per_rank
    std::vector<uint32_t> tokens_per_expert(experts_per_rank, 0);

    // per rank offset of tokens FROM current rank
    std::vector<uint32_t> dispatch_src_offset(this->world_size, 0);

    // per expert offset (in tokens) FROM current rank
    std::vector<uint32_t> source_expert_offset(this->num_experts, 0);
    std::vector<uint32_t> tokens_to_rank(this->world_size, 0);

    uint32_t num_recv_tokens = 0; // tracker of receiving tokens and offsets
    {
      uint32_t rank_offset = 0;
      for (uint32_t dp_group = 0; dp_group < num_dp_groups; dp_group++) {
        uint32_t num_tokens = 0;

        for (uint32_t i = 0; i < this->dp_size; i++) {
          uint32_t rank = dp_group * this->dp_size + i;
          dispatch_src_offset[rank] = rank_offset;

          uint32_t first_expert = rank * experts_per_rank;
          uint32_t last_expert =
              std::min(first_expert + experts_per_rank, this->num_experts);

          // for current rank's experts, we accumulate
          for (uint32_t expert = first_expert; expert < last_expert; expert++) {
            uint32_t n = this->get_num_routed(
                this->dp_group,
                expert); // this tells the number route FROM (dp_group, expert)
            source_expert_offset[expert] =
                rank_offset; // cumulative sum for offsets
            tokens_to_rank[rank] += n;
            rank_offset += n;
          }
        }

        // calculate group offset
        uint32_t offset = 0;
        for (uint32_t expert = 0; expert < first_local_expert; expert++) {
          offset += this->get_num_routed(dp_group, expert);
        }
        dst_group_offset[dp_group] =
            offset; // for local device i guess? since this can vary across
                    // ranks in dp_group

        // for each dp_group, we get the number of tokens FROM that dp_group
        // coming into current rank
        for (uint32_t expert = first_local_expert; expert < last_local_expert;
             expert++) {
          uint32_t n = this->get_num_routed(dp_group, expert);
          num_tokens += n;
          tokens_per_expert[expert - first_local_expert] += n;
        }

        tokens_from_group[dp_group] += num_tokens;
        src_group_offset[dp_group] =
            num_recv_tokens; // cumulative sum of tokens_from_group
        num_recv_tokens += num_tokens;
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        this->tokens_per_expert, tokens_per_expert.data(),
        sizeof(uint32_t) * num_local_experts, cudaMemcpyHostToDevice, stream));
    // padded offset for per-expert counts
    std::vector<uint32_t> padded_offsets;
    padded_offsets.reserve(num_local_experts);
    uint32_t base_expert_offset = 0;

    for (auto count : tokens_per_expert) {
      uint32_t padded_expert_count =
          ((count + this->expert_padding - 1) / this->expert_padding) *
          this->expert_padding;
      padded_offsets.push_back(base_expert_offset);
      base_expert_offset += padded_expert_count;
    }
    // where in the peer send buffer to read from
    std::vector<uint32_t> source_dispatch_offset =
        std::vector<uint32_t>(num_recv_tokens, 0);
    // ? don't know yet
    std::vector<uint32_t> combine_send_offset =
        std::vector<uint32_t>(num_recv_tokens, 0);
    std::vector<uint32_t> source_rank =
        std::vector<uint32_t>(num_recv_tokens, 0);
    std::vector<uint32_t> padded_index =
        std::vector<uint32_t>(num_recv_tokens, 0);

    // uint32_t base_offset = this->max_private_tokens * num_dp_groups;
    uint32_t last = 0;

    std::vector<uint32_t> src_dispatch_count =
        std::vector<uint32_t>(num_dp_groups, 0);
    std::vector<uint32_t> src_combine_count =
        std::vector<uint32_t>(num_dp_groups, 0);
    std::vector<uint32_t> expert_count =
        std::vector<uint32_t>(num_local_experts, 0);

    // routes the tokens coming into current rank from a given peer group
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

        uint32_t peer_rank = peer_group * this->dp_size + this->dp_rank;

        // per token assigned to expert
        for (uint32_t i = 0; i < routed; i++) {
          if (peer_rank == this->rank) {
            // if self copy, then we use the per expert count from our own rank
            uint32_t local_offset = source_expert_offset[expert];
            source_expert_offset[expert] +=
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
              source_dispatch_offset[last] = index_on_rank + private_offset;
            } else if (peer_rank / this->node_size == rank_node) { // same node
              // add 31-bit flag for NV Link, this is also an offset into the
              // peer group's send buffer
              source_dispatch_offset[last] =
                  (dst_offset + index_on_rank) | (1u << 31);
            } else {
              // not supported
            }
          }

          // combine section - not looked at yet
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
    // we skip the inter-node routings in pplx-garden
    // intra-node routings
    for (uint32_t local_group = 0; local_group < groups_per_node;
         local_group++) {
      route_group(rank_node * groups_per_node +
                  (this->dp_group + local_group) % groups_per_node);
    }
    // self group copy as well
    route_group(this->dp_group);
    CUDA_CHECK(cudaMemcpyAsync(this->padded_index, padded_index.data(),
                               sizeof(uint32_t) * num_recv_tokens,
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
    CUDA_CHECK(cudaMemcpyAsync(this->num_recv_tokens, &num_recv_tokens,
                               sizeof(uint32_t), cudaMemcpyHostToDevice,
                               stream));
  }
};
