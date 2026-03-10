#pragma once
#include "a2a_kernels.h"
#include <moe_cuda/kernels/common/launch_utils.cuh>

#include <runtime/utils.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>

// reverse of dispatch_recv, we need to reverse the token order

template <size_t kNumThreads, uint32_t NODE_SIZE, uint32_t DP_SIZE, typename TokenDim_t>
__global__ void __launch_bounds__(kNumThreads, 1)
    a2a_combine_send_kernel(const size_t token_dim, const size_t rank, const std::byte *__restrict__ expert_x_ptr,
                            size_t expert_x_stride, std::byte *__restrict__ send_buffer,
                            std::byte *__restrict__ recv_buffer, uint32_t *__restrict__ source_rank,
                            uint32_t *__restrict__ combine_send_offset, uint32_t *__restrict__ padded_index,
                            const uint32_t *__restrict__ num_recv_tokens_ptr, uint32_t *__restrict__ sync_counter,
                            uint32_t **__restrict__ sync_ptrs, std::byte **recv_ptrs) {
  TokenDim_t token_bound_fixed(token_dim);

  constexpr size_t NUM_STAGES = 8;

  struct Stage {
    uint32_t offset;
    uint32_t index;
    uint32_t rank;
  };

  __shared__ Stage shared_stages[NUM_STAGES];
  Stage local_stages[NUM_STAGES];

  std::byte *recv_ptrs_local[NODE_SIZE];
#pragma unroll
  for (uint32_t i = 0; i < NODE_SIZE; i++) {
    recv_ptrs_local[i] = recv_ptrs[i];
  }

  auto grid = cooperative_groups::this_grid();
  const unsigned rank_node = rank / NODE_SIZE;
  const unsigned warp_id = threadIdx.x / 32;
  const unsigned lane_id = threadIdx.x & 0x1f;

  const unsigned num_recv_tokens = __ldg(num_recv_tokens_ptr);

  // tokens <= num_blocks?
  unsigned token = blockIdx.x;
  auto counter = *sync_counter;

  auto shared_to_local = [&]() {
#pragma unroll NUM_STAGES
    for (unsigned s = 0; s < NUM_STAGES; s++) {
      local_stages[s] = shared_stages[s];
    }
    __syncthreads();
  };

  if (warp_id == 0) {
    if constexpr (NODE_SIZE > 1) {
      auto local_rank = rank % NODE_SIZE;
      if (lane_id < NODE_SIZE) {
        auto *flag = &sync_ptrs[lane_id][local_rank];
        while (ld_volatile_u32(flag) != counter)
          ;
      }
    }
  } else if (warp_id == 1) {
  }

  __syncthreads();

  // prefetch into shared stages
  if (warp_id == 0) {
    uint32_t next_token = token + lane_id * gridDim.x; // blockIdx.x + lane_id * gridDim.x, this is blockwide but we're
                                                       // only filling the first NUM_STAGES at each block iteration
    if (next_token < num_recv_tokens && lane_id < NUM_STAGES) {
      shared_stages[lane_id].offset = combine_send_offset[next_token];
      shared_stages[lane_id].index = padded_index[next_token];
      shared_stages[lane_id].rank = source_rank[next_token];
    }
  }
  __syncthreads();

  shared_to_local();

  grid.sync();

  while (token < num_recv_tokens) {
    uint32_t next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
    if (threadIdx.x < NUM_STAGES && next_token < num_recv_tokens) {
      shared_stages[threadIdx.x].offset = combine_send_offset[next_token];
      shared_stages[threadIdx.x].index = padded_index[next_token];
      shared_stages[threadIdx.x].rank = source_rank[next_token];
    }
    __syncthreads();

    // pipelined copy, we already loaded into shared, now we process local_stages information
    uint4 values[NUM_STAGES];

    for (uint32_t i = threadIdx.x; i * sizeof(uint4) < token_bound_fixed; i += kNumThreads) {
#pragma unroll NUM_STAGES
      for (uint32_t s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
        auto *ptr = (uint4 *)(expert_x_ptr + expert_x_stride * local_stages[s].index);
        values[s] = ld_global_nc_uint4(ptr);
      }

#pragma unroll NUM_STAGES
      for (uint32_t s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
        uint32_t offset = local_stages[s].offset;
        uint32_t token_rank = local_stages[s].rank;
        uint32_t token_node = token_rank / NODE_SIZE;

        if (token_node == rank_node) {
          unsigned first_dp_rank = (token_rank / DP_SIZE) * DP_SIZE;
#pragma unroll DP_SIZE
          for (uint32_t dp_peer = 0; dp_peer < DP_SIZE; dp_peer++) { // cycle through ranks in dp group
            auto token_peer = (first_dp_rank + dp_peer) % NODE_SIZE;
            auto *x_token_dst = (uint4 *)(recv_ptrs_local[token_peer] + offset * token_bound_fixed);
            st_global_nc_uint4(values[s], &x_token_dst[i]);
          }
        }
      }
    }

#pragma unroll NUM_STAGES
    for (uint32_t s = 0; s < NUM_STAGES && token < num_recv_tokens; s++) {
      token += gridDim.x;
    }
    shared_to_local();
  }

  grid.sync();

  if (blockIdx.x == 0) {

    if (warp_id == 0 && ti_elect_one_sync()) {
      *sync_counter = counter + 1;

    } else if (warp_id == 1) {
      if constexpr (NODE_SIZE > 1) {
        auto local_rank = rank % NODE_SIZE;
        if (lane_id < NODE_SIZE) {
          st_release_u32(&sync_ptrs[lane_id][local_rank + NODE_SIZE], counter + 1);
        }
      }
    }
  }
}

cudaError_t a2a_kernels::a2a_combine_send(size_t num_blocks, size_t hidden_dim,
                                          size_t x_elemsize, // bf16 here, or what output activation size
                                          size_t rank, size_t node_size, size_t dp_size, uint8_t *expert_x_ptr,
                                          size_t expert_x_stride, uint8_t *send_buffer, uint8_t *recv_buffer,
                                          uint32_t *source_rank, uint32_t *combine_send_offset, uint32_t *padded_index,
                                          uint32_t *num_recv_tokens_ptr, uint32_t *sync_counter, uint32_t **sync_ptrs,
                                          uint8_t **recv_ptrs, cudaStream_t stream) {
  const size_t token_dim = ti_align(hidden_dim * x_elemsize, sizeof(int4));

  void *args[] = {
      const_cast<size_t *>(&token_dim),
      &rank,
      &expert_x_ptr,
      &expert_x_stride,
      &send_buffer,
      &recv_buffer,
      &source_rank,
      &combine_send_offset,
      &padded_index,
      &num_recv_tokens_ptr,
      &sync_counter,
      &sync_ptrs,
      &recv_ptrs,
  };

  dim3 dimGrid(num_blocks, 1, 1);
  constexpr size_t kNumThreads = 512; // at least 512 experts <-> tokens
  dim3 dimBlock(kNumThreads, 1, 1);
  cudaError_t status;

  nvtxRangePush("combine_send");

  LAUNCH_WORLD_SIZE(dp_size, DP_SIZE, {
    LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
      LAUNCH_TOKEN_DIM(token_dim, TokenDim_t, {
        status =
            cudaLaunchCooperativeKernel((void *)&a2a_combine_send_kernel<kNumThreads, NODE_SIZE, DP_SIZE, TokenDim_t>,
                                        dimGrid, dimBlock, args, 0, stream);
      });
    });
  });

  nvtxRangePop();

  return status;
}