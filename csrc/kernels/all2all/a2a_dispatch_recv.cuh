#pragma once
#include "a2a_kernels.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <moe_cuda/kernels/common/common.hpp>
#include <moe_cuda/kernels/common/launch_utils.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <runtime/utils.h>

/*
This kernel checks the sync ptrs that the dispatch recv sent, and then copies
from its recv pointers into its local output rows, preparing for MoE

notes on some of the arguments that are modified host side:

look at process_routing_info for these:
source_rank
source_offset
padded_index
num_routed is of size (num_dp_groups, num_experts) (uint32)
num_recv_tokens_ptr
*/

template <size_t kNumThreads, uint32_t NODE_SIZE, typename TokenDim_t,
          typename HiddenDimScale_t>
__global__ void __launch_bounds__(kNumThreads, 1) a2a_dispatch_recv_kernel(
    const size_t token_dim, const size_t token_scale_dim,
    const size_t token_stride, size_t hidden_dim, size_t hidden_dim_scale,
    size_t x_elemsize, size_t x_scale_elemsize, size_t num_experts, size_t rank,
    size_t world_size, int32_t *__restrict__ out_num_tokens_ptr,
    std::byte *__restrict__ out_x_ptr, size_t out_x_stride,
    float *__restrict__ out_x_scale_ptr, size_t out_x_scale_stride_elem,
    size_t out_x_scale_stride_token, uint32_t *__restrict__ tokens_per_expert,
    std::byte *__restrict__ send_buffer, std::byte *__restrict__ recv_buffer,
    uint32_t *__restrict__ source_rank, uint32_t *__restrict__ source_offset,
    uint32_t *__restrict__ padded_index, uint32_t *__restrict__ num_routed,
    uint32_t *__restrict__ num_recv_tokens_ptr,
    uint32_t *__restrict__ sync_counter, uint32_t **__restrict__ sync_ptrs,
    std::byte **send_ptrs) {

  TokenDim_t token_dim_fixed(token_dim);
  HiddenDimScale_t hidden_dim_scale_fixed(hidden_dim_scale);

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x & 0x1f;
  const uint32_t pred = ti_elect_one_sync();
  const size_t experts_per_rank = ti_ceil_div(num_experts, world_size);

  // creates send_ptrs_local to avoid repeated GBM accesses

  std::byte *send_ptrs_local[NODE_SIZE];
  for (uint32_t i = 0; i < NODE_SIZE; i++) {
    send_ptrs_local[i] = send_ptrs[i];
  }

  auto counter = *sync_counter;
  // nv link sync using sync_ptrs
  if (warp_id == 0) {
    if constexpr (NODE_SIZE > 1) {
      auto local_rank = rank % NODE_SIZE;
      if (lane_id < NODE_SIZE) {
        auto *flag_ptr = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
        while (ld_acquire_u32(flag_ptr) !=
               counter) // we're load_acquiring the store relase from the
                        // dispatch_recv, to ensure explicit memory fence
          ;
      }
    }
  }
  __syncthreads(); // more efficient than a grid.sync

  const unsigned num_recv_tokens = ld_volatile_u32(num_recv_tokens_ptr);
  for (uint32_t token = blockIdx.x; token < num_recv_tokens;
       token += gridDim.x) {
    auto padded_token = padded_index[token]; // padded index into
    auto token_rank = source_rank[token];    // where the token is coming from

    auto local_rank = token_rank % NODE_SIZE;
    auto position =
        source_offset[token]; // source_dispatch_offset in worker.hpp, gives us
                              // absolute position

    uint4 *x_token_src;
    // copy from send buffer or send_ptrs
    if (token_rank == rank) {
      x_token_src = (uint4 *)(send_buffer + position * token_stride);
    } else if (position &
               (1u << 31)) { // nv link -> send buffer copy, but it's overflow
                             // over the max_private_tokens, so we take from the
                             // peer's send buffer
      x_token_src = (uint4 *)(send_ptrs_local[local_rank] +
                              (position & ~(1u << 31)) * token_stride);
    } else {
      // position < max_private_tokens, directly copied into our recv buffer
      // from dispatch_send
      x_token_src = (uint4 *)(recv_buffer + position * token_stride);
    }

    uint4 *x_token_dst = (uint4 *)(out_x_ptr + padded_token * out_x_stride);
    float *x_scale_src = (float *)((std::byte *)x_token_src + token_dim_fixed);
    float *x_scale_dst =
        (float *)(out_x_scale_ptr + padded_token * out_x_scale_stride_token);

    for (uint32_t i = threadIdx.x; i * sizeof(uint4) < token_dim;
         i += kNumThreads) {
      const bool has_scale = out_x_scale_ptr && i < hidden_dim_scale_fixed;

      auto val = ld_global_nc_uint4(&x_token_src[i]);
      float scale;
      if (has_scale) {
        scale = x_scale_src[i];
      }

      st_global_nc_uint4(val, &x_token_dst[i]);
      if (has_scale) {
        x_scale_dst[i * out_x_scale_stride_elem] = scale;
      }
    }
  }

  const size_t first_expert = rank * experts_per_rank;
  const size_t last_expert = min(first_expert + experts_per_rank, num_experts);
  if (blockIdx.x == 0) {
    for (unsigned expert = threadIdx.x; expert < last_expert - first_expert;
         expert += blockDim.x) {
      out_num_tokens_ptr[expert] =
          tokens_per_expert[expert]; // copy the output counts per expert that
                                     // were calculated in dispatch_send, this
                                     // tells the number, maybe for debugging or
                                     // load balancing
    }
  }

  if constexpr (NODE_SIZE >= 1) {
    grid.sync();

    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        *sync_counter = counter + 1;
      }
      auto local_rank = rank % NODE_SIZE;
      for (int peer = threadIdx.x; peer < NODE_SIZE; peer += blockDim.x) {
        // signal to all the other peer nodes that current rank is done
        st_volatile_u32(&sync_ptrs[peer][local_rank], counter + 1);
      }
    }
  }
  // structure:
  /*
  we're not doing RDMAs, but the original authors overlap nvlink (Faster) with
  the longer RDMA writes into recv_buffer we load in the number of received
  tokens from num_recv_tokens_ptr, once we know it is written into, and then we
  start the transfer

  we don't need to use the shared_to_local - just remember to put a 31st bit
  flag on the source_offset of a given token so we know it's from another rank

  performs writes, ensure tokens are packed together by expert if possible
    look at a2a_worker.rs to understand how padded_index is constructed

  then updates the counter and stores to sync_ptrs to ensure combine kernel
  won't overwrite
  */
}

cudaError_t a2a_kernels::a2a_dispatch_recv(
    size_t num_blocks, size_t hidden_dim, size_t hidden_dim_scale,
    size_t x_elemsize, size_t x_scale_elemsize, size_t num_experts, size_t rank,
    size_t node_size, size_t world_size, int32_t *out_num_tokens_ptr,
    uint8_t *out_x_ptr, size_t out_x_stride, uint8_t *out_x_scale_ptr,
    size_t out_x_scale_stride_elem, size_t out_x_scale_stride_token,
    uint32_t *tokens_per_expert, uint8_t *send_buffer, uint8_t *recv_buffer,
    uint32_t *source_rank, uint32_t *source_offset, uint32_t *padded_index,
    uint32_t *num_routed, uint32_t *num_recv_tokens_ptr, uint32_t *sync_counter,
    uint32_t **sync_ptrs, uint8_t **send_ptrs, cudaStream_t stream) {
  constexpr size_t kNumThreads = 512;

  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(kNumThreads, 1, 1);

  const size_t token_dim = ti_align(hidden_dim * x_elemsize, sizeof(float4));
  const size_t token_scale_dim =
      ti_align(hidden_dim_scale * x_scale_elemsize, sizeof(float4));
  const size_t token_stride = token_dim + token_scale_dim + 16;

  HOST_ASSERT(token_stride % sizeof(float4) == 0, "Token stride not divisible");

  void *args[] = {
      const_cast<size_t *>(&token_dim),
      const_cast<size_t *>(&token_scale_dim),
      const_cast<size_t *>(&token_stride),
      &hidden_dim,
      &hidden_dim_scale,
      &x_elemsize,
      &x_scale_elemsize,
      &num_experts,
      &rank,
      &world_size,
      &out_num_tokens_ptr,
      &out_x_ptr,
      &out_x_stride,
      &out_x_scale_ptr,
      &out_x_scale_stride_elem,
      &out_x_scale_stride_token,
      &tokens_per_expert,
      &send_buffer,
      &recv_buffer,
      &source_rank,
      &source_offset,
      &padded_index,
      &num_routed,
      &num_recv_tokens_ptr,
      &sync_counter,
      &sync_ptrs,
      &send_ptrs,
  };

  nvtxRangePush("dispatch_recv");
  cudaError_t status;

  LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
    LAUNCH_TOKEN_DIM(token_dim, TokenDim_t, {
      LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, HiddenDimScale_t, {
        status = cudaLaunchCooperativeKernel(
            (void *)&a2a_dispatch_recv_kernel<kNumThreads, NODE_SIZE,
                                              TokenDim_t, HiddenDimScale_t>,
            dimGrid, dimBlock, args, 0, stream);
      });
    });
  });

  nvtxRangePop();
  return status;
}