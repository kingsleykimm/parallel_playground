#pragma once
#include "a2a_kernels.h"
#include "pyutils/parallel_tensor.cuh"
#include <cooperative_groups.h>
#include <cuda.h>
#include <kittens.cuh>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/launch_utils.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <pyutils/torchutils.cuh>
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

using namespace kittens;

namespace a2a_dispatch_recv {

struct config {
  static constexpr int CLUSTER_SIZE = 1;
  static constexpr int NUM_THREADS = 512;
  static constexpr int NUM_WARPS = NUM_THREADS / 32;
  static constexpr int NUM_DEVICES = 4;
  static constexpr int NUM_BLOCKS = 132; // for now
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
struct globals {

  static constexpr int ROW_BLOCK_SIZE = 16; // tunable, must be divisible by 16
  static constexpr int COL_BLOCK_SIZE = 128;

  using barrier_layout = pgl<gl<int, -1, -1, -1, config::NUM_DEVICES * 2>,
                             config::NUM_DEVICES, false>;
  using token_tile = st_fp8e4m3<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
  // using scale_tile = sv_fl<ROW_BLOCK_SIZE>;

  using in_tokens_layout =
      pgl<gl<fp8e4m3, -1, -1, -1, TOKEN_DIM>, config::NUM_DEVICES, false>;
  using in_scale_layout =
      pgl<gl<float, -1, -1, -1, TOKEN_DIM / 128>, config::NUM_DEVICES, false>;
  using send_buffer_layout =
      pgl<gl<fp8e4m3, -1, -1, -1, TOKEN_DIM>, config::NUM_DEVICES, false>;
  using send_scale_buffer_layout =
      pgl<gl<float, -1, -1, -1, TOKEN_DIM / 128>, config::NUM_DEVICES, false>;

  using out_tokens_layout = gl<fp8e4m3, -1, -1, -1, TOKEN_DIM, token_tile>;
  using out_scale_layout = gl<float, -1, -1, -1, TOKEN_DIM / 128>;
  barrier_layout barrier;
  in_tokens_layout in_tokens;
  in_scale_layout in_scales;
  send_buffer_layout send_buffer;
  send_scale_buffer_layout send_scale_buffer;
  out_tokens_layout out_tokens;
  out_scale_layout out_scales;
  uint32_t *sync_counter;

  uint32_t *__restrict__ source_rank;
  uint32_t *__restrict__ source_offset;
  uint32_t *__restrict__ padded_index;
  const uint32_t num_recv_tokens;
  int rank;
  int dp_size;

  __host__ inline int dynamic_shared_memory() const {
    return 0; // subject to change
  }
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__device__ inline void
kernel(const globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &G) {

  using GLOBALS = decltype(G);
  const size_t warp_id = ::warpid();
  const size_t lane_id = ::laneid();
  const int dp_group = G.rank / G.dp_size; // dp_size is almost always 1

  constexpr int NUM_DEVICES = config::NUM_DEVICES;
  constexpr int NUM_THREADS = config::NUM_THREADS;
  constexpr int num_experts_per_rank =
      constexpr_ti_ceil_div(NUM_EXPERTS, NUM_DEVICES);
  auto grid = cooperative_groups::this_grid();

  const uint32_t pred = ti_elect_one_sync();

  uint32_t counter = *G.sync_counter;

  if constexpr (NUM_DEVICES > 1) {
    if (warp_id == 0) {
      if (lane_id < NUM_DEVICES) {
        int local_rank = lane_id % NUM_DEVICES;
        node_sync::wait(G.barrier, {local_rank + NUM_DEVICES}, G.rank, counter);
      }
    }
  }
  __syncthreads();

  const unsigned num_recv_tokens = G.num_recv_tokens;
  for (uint32_t token = blockIdx.x; token < num_recv_tokens;
       token += gridDim.x) {

    auto padded_index = G.padded_index[token];
    auto source_rank = G.source_rank[token];

    auto local_rank = source_rank % NUM_DEVICES;
    auto position =
        G.source_offset[token]; // this is the source_dispatch_offset calculated
                                // in worker.hpp
    uint4 *x_token_src;
    float *x_scale_src;
    if (source_rank == G.rank) { // our own send buffer
      x_token_src =
          (uint4 *)(G.send_buffer[G.rank].raw_ptr + position * TOKEN_DIM);
      x_scale_src =
          G.send_scale_buffer[G.rank].raw_ptr + position * TOKEN_DIM / 128;
    } else if (position & (1u << 31)) { // overflow copy from peer's send buffer
      x_token_src = (uint4 *)(G.send_buffer[local_rank].raw_ptr +
                              (position & ~(1u << 31)) * TOKEN_DIM);
      x_scale_src = G.send_scale_buffer[local_rank].raw_ptr +
                    (position & ~(1u << 31)) * TOKEN_DIM / 128;
    } else {
      // recv buffer case, already copied into our recv buffer
      x_token_src =
          (uint4 *)(G.in_tokens[G.rank].raw_ptr + position * TOKEN_DIM);
      x_scale_src = G.in_scales[G.rank].raw_ptr + position * TOKEN_DIM / 128;
    }

    uint4 *x_token_dst =
        (uint4 *)(G.out_tokens.raw_ptr + padded_index * TOKEN_DIM);
    float *x_scale_dst = G.out_scales.raw_ptr + padded_index * TOKEN_DIM / 128;

    for (uint32_t i = threadIdx.x; i * sizeof(uint4) < TOKEN_DIM;
         i += NUM_THREADS) {
      const bool has_scale = i < TOKEN_DIM / 128;
      auto val = ld_global_nc_uint4(&x_token_src[i]);
      float scale;
      if (has_scale) {
        scale = x_scale_src[i];
        x_scale_dst[i] = scale;
      }
      st_global_nc_uint4(val, &x_token_dst[i]);
    }
  }
  // const int first_expert = G.rank * num_experts_per_rank;
  // const int last_expert =
  //     ti_min(first_expert + num_experts_per_rank, NUM_EXPERTS);
  // if (blockIdx.x == 0) {
  //   for (int expert = threadIdx.x; expert < last_expert - first_expert;
  //        expert += blockDim.x) {
  //     out_num_tokens_ptr[expert] =
  //         G.tokens_per_expert[expert]; // copy the output counts per expert
  //         that
  //                                    // were calculated in dispatch_send,
  //                                    this
  //                                    // tells the number, maybe for debugging
  //                                    or
  //                                    // load balancing
  //   }
  // }
  if constexpr (NUM_DEVICES > 1) {
    grid.sync();
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        *G.sync_counter = counter + 1;
      }
      auto local_rank = G.rank % NUM_DEVICES;
      for (int peer = threadIdx.x; peer < NUM_DEVICES; peer += blockDim.x) {
        // signal to all the other peer nodes that current rank is done
        // dst_dev_idx is peer rank, source is from current rank signaling
        // completion
        node_sync::signal(G.barrier, {local_rank}, peer, 1);
      }
    }
  }
}
} // namespace a2a_dispatch_recv

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__global__ void __launch_bounds__(a2a_dispatch_recv::config::NUM_THREADS)
    dispatch_recv_global_kernel(
        a2a_dispatch_recv::globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>
            G) {
  a2a_dispatch_recv::kernel(G);
}

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t a2a_kernels::fp8e4m3_a2a_dispatch_recv(
    kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_scales,
    kittens::py::TKParallelTensor &send_buffer,
    kittens::py::TKParallelTensor &send_scale_buffer,
    at::Tensor &out_tokens_tensor, at::Tensor &out_scales_tensor,
    kittens::py::TKParallelTensor &barrier, uint32_t *sync_counter,
    uint32_t *source_rank, uint32_t *source_offset, uint32_t *padded_index,
    uint32_t num_recv_tokens, int rank, int dp_size, cudaStream_t stream) {
  using globals = typename a2a_dispatch_recv::globals<EXPERTS_PER_TOKEN,
                                                      NUM_EXPERTS, TOKEN_DIM>;

  using in_tokens_layout = typename globals::in_tokens_layout;
  using in_scale_layout = typename globals::in_scale_layout;
  using send_buffer_layout = typename globals::send_buffer_layout;
  using send_scale_buffer_layout = typename globals::send_scale_buffer_layout;
  using out_tokens_layout = typename globals::out_tokens_layout;
  using out_scale_layout = typename globals::out_scale_layout;
  using barrier_layout = typename globals::barrier_layout;
  globals args = {
      .barrier = kittens::py::parallel_tensor_to_pgl<barrier_layout>(barrier),
      .in_tokens =
          kittens::py::parallel_tensor_to_pgl<in_tokens_layout>(in_tokens),
      .in_scales =
          kittens::py::parallel_tensor_to_pgl<in_scale_layout>(in_scales),
      .send_buffer =
          kittens::py::parallel_tensor_to_pgl<send_buffer_layout>(send_buffer),
      .send_scale_buffer =
          kittens::py::parallel_tensor_to_pgl<send_scale_buffer_layout>(
              send_scale_buffer),
      .out_tokens =
          kittens::py::tensor_to_gl<out_tokens_layout>(out_tokens_tensor),
      .out_scales =
          kittens::py::tensor_to_gl<out_scale_layout>(out_scales_tensor),
      .sync_counter = sync_counter,
      .source_rank = source_rank,
      .source_offset = source_offset,
      .padded_index = padded_index,
      .num_recv_tokens = num_recv_tokens,
      .rank = rank,
      .dp_size = dp_size,
  };
  void *kernel_args[] = {&args};

  nvtxRangePush("a2a_dispatch_recv_tk");
  cudaError_t status = cudaLaunchCooperativeKernel(
      (void *)&dispatch_recv_global_kernel<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                           TOKEN_DIM>,
      dim3(a2a_dispatch_recv::config::NUM_BLOCKS, 1, 1),
      dim3(a2a_dispatch_recv::config::NUM_THREADS, 1, 1), kernel_args,
      args.dynamic_shared_memory(), stream);

  nvtxRangePop();
  return status;
}
