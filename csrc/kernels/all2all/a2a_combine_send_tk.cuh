#pragma once
#include "a2a_kernels.h"
#include "pyutils/torchutils.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <kittens.cuh>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <pyutils/parallel_tensor.cuh>
#include <runtime/utils.h>

using namespace kittens;
struct Stage {
  uint32_t offset;
  uint32_t index;
  uint32_t rank;
};
namespace a2a_combine_send {

struct config {

  static constexpr int NUM_THREADS = 512;
  static constexpr int NUM_WARPS = 512 / 16;
  static constexpr int NUM_DEVICES = 4;
  static constexpr int NUM_BLOCKS = 132;
  static constexpr int NUM_STAGES = 8; // for shared memory
};

// this TOKEN_DIM needs to be multiplied by out_elemsize, honestly all of them
// do
template <int TOKEN_DIM> struct globals {

  using barrier_layout = pgl<gl<int, -1, -1, -1, config::NUM_DEVICES * 2>,
                             config::NUM_DEVICES, false>;

  using in_tokens_layout =
      gl<bf16, -1, -1, -1,
         TOKEN_DIM>; // converted to a std::byte format for easier transferring
  using recv_buffer_layout =
      pgl<gl<bf16, -1, -1, -1, -1>, config::NUM_DEVICES, false>;

  in_tokens_layout in_tokens;
  recv_buffer_layout recv_buffer;

  barrier_layout barrier;

  uint32_t *__restrict__ combine_send_offset;
  uint32_t *__restrict__ source_rank;
  uint32_t *__restrict__ padded_index;
  uint32_t *__restrict__ sync_counter;
  const int num_recv_tokens;
  const int rank;
  const int dp_group;
  const int dp_size;
  __host__ inline int dynamic_shared_memory() {
    return sizeof(Stage) * config::NUM_STAGES;
  }
};

template <int TOKEN_DIM> __device__ inline void kernel(globals<TOKEN_DIM> &G) {

  constexpr int NUM_DEVICES = config::NUM_DEVICES;
  constexpr size_t NUM_STAGES = 8;

  using GLOBALS = decltype(G);
  __shared__ Stage shared_stages[NUM_STAGES];

  Stage local_stages[NUM_STAGES];

  auto grid = cooperative_groups::this_grid();

  auto shared_to_local = [&]() {
#pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
      local_stages[stage] = shared_stages[stage];
    }
    everyone::sync(1);
  };

  int token = blockIdx.x;
  auto counter = *G.sync_counter;
  const int warp_id = kittens::warpid();
  const int lane_id = kittens::laneid();

  if (warp_id == 0) {
    if constexpr (NUM_DEVICES > 1) {
      auto local_rank = G.rank % NUM_DEVICES;
      if (lane_id < NUM_DEVICES) {
        node_sync::wait(G.barrier, {lane_id}, local_rank, counter);
      }
    }
  } else if (warp_id == 1) {
    int next_token = token + lane_id * gridDim.x;
    if (next_token < G.num_recv_tokens && lane_id < NUM_STAGES) {
      shared_stages[lane_id].offset = G.combine_send_offset[next_token];
      shared_stages[lane_id].rank = G.source_rank[next_token];
      shared_stages[lane_id].index = G.padded_index[next_token];
    }
  }
  everyone::sync(1);

  shared_to_local();
  grid.sync();

  while (token < G.num_recv_tokens) {
    // fetch next pipeline batch, we do NUM_STAGES + threadIdx.x because we're
    // always keeping 2 x STAGES in transit
    int next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
    if (threadIdx.x < NUM_STAGES && next_token < G.num_recv_tokens) {
      shared_stages[threadIdx.x].offset = G.combine_send_offset[next_token];
      shared_stages[threadIdx.x].rank = G.source_rank[next_token];
      shared_stages[threadIdx.x].index = G.padded_index[next_token];
    }

    everyone::sync(1);

    uint4 values[NUM_STAGES];
    // loop across token dimension
    for (int i = threadIdx.x; i * sizeof(uint4) < TOKEN_DIM * 2;
         i += config::NUM_THREADS) {

      // load loop per stage
#pragma unroll NUM_STAGES
      for (int s = 0;
           s < NUM_STAGES && token + s * gridDim.x < G.num_recv_tokens; s++) {
        uint4 *ptr =
            (uint4
                 *)(&G.in_tokens[{static_cast<int>(local_stages[s].index), 0}]);
        values[s] = ld_global_nc_uint4(&ptr[i]);
      }

      // we separate the loops here in order to allow the compiler to order
      // coalesce the ld_globals

      // store loop per stage
#pragma unroll NUM_STAGES
      for (int s = 0;
           s < NUM_STAGES && token + s * gridDim.x < G.num_recv_tokens; s++) {

        int offset = local_stages[s].offset;
        int token_rank = local_stages[s].rank;
        // auto token_node = token_rank / NUM_DEVICES;
        int first_peer = (token_rank / G.dp_size) * G.dp_size;
        for (int dp_peer = 0; dp_peer < G.dp_size; dp_peer++) {
          int cur_rank = (first_peer + dp_peer) % NUM_DEVICES;
          uint4 *dst_ptr = (uint4 *)((&G.recv_buffer[cur_rank][{offset, 0}]));
          st_global_nc_uint4(values[s], &dst_ptr[i]);
        }
      }
    }

#pragma unroll NUM_STAGES
    for (int s = 0; s < NUM_STAGES && token < G.num_recv_tokens;
         s++) { // advance to next token
      token += gridDim.x;
    }

    // after we're finished using the current local_stages, copy the next
    // pipeline into shared
    shared_to_local();
  }

  if constexpr (NUM_DEVICES > 1) {
    grid.sync();
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        *G.sync_counter = counter + 1;
      }
      if (threadIdx.x < NUM_DEVICES) {
        auto local_rank = G.rank % NUM_DEVICES;
        node_sync::signal(
            G.barrier, {local_rank + NUM_DEVICES}, threadIdx.x,
            counter + 1); // this rank is done, signal to all other ranks
      }
    }
  }
}
}; // namespace a2a_combine_send

template <int TOKEN_DIM>
__global__ void __launch_bounds__(a2a_combine_send::config::NUM_THREADS, 1)
    combine_send_global_kernel(a2a_combine_send::globals<TOKEN_DIM> G) {
  a2a_combine_send::kernel(G);
}

template <int TOKEN_DIM>
cudaError_t a2a_kernels::a2a_combine_send_tk(
    at::Tensor &in_tokens_tensor,
    kittens::py::TKParallelTensor &recv_buffer_tensor,
    kittens::py::TKParallelTensor &barrier_tensor,
    uint32_t *combine_send_offset, uint32_t *source_rank,
    uint32_t *padded_index, uint32_t *sync_counter, int num_recv_tokens,
    int rank, int dp_group, int dp_size, cudaStream_t stream) {
  const size_t token_dim = TOKEN_DIM * 2; // bf16 / fp16 types only
  HOST_ASSERT(token_dim % sizeof(uint4) == 0,
              "TOKEN_DIM does not fit alignment requirements");

  using GLOBALS = a2a_combine_send::globals<TOKEN_DIM>;
  using barrier_layout = typename GLOBALS::barrier_layout;
  using in_tokens_layout = typename GLOBALS::in_tokens_layout;
  using recv_buffer_layout = typename GLOBALS::recv_buffer_layout;

  GLOBALS global_args = {
      .in_tokens =
          kittens::py::tensor_to_gl<in_tokens_layout, false>(in_tokens_tensor),
      .recv_buffer =
          kittens::py::parallel_tensor_to_pgl<recv_buffer_layout, false>(
              recv_buffer_tensor),
      .barrier = kittens::py::parallel_tensor_to_pgl<barrier_layout, false>(
          barrier_tensor),
      .combine_send_offset = combine_send_offset,
      .source_rank = source_rank,
      .padded_index = padded_index,
      .sync_counter = sync_counter,
      .num_recv_tokens = num_recv_tokens,
      .rank = rank,
      .dp_group = dp_group,
      .dp_size = dp_size};

  void *args[] = {&global_args};

  nvtxRangePush("combine_send");
  cudaError_t status = cudaLaunchCooperativeKernel(
      &combine_send_global_kernel<TOKEN_DIM>,
      dim3(a2a_combine_send::config::NUM_BLOCKS, 1, 1),
      dim3(a2a_combine_send::config::NUM_THREADS, 1, 1), args,
      global_args.dynamic_shared_memory(), stream);

  nvtxRangePop();

  return status;
}