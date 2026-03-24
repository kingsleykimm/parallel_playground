/*
Note: add compile time specialization dispatch functions for more of the common
ones

*/
#pragma once
#include "a2a_kernels.h"
#include "common/base_types.cuh"
#include "pyutils/parallel_tensor.cuh"
#include "pyutils/torchutils.cuh"
#include <cooperative_groups.h>
#include <cuda.h>
#include <kittens.cuh>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <runtime/utils.h>

// each rank has a set number of tokens -> tokens * num_experts
using namespace kittens;
namespace a2a_dispatch_send {

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

  using token_tile = st_fp8e4m3<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
  using scale_tile = sv_fl<ROW_BLOCK_SIZE>;
  using tokens_per_expert_layout = st<uint32_t, 1, NUM_EXPERTS, false>;
  using experts_offset_layout = st<uint32_t, 1, NUM_EXPERTS, false>;
  using barrier_layout = pgl<gl<int, -1, -1, -1, config::NUM_DEVICES * 2>,
                             config::NUM_DEVICES, false>;

  using tokens_layout = gl<fp8e4m3, -1, -1, -1, TOKEN_DIM, token_tile>;
  using scale_layout = gl<float, -1, -1, TOKEN_DIM / 128, -1>;
  using out_tokens_layout =
      pgl<gl<fp8e4m3, -1, -1, -1, -1>, config::NUM_DEVICES, false>;
  using out_scale_layout =
      pgl<gl<float, -1, -1, TOKEN_DIM / 128, -1>, config::NUM_DEVICES, false>;
  using send_buffer_layout =
      pgl<gl<fp8e4m3, -1, -1, -1, -1>, config::NUM_DEVICES, false>;
  using send_scale_layout =
      pgl<gl<float, -1, -1, TOKEN_DIM / 128, -1>, config::NUM_DEVICES, false>;
  using num_routed_layout = gl<uint32_t, 1, 1, -1, NUM_EXPERTS>;
  using indices_layout = gl<uint32_t, -1, -1, -1, EXPERTS_PER_TOKEN>;
  using weights_layout = gl<float, -1, -1, -1, EXPERTS_PER_TOKEN>;

  using token_offset_layout =
      gl<uint32_t, -1, -1, -1,
         EXPERTS_PER_TOKEN>; // (num_tokens, EXPERTS_PER_TOKEN)
  using expert_offsets_layout =
      gl<uint32_t, 1, 1, 1, NUM_EXPERTS>; // (NUM_EXPERTS)

  tokens_layout input_tokens; // shared tile we load in with for
  scale_layout input_scales;  // shared tile we load in with for
  barrier_layout barrier;
  out_tokens_layout out_tokens;
  out_scale_layout out_scales;
  send_buffer_layout send_buffer;
  send_scale_layout send_scale_buffer;
  num_routed_layout num_routed;
  uint32_t num_tokens;
  indices_layout indices;
  weights_layout weights;
  token_offset_layout token_offset;     // size (num_tokens * EXPERTS_PER_TOKEN)
  expert_offsets_layout expert_offsets; // size (NUM_EXPERTS)
  uint32_t *__restrict__ sync_counter;  // intra-node sync counter
  uint32_t max_private_tokens; // max padded amount any device will take
  uint8_t *__restrict__ dispatch_route_done; // signal to the worker to start
                                             // copying over

  __host__ inline int dynamic_shared_memory() const {
    return sizeof(tokens_per_expert_layout);
  }

  template <bool PERSISTENT_GRID> __host__ inline dim3 grid() const {
    return PERSISTENT_GRID
               ? 132
               : (TOKEN_DIM / COL_BLOCK_SIZE) *
                     (tokens_layout::rows() / ROW_BLOCK_SIZE) *
                     tokens_layout::depth() * tokens_layout::batch();
  }

  const int rank;
  const int dp_size;
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__device__ inline void
kernel(const globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &G) {

  using GLOBALS = globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>;
  const size_t warp_id = ::warpid();
  const size_t lane_id = ::laneid();
  const int dp_group = G.rank / G.dp_size; // dp_size is almost always 1

  constexpr int NUM_DEVICES = config::NUM_DEVICES;
  constexpr int NUM_THREADS = config::NUM_THREADS;
  constexpr int num_experts_per_rank =
      constexpr_ti_ceil_div(NUM_EXPERTS, NUM_DEVICES);
  auto grid = cooperative_groups::this_grid();

  // using input_tokens_layout = typename GLOBALS::tokens_layout;
  // using input_scales_layout = typename GLOBALS::scale_layout;
  using indices_layout = typename GLOBALS::indices_layout;
  using weights_layout = typename GLOBALS::weights_layout;
  using tokens_per_expert_layout = typename GLOBALS::tokens_per_expert_layout;
  using token_offset_layout = typename GLOBALS::token_offset_layout;
  using expert_offsets_layout = typename GLOBALS::expert_offsets_layout;

  extern __shared__ int __shm[];
  shared_allocator alloc(&__shm[0]);
  // input_tokens_layout(&input_tokens) = alloc.allocate<input_tokens_layout>();
  // input_scales_layout(&input_scales) = alloc.allocate<input_scales_layout>();
  // indices_layout(&indices) = alloc.allocate<indices_layout>();
  // weights_layout(&weights) = alloc.allocate<weights_layout>();
  tokens_per_expert_layout(&tokens_per_expert) = // size (NUM_EXPERTS)
      alloc.allocate<tokens_per_expert_layout>();

  if (blockIdx.x == 0) {
    warp::zero(tokens_per_expert); // -> don't assume TK warp operations are
                                   // auto sychronized across a CTA!
    everyone::sync(1);
    for (uint32_t i = threadIdx.x; i < G.num_tokens * EXPERTS_PER_TOKEN;
         i += blockDim.x) {
      const uint32_t token = i / EXPERTS_PER_TOKEN;
      const uint32_t index = i % EXPERTS_PER_TOKEN;
      const uint32_t expert =
          G.indices[{static_cast<int>(token), static_cast<int>(index)}];

      // token offset describes the expert-local offsets per token, used later
      // in expert iterator as well as in the combien_send kernel in order to
      // place the gemmed token in its correct position
      G.token_offset[{static_cast<int>(token), static_cast<int>(index)}] =
          atomicAdd(&tokens_per_expert[expert], 1);
    }
    __syncthreads();
    // number routed to each rank, put this in every dp-rank

    // we can overlap host side NVLink scatters with the kernel once we signal
    uint32_t expert_offset = 0;
    uint32_t *local_num_routed = G.num_routed.raw_ptr + dp_group * NUM_EXPERTS;

    if (threadIdx.x < NUM_EXPERTS) {
      expert_offset = tokens_per_expert[threadIdx.x];
      local_num_routed[threadIdx.x] = expert_offset;
    }
    everyone::sync(1);
    uint32_t *expert_sums = (uint32_t *)(tokens_per_expert.data);

    if (threadIdx.x == 0) {
      __threadfence_system();
      *G.dispatch_route_done = 1;
    }
    everyone::sync(1);

    // block-wide prefix scan to ensure each expert_sums is at its absolute
    // offset relative to current block. we overwrite tokens_per_expert since we
    // don't need its data anymore

    for (int offset = 1; offset < 32; offset <<= 1) {
      uint32_t warp_sum_expert =
          __shfl_up_sync(0xffffffff, expert_offset, offset);
      if (lane_id >= offset) {
        expert_offset += warp_sum_expert;
      }
    }

    if (lane_id == 31) {
      expert_sums[warp_id] = expert_offset;
    }

    everyone::sync(1);

    if (warp_id == 0) {
      uint32_t total_expert_sum =
          (lane_id < config::NUM_WARPS) ? expert_sums[lane_id] : 0;
      for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        uint32_t warp_sum =
            __shfl_up_sync(0xffffffff, total_expert_sum, offset);
        if (lane_id >= offset) {
          total_expert_sum += warp_sum;
        }
      }

      if (lane_id < config::NUM_WARPS) {
        expert_sums[lane_id] = total_expert_sum;
      }
    }
    everyone::sync(1);

    // more global stores for metadata, will be used later in combine
    if (threadIdx.x < NUM_EXPERTS) {
      if (warp_id > 0) {
        G.expert_offsets[{static_cast<int>(threadIdx.x)}] =
            expert_sums[warp_id - 1] + expert_offset;
      } else {
        G.expert_offsets[{static_cast<int>(threadIdx.x)}] = expert_offset;
      }
    }
    everyone::sync(1);
  }

  // NVLink barrier set up to wait for previous combine kernels to finish before
  // writing into send buffer
  int counter = *G.sync_counter;
  if constexpr (NUM_DEVICES > 1) {
    if (blockIdx.x == 0) {
      for (int peer = threadIdx.x; peer < NUM_DEVICES; peer += NUM_THREADS) {
        node_sync::wait(G.barrier, {peer}, G.rank, counter);
      }
    }
  }
  grid.sync();

  if constexpr (NUM_DEVICES > 1) {
    for (uint32_t token = blockIdx.x; token < G.num_tokens;
         token += gridDim.x) {
      uint4 *tokens_ptr =
          (uint4 *)(G.input_tokens.raw_ptr + (token * TOKEN_DIM));
      constexpr size_t NUM_STRIDES =
          constexpr_ti_ceil_div(TOKEN_DIM, NUM_THREADS);

      uint4 vals[NUM_STRIDES];
      float scales[NUM_STRIDES];

#pragma unroll NUM_STRIDES
      for (int i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM;
           i += NUM_THREADS, s++) {
        const bool has_scale = i < TOKEN_DIM / 128;
        vals[s] = ld_global_nc_uint4(&tokens_ptr[i]);
        if (has_scale) {
          scales[s] = G.input_scales[{i, static_cast<int>(token)}];
          // printf("input token : %d, input scale : %f \n", token, scales[s]);
        }
      }

      // perform scatter to each node
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TOKEN; e++) {
        const uint32_t expert = G.indices[{static_cast<int>(token), e}];
        const uint32_t offset = G.token_offset[{static_cast<int>(token), e}];
        const uint32_t dst_rank = expert / num_experts_per_rank;

        const uint32_t position =
            ((expert > 0) ? G.expert_offsets[{(int)expert - 1}] : 0) + offset;
        const uint32_t rank_offset =
            dst_rank > 0 ? G.expert_offsets[{static_cast<int>(
                               dst_rank * num_experts_per_rank - 1)}]
                         : 0;
        const uint32_t local_rank_offset = position - rank_offset;
        if (dst_rank != G.rank && local_rank_offset < G.max_private_tokens) {
          if (dst_rank % G.dp_size == G.rank % G.dp_size) {
            const int token_position =
                dp_group * (int)G.max_private_tokens + (int)local_rank_offset;
            uint4 *recv_token_ptr =
                (uint4 *)(&G.out_tokens[dst_rank][{token_position, 0}]);
            for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
                 i += config::NUM_THREADS, s++) {
              const bool has_scale = i < TOKEN_DIM / 128;
              st_global_nc_uint4(vals[s], &recv_token_ptr[i]);
              if (has_scale) {
                move<float>::stg(&G.out_scales[dst_rank][{i, token_position}],
                                 scales[s]);
              }
            }
          }
        } else {
          // for either same node copies or overflow, we use our
          // send_buffer
          uint4 *send_token_ptr =
              (uint4 *)(&G.send_buffer[G.rank][{(int)position, 0}]);
          for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
               i += config::NUM_THREADS, s++) {
            const bool has_scale = i < TOKEN_DIM / 128;
            st_global_nc_uint4(vals[s], &send_token_ptr[i]);
            if (has_scale) {
              move<float>::stg(&G.send_scale_buffer[G.rank][{i, (int)position}],
                               scales[s]);
            }
          }
        }
      }
    }
  }

  if constexpr (NUM_DEVICES > 1) {
    grid.sync();
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        *G.sync_counter = counter + 1;
      }
      if (threadIdx.x < NUM_DEVICES) {
        node_sync::signal(G.barrier, {G.rank + NUM_DEVICES},
                          static_cast<int>(threadIdx.x), counter + 1);
      }
    }
  }
}
} // namespace a2a_dispatch_send

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__global__ __launch_bounds__(
    a2a_dispatch_send::config::NUM_THREADS,
    1) void dispatch_send_global_kernel(const __grid_constant__
                                            a2a_dispatch_send::globals<
                                                EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                                TOKEN_DIM>
                                                G) {
  a2a_dispatch_send::kernel(G);
}

// input tokens MUST BE 2D here, we can group later
template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t a2a_kernels::fp8e4m3_a2a_dispatch_send(
    at::Tensor &input_tokens_tensor, at::Tensor &input_scales_tensor,
    at::Tensor &indices_tensor, at::Tensor &weights_tensor,
    uint32_t *token_offsets_ptr, uint32_t *expert_offsets_ptr,
    kittens::py::TKParallelTensor &out_tokens,
    kittens::py::TKParallelTensor &out_scales, at::Tensor &num_routed_tensor,
    kittens::py::TKParallelTensor &send_buffer,
    kittens::py::TKParallelTensor &send_scale_buffer,
    kittens::py::TKParallelTensor &barrier, uint32_t *sync_counter_ptr,
    uint32_t max_private_tokens, uint8_t *dispatch_route_done, int local_rank,
    int dp_size, cudaStream_t stream) {
  HOST_ASSERT(
      NUM_EXPERTS <= a2a_dispatch_send::config::NUM_THREADS,
      "needed for block wide parallel scan to ensure all experts are covered");
  const size_t token_dim = ti_align(TOKEN_DIM, sizeof(int4));
  const size_t token_scale_dim = ti_align(TOKEN_DIM / 128 * 4, sizeof(int4));
  const size_t token_stride = token_dim + token_scale_dim + 16; // padding

  const uint32_t num_tokens =
      static_cast<uint32_t>(input_tokens_tensor.size(0));

  using kernel_globals =
      a2a_dispatch_send::globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>;

  using tokens_layout = typename kernel_globals::tokens_layout;
  using scale_layout = typename kernel_globals::scale_layout;
  using indices_layout = typename kernel_globals::indices_layout;
  using weights_layout = typename kernel_globals::weights_layout;
  using token_offset_layout = typename kernel_globals::token_offset_layout;
  using expert_offsets_layout = typename kernel_globals::expert_offsets_layout;
  using barrier_layout = typename kernel_globals::barrier_layout;
  using out_tokens_layout = typename kernel_globals::out_tokens_layout;
  using out_scale_layout = typename kernel_globals::out_scale_layout;
  using send_buffer_layout = typename kernel_globals::send_buffer_layout;
  using send_scale_layout = typename kernel_globals::send_scale_layout;
  using num_routed_layout = typename kernel_globals::num_routed_layout;

  token_offset_layout token_offset = kittens::make_gl<token_offset_layout>(
      (uint64_t)token_offsets_ptr, 1, 1, (size_t)num_tokens, EXPERTS_PER_TOKEN);
  expert_offsets_layout expert_offsets =
      kittens::make_gl<expert_offsets_layout>((uint64_t)expert_offsets_ptr, 1,
                                              1, 1, NUM_EXPERTS);
  HOST_ASSERT(token_stride % sizeof(int4) == 0,
              "token_stride must be divisible by sizeof(int4)");

  kernel_globals args = {
      .input_tokens =
          kittens::py::tensor_to_gl<tokens_layout>(input_tokens_tensor),
      .input_scales =
          kittens::py::tensor_to_gl<scale_layout>(input_scales_tensor),
      .barrier = kittens::py::parallel_tensor_to_pgl<barrier_layout>(barrier),
      .out_tokens =
          kittens::py::parallel_tensor_to_pgl<out_tokens_layout, false>(
              out_tokens),
      .out_scales =
          kittens::py::parallel_tensor_to_pgl<out_scale_layout>(out_scales),
      .send_buffer =
          kittens::py::parallel_tensor_to_pgl<send_buffer_layout, false>(
              send_buffer),
      .send_scale_buffer =
          kittens::py::parallel_tensor_to_pgl<send_scale_layout>(
              send_scale_buffer),
      .num_routed = kittens::py::tensor_to_gl<num_routed_layout, false>(
          num_routed_tensor),
      .num_tokens = num_tokens,
      .indices =
          kittens::py::tensor_to_gl<indices_layout, false>(indices_tensor),
      .weights = kittens::py::tensor_to_gl<weights_layout>(weights_tensor),
      .token_offset = token_offset,
      .expert_offsets = expert_offsets,
      .sync_counter = sync_counter_ptr,
      .max_private_tokens = max_private_tokens,
      .dispatch_route_done = dispatch_route_done,
      .rank = local_rank,
      .dp_size = dp_size,
  };
  void *kernel_args[] = {&args};

  nvtxRangePush("a2a_dispatch_send_tk");

  cudaError_t status = cudaLaunchCooperativeKernel(
      (void *)&dispatch_send_global_kernel<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                           TOKEN_DIM>,
      dim3(a2a_dispatch_send::config::NUM_BLOCKS, 1, 1),
      dim3(a2a_dispatch_send::config::NUM_THREADS, 1, 1), kernel_args,
      args.dynamic_shared_memory(), stream);
  nvtxRangePop();
  return status;

  // kittens::py::launch_kernel<
  //     a2a_dispatch_send::config, kernel_globals,
  //     a2a_dispatch_send::kernel<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>>(
  //     args);
}
