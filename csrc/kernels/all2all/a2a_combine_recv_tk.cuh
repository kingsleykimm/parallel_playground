#pragma once
#include "a2a_kernels.h"
#include "runtime/device.hpp"
#include "types/register/rv_layout.cuh"
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <kittens.cuh>
#include <moe_cuda/dtype.h>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <pyutils/parallel_tensor.cuh>
#include <pyutils/torchutils.cuh>
#include <runtime/utils.h>

// assume tokens are type __nv_bfloat16

using namespace kittens;

namespace a2a_combine_recv {

struct config {
  static constexpr int NUM_BLOCKS = 132;
  static constexpr int NUM_THREADS = 512;
  static constexpr int NUM_DEVICES = 4;
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
struct globals {

  using barrier_layout = pgl<gl<int, -1, -1, -1, config::NUM_DEVICES * 2>,
                             config::NUM_DEVICES, false>;
  // i don't like making it an fp8, but i just did this since it's equivalent in
  // size to a byte
  using indices_layout = gl<int, -1, -1, -1, EXPERTS_PER_TOKEN>;
  using weights_layout = gl<float, -1, -1, -1, EXPERTS_PER_TOKEN>;
  using recv_buffer_layout =
      pgl<gl<bf16, -1, -1, -1, -1>, config::NUM_DEVICES, false>;
  using out_tokens_layout = gl<bf16, -1, -1, -1, TOKEN_DIM>;

  barrier_layout barrier;
  recv_buffer_layout recv_buffer;
  out_tokens_layout out_tokens;
  indices_layout indices;
  weights_layout weights;

  uint32_t *__restrict__ token_offset;
  uint32_t *__restrict__ expert_offsets;
  uint32_t *__restrict__ sync_counter;
  const int num_tokens;
  const bool accumulate;
  const int rank;
  const int dp_group;
  __host__ inline int dynamic_shared_memory() {
    auto num_tokens_per_block = ti_ceil_div(num_tokens, config::NUM_BLOCKS);
    return num_tokens_per_block * EXPERTS_PER_TOKEN * (int)sizeof(int);
  }
};

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__device__ void kernel(globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &G) {

  extern __shared__ int shm[];
  auto grid = cooperative_groups::this_grid();
  auto counter = *G.sync_counter;

  auto warp_id = kittens::warpid();
  auto lane_id = kittens::laneid();
  int *positions = &shm[0];

  {
    uint32_t i = threadIdx.x;
    for (;;) {
      const uint32_t local_token = i / EXPERTS_PER_TOKEN;
      // this token has multiple dimensions - we're striding across the grid AND
      // the block. On each block stride, we get a new local linearized (token +
      // expert) index that we convert into the global token.

      // with 512 threads, a single CTA will process 512 / EXPERTS_PER_TOKEN
      // tokens per block stride, and an entire grid processes floor(512/
      // EXPERTS_PER_TOKEN) * NUM_SMS
      const uint32_t token = blockIdx.x + local_token * gridDim.x;
      const uint32_t route = i % EXPERTS_PER_TOKEN;

      if (token >= G.num_tokens) {
        break;
      }

      const int global_slot = EXPERTS_PER_TOKEN * token + route;
      const int local_slot = EXPERTS_PER_TOKEN * local_token + route;

      const uint32_t expert = G.indices[{
          static_cast<int>(token),
          static_cast<int>(route)}]; // this uses the global expert, needed
      const uint32_t offset = G.token_offset[global_slot];
      DEVICE_ASSERT(expert < NUM_EXPERTS);
      const uint32_t position =
          (expert > 0 ? G.expert_offsets[expert - 1] : 0) + offset;
      positions[local_slot] = position;
      i += blockDim.x;
    }

    everyone::sync(1);
  }

  if (warp_id == 0 && config::NUM_DEVICES > 1) {
    auto local_rank = G.rank % config::NUM_DEVICES;
    if (lane_id < config::NUM_DEVICES) {
      node_sync::wait(G.barrier, {lane_id + config::NUM_DEVICES},
                      local_rank, // make sure all other ranks have signaled
                                  // that G.rank is ready
                      counter);
    }
  }
  everyone::sync(1);

  for (int token = blockIdx.x, local_token = 0; token < G.num_tokens;
       token += gridDim.x, local_token++) {
    __nv_bfloat16 *out_ptr = &G.out_tokens[{token, 0}];
    using vec_loader =
        VEC_LOAD<sizeof(__nv_bfloat16) * TOKEN_DIM * 8, __nv_bfloat16>;
    vec_loader vec_load;
    using vec_load_ptr_t = typename vec_loader::ptr_type;

    __nv_bfloat16 *topk_tokens[EXPERTS_PER_TOKEN];
    float weights[EXPERTS_PER_TOKEN];
    constexpr int VEC_SIZE = vec_load.SIZE;
// reduce weights + indices here
#pragma unroll EXPERTS_PER_TOKEN
    for (int e = 0; e < EXPERTS_PER_TOKEN; ++e) {
      const int position = positions[local_token * EXPERTS_PER_TOKEN + e];
      topk_tokens[e] = (__nv_bfloat16 *)(&G.recv_buffer[G.rank][{position, 0}]);
      weights[e] = G.weights[{token, e}];
    }

    for (int j = threadIdx.x * VEC_SIZE; j < TOKEN_DIM;
         j += blockDim.x * VEC_SIZE) {

      __nv_bfloat16 acc[VEC_SIZE] = {0};
      float sum[VEC_SIZE];
      if (G.accumulate) {
        vec_load_ptr_t load_val = ld_global_uint_dispatch(
            reinterpret_cast<vec_load_ptr_t *>(out_ptr + j));
        VEC_LOAD_CVT<vec_load_ptr_t, __nv_bfloat16>::convert(load_val, &acc[0]);
      }

#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
        sum[v] = __bfloat162float(acc[v]);
      }

      __nv_bfloat16 srcs[EXPERTS_PER_TOKEN][VEC_SIZE];
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TOKEN; e++) {
        vec_load_ptr_t load_val = ld_global_uint_dispatch(
            reinterpret_cast<vec_load_ptr_t *>(topk_tokens[e] + j));
        VEC_LOAD_CVT<vec_load_ptr_t, __nv_bfloat16>::convert(load_val,
                                                             &srcs[e][0]);
      }

#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
#pragma unroll
        for (int e = 0; e < EXPERTS_PER_TOKEN; e++) {
          sum[v] += weights[e] * __bfloat162float(srcs[e][v]);
        }
      }

#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
        acc[v] = __float2bfloat16(sum[v]);
      }
      vec_load_ptr_t out_val;
      VEC_LOAD_CVT<__nv_bfloat16, vec_load_ptr_t>::convert(&acc[0], &out_val);
      st_global_uint_dispatch(out_val,
                              reinterpret_cast<vec_load_ptr_t *>(out_ptr + j));
    }
  }

  if constexpr (config::NUM_DEVICES > 1) {
    grid.sync();
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        *G.sync_counter = counter + 1;
      }
      if (threadIdx.x < config::NUM_DEVICES) {
        auto local_rank = threadIdx.x % config::NUM_DEVICES;
        node_sync::signal(G.barrier, {G.rank}, local_rank, counter + 1);
      }
    }
  }
}

} // namespace a2a_combine_recv

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
__global__ void a2a_combine_recv_global_kernel(
    a2a_combine_recv::globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> G) {
  a2a_combine_recv::kernel(G);
}

template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
cudaError_t a2a_kernels::a2a_combine_recv_tk(
    kittens::py::TKParallelTensor &barrier_tensor,
    kittens::py::TKParallelTensor &recv_buffer_tensor,
    at::Tensor &out_tokens_tensor, at::Tensor &indices_tensor,
    at::Tensor &weights_tensor, uint32_t *token_offset,
    uint32_t *expert_offsets, uint32_t *sync_counter, int num_tokens,
    bool accumulate, int rank, int dp_group, cudaStream_t stream) {

  using GLOBALS =
      a2a_combine_recv::globals<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>;

  using barrier_layout = typename GLOBALS::barrier_layout;
  using out_tokens_layout = typename GLOBALS::out_tokens_layout;
  using indices_layout = typename GLOBALS::indices_layout;
  using weights_layout = typename GLOBALS::weights_layout;
  using recv_buffer_layout = typename GLOBALS::recv_buffer_layout;

  GLOBALS g = {
      .barrier = kittens::py::parallel_tensor_to_pgl<barrier_layout, false>(
          barrier_tensor),
      .recv_buffer =
          kittens::py::parallel_tensor_to_pgl<recv_buffer_layout, false>(
              recv_buffer_tensor),
      .out_tokens =
          kittens::py::tensor_to_gl<out_tokens_layout>(out_tokens_tensor),
      .indices = kittens::py::tensor_to_gl<indices_layout>(indices_tensor),
      .weights = kittens::py::tensor_to_gl<weights_layout>(weights_tensor),
      .token_offset = token_offset,
      .expert_offsets = expert_offsets,
      .sync_counter = sync_counter,
      .num_tokens = num_tokens,
      .accumulate = accumulate,
      .rank = rank,
      .dp_group = dp_group};

  cudaError_t status;
  void *args[] = {&g};

  HOST_ASSERT(g.dynamic_shared_memory() < device_prop->get_smem_size(),
              "Maximum shared meomry per SM exceede");
  nvtxRangePush("combine_recv");
  status = cudaLaunchCooperativeKernel(
      &a2a_combine_recv_global_kernel<EXPERTS_PER_TOKEN, NUM_EXPERTS,
                                      TOKEN_DIM>,
      dim3(a2a_combine_recv::config::NUM_BLOCKS),
      dim3(a2a_combine_recv::config::NUM_THREADS), args,
      g.dynamic_shared_memory(), stream);

  nvtxRangePop();
  return status;
}
