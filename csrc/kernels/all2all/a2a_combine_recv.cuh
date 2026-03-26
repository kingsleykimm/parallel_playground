#pragma once
#include "a2a_kernels.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <moe_cuda/dtype.h>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/launch_utils.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <runtime/utils.h>

template <uint32_t kNumThreads, uint32_t NODE_SIZE, typename T, typename U,
          typename NumExpertsPerToken_t>
__global__ void __launch_bounds__(kNumThreads, 1) a2a_combine_recv_kernel(
    const size_t token_dim, size_t hidden_dim, size_t num_experts,
    size_t num_experts_per_token, size_t rank, size_t world_size,
    size_t num_tokens, const int32_t *bound_m_ptr, const int32_t *indices_ptr,
    const size_t indices_stride, const float *weights_ptr,
    const size_t weights_stride, U *out_tokens_ptr, size_t out_tokens_stride,
    uint8_t accumulate, std::byte *recv_buffer, uint32_t *token_offset,
    uint32_t *expert_offsets, uint32_t *sync_counter, uint32_t **sync_ptrs) {

  extern __shared__ std::byte shared_memory[];
  auto grid = cooperative_groups::this_grid();
  const auto warp_id = threadIdx.x / 32;
  const auto lane_id = threadIdx.x & 0x1f;

  const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

  uint32_t *positions = reinterpret_cast<uint32_t *>(shared_memory);

  { // the i = tix indexing is just to fill the shared memory
    uint32_t i = threadIdx.x;

    for (;;) {
      const uint32_t local_token = i / num_experts_per_token;
      const uint32_t token =
          blockIdx.x +
          local_token * gridDim.x; // this is the actual coordinate, that
                                   // corresponds to the dispatch kernel as well
      const uint32_t route = i % num_experts_per_token;

      if (token >= num_send_tokens) {
        break;
      }

      const uint32_t global_slot = token * num_experts_per_token + route;
      const uint32_t local_slot = local_token * num_experts_per_token + route;
      const uint32_t expert = indices_ptr[token * indices_stride + route];
      const uint32_t offset = token_offset[global_slot];
      const uint32_t position =
          (expert > 0 ? expert_offsets[expert - 1] : 0) + offset;
      positions[local_slot] = position;
      i += blockDim.x;
    }
    __syncthreads();
  }

  auto counter = *sync_counter;

  if constexpr (NODE_SIZE > 1) {
    if (warp_id == 0) {
      if (lane_id < NODE_SIZE) {
        auto local_rank = rank % NODE_SIZE;
        auto *flag = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
        while (ld_volatile_u32(flag) != counter)
          ;
      }
    }
  }
  __syncthreads();

  // accumulate across topk experts per token using grid strided loop
  for (uint32_t token = blockIdx.x, local_token = 0; token < num_send_tokens;
       token += gridDim.x, local_token++) {
    U *dst_ptr = out_tokens_ptr + token * out_tokens_stride;
    NumExpertsPerToken_t num_experts_per_token_bound(num_experts_per_token);

    using DstTy = VEC_LOAD<sizeof(U) * 8, U>;
    using SrcTy = VEC_LOAD<sizeof(T) * 8, T>;
    using AccTy = VEC_LOAD<sizeof(float) * 8, float>;

    using SrcTy_ptr = typename SrcTy::ptr_type;
    using AccTy_ptr = typename AccTy::ptr_type;

    constexpr uint32_t VEC_SIZE = SrcTy::SIZE;

    if constexpr (std::is_same_v<NumExpertsPerToken_t, NotFixed>) {
      for (unsigned j = threadIdx.x * DstTy::SIZE; j < hidden_dim;
           j += blockDim.x * DstTy::SIZE) {

        // we load accumulators in float
        uint4 acc = accumulate ? ld_global_nc_uint4((uint4 *)(dst_ptr + j))
                               : make_uint4(0, 0, 0, 0);

#pragma unroll 8
        for (uint32_t k = 0; k < num_experts_per_token_bound; k++) {
          const float weight = weights_ptr[token * weights_stride + k];
          const uint32_t position =
              positions[local_token * num_experts_per_token + k];

          T *buffer = (T *)(recv_buffer + position * token_dim);
          // this offsetse by position
          // vectorized load  first

          uint4 vec_load = ld_global_nc_uint4((uint4 *)(buffer + j));

          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // each of the uint -> 2 bfloats
            __nv_bfloat162 *x = reinterpret_cast<__nv_bfloat162 *>(&vec_load.x);
            float2 x_f = __bfloat1622float2(x[0]);
            __nv_bfloat162 *y = reinterpret_cast<__nv_bfloat162 *>(&vec_load.y);
            float2 y_f = __bfloat1622float2(y[0]);
            __nv_bfloat162 *z = reinterpret_cast<__nv_bfloat162 *>(&vec_load.z);
            float2 z_f = __bfloat1622float2(z[0]);
            __nv_bfloat162 *w = reinterpret_cast<__nv_bfloat162 *>(&vec_load.w);
            float2 w_f = __bfloat1622float2(w[0]);

            __nv_bfloat162 *acc_x = reinterpret_cast<__nv_bfloat162 *>(&acc.x);
            float2 acc_x_f = __bfloat1622float2(acc_x[0]);
            __nv_bfloat162 *acc_y = reinterpret_cast<__nv_bfloat162 *>(&acc.y);
            float2 acc_y_f = __bfloat1622float2(acc_y[0]);
            __nv_bfloat162 *acc_z = reinterpret_cast<__nv_bfloat162 *>(&acc.z);
            float2 acc_z_f = __bfloat1622float2(acc_z[0]);
            __nv_bfloat162 *acc_w = reinterpret_cast<__nv_bfloat162 *>(&acc.w);
            float2 acc_w_f = __bfloat1622float2(acc_w[0]);

            acc_x_f.x += x_f.x * weight;
            acc_x_f.y += x_f.y * weight;
            acc_y_f.x += y_f.x * weight;
            acc_y_f.y += y_f.y * weight;
            acc_z_f.x += z_f.x * weight;
            acc_z_f.y += z_f.y * weight;
            acc_w_f.x += w_f.x * weight;
            acc_w_f.y += w_f.y * weight;

            st_global_nc_uint4(make_uint4(pack_float_2(acc_x_f.x, acc_x_f.y),
                                          pack_float_2(acc_y_f.x, acc_y_f.y),
                                          pack_float_2(acc_z_f.x, acc_z_f.y),
                                          pack_float_2(acc_w_f.x, acc_w_f.y)),
                               (uint4 *)(dst_ptr + j));
          }

          else if constexpr (std::is_same_v<T, __half>) {
            __half2 *x = reinterpret_cast<__half2 *>(&vec_load.x);
            float2 x_f = __half22float2(x[0]);
            __half2 *y = reinterpret_cast<__half2 *>(&vec_load.y);
            float2 y_f = __half22float2(y[0]);
            __half2 *z = reinterpret_cast<__half2 *>(&vec_load.z);
            float2 z_f = __half22float2(z[0]);
            __half2 *w = reinterpret_cast<__half2 *>(&vec_load.w);
            float2 w_f = __half22float2(w[0]);

            __half2 *acc_x = reinterpret_cast<__half2 *>(&acc.x);
            float2 acc_x_f = __half22float2(acc_x[0]);
            __half2 *acc_y = reinterpret_cast<__half2 *>(&acc.y);
            float2 acc_y_f = __half22float2(acc_y[0]);
            __half2 *acc_z = reinterpret_cast<__half2 *>(&acc.z);
            float2 acc_z_f = __half22float2(acc_z[0]);
            __half2 *acc_w = reinterpret_cast<__half2 *>(&acc.w);
            float2 acc_w_f = __half22float2(acc_w[0]);

            acc_x_f.x += x_f.x * weight;
            acc_x_f.y += x_f.y * weight;
            acc_y_f.x += y_f.x * weight;
            acc_y_f.y += y_f.y * weight;
            acc_z_f.x += z_f.x * weight;
            acc_z_f.y += z_f.y * weight;
            acc_w_f.x += w_f.x * weight;
            acc_w_f.y += w_f.y * weight;

            acc_x[0] = __float22half2_rn(acc_x_f);
            acc_y[0] = __float22half2_rn(acc_y_f);
            acc_z[0] = __float22half2_rn(acc_z_f);
            acc_w[0] = __float22half2_rn(acc_w_f);

            st_global_nc_uint4(acc, (uint4 *)(dst_ptr + j));
          }

          else {
            DEVICE_ASSERT(false);
          }
        }
      }
    }

    else {
      static constexpr size_t NUM_EXPERTS = NumExpertsPerToken_t::Value;
      T *tokens[NUM_EXPERTS];
      float weights[NUM_EXPERTS];

#pragma unroll NUM_EXPERTS
      for (uint32_t k = 0; k < NUM_EXPERTS; k++) {
        const uint32_t position =
            positions[local_token * num_experts_per_token + k];
        tokens[k] = (T *)(recv_buffer + position * token_dim);
        weights[k] = weights_ptr[token * weights_stride + k];
      }

      for (uint32_t j = threadIdx.x * VEC_SIZE; j < hidden_dim;
           j += blockDim.x * VEC_SIZE) {
        uint4 acc = accumulate ? ld_global_nc_uint4((uint4 *)(dst_ptr + j))
                               : make_uint4(0, 0, 0, 0);

        SrcTy srcs[NUM_EXPERTS];

#pragma unroll NUM_EXPERTS
        for (uint32_t k = 0; k < NUM_EXPERTS; k++) {
          uint4 vec_load = ld_global_nc_uint4((uint4 *)(tokens[k] + j));
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // each of the uint -> 2 bfloats
            __nv_bfloat162 *x = reinterpret_cast<__nv_bfloat162 *>(&vec_load.x);
            float2 x_f = __bfloat1622float2(x[0]);
            __nv_bfloat162 *y = reinterpret_cast<__nv_bfloat162 *>(&vec_load.y);
            float2 y_f = __bfloat1622float2(y[0]);
            __nv_bfloat162 *z = reinterpret_cast<__nv_bfloat162 *>(&vec_load.z);
            float2 z_f = __bfloat1622float2(z[0]);
            __nv_bfloat162 *w = reinterpret_cast<__nv_bfloat162 *>(&vec_load.w);
            float2 w_f = __bfloat1622float2(w[0]);

            __nv_bfloat162 *acc_x = reinterpret_cast<__nv_bfloat162 *>(&acc.x);
            float2 acc_x_f = __bfloat1622float2(acc_x[0]);
            __nv_bfloat162 *acc_y = reinterpret_cast<__nv_bfloat162 *>(&acc.y);
            float2 acc_y_f = __bfloat1622float2(acc_y[0]);
            __nv_bfloat162 *acc_z = reinterpret_cast<__nv_bfloat162 *>(&acc.z);
            float2 acc_z_f = __bfloat1622float2(acc_z[0]);
            __nv_bfloat162 *acc_w = reinterpret_cast<__nv_bfloat162 *>(&acc.w);
            float2 acc_w_f = __bfloat1622float2(acc_w[0]);

            acc_x_f.x += x_f.x * weights[k];
            acc_x_f.y += x_f.y * weights[k];
            acc_y_f.x += y_f.x * weights[k];
            acc_y_f.y += y_f.y * weights[k];
            acc_z_f.x += z_f.x * weights[k];
            acc_z_f.y += z_f.y * weights[k];
            acc_w_f.x += w_f.x * weights[k];
            acc_w_f.y += w_f.y * weights[k];

            st_global_nc_uint4(make_uint4(pack_float_2(acc_x_f.x, acc_x_f.y),
                                          pack_float_2(acc_y_f.x, acc_y_f.y),
                                          pack_float_2(acc_z_f.x, acc_z_f.y),
                                          pack_float_2(acc_w_f.x, acc_w_f.y)),
                               (uint4 *)(dst_ptr + j));
          }

          else if constexpr (std::is_same_v<T, __half>) {
            __half2 *x = reinterpret_cast<__half2 *>(&vec_load.x);
            float2 x_f = __half22float2(x[0]);
            __half2 *y = reinterpret_cast<__half2 *>(&vec_load.y);
            float2 y_f = __half22float2(y[0]);
            __half2 *z = reinterpret_cast<__half2 *>(&vec_load.z);
            float2 z_f = __half22float2(z[0]);
            __half2 *w = reinterpret_cast<__half2 *>(&vec_load.w);
            float2 w_f = __half22float2(w[0]);

            __half2 *acc_x = reinterpret_cast<__half2 *>(&acc.x);
            float2 acc_x_f = __half22float2(acc_x[0]);
            __half2 *acc_y = reinterpret_cast<__half2 *>(&acc.y);
            float2 acc_y_f = __half22float2(acc_y[0]);
            __half2 *acc_z = reinterpret_cast<__half2 *>(&acc.z);
            float2 acc_z_f = __half22float2(acc_z[0]);
            __half2 *acc_w = reinterpret_cast<__half2 *>(&acc.w);
            float2 acc_w_f = __half22float2(acc_w[0]);

            acc_x_f.x += x_f.x * weights[k];
            acc_x_f.y += x_f.y * weights[k];
            acc_y_f.x += y_f.x * weights[k];
            acc_y_f.y += y_f.y * weights[k];
            acc_z_f.x += z_f.x * weights[k];
            acc_z_f.y += z_f.y * weights[k];
            acc_w_f.x += w_f.x * weights[k];
            acc_w_f.y += w_f.y * weights[k];

            acc_x[0] = __float22half2_rn(acc_x_f);
            acc_y[0] = __float22half2_rn(acc_y_f);
            acc_z[0] = __float22half2_rn(acc_z_f);
            acc_w[0] = __float22half2_rn(acc_w_f);
            st_global_nc_uint4(acc, (uint4 *)(dst_ptr + j));
          }

          else {
            DEVICE_ASSERT(false);
          }
        }
      }
    }
  }

  grid.sync();

  if (blockIdx.x == 0) {
    if (warp_id == 0) {
      if constexpr (NODE_SIZE > 1) {
        if (lane_id < NODE_SIZE) {
          uint32_t local_rank = rank % NODE_SIZE;
          auto *flag = &sync_ptrs[local_rank][lane_id];
          st_volatile_u32(flag, counter + 1);
        }
      }
    } else if (warp_id == 1) {
      if (ti_elect_one_sync()) {
        *sync_counter = counter + 1;
      }
    }
  }
}

cudaError_t a2a_kernels::a2a_combine_recv(
    size_t num_blocks, size_t hidden_dim, size_t x_elemsize,
    c10::ScalarType in_dtype, c10::ScalarType out_dtype, size_t num_experts,
    size_t num_experts_per_token, size_t rank, size_t node_size,
    size_t world_size, size_t num_tokens, int32_t *bound_m_ptr,
    int32_t *indices_ptr, size_t indices_stride, float *weights_ptr,
    size_t weights_stride, uint8_t *out_tokens_ptr, size_t out_tokens_stride,
    bool accumulate, uint8_t *recv_buffer, uint32_t *token_offset,
    uint32_t *expert_offsets, uint32_t *sync_counter, uint32_t **sync_ptrs,
    cudaStream_t stream) {
  constexpr size_t kNumThreads = 512;
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(kNumThreads, 1, 1);

  const size_t token_dim = ti_align(hidden_dim * x_elemsize, sizeof(uint4));
  const size_t tokens_per_block = ti_ceil_div(num_tokens, num_blocks);

  void *args[] = {
      const_cast<size_t *>(&token_dim),
      &hidden_dim,
      &num_experts,
      &num_experts_per_token,
      &rank,
      &world_size,
      &num_tokens,
      &bound_m_ptr,
      &indices_ptr,
      &indices_stride,
      &weights_ptr,
      &weights_stride,
      &out_tokens_ptr,
      &out_tokens_stride,
      &accumulate,
      &recv_buffer,
      &token_offset,
      &expert_offsets,
      &sync_counter,
      &sync_ptrs,
  };

  const size_t shared_memory =
      tokens_per_block * num_experts_per_token * sizeof(uint32_t);

  cudaError_t status;
  nvtxRangePush("combine_recv");
  LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
    LAUNCH_ACT_TYPE(in_dtype, InTy, {
      LAUNCH_ACT_TYPE(out_dtype, OutTy, {
        LAUNCH_NUM_EXPERTS_PER_TOKEN(
            num_experts_per_token, NumExpertsPerToken_t, {
              status = cudaLaunchCooperativeKernel(
                  (void *)&a2a_combine_recv_kernel<kNumThreads, NODE_SIZE, InTy,
                                                   OutTy, NumExpertsPerToken_t>,
                  dimGrid, dimBlock, args, shared_memory, stream);
            });
      });
    });
  });

  nvtxRangePop();
  return status;
}
