/*
Note: add compile time specialization dispatch functions for more of the common
ones

*/
#pragma once
#include "a2a_kernels.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/launch_utils.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <runtime/utils.h>

// notes for future reference
// struct All2AllKernelParams {
//   size_t token_dim;       // aligned token size to int4 for vec load =
//   align(hidden_dim * sizeof(type), 16) size_t token_scale_dim; // aligned
//   scale dim for int4 load size_t token_stride;    // token_dim +
//   token_scale_dim + 16 (extra bytes for padding) size_t hidden_dim; size_t
//   hidden_dim_scale; // number of quant scale groups in the quant direction
//   for 1D block size_t x_elemsize; size_t x_scale_elemsize; size_t
//   num_experts; size_t num_experts_per_token; size_t
//       max_private_tokens; // unused - ported over in case RDMA is added
//       later, but I'm only writing for NVlink right now
//   size_t rank;            // GPU rank
//   size_t dp_size;         // DP group size
//   size_t node_size;
//   size_t world_size;

//   // input data
//   size_t num_tokens;           // static batch size
//   int32_t *bound_m_ptr;        // dynamic batch size, overrides num_tokens
//   const std::byte *x_ptr;      // input token embedding pointers
//   size_t x_stride;             // byte stride between tokens
//   float *x_scale_ptr;          // size (num_tokens, hidden_dim_scale)
//   size_t x_scale_stride_elem;  // distance between cons scales in token
//   size_t x_scale_stride_token; // distance between tokens in scale
//   int32_t *indices;            // [num_tokens, num_experts_per_token] expert
//   index size_t indices_stride;       // usually num_experts_per_token float
//   *weights;              // [num_tokens, num_expert_per_token] // top k
//   weights size_t weight_stride;

//   // scratchpad buffers
//   uint32_t *token_offset; // [num_tokens, num_experts_per_token], atomic
//   counts, where this is the position of token
//   t
//                           // in an expert's buffer
//   uint32_t *num_routed; // [dp_size, num_experts] number of tokens allocated
//   to each expert, needed for space allocation uint32_t *expert_offsets; //
//   [num_experts] - cumulative of token counts per expert for each rank, so we
//   can quickly
//                             // offset into an expert's contiguous slice of
//                             tokens
//   // std::byte * send_buffer (removed)

//   uint32_t *sync_counter; // used to sync across nvlink
//   uint32_t **sync_ptrs;
//   std::byte **recv_ptrs;
// };

struct ExpertAndOffset {
  uint32_t expert;
  uint32_t offset;
  uint32_t position;
  float weight;
};

// from pplx-garden all2all, a helper class to efficiently access expert indices
// and offsets, once computed by the kernel
template <typename ExpertsPerToken_t> class ExpertIterator {

public:
  __forceinline__ __device__ ExpertIterator(
      ExpertsPerToken_t num_experts_per_token, const int32_t *indices,
      const size_t indices_stride, const float *weights,
      const size_t weights_stride, const uint32_t *token_offset,
      const uint32_t *expert_offsets, uint32_t token, uint32_t experts_per_rank)
      : num_experts_per_token_(num_experts_per_token), indices_(indices),
        indices_stride_(indices_stride), weights_(weights),
        weights_stride_(weights_stride), token_offset_(token_offset),
        expert_offsets_(expert_offsets), token_(token),
        experts_per_rank_(experts_per_rank) {}

  __forceinline__ __device__ ExpertAndOffset operator[](uint32_t expert_idx) {
    const uint32_t expert = indices_[token_ * indices_stride_ + expert_idx];
    const float weight = weights_[token_ + weights_stride_ + expert_idx];

    // offset with expert_idx buffer
    const uint32_t offset =
        token_offset_[token_ * num_experts_per_token_ + expert_idx];
    const uint32_t dst_rank = expert / experts_per_rank_;
    const uint32_t position =
        ((expert > 0) ? expert_offsets_[expert - 1] : 0) + offset;
    const uint32_t rank_offset =
        dst_rank > 0 ? expert_offsets_[dst_rank * experts_per_rank_ - 1] : 0;

    // offset is used when accessing private buffer, it is a relative position
    // (direct NVlink write) position is the absolute offset, used for accessing
    // the send buffer
    return {expert, position - rank_offset, position, weight};
  }

private:
  ExpertsPerToken_t num_experts_per_token_;
  const int32_t *indices_;
  const size_t indices_stride_;
  const float *weights_;
  const size_t weights_stride_;
  const uint32_t *token_offset_;
  const uint32_t *expert_offsets_;
  uint32_t token_;
  uint32_t experts_per_rank_;
};

// partial template specialization of ExpertIterator for compiler optimization
template <size_t N> class ExpertIterator<Fixed<N>> {

public:
  __forceinline__ __device__
  ExpertIterator(Fixed<N> num_experts_per_token, const int32_t *indices,
                 const size_t indices_stride, const float *weights,
                 const size_t weights_stride, const uint32_t *token_offset,
                 const uint32_t *expert_offsets, uint32_t token,
                 uint32_t experts_per_rank) {

#pragma unroll N
    for (int e = 0; e < num_experts_per_token; e++) {
      const uint32_t expert = indices[token * indices_stride + e];
      const float weight = weights_[token + weights_stride + e];
      // offset with expert_idx buffer
      const uint32_t offset = token_offset[token * num_experts_per_token + e];
      const uint32_t dst_rank = expert / experts_per_rank;
      const uint32_t position =
          ((expert > 0) ? expert_offsets[expert - 1] : 0) + offset;
      const uint32_t rank_offset =
          dst_rank > 0 ? expert_offsets[dst_rank * experts_per_rank - 1] : 0;

      weights_[e] = weight;
      offsets_[e] = offset;
      positions_[e] = position;
      offsets_[e] = position - rank_offset;
    }
  }

  __forceinline__ __device__ ExpertAndOffset operator[](uint32_t e) {
    return {experts_[e], offsets_[e], positions_[e], weights_[e]};
  }

private:
  uint32_t experts_[N];
  uint32_t offsets_[N];
  uint32_t positions_[N];
  float weights_[N];
};

// QUICK tells us whether num_blocks >= num_tokens
template <bool QUICK, uint32_t kNumThreads, uint32_t NODE_SIZE,
          typename TokenDim_t, typename NumExpertsPerToken_t,
          typename HiddenDimScale_t>
__global__ void __launch_bounds__(kNumThreads, 1) a2a_dispatch_send_kernel(
    const size_t token_dim, const size_t token_scale_dim,
    const size_t token_stride, size_t hidden_dim, size_t hidden_dim_scale,
    size_t num_experts, size_t num_experts_per_token, size_t max_private_tokens,
    size_t rank, size_t dp_size, size_t node_size, size_t world_size,
    size_t num_tokens_, const int32_t *__restrict__ bound_m_ptr,
    const std::byte *__restrict__ x_ptr, size_t x_elemsize, size_t x_stride,
    const float *__restrict__ x_scale_ptr, size_t x_scale_elemsize,
    size_t x_scale_stride_elem, size_t x_scale_stride_token,
    const int32_t *__restrict__ indices, size_t indices_stride,
    const float *__restrict__ weights, size_t weights_stride,
    uint32_t *__restrict__ token_offset, uint32_t *__restrict__ num_routed,
    uint32_t *__restrict__ expert_offsets, std::byte *__restrict__ send_buffer,
    uint32_t *__restrict__ sync_counter, uint32_t **__restrict__ sync_ptrs,
    std::byte **__restrict__ recv_ptrs) {
  TokenDim_t token_dim_fixed(token_dim);
  NumExpertsPerToken_t num_experts_per_token_fixed(num_experts_per_token);
  HiddenDimScale_t hidden_dim_scale_fixed(hidden_dim_scale);
  extern __shared__ std::byte shared_memory[];

  auto grid = cooperative_groups::this_grid();
  auto thread_block = cooperative_groups::this_thread_block();

  const size_t warp_id = threadIdx.x / 32;
  const size_t lane_id = threadIdx.x & 0x1f;

  std::byte *recv_ptrs_local[NODE_SIZE];
  for (uint32_t i = 0; i < NODE_SIZE; i++) {
    recv_ptrs_local[i] = recv_ptrs[i];
  }

  uint32_t counter = *sync_counter;

  const size_t node_rank = rank / NODE_SIZE; // should always be 0
  const size_t local_rank = rank % NODE_SIZE;
  const size_t dp_group = rank / dp_size;
  const size_t experts_per_rank = ti_ceil_div(num_experts, world_size);
  const size_t first_expert = rank * experts_per_rank;
  const size_t last_expert = min(first_expert + experts_per_rank, num_experts);

  const size_t num_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens_;
  if (blockIdx.x == 0) {

    uint32_t *tokens_per_expert = (uint32_t *)shared_memory;
    for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
      tokens_per_expert[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_tokens * num_experts_per_token;
         i += blockDim.x) {
      const uint32_t token = i / num_experts_per_token;
      const uint32_t index = i % num_experts_per_token;
      const uint32_t expert = __ldg(&indices[token * indices_stride + index]);

      // token offset is of length num_tokens * params.num_experts_per_token, so
      // this gives the absolute offset, for each expert for each (token_idx,
      // in_expert_token_idx) -> maps to token offset in actual expert, which
      // ranges from (0, params.num_experts_per_token)
      token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
    }
    __syncthreads();

    const uint32_t i = threadIdx.x;
    const uint32_t num_warps = ti_ceil_div(num_experts, 32);
    uint32_t *expert_sums = (uint32_t *)shared_memory;

    uint32_t *local_num_routed = num_routed + dp_group * num_experts;
    uint32_t expert_offset = 0;

    if (i < num_experts) {
      expert_offset = tokens_per_expert[i];
      local_num_routed[i] =
          expert_offset; // for each token, we write in the number of tokens for
                         // the current expert - used later for allocations
    }
    __syncthreads();

    // warp-local parallel scan
    for (int offset = 1; offset < 32; offset <<= 1) {
      uint32_t warp_sum_expert =
          __shfl_up_sync(uint32_t(-1), expert_offset, offset);
      if (lane_id >= offset) {
        expert_offset += warp_sum_expert;
      }
    }
    if (lane_id == 31) {
      expert_sums[warp_id] =
          expert_offset; // store end cumsum, expert_sums is size NUM_WARPS
    }
    __syncthreads();

    if (warp_id == 0) {
      uint32_t total_expert_sum =
          (lane_id < num_warps) ? expert_sums[lane_id] : 0;
      for (uint32_t offset = 0; offset < 32; offset <<= 1) {
        uint32_t warp_sum =
            __shfl_down_sync(uint32_t(-1), total_expert_sum, offset);
        if (lane_id >= offset) {
          total_expert_sum += warp_sum;
        }
      }
      if (lane_id < num_warps) {
        expert_sums[lane_id] = total_expert_sum; // we store per-warp cumsum
      }
    }
    __syncthreads();

    if (i < num_experts) {
      if (warp_id > 0) {
        expert_offsets[i] = expert_sums[warp_id - 1] + expert_offset;
      } else {
        expert_offsets[i] = expert_offset;
      }
    }
  }
  __syncthreads();

  // sychronize across NVlink barrier for any previous combine stages
  if constexpr (NODE_SIZE > 1) {
    if (blockIdx.x == 0) {
      // we check the sync_ptrs[local_rank][1 * NODE_SIZE] elements to see if
      // they've reached the counter - this signals they are ready to recieve
      // more tokens in their recv_ptr buffers
      for (int peer = threadIdx.x; peer < NODE_SIZE; peer += kNumThreads) {
        while ((ld_volatile_u32(&sync_ptrs[local_rank][peer]) != counter))
          ;
      }
    }
  }

  grid.sync(); // sync after barriers are filled

  // increment counter across all devices
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sync_counter = counter + 1;
  }

  if constexpr (QUICK) {
    // we don't have to use CTA strided loop here, one CTA per token
    int token = blockIdx.x;
    if (token < num_tokens) {
      uint4 *x_token_src = (uint4 *)(x_ptr + token * x_stride);
      float *x_scale_src =
          (float *)(x_scale_ptr + token * x_scale_stride_token);
      ExpertIterator<NumExpertsPerToken_t> expert_iterator(
          num_experts_per_token_fixed, indices, indices_stride, weights,
          weights_stride, token_offset, expert_offsets, token,
          experts_per_rank);

      if constexpr (std::is_same_v<TokenDim_t, NotFixed>) {
        // we don't need to worry about misalignment here - we force tokendim to
        // be padded to sizeof(uint4) (16) this loop is across token / hidden
        // dim, for a given token we are copying all of its dimensions
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim;
             i += kNumThreads) {
          const bool has_scale = x_scale_ptr && i < hidden_dim_scale;

          uint4 token_val = ld_global_nc_uint4(&x_token_src[i]);
          float scale_val;
          if (has_scale) {
            scale_val = *(x_scale_src + i * x_scale_stride_elem);
          }

          // scatter to each of the experts assigned to one token
#pragma unroll
          for (uint32_t e = 0; e < num_experts_per_token_fixed; e++) {
            auto route = expert_iterator[e];
            const uint32_t dst_rank = route.expert / experts_per_rank;
            const uint32_t dst_node = dst_rank / NODE_SIZE;
            if (dst_node == node_rank && dst_rank != rank &&
                route.offset < max_private_tokens) {
              if (dst_rank % dp_size == rank % dp_size) {
                // NV link write
                const uint32_t local_peer = dst_rank % NODE_SIZE;
                // advance token pointer to the appropriate expert slice
                std::byte *token_ptr =
                    recv_ptrs_local[local_peer] +
                    (dp_group * max_private_tokens + route.offset) *
                        token_stride;
                uint4 *x_token_dst = (uint4 *)token_ptr;
                // store current iteration of expert dim
                st_global_nc_uint4(token_val, &x_token_dst[i]);
                if (has_scale) {
                  *((float *)(token_ptr + token_dim_fixed + i)) =
                      scale_val; // offset after the token_dim_fixed, this means
                                 // receive pointers should be padded enough for
                                 // both scale and activations
                }
              } else {
                // local write, self-copy in same rank
                // since we don't use RDMA the only tokens being written to send
                // buffer are local copies
                std::byte *token_ptr =
                    send_buffer + route.position * token_stride;
                uint4 *x_token_dst = (uint4 *)token_ptr;
                st_global_nc_uint4(token_val, &x_token_dst[i]);
                if (has_scale) {
                  *(float *)(token_ptr + token_dim_fixed + i) = scale_val;
                }
              }
            }
          }
        }
      } else {
        constexpr size_t TOKEN_DIM = TokenDim_t::Value;
        constexpr size_t NUM_STRIDES =
            constexpr_ti_ceil_div(TOKEN_DIM, kNumThreads);

        uint4 vals[NUM_STRIDES];
        float scales[NUM_STRIDES];

        // GMEM -> RMEM
#pragma unroll NUM_STRIDES
        for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
             i += kNumThreads, s++) {
          const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
          vals[s] = ld_global_nc_uint4(&x_token_src[i]);
          if (has_scale) {
            scales[s] = *(float *)(x_scale_src + i * x_scale_stride_elem);
          }
        }

// RMEM -> peer buffers
#pragma unroll
        for (uint32_t e = 0; e < num_experts_per_token_fixed; e++) {
          auto route = expert_iterator[e];
          const uint32_t dst_rank = route.expert / experts_per_rank;
          const uint32_t dst_node = dst_rank / NODE_SIZE;
          DEVICE_ASSERT(route.offset <
                        max_private_tokens); // since we're only doing NVLink,
                                             // we're kind of limited here

          if (dst_node == node_rank && dst_rank != rank &&
              route.offset < max_private_tokens) {
            if (dst_rank % dp_size == rank % dp_size) {
              const uint32_t local_peer = dst_rank % NODE_SIZE;
              std::byte *token_ptr =
                  recv_ptrs_local[local_peer] +
                  (dp_group * max_private_tokens + route.offset) * token_stride;
              uint4 *token_dst = (uint4 *)token_ptr;
              for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
                   i += kNumThreads, s++) {
                const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
                st_global_nc_uint4(vals[s], &token_dst[i]);
                if (has_scale) {
                  *(float *)(token_dst + token_dim_fixed + i) = scales[s];
                }
              }
            }
          } else { // for either same node copies or overflow, we use our
                   // send_buffer
            // for same copies, we use the absolute offset
            for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
                 i += kNumThreads, s++) {
              const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
              std::byte *token_ptr =
                  send_buffer + route.position * token_stride;
              uint4 *x_token_dst = (uint4 *)token_ptr;
              st_global_nc_uint4(vals[s], &x_token_dst[i]);
              if (has_scale) {
                *(float *)(token_ptr + token_dim_fixed + i) = scales[s];
              }
            }
          }
        }
      }
    }

    else {
    }
  } else {
    // num_blocks < num_tokens, prefill case
    if constexpr (NODE_SIZE >= 1) {
      uint32_t num_local_tokens = 0;
      for (uint32_t token = blockIdx.x; token < num_tokens;
           token += gridDim.x, num_local_tokens++) {

        uint4 *x_token_src = (uint4 *)(x_ptr + token * x_stride);
        float *x_scale_src =
            (float *)(x_scale_ptr + token * x_scale_stride_token);

        ExpertIterator<NumExpertsPerToken_t> expert_iterator(
            num_experts_per_token_fixed, indices, indices_stride, weights,
            weights_stride, token_offset, expert_offsets, token,
            experts_per_rank);

        if constexpr (std::is_same_v<TokenDim_t, NotFixed>) {
          // we don't need to worry about misalignment here - we force tokendim
          // to be padded to sizeof(uint4) (16) this loop is across token /
          // hidden dim, for a given token we are copying all of its dimensions
          for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim;
               i += kNumThreads) {
            const bool has_scale = x_scale_ptr && i < hidden_dim_scale;

            uint4 token_val = ld_global_nc_uint4(&x_token_src[i]);
            float scale_val;
            if (has_scale) {
              scale_val = *(x_scale_src + i * x_scale_stride_elem);
            }

            // scatter to each of the experts assigned to one token
#pragma unroll
            for (uint32_t e = 0; e < num_experts_per_token_fixed; e++) {
              auto route = expert_iterator[e];
              const uint32_t dst_rank = route.expert / experts_per_rank;
              const uint32_t dst_node = dst_rank / NODE_SIZE;
              DEVICE_ASSERT(route.offset < max_private_tokens);
              // NV link write
              if (dst_node == node_rank && dst_rank != rank &&
                  route.offset < max_private_tokens) {
                if (dst_rank % dp_size == rank % dp_size) {
                  const uint32_t local_peer = dst_rank % NODE_SIZE;

                  // advance token pointer to the appropriate expert slice
                  std::byte *token_ptr =
                      recv_ptrs_local[local_peer] +
                      (dp_group * max_private_tokens + route.offset) *
                          token_stride;
                  uint4 *x_token_dst = (uint4 *)token_ptr;
                  // store current iteration of expert dim
                  st_global_nc_uint4(token_val, &x_token_dst[i]);
                  if (has_scale) {
                    *((float *)(token_ptr + token_dim_fixed + i)) =
                        scale_val; // offset after the token_dim_fixed, this
                                   // means receive pointers should be padded
                                   // enough for both scale and activations
                  }
                } else {
                  // for same GPU to GPU copies, we use the send buffer
                  std::byte *token_ptr =
                      send_buffer + route.position * token_stride;
                  uint4 *x_token_dst = (uint4 *)token_ptr;
                  st_global_nc_uint4(token_val, &x_token_dst[i]);
                  if (has_scale) {
                    *(float *)(token_ptr + token_dim_fixed + i) = scale_val;
                  }
                }
              }
            }
          }
        } else {
          constexpr size_t TOKEN_DIM = TokenDim_t::Value;
          constexpr size_t NUM_STRIDES =
              constexpr_ti_ceil_div(TOKEN_DIM, kNumThreads);

          uint4 vals[NUM_STRIDES];
          float scales[NUM_STRIDES];

          // GMEM -> RMEM
#pragma unroll NUM_STRIDES
          for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
               i += kNumThreads, s++) {
            const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
            vals[s] = ld_global_nc_uint4(&x_token_src[i]);
            if (has_scale) {
              scales[s] = *(float *)(x_scale_src + i * x_scale_stride_elem);
            }
          }

// RMEM -> peer buffers
#pragma unroll
          for (uint32_t e = 0; e < num_experts_per_token_fixed; e++) {
            auto route = expert_iterator[e];
            const uint32_t dst_rank = route.expert / experts_per_rank;
            const uint32_t dst_node = dst_rank / NODE_SIZE;
            DEVICE_ASSERT(route.offset <
                          max_private_tokens); // since we're only doing NVLink,
                                               // we're kind of limited here
            if (dst_node == node_rank && dst_rank != rank &&
                route.offset < max_private_tokens) {
              if (dst_rank % dp_size == rank % dp_size) {
                const uint32_t local_peer = dst_rank % NODE_SIZE;
                std::byte *token_ptr =
                    recv_ptrs_local[local_peer] +
                    (dp_group * max_private_tokens + route.offset) *
                        token_stride;
                uint4 *token_dst = (uint4 *)token_ptr;
                for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
                     i += kNumThreads, s++) {
                  const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
                  st_global_nc_uint4(vals[s], &token_dst[i]);
                  if (has_scale) {
                    *(float *)(token_dst + token_dim_fixed + i) = scales[s];
                  }
                }
              }

              else {
                // self copies to send buffer
                for (int i = threadIdx.x, s = 0; i * sizeof(int4) < TOKEN_DIM;
                     i += kNumThreads, s++) {
                  const bool has_scale = x_scale_ptr && i < hidden_dim_scale;
                  std::byte *token_ptr =
                      send_buffer + route.position * token_stride;
                  uint4 *x_token_dst = (uint4 *)token_ptr;
                  st_global_nc_uint4(vals[s], &x_token_dst[i]);
                  if (has_scale) {
                    *(float *)(token_ptr + token_dim_fixed + i) = scales[s];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if constexpr (NODE_SIZE >= 1) {
    grid.sync(); // needed to ensure next if loop's writes are visible
    if (blockIdx.x == 0) {
      auto local_rank = rank % NODE_SIZE;
      if (threadIdx.x < NODE_SIZE) {
        auto *flag = &sync_ptrs[threadIdx.x][local_rank + NODE_SIZE];
        st_release_u32(flag, counter + 1);
      }
    }
  }
}

cudaError_t a2a_kernels::a2a_dispatch_send(
    const uint32_t num_blocks, size_t hidden_dim, size_t hidden_dim_scale,
    size_t x_elemsize, size_t x_scale_elemsize, size_t num_experts,
    size_t num_experts_per_token, size_t num_tokens, size_t max_private_tokens,
    size_t rank, size_t dp_size, size_t node_size, size_t world_size,
    int32_t *bound_m_ptr, std::byte *x_ptr, size_t x_stride, float *x_scale_ptr,
    size_t x_scale_stride_elem, size_t x_scale_stride_token, int32_t *indices,
    size_t indices_stride, float *weights, size_t weight_stride,
    uint32_t *token_offset, uint32_t *num_routed, uint32_t *expert_offsets,
    uint8_t *send_buffer,
    uint32_t *sync_counter, // used to sync across nvlink
    uint32_t **sync_ptrs, std::byte **recv_ptrs, cudaStream_t stream) {
  // number of
  constexpr size_t numThreads = 512;
  constexpr size_t numWarps = numThreads / 32;
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);

  //
  HOST_ASSERT(
      num_experts <= numThreads,
      "needed for block wide parallel scan to ensure all experts are covered");
  const size_t token_dim = ti_align(hidden_dim * x_elemsize, sizeof(int4));
  const size_t token_scale_dim =
      ti_align(hidden_dim_scale * x_scale_elemsize, sizeof(int4));
  const size_t token_stride = token_dim + token_scale_dim + 16; // padding

  HOST_ASSERT(token_stride % sizeof(int4) == 0,
              "token_stride must be divisible by sizeof(int4)");
  void *args[] = {
      const_cast<size_t *>(&token_dim),
      const_cast<size_t *>(&token_scale_dim),
      const_cast<size_t *>(&token_stride),
      &hidden_dim,
      &hidden_dim_scale,
      &num_experts,
      &num_experts_per_token,
      &max_private_tokens,
      &rank,
      &dp_size,
      &node_size,
      &world_size,
      &num_tokens,
      &bound_m_ptr,
      &x_ptr,
      &x_elemsize,
      &x_stride,
      &x_scale_ptr,
      &x_scale_elemsize,
      &x_scale_stride_elem,
      &x_scale_stride_token,
      &indices,
      &indices_stride,
      &weights,
      &weight_stride,
      &token_offset,
      &num_routed,
      &expert_offsets,
      &send_buffer,
      &sync_counter,
      &sync_ptrs,
      &recv_ptrs,

  };

  const size_t smem_size = std::max(num_experts, numWarps) * sizeof(uint32_t);
  cudaError_t status;
  nvtxRangePush("a2a_dispatch_send");

  LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, NumExpertsPerToken_t, {
    LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, HiddenDimScale_t, {
      LAUNCH_TOKEN_DIM(token_dim, TokenDim_t, {
        LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
          if (num_blocks >= num_tokens) {
            status = cudaLaunchCooperativeKernel(
                (void *)&a2a_dispatch_send_kernel<
                    true, numThreads, NODE_SIZE, TokenDim_t,
                    NumExpertsPerToken_t, HiddenDimScale_t>,
                dimGrid, dimBlock, args, smem_size, stream);
          } else {
            status = cudaLaunchCooperativeKernel(
                (void *)&a2a_dispatch_send_kernel<
                    false, numThreads, NODE_SIZE, TokenDim_t,
                    NumExpertsPerToken_t, HiddenDimScale_t>,
                dimGrid, dimBlock, args, smem_size, stream);
          }
        });
      });
    });
  });
  nvtxRangePop();
  return status;
}
