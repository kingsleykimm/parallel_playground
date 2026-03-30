/**
 * @file : kernel5_2.cuh
 * @brief: Down Proj (bf16 output) -> into combine with bf16 precision.
 **/

#pragma once

#include "common/sm90_utils.cuh"
#include "ops/thread/memory/tile/tma.cuh"
#include <cooperative_groups.h>
#include <kittens.cuh>

using namespace kittens;
namespace kernel5_2 {

// persistent kernel settings
static constexpr int SM_COUNT = 132;
static constexpr int WORLD_SIZE = 4;
// flow - the first GEMMs must have a block N (intermediate size) of 128. then
// the output block size

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
struct globals {
  static constexpr int BM = _BM, BN = _BN;
  // always uniform scales
  static constexpr int NUM_TILES = (_BM + 63) / 64;
  static constexpr int NUM_THREADS = NUM_TILES > 1 ? 384 : 256;

  static constexpr int M = _M;
  static constexpr int I = _I;
  static constexpr int H = _H;

  static constexpr bool kIsUniformScales = (BN % 128) != 0;

  // we can actually calculate this since we know how much each token will take
  // up in shared memory, with scales
  // static constexpr int TOKENS_PER_BLOCK = constexpr_min(
  //     (KERNEL_SMEM_SIZE - 1024) / (H + sizeof(kittens::semaphore)),
  //     MAIN_THREADS);
  static constexpr int TOKENS_PER_BLOCK = 16;

  static constexpr int NUM_PRODUCER_WARPS = NUM_PRODUCER_WARPS_;
  static constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPS_;
  static constexpr int NUM_STAGES = NUM_STAGES_;

  static_assert(BN == 128,
                "BN Must be divisible by 128 for fused silu-mul-quant");
  // =========== SHARED MEMORY TILES ============== //
  // These can actually have templated constants
  using a_tile =
      st_fp8e4m3<64, 128>; // this is static, since this has to fit the TK WGMMA
                           // requirement, see warpgroup.cuh
  using down_tile = st_fp8e4m3<128, 128>;
  using c_tile = st<bf16, 64, BN>;
  using sfa_tile = sv<float, 64>;
  using token_vec_tile = sv_bf<H>;
  // assume this is k-major here, with shape (N / 128, K / 128)

  // all shared vector lengths must be padded to the tile dimension of 16, even
  // if we aren't going to use allof them

  // ===== GLOBAL MEMORY LAYOUTS ===== //
  // Because we pull these layouts into host side for the factory function, and
  // use a convertor method, we want to avoid using templated layouts here
  using comm_comp_barrier_layout = gl<int, 1, 1, 1, -1>;
  using out_tokens_layout =
      pgl<gl<bf16, 1, 1, -1, -1, token_vec_tile>, WORLD_SIZE, false>;

  // these are the outputs from the swiglu (kernel 5_1)
  using expert_y_tokens_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
  using expert_y_tokens_scale_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using tokens_per_expert_layout =
      sv<int, constexpr_ti_align(NUM_EXPERTS / WORLD_SIZE, 16)>;
  using padded_expert_counts_layout = gl<int, 1, 1, -1, -1>;
  using src_token_idx_layout = gl<int, 1, 1, -1, -1>;
  using src_dev_idx_layout = gl<int, 1, 1, -1, -1>;
  using src_slot_idx_layout = gl<int, 1, 1, -1, -1>;

  // ================ GLOBAL MEMORY ============= //

  using a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
  using down_layout = gl<fp8e4m3, 1, 1, -1, -1, down_tile>;
  using c_layout = gl<bf16, 1, 1, -1, -1, c_tile>;

  // only sfa_tile since this is the only one we do TMA load on
  using scale_a_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using scale_down_layout = gl<float, 1, 1, -1, -1>;
  using weights_layout = gl<float, 1, 1, -1, -1>;

  template <typename T = float> using accum_tile = rt<T, 16, c_tile::cols>;

  out_tokens_layout out_tokens;
  expert_y_tokens_layout expert_y_tokens;
  expert_y_tokens_scale_layout expert_y_tokens_scale;
  comm_comp_barrier_layout comm_comp_barrier;

  down_layout down;
  scale_down_layout scale_down;
  c_layout C;
  weights_layout weights;
  padded_expert_counts_layout padded_expert_counts;
  // important: this maps from the destination expert + ith token (out of
  // NUM_EXPERTS_PER_TOKEN) to the source token idx
  src_token_idx_layout src_token_idx;
  src_dev_idx_layout src_dev_idx;
  src_slot_idx_layout src_slot_idx;

  int num_tokens;
  int *num_recv_tokens;
  int dp_rank;
  int rank;
  int dp_size;
  int cur_dp_group;
  int num_dp_groups;

  const int num_comm_sms;
  const int num_comp_sms;
  // these are conditional on the number of warps
  // remember that sfa is loaded in MN-major
  struct pipeline_inputs {
    a_tile a[BM > 64 ? 2 : 1];
    down_tile down;
    sfa_tile sfa[BM > 64 ? 2 : 1];
  };

  struct pipeline_outputs {
    c_tile c[BM > 64 ? 2 : 1];
  };

  struct common_state {
    int2 coord;
    int Rblocks;
  };

  struct consumer_state {
    accum_tile<float> down_accum, per_k_down_accum;
  };
};
template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
__device__ inline void
compute_path(const globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                           NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                           NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M> &G,
             const int sm_idx) {

  using GLOBALS = globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                          NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                          NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>;
  using pipeline_inputs = typename GLOBALS::pipeline_inputs;
  using pipeline_outputs = typename GLOBALS::pipeline_outputs;
  using common_state = typename GLOBALS::common_state;
  using consumer_state = typename GLOBALS::consumer_state;
  using tokens_per_expert_layout = typename GLOBALS::tokens_per_expert_layout;
  constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / WORLD_SIZE;
  const int lane_id = laneid();
  const int warp_id = warpid();

  extern __shared__ int __shm[];
  tma_swizzle_allocator allocator((int *)&__shm[0]);

  pipeline_inputs(&inputs)[NUM_STAGES_] =
      allocator.allocate<pipeline_inputs, NUM_STAGES_>();
  constexpr int FINISH_BLOCK_OFFSET =
      KERNEL_SMEM_SIZE - sizeof(pipeline_outputs); // single SM persistent
  static_assert(FINISH_BLOCK_OFFSET >= 0, "not enough shared memory");
  constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024;
  // round down number of stages if necessary
  constexpr int SAFE_STAGES_BETWEEN_BLOCKS =
      (NON_FINISH_BLOCK_SPACE / sizeof(pipeline_inputs));
  // will integrate finish_finished signalling later for pingpong schedulign
  static_assert(SAFE_STAGES_BETWEEN_BLOCKS >= NUM_STAGES_);

  pipeline_outputs &outputs = allocator.allocate<pipeline_outputs>();

  tokens_per_expert_layout &tokens_per_expert =
      allocator.allocate<tokens_per_expert_layout>();
  __shared__ semaphore inputs_arrived[NUM_STAGES_];
  __shared__ semaphore inputs_finished[NUM_STAGES_];
  uint32_t semaphore_bitfield = 0xFFFF0000;

  for (int expert = threadIdx.x; expert < NUM_EXPERTS_PER_DEV;
       expert += blockDim.x) {
    tokens_per_expert.data[expert] = G.padded_expert_counts[{expert}];
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < GLOBALS::NUM_STAGES; stage++) {
      init_semaphore(inputs_arrived[stage], 1, 0);
      init_semaphore(inputs_finished[stage], NUM_CONSUMER_WARPS_, 0);
    }
  }
  everyone::sync(1);

  constexpr int NUM_C_BLOCKS = _H / 128;
  constexpr int num_iters = _I / 128;

  common_state common;
  int input_ring = 0;
  if (warpid() >= NUM_CONSUMER_WARPS_) {
    warpgroup::decrease_registers<40>();
    using producers = group<NUM_PRODUCER_WARPS_>;
// load in the tokens for a certain block
#pragma unroll 8
    for (int expert = 0, cumsum_tokens = 0, task_id = sm_idx;
         expert < NUM_EXPERTS_PER_DEV; expert++) {
      int row_block_start = cumsum_tokens / _BM;
      cumsum_tokens += tokens_per_expert[expert];
      int row_block_end = (cumsum_tokens + _BM - 1) / _BM;
      int num_blocks = (row_block_end - row_block_start) * NUM_C_BLOCKS;
      int r_blocks = row_block_end - row_block_start;
      int super_rows = (r_blocks / SUPER_M) * SUPER_M; // round down
      int remaining_rows = r_blocks - super_rows;
      int super_repeat = SUPER_M * NUM_C_BLOCKS;

      for (; task_id < num_blocks; task_id += G.num_comp_sms) {
        // this is persistent sm scheduling here now
        if (task_id < super_rows * NUM_C_BLOCKS) {
          common.coord.x =
              (task_id / super_repeat) * SUPER_M + task_id % SUPER_M;
          common.coord.y = (task_id % super_repeat) /
                           SUPER_M; // super_repeat includes all the c blocks
        } else {
          int remainder = task_id - super_rows * NUM_C_BLOCKS;
          common.coord.x = super_rows + (remainder % remaining_rows);
          common.coord.y = remainder / remaining_rows;
        }
        common.coord.x += row_block_start;
        if (warpgroup::laneid() == 0) {

          for (int iter = 0; iter < num_iters; iter++) {
            wait(inputs_finished[input_ring],
                 get_phasebit<1>(semaphore_bitfield, input_ring));
            update_phasebit<1>(semaphore_bitfield, input_ring);

            tma::expect(inputs_arrived[input_ring], inputs[input_ring]);

#pragma unroll
            for (int wg_id = 0; wg_id < GLOBALS::NUM_TILES; wg_id++) {
              tma::load_async(
                  inputs[input_ring].a[wg_id], G.expert_y_tokens,
                  coord<ducks::default_type>{common.coord.x * _BM + wg_id * 64,
                                             iter * 128},
                  inputs_arrived[input_ring]);
              tma::load_async(inputs[input_ring].sfa[wg_id],
                              G.expert_y_tokens_scale,
                              coord<ducks::default_type>{
                                  iter, (common.coord.x + wg_id) * 64},
                              inputs_arrived[input_ring]);
            }
            tma::load_async(inputs[input_ring].down, G.down,
                            coord<ducks::default_type>{_H, common.coord.y * _BN,
                                                       iter * 128},
                            inputs_arrived[input_ring]);
            input_ring = ring_advance<NUM_STAGES_>(input_ring);
          }
        }
      }
      task_id -= num_blocks;
    }
  } else {
    using producers = group<NUM_CONSUMER_WARPS_ / 2>;
    consumer_state consumer;
    warpgroup::increase_registers<232>();

    const int wg_id = warpgroup::groupid();
    static constexpr int shape_k_scales = _H / 128;
    static constexpr int shape_n_sfb = _I / 128;
    static constexpr uint32_t stride_n_sfb = shape_k_scales;
    static constexpr uint32_t stride_k_sfb = 1;
    int num_former_iters, num_full_iters;
#pragma unroll 8
    for (int expert = 0, cumsum_tokens = 0, task_id = sm_idx;
         expert < NUM_EXPERTS_PER_DEV; expert++) {
      int row_block_start = cumsum_tokens / (_BM);
      cumsum_tokens += tokens_per_expert[expert];
      int row_block_end = (cumsum_tokens + _BM - 1) / _BM;
      int num_blocks = (row_block_end - row_block_start) * NUM_C_BLOCKS;
      int r_blocks = row_block_end - row_block_start;
      int super_rows = (r_blocks / SUPER_M) * SUPER_M; // round down
      int remaining_rows = r_blocks - super_rows;
      int super_repeat = SUPER_M * NUM_C_BLOCKS;

      for (; task_id < num_blocks; task_id += G.num_comp_sms) {
        warp::zero(consumer.down_accum);
        if (task_id < super_rows * NUM_C_BLOCKS) {
          common.coord.x =
              (task_id / super_repeat) * SUPER_M + task_id % SUPER_M;
          common.coord.y = (task_id % super_repeat) /
                           SUPER_M; // super_repeat includes all the cblocks
        } else {
          int remainder = task_id - super_rows * NUM_C_BLOCKS;
          common.coord.x = super_rows + (remainder % remaining_rows);
          common.coord.y = remainder / remaining_rows;
        }

        common.coord.x += row_block_start;
        // for the current task id, we just need to see whether the current
        // assigned row is ready on the barrier

        // get scale meta data
        const auto group_offset = shape_k_scales * shape_n_sfb * expert;

#pragma unroll 8
        for (int iter = 0; iter < num_iters; iter++) {
          const auto scale_b_offset = group_offset + iter * stride_k_sfb +
                                      common.coord.y * stride_n_sfb;
          wait(inputs_arrived[input_ring],
               get_phasebit<0>(semaphore_bitfield, input_ring));
          update_phasebit<0>(semaphore_bitfield, input_ring);
          warp::zero(consumer.per_k_down_accum);
          if constexpr (GLOBALS::kIsUniformScales) {
            num_former_iters = num_full_iters = _BN / 8;
          } else {
            num_full_iters = min(_H - common.coord.y * _BN, _BN) / 8;
            num_former_iters = min(_BN, 128 - (common.coord.y * _BN) % 128) / 8;
          }
          float *local_down_sfb = G.scale_down.raw_ptr + scale_b_offset;
          warpgroup::mma_ABt(consumer.per_k_down_accum,
                             inputs[input_ring].a[wg_id],
                             inputs[input_ring].down);

          float down_scale_0, down_scale_1;
          move<float>::ldg(down_scale_0, local_down_sfb);
          if constexpr (!GLOBALS::kIsUniformScales) {
            if (num_full_iters > num_former_iters) {
              move<float>::ldg(down_scale_1, local_down_sfb + stride_n_sfb);
            }
          }
          col_vec<decltype(consumer.per_k_down_accum)> scale_a_rv;
          warpgroup::load(scale_a_rv, inputs[input_ring].sfa[wg_id]);
          warpgroup::mma_async_wait();
          warp::mul_row(consumer.per_k_down_accum, consumer.per_k_down_accum,
                        scale_a_rv);
          if constexpr (GLOBALS::kIsUniformScales) {
            consumer.per_k_down_accum *= down_scale_0;
          } else {
            row_vec<decltype(consumer.per_k_down_accum)> scale_b;
#pragma unroll
            for (int i = 0; i < _BN / 8;
                 i++) { // the thunderkittens rt naturally splits the length of
                        // the rv into subtiles of length 16
              int tile_idx = i / 2;
              int inner_dim_idx = i % 2;
              if (i < num_former_iters) {
                scale_b.data[tile_idx][inner_dim_idx] =
                    make_float2(down_scale_0, down_scale_1);
              } else {
                scale_b.data[tile_idx][inner_dim_idx] =
                    make_float2(down_scale_1, down_scale_1);
              }
            }
            warp::mul_col(consumer.per_k_down_accum, consumer.per_k_down_accum,
                          scale_b);
          }
          consumer.down_accum += consumer.per_k_down_accum;
          if (lane_id == 0) {
            arrive(inputs_finished[input_ring]);
          }
          input_ring = ring_advance<NUM_STAGES_>(input_ring);
        }

        // epilogue
        warpgroup::store(outputs.c[wg_id], consumer.down_accum);
        warpgroup::sync(wg_id + 3);
        if (warpgroup::laneid() == 0) {
          tma::store_async(
              G.C, outputs.c[wg_id],
              coord<ducks::default_type>(common.coord.x * _BM + wg_id * 64,
                                         common.coord.y * _BN));
          tma::store_async_wait();
          asm volatile("red.release.gpu.global.s32 [%0], %1" ::"l"(
                           &G.comm_comp_barrier[{common.coord.x}]),
                       "r"(64)
                       : "memory");
        }
      }
    }
  }
}

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
__device__ inline void
combine_path(const globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                           NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                           NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M> &G,
             const int sm_idx) {

  using GLOBALS = globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                          NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                          NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>;
  constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / WORLD_SIZE;
  extern __shared__ int __shm[];
  tma_allocator st_allocator((int *)&__shm[0]);

  typename GLOBALS::token_vec_tile(&send_token)[GLOBALS::TOKENS_PER_BLOCK] =
      st_allocator.allocate<typename GLOBALS::token_vec_tile,
                            GLOBALS::TOKENS_PER_BLOCK>();
  semaphore(&token_sent)[GLOBALS::TOKENS_PER_BLOCK] =
      st_allocator.allocate<semaphore, GLOBALS::TOKENS_PER_BLOCK>();
  if (threadIdx.x < GLOBALS::TOKENS_PER_BLOCK) {
    int token = sm_idx * GLOBALS::TOKENS_PER_BLOCK + threadIdx.x;
    init_semaphore(token_sent[threadIdx.x], 0, 1);
    int phase = 0;
    int src_token_idx, src_dev_idx, src_slot_idx, dst_dp_group;
    int dst_rank;
    int num_recv_tokens = *G.num_recv_tokens;
    while (token < num_recv_tokens) {

      src_token_idx = G.src_token_idx[token];
      src_dev_idx = G.src_dev_idx[token];
      dst_dp_group = src_dev_idx / G.dp_size;
      src_slot_idx = G.src_slot_idx[token];
      // we actually need to send it to every single dp group that owns the
      // tokens

      if (src_token_idx >= 0 && src_dev_idx >= 0) {
        tma::expect(token_sent[threadIdx.x], GLOBALS::token_vec_tile);
        tma::load_async(send_token[threadIdx.x], G.C[token], {token, 0},
                        token_sent[threadIdx.x]);
        wait(token_sent[threadIdx.x], phase);
        phase ^= 1;
        // for the weights loading - we need the src_token_idx, and then the
        // intra expert idx
        float weight = G.weights[{src_token_idx, src_slot_idx}];
        send_token[threadIdx.x] *= weight;
        for (int intra_group_rank = 0; intra_group_rank < G.dp_size;
             intra_group_rank++) {
          dst_rank = dst_dp_group * G.dp_size + intra_group_rank;
          tma::store_add_async(G.out_tokens[dst_rank], send_token[threadIdx.x],
                               {src_token_idx, 0});
        }
        tma::store_async_wait();
      }

      token += sm_idx * GLOBALS::TOKENS_PER_BLOCK;
    }
  }
}

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
__global__ void __launch_bounds__(_BM > 64 ? 384 : 256, 1) global_kernel(
    const globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                  NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                  NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>
        G) {
  if (blockIdx.x < G.num_comp_sms) {
    compute_path(G, blockIdx.x);
  } else {
    combine_path(G, blockIdx.x - G.num_comp_sms);
  }
}
} // namespace kernel5_2