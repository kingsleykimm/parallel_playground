/**
 * @file : kernel5_2.cuh
 * @brief: Down Proj (bf16 output) -> into combine with bf16 precision.
 **/

#pragma once

#include "common/sm90_utils.cuh"
#include <cooperative_groups.h>
#include <kittens.cuh>

using namespace kittens;
namespace kernel5_2 {

// persistent kernel settings
static constexpr int SM_COUNT = 132;
static constexpr int MAIN_THREADS = 384; // 1 P_WG + 2 C_WG
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

  static constexpr int M = _M;
  static constexpr int I = _I;
  static constexpr int H = _H;

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

  using gate_tile = st_fp8e4m3<128, 128>;
  using up_tile = st_fp8e4m3<128, 128>;
  using c_tile = st<fp8e4m3, 64, BN>;
  using sfa_tile = sv<float, 64>;

  using num_routed_layout = sv<int, constexpr_ti_align(NUM_EXPERTS, 16)>;
  using tokens_per_expert_layout =
      sv<int, constexpr_ti_align(NUM_EXPERTS / WORLD_SIZE, 16)>;
  using expert_sums_layout = sv<int, constexpr_ti_align(MAIN_THREADS / 32, 16)>;
  using expert_offsets_layout =
      sv<int, constexpr_ti_align(NUM_EXPERTS / WORLD_SIZE, 16)>;
  using token_vec_tile = sv_fp8e4m3<H>; // char is same size as fp8e4m3 but
                                        // avoids TK's sv fp8 static_assert
  // assume this is k-major here, with shape (N / 128, K / 128)

  // all shared vector lengths must be padded to the tile dimension of 16, even
  // if we aren't going to use allof them

  // ===== GLOBAL MEMORY LAYOUTS ===== //
  // Because we pull these layouts into host side for the factory function, and
  // use a convertor method, we want to avoid using templated layouts here
  using padded_expert_counts_layout = gl<int, 1, 1, -1, -1>;
  using global_num_routed_layout =
      pgl<gl<int, 1, 1, -1, -1>, WORLD_SIZE, false>;
  using expert_to_token_map_layout =
      pgl<gl<int, 1, 1, -1, -1>, WORLD_SIZE, false>;

  using comm_comp_barrier_layout = gl<int, 1, 1, 1, -1>;

  using in_tokens_layout =
      pgl<gl<fp8e4m3, 1, 1, -1, -1, token_vec_tile>, WORLD_SIZE, false>;
  using in_tokens_scales_layout =
      pgl<gl<float, 1, 1, -1, -1>, WORLD_SIZE, false>;
  using expert_x_tokens_layout =
      gl<fp8e4m3, 1, 1, -1, -1, a_tile, token_vec_tile>;
  using expert_x_tokens_scale_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using src_token_idx_layout = gl<int, 1, 1, -1, -1>;
  using src_dev_idx_layout = gl<int, 1, 1, -1, -1>;
  using barrier_layout = pgl<gl<int, -1, -1, -1, 1>, WORLD_SIZE, false>;

  // ================ GLOBAL MEMORY ============= //

  using a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
  using gate_layout = gl<fp8e4m3, 1, 1, -1, -1, gate_tile>;
  using up_layout = gl<fp8e4m3, 1, 1, -1, -1, up_tile>;
  using c_layout = gl<fp8e4m3, 1, 1, -1, -1, c_tile>;

  // only sfa_tile since this is the only one we do TMA load on
  using scale_a_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using scale_gate_layout = gl<float, 1, 1, -1, -1>;
  using scale_up_layout = gl<float, 1, 1, -1, -1>;
  using out_scales_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using indices_layout = gl<int, 1, 1, -1, -1>;
  using gl_src_group_offset_layout = gl<int, 1, 1, -1, -1>;

  template <typename T = float> using accum_tile = rt<T, 16, c_tile::cols>;

  in_tokens_layout in_tokens;
  in_tokens_scales_layout in_tokens_scales;
  expert_x_tokens_layout expert_x_tokens;
  expert_x_tokens_scale_layout expert_x_tokens_scale;

  comm_comp_barrier_layout comm_comp_barrier;

  gate_layout gate;
  up_layout up;
  c_layout C;
  scale_gate_layout scale_gate;
  scale_up_layout scale_up;
  out_scales_layout out_scales;
  indices_layout indices;
  global_num_routed_layout global_num_routed;
  // important: this maps from the destination expert + ith token (out of
  // NUM_EXPERTS_PER_TOKEN) to the source token idx
  expert_to_token_map_layout expert_to_token_map;
  gl_src_group_offset_layout gl_src_group_offset;
  padded_expert_counts_layout padded_expert_counts;
  src_token_idx_layout src_token_idx;
  src_dev_idx_layout src_dev_idx;
  barrier_layout barrier;

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
    gate_tile gate;
    up_tile up;
    sfa_tile sfa[BM > 64 ? 2 : 1];
  };

  struct pipeline_outputs {
    c_tile c[BM > 64 ? 2 : 1];
    sfa_tile out_scales[BM > 64 ? 2 : 1];
  };

  struct common_state {
    int2 coord;
    int Rblocks;
    int interC_blocks;
    int finalC_Cblocks;
    int cur_group_idx = 0;
    int current_m_cumsum = 0;
    int m_block_idx;
    bool computation_valid = false;
  };

  struct consumer_state {
    accum_tile<float> gate_accum, up_accum;
    accum_tile<float> per_k_gate_accum, per_k_up_accum;
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
        // if (task_id < super_rows * NUM_C_BLOCKS) {
        //   common.coord.x =
        //       (task_id / super_repeat) * SUPER_M + task_id % SUPER_M;
        //   common.coord.y = (task_id % super_repeat) /
        //                    SUPER_M; // super_repeat includes all the c blocks
        // } else {
        //   int remainder = task_id - super_rows * NUM_C_BLOCKS;
        //   common.coord.x = super_rows + (remainder % remaining_rows);
        //   common.coord.y = remainder / remaining_rows;
        // }

        common.coord.x = task_id / NUM_C_BLOCKS + row_block_start;
        common.coord.y = task_id % NUM_C_BLOCKS;
        if (warpgroup::laneid() == 0) {

          for (int iter = 0; iter < num_iters; iter++) {
            wait(inputs_finished[input_ring],
                 get_phasebit<1>(semaphore_bitfield, input_ring));
            update_phasebit<1>(semaphore_bitfield, input_ring);

            tma::expect(inputs_arrived[input_ring], inputs[input_ring]);

#pragma unroll
            for (int wg_id = 0; wg_id < GLOBALS::NUM_TILES; wg_id++) {
              tma::load_async(
                  inputs[input_ring].a[wg_id], G.expert_x_tokens,
                  coord<ducks::default_type>{common.coord.x * _BM + wg_id * 64,
                                             iter * 128},
                  inputs_arrived[input_ring]);
              tma::load_async(inputs[input_ring].sfa[wg_id],
                              G.expert_x_tokens_scale,
                              coord<ducks::default_type>{
                                  iter, (common.coord.x + wg_id) * 64},
                              inputs_arrived[input_ring]);
            }
            tma::load_async(inputs[input_ring].up, G.up,
                            coord<ducks::default_type>{
                                _H * expert + common.coord.y * _BN, iter},
                            inputs_arrived[input_ring]);
            tma::load_async(inputs[input_ring].gate, G.gate,
                            coord<ducks::default_type>{
                                _H * expert + common.coord.y * _BN, iter},
                            inputs_arrived[input_ring]);
            input_ring = ring_advance<NUM_STAGES_>(input_ring);
          }
        }
      }
      task_id -= num_blocks;
    }
  }
}
} // namespace kernel5_2
