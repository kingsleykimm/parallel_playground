/**
 * @file
 * @brief: Dispatch + GEMM, using padded_index or expert offsets for contiguous
 * or masked
 **/
#pragma once

#include "common/base_types.cuh"
#include "common/common.cuh"
#include "common/sm90_utils.cuh"
#include "common/util.cuh"
#include "kittens.cuh"
#include "ops/thread/util/tma.cuh"
#include <cooperative_groups.h>

using namespace kittens;
namespace kernel5_1 {

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
  using c_tile = st<fp8e4m3, 64, 128>;
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
  // do some checking first

  // TODO: stages heuristics needs to be changed here now
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

  constexpr int NUM_C_BLOCKS = _I / 128;
  constexpr int num_iters = _H / 128;

  common_state common;
  int input_ring = 0;
  // producer path
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

        int counter;
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                     : "=r"(counter)
                     : "l"(&G.comm_comp_barrier[{common.coord.x}])
                     : "memory");
        while (counter != _BM) {
          __nanosleep(16);
          asm volatile("{ld.acquire.gpu.global.s32 %0, [%1];}"
                       : "=r"(counter)
                       : "l"(&G.comm_comp_barrier[{common.coord.x}])
                       : "memory");
        }
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
                                  iter, (common.coord.x * _BM + wg_id * 64)},
                              inputs_arrived[input_ring]);
            }
            tma::load_async(inputs[input_ring].up, G.up,
                            coord<ducks::default_type>{
                                _I * expert + common.coord.y * _BN, iter * 128},
                            inputs_arrived[input_ring]);
            tma::load_async(inputs[input_ring].gate, G.gate,
                            coord<ducks::default_type>{
                                _I * expert + common.coord.y * _BN, iter * 128},
                            inputs_arrived[input_ring]);
            input_ring = ring_advance<NUM_STAGES_>(input_ring);
          }
        }
      }
      task_id -= num_blocks;
    }
  }

  else {
    consumer_state state;
    warpgroup::increase_registers<232>();
    using consumer = group<NUM_CONSUMER_WARPS_>;
    const int wg_id = warpgroup::groupid();
    static constexpr int shape_k_scales = _H / 128;
    static constexpr int shape_n_sfb = _I / 128;
    static constexpr uint32_t stride_n_sfb = shape_k_scales;
    static constexpr uint32_t stride_k_sfb = 1;
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
        warp::zero(state.up_accum);
        warp::zero(state.gate_accum);
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

          warp::zero(state.per_k_gate_accum);
          warp::zero(state.per_k_up_accum);
          float *local_gate_sfb = G.scale_gate.raw_ptr + scale_b_offset;
          float *local_up_sfb = G.scale_up.raw_ptr + scale_b_offset;
          warpgroup::mma_ABt(state.per_k_gate_accum,
                             inputs[input_ring].a[warpgroup::groupid()],
                             inputs[input_ring].gate);
          float gate_scale_0, up_scale_0;
          move<float>::ldg(gate_scale_0, local_gate_sfb);
          typename decltype(state.per_k_gate_accum)::col_vec scale_a_rv;
          warpgroup::load(scale_a_rv,
                          inputs[input_ring].sfa[warpgroup::groupid()]);

          // === Phase 3: wait for gate WGMMA ===
          warpgroup::mma_async_wait();
          warpgroup::mma_ABt(state.per_k_up_accum,
                             inputs[input_ring].a[warpgroup::groupid()],
                             inputs[input_ring].up);
          warp::mul_row(state.per_k_gate_accum, state.per_k_gate_accum,
                        scale_a_rv);
          state.per_k_gate_accum *= gate_scale_0;
          state.gate_accum += state.per_k_gate_accum;

          move<float>::ldg(up_scale_0, local_up_sfb);

          warpgroup::mma_async_wait();
          warp::mul_row(state.per_k_up_accum, state.per_k_up_accum, scale_a_rv);
          state.per_k_up_accum *= up_scale_0;
          state.up_accum += state.per_k_up_accum;
          if (lane_id == 0) {
            arrive(inputs_finished[input_ring]);
          }
          input_ring = ring_advance<NUM_STAGES_>(input_ring);
        }
        // epilogue
        state.up_accum = (state.up_accum * state.gate_accum) /
                         (warp::exp(state.gate_accum * -1.0f) + 1.0f);
        warp::abs(
            state.gate_accum,
            state.up_accum); // we're reusing the gate_accum to save registers
        col_vec<rt<float, 16, _BN>> row_amaxes_rv;
        warp::row_reduce<base_ops::max, decltype(row_amaxes_rv),
                         decltype(state.gate_accum), true>(
            row_amaxes_rv, state.gate_accum, row_amaxes_rv);
        row_amaxes_rv /= 448.0f; // scale factors
        // need to quantize to FP8, which is state.up_accum / row_amaxes_rv
        // across rows
        warp::div_row(state.up_accum, state.up_accum, row_amaxes_rv);

        rt<fp8e4m3, 16, _BN> fp8_output_tile;
        warp::copy(fp8_output_tile, state.up_accum);
        tma::store_async_read_wait(); // wait for previous stores from shared
                                      // memory to finish
        warpgroup::store(outputs.c[warpgroup::groupid()], fp8_output_tile);
        warpgroup::store(outputs.out_scales[warpgroup::groupid()],
                         row_amaxes_rv);

        warpgroup::sync(wg_id + 3);
        if (warpgroup::laneid() == 0) {
          tma::store_async(
              G.C, outputs.c[warpgroup::groupid()],
              coord<ducks::default_type>{common.coord.x * _BM + wg_id * 64,
                                         common.coord.y * _BN});
          tma::store_async(
              G.out_scales, outputs.out_scales[warpgroup::groupid()],
              coord<ducks::default_type>{common.coord.y,
                                         common.coord.x * _BM + wg_id * 64});
        }
      }
      task_id -= num_blocks;
    }
  }
}

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 8>
__device__ inline void compute_routing_info(
    const globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                  NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                  NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M> &G) {

  using GLOBALS = globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                          NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                          NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>;
  // in the a2a, we need to communicate between the dp groups as well, so we
  // first count the number of tokens that will be sent to each dp group -
  // communicate between same dp ranks to find out the

  constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / WORLD_SIZE;
  constexpr int EXPERT_PADDING = 128;
  DEVICE_ASSERT(NUM_EXPERTS_PER_DEV <= MAIN_THREADS);

  extern __shared__ int __shm[];
  shared_allocator<16> allocator((int *)&__shm[0]);

  using num_routed_layout = typename GLOBALS::num_routed_layout;
  using tokens_per_expert_layout = typename GLOBALS::tokens_per_expert_layout;
  using expert_sums_layout = typename GLOBALS::expert_sums_layout;
  using expert_offsets_layout = typename GLOBALS::expert_offsets_layout;
  num_routed_layout &num_routed = allocator.allocate<num_routed_layout>();
  tokens_per_expert_layout &tokens_per_expert =
      allocator.allocate<tokens_per_expert_layout>();
  expert_sums_layout &expert_sums = allocator.allocate<expert_sums_layout>();
  expert_offsets_layout &expert_offsets =
      allocator.allocate<expert_offsets_layout>();
  int *src_group_offset = (int *)allocator.ptr;
  // check that we're not running out of dynamic shared memory
  DEVICE_ASSERT((allocator.ptr - &__shm[0]) * sizeof(int) +
                    NUM_EXPERTS_PER_DEV * G.num_dp_groups * sizeof(int) <=
                KERNEL_SMEM_SIZE);
  int lane_id = kittens::laneid();
  int warp_id = kittens::warpid();

  for (int i = threadIdx.x; i < NUM_EXPERTS; i += blockDim.x) {
    num_routed.data[i] = 0;
  }
  everyone::sync(1);

  // for the current local dp group
  for (int i = threadIdx.x; i < G.num_tokens * EXPERTS_PER_TOKEN;
       i += blockDim.x) {
    int token = i / EXPERTS_PER_TOKEN;
    int slot = i % EXPERTS_PER_TOKEN;
    int routed_expert = G.indices[{token, slot}];
    int ith_routed_token = atomicAdd(&num_routed.data[routed_expert], 1);
    G.expert_to_token_map[G.rank][{routed_expert, ith_routed_token}] = token;

    // expert -> token map for the rounting map
  }
  everyone::sync(1);
  // we need the thread fence to ensure the expert to token maps will be
  // written to system scope as well

  for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += blockDim.x) {
    for (int dp_group = 0; dp_group < G.num_dp_groups; dp_group++) {
      int cur_rank = dp_group * G.dp_size + G.dp_rank;
      G.global_num_routed[cur_rank][{G.cur_dp_group, expert}] =
          num_routed.data[expert];
    }
  }

  __syncthreads();
  __threadfence_system();

  if (threadIdx.x < G.num_dp_groups) {
    int cur_rank = threadIdx.x * G.dp_size + G.dp_rank;
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" ::"l"(
                     &G.barrier[cur_rank][0]),
                 "r"(1)
                 : "memory");
  }
  if (threadIdx.x == 0) {
    node_sync::wait(G.barrier, {0}, G.rank, G.num_dp_groups);
  }
  everyone::sync(1);
  // calculate expert padding
  // iterate across the num routed, and then find the local experts

  int first_local_expert = G.rank * NUM_EXPERTS_PER_DEV;
  int last_local_expert = min(first_local_expert + NUM_EXPERTS_PER_DEV,
                              NUM_EXPERTS); // not inclusive
  everyone::sync(1);

  // what do we need now? we need to set up the padded indices from each
  // source dp group, and also set up the returning source ranks let's first
  // collect the number of incoming total tokens, as well as the offsets, so
  // that we can pad them and the initialize the padded indices for ALL tokens

  // we just need to get src token idx and src group idx for every token that
  // will come into our dp group. so let's iterate through through the num
  // routed + dp groups

  for (int expert = threadIdx.x; expert < NUM_EXPERTS_PER_DEV;
       expert += blockDim.x) {
    int src_rank_offset = 0;
    // this linearity is IMPORTANT.
    for (int dp_group = 0; dp_group < G.num_dp_groups; dp_group++) {

      int num_tokens_from_group =
          G.global_num_routed[G.rank][{dp_group, first_local_expert + expert}];

      // so we know this is the number of experts from a specific dp group.
      // we're going to use these to get the total number of tokens
      // this is within an expert on the CURRENT rank, it tells us the number
      // of tokens that will come
      src_rank_offset += num_tokens_from_group;
      src_group_offset[expert * G.num_dp_groups + dp_group] = src_rank_offset;
    }
    // tokens_per_expert[{expert}] = src_rank_offset;
    // we can actually just pad this and put it in, since src_group_offset
    // will hold the real number of tokens
    tokens_per_expert.data[expert] = ((src_rank_offset + 127) / 128) * 128;
    G.padded_expert_counts[{expert}] = tokens_per_expert[expert];
  }

  everyone::sync(1);

  // warp cumsum to get absolute padded expert offsets

  // each warp will take its warp_id * 32, (warp_id + 1) * 32 range to cumsum
  // first
  // we need to handle the case where num_experts_per_dev < 32
  int expert_count =
      threadIdx.x < NUM_EXPERTS_PER_DEV ? tokens_per_expert[threadIdx.x] : 0;
  if constexpr (NUM_EXPERTS_PER_DEV <= 32) {
    if (warp_id == 0) {
      for (int offset = 1; offset < 32; offset <<= 1) {
        int below_expert_count =
            __shfl_up_sync(uint32_t(-1), expert_count, offset);
        if (lane_id >= offset) {
          expert_count += below_expert_count;
        }
      }
      if (lane_id < NUM_EXPERTS_PER_DEV) {
        expert_offsets.data[lane_id] = expert_count;
      }
      if (lane_id == NUM_EXPERTS_PER_DEV - 1) {
        *G.num_recv_tokens = expert_count;
      }
    }
  } else {
    if (warp_id < NUM_EXPERTS_PER_DEV / 32) {
      for (int offset = 1; offset < 32; offset <<= 1) {
        int below_expert_count =
            __shfl_up_sync(uint32_t(-1), expert_count, offset);
        if (lane_id >= offset) {
          expert_count += below_expert_count;
        }
      }
      if (lane_id == 31) {
        expert_sums.data[warp_id] = expert_count;
      }
      // expert_count is cumsum per warp now
    }

    everyone::sync(1);
    if (warp_id == 0) {
      int expert_sum =
          lane_id < NUM_EXPERTS_PER_DEV / 32 ? expert_sums[lane_id] : 0;
      for (int offset = 1; offset < 32; offset <<= 1) {
        int below_expert_count =
            __shfl_up_sync(uint32_t(-1), expert_sum, offset);
        if (lane_id >= offset) {
          expert_sum += below_expert_count;
        }
      }

      if (lane_id < NUM_EXPERTS_PER_DEV / 32) {
        expert_sums.data[lane_id] =
            expert_sum; // this is now the absolute expert offset
      }
    }
    everyone::sync(1);
    if (threadIdx.x < NUM_EXPERTS_PER_DEV) {
      expert_offsets.data[threadIdx.x] =
          ((warp_id > 0) ? expert_sums[warp_id - 1] : 0) + expert_count;
    }
    if (threadIdx.x == NUM_EXPERTS_PER_DEV - 1) {
      *G.num_recv_tokens = expert_offsets[threadIdx.x];
    }
  }
  everyone::sync(1);

  // now we need to set up the src dev idx and src token idx
  int num_tokens = *G.num_recv_tokens;

  // now in the padded offsets, for each expert, we find the src_rank_offset
  // and iterate through that for the number of tokens, and map this to the
  // padded expert offset index

  // quick debug
  // if (G.rank == 0 && threadIdx.x < G.num_dp_groups * NUM_EXPERTS_PER_DEV) {
  //   printf("src_group_offset[%d] = %d\n", threadIdx.x,
  //          src_group_offset[threadIdx.x]);
  // }
  // everyone::sync(1);

  int token = threadIdx.x;
  while (token < num_tokens) {
    // expert_offsets[expert] describes the exclusive end index for the token
    int local_expert = 0;
    while (token >= expert_offsets[local_expert] &&
           local_expert < NUM_EXPERTS_PER_DEV) {
      local_expert++;
    }
    if (local_expert >= NUM_EXPERTS_PER_DEV) {
      break;
    }
    int src_group = 0;
    int expert_start = local_expert > 0 ? expert_offsets[local_expert - 1] : 0;
    int intra_expert_token = token - expert_start;
    int src_rank_token_offset = intra_expert_token; // start at t

    while (intra_expert_token >=
               src_group_offset[local_expert * G.num_dp_groups + src_group] &&
           src_group < G.num_dp_groups) {
      src_rank_token_offset =
          intra_expert_token -
          src_group_offset[local_expert * G.num_dp_groups + src_group];
      src_group++;
    }

    // now, intra_expert_token here is either a padded token, or a real one
    int src_dev_idx =
        src_group < G.num_dp_groups ? src_group * G.dp_size + G.dp_rank : -1;
    int src_token_idx =
        src_group < G.num_dp_groups
            ? G.expert_to_token_map[src_dev_idx][{
                  first_local_expert + local_expert, src_rank_token_offset}]
            : -1;
    G.src_dev_idx[{token}] = src_dev_idx;
    G.src_token_idx[{token}] = src_token_idx;
    token += blockDim.x;
  }
}

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
__device__ inline void
comm_path(const globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                        NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                        NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M> &G,
          const int sm_idx) {
  using GLOBALS = globals<_M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_,
                          NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                          NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>;
  int num_recv_tokens = *G.num_recv_tokens;
  extern __shared__ int __shm[];
  // we're also going to rewrite the shared memory with a new allocator now
  tma_allocator tl((int *)&__shm[0]);
  typename GLOBALS::token_vec_tile(&token)[GLOBALS::TOKENS_PER_BLOCK] =
      tl.allocate<typename GLOBALS::token_vec_tile,
                  GLOBALS::TOKENS_PER_BLOCK>();
  semaphore(&token_arrived)[GLOBALS::TOKENS_PER_BLOCK] =
      tl.allocate<semaphore, GLOBALS::TOKENS_PER_BLOCK>();
  if (threadIdx.x < GLOBALS::TOKENS_PER_BLOCK) {
    // while (token_idx < num_recv_tokens) {
    int phase = 0;

    init_semaphore(token_arrived[threadIdx.x], 0, 1);
    for (int token_idx = sm_idx * GLOBALS::TOKENS_PER_BLOCK + threadIdx.x;
         token_idx < num_recv_tokens;
         token_idx += GLOBALS::TOKENS_PER_BLOCK * G.num_comm_sms) {
      int src_dev_idx = G.src_dev_idx[{token_idx}];
      int src_token_idx = G.src_token_idx[{token_idx}];

      if (src_dev_idx >= 0 && src_token_idx >= 0) {

        tma::expect_bytes(token_arrived[threadIdx.x],
                          sizeof(typename GLOBALS::token_vec_tile));
        tma::load_async(token[threadIdx.x], G.in_tokens[src_dev_idx],
                        coord<ducks::default_type>{src_token_idx, 0},
                        token_arrived[threadIdx.x]);

        wait(token_arrived[threadIdx.x], phase);
        phase ^= 1;

        tma::store_async(G.expert_x_tokens, token[threadIdx.x], {token_idx, 0});
        tma::store_async_wait();
        float *scale_ptr =
            &(G.in_tokens_scales[src_dev_idx][{0, src_token_idx}]);
        int stride = G.in_tokens_scales[src_dev_idx].cols();
#pragma unroll
        for (int s = 0; s < _H / 128; s++) {
          G.expert_x_tokens_scale[{s, token_idx}] = *(scale_ptr + s * stride);
        }
      }
      // add to the barrier, even padded tokens
      asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                   :
                   : "l"(&G.comm_comp_barrier[{token_idx / _BM}]), "r"(1)
                   : "memory");
    }
  }
}

template <int _M, int _I, int _H, int _BM, int _BN, int NUM_CONSUMER_WARPS_,
          int NUM_PRODUCER_WARPS_, int NUM_STAGES_, int KERNEL_SMEM_SIZE,
          int NUM_EXPERTS, int EXPERTS_PER_TOKEN, int SUPER_M = 12>
__global__ void __launch_bounds__(MAIN_THREADS, 1) global_kernel5_1(
    const __grid_constant__ globals<
        _M, _I, _H, _BM, _BN, NUM_CONSUMER_WARPS_, NUM_PRODUCER_WARPS_,
        NUM_STAGES_, KERNEL_SMEM_SIZE, NUM_EXPERTS, EXPERTS_PER_TOKEN, SUPER_M>
        G) {
  auto grid = cooperative_groups::this_grid();
  if (blockIdx.x == 0) {
    compute_routing_info(G);
  }
  grid.sync();
  if (blockIdx.x < G.num_comp_sms) {
    compute_path(G, blockIdx.x);
  } else {
    comm_path(G, blockIdx.x - G.num_comp_sms);
  }
}
} // namespace kernel5_1
