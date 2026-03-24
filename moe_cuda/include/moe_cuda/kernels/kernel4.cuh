/**
 * @file
 * @brief: Extension of kernel3 with consumer ping pong scheduling to keep
 * tensor cores hot
 **/
#pragma once

#include "common/common.cuh"
#include "common/util.cuh"
#include "kittens.cuh"
#include "ops/group/group.cuh"
#include "prototype.cuh"

using namespace kittens;
namespace kernel4 {

// persistent kernel settings
static constexpr int SM_COUNT = 132;
static constexpr int MAIN_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

template <int _M, int _N, int _K, int _BM, int _BN, int _BK, int NUM_GROUPS,
          int NUM_CONSUMER_WARPS_, int NUM_PRODUCER_WARPS_, int NUM_STAGES_,
          int KERNEL_SMEM_SIZE, int GEMM_TYPE, typename c_dtype,
          int SUPER_N = 12>
struct globals {
  static constexpr int BM = _BM, BN = _BN, BK = _BK;
  static constexpr bool kIsUniformScales = (BK % BN) == 0;

  static constexpr int M = _M;
  static constexpr int N = _N;
  static constexpr int K = _K;

  static constexpr int NUM_PRODUCER_WARPS = NUM_PRODUCER_WARPS_;
  static constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPS_;
  static constexpr int NUM_STAGES = NUM_STAGES_;

  static constexpr int isGroupedMasked = GEMM_TYPE == 0;
  static constexpr int isGroupedContiguous = GEMM_TYPE == 1;

  static_assert(BM == 64, "Pingpong scheduler only allow BM = 64 for H100");
  static_assert(BN % 128 == 0,
                "BN Must be divisible by 128 for fused silu-mul-quant");
  // =========== SHARED MEMORY TILES ============== //
  using a_tile =
      st_fp8e4m3<64, BK>; // this is static, since this has to fit the TK WGMMA
                          // requirement, see warpgroup.cuh
  using gate_tile = st_fp8e4m3<BN, BK>;
  using up_tile = st_fp8e4m3<BN, BK>;
  using c_tile = st<c_dtype, 64, BN>;
  using sfa_tile = sv<float, 64>;
  // assume this is k-major here, with shape (N / 128, K / 128)

  // ================ GLOBAL MEMORY ============= //
  using a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
  using gate_layout = gl<fp8e4m3, 1, 1, -1, -1, gate_tile>;
  using up_layout = gl<fp8e4m3, 1, 1, -1, -1, up_tile>;
  using c_layout = gl<c_dtype, 1, 1, -1, -1, c_tile>;

  // only sfa_tile since this is the only one we do TMA load on
  using scale_a_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using scale_gate_layout = gl<float, 1, 1, -1, -1>;
  using scale_up_layout = gl<float, 1, 1, -1, -1>;

  using out_scale_layout =
      gl<float, 1, 1, -1, -1,
         sfa_tile>; // this will actually be transposed to column major again,
                    // for input to next grouped gemm

  template <typename T = float> using accum_tile = rt<T, 16, c_tile::cols>;

  a_layout A;
  gate_layout gate;
  up_layout up;
  c_layout C;
  scale_a_layout scale_a;
  scale_gate_layout scale_gate;
  scale_up_layout scale_up;
  out_scale_layout out_scales;
  int *grouped_layout;

  // these are conditional on the number of warps
  // remember that sfa is loaded in MN-major
  struct pipeline_inputs {
    a_tile a[2];
    gate_tile gate[2];
    up_tile up[2];
    sfa_tile sfa[2];
  };

  struct pipeline_outputs {
    c_tile c[2];
    sfa_tile out_scales[2];
  };

  struct common_state {
    int2 coord;
    int Rblocks;
    int Cblocks;
    int cur_group_idx = 0;
    int current_m_cumsum = 0;
    int m_block_idx;
    bool computation_valid = false;
    int cur_mma_wg_idx = 0;
  };

  struct consumer_state {
    accum_tile<float> gate_accum, up_accum;
    accum_tile<float> per_k_gate_accum, per_k_up_accum;
    int num_former_iters = 0, num_full_iters = 0;
  };

  __device__ inline void scheduler(common_state &common, int &task_iter,
                                   int &num_iters) const {
    int Rblocks = constexpr_ti_ceil_div(
            _M, 64), // two warpgroups that each take BM = 64 tile
        Cblocks = constexpr_ti_ceil_div(_N, BN);
    int task_id = task_iter * gridDim.x + blockIdx.x;

    common.Rblocks = Rblocks;
    common.Cblocks = Cblocks;
    // M grouped masked
    if constexpr (isGroupedMasked) {
      while (true) {
        if (common.cur_group_idx == NUM_GROUPS) {
          num_iters = -1;
          return;
        }

        Rblocks = ti_ceil_div(__ldg(grouped_layout + common.cur_group_idx), BM);

        auto cur_cumsum_m_blocks = common.current_m_cumsum + Rblocks;
        if (task_id < cur_cumsum_m_blocks * Cblocks) {
          break;
        }

        ++common.cur_group_idx;
        common.current_m_cumsum = cur_cumsum_m_blocks;
      }
      // after we find the current group, we just need to make sure the task_id
      // is back in the 'local' mnk shape
      task_id -= common.current_m_cumsum * Cblocks;
    }

    // thread-block swizzling based off of Multicast direction

    int super_cols = (Cblocks / SUPER_N) * SUPER_N,
        final_cols = Cblocks - super_cols, super_repeat = SUPER_N * Rblocks;

    if (task_id < super_cols * Rblocks)
      common.coord = {
          (task_id % super_repeat) / SUPER_N,
          SUPER_N * (task_id / super_repeat) + task_id % SUPER_N,
      };
    else if (task_id < Rblocks * Cblocks) {
      int remainder_id = task_id - super_cols * Rblocks;
      common.coord = {remainder_id / final_cols,
                      super_cols + (remainder_id % final_cols)};
    } else {
      num_iters = -1;
      return;
    }
    num_iters = K / BK;

    // for ping pong scheduling, bM is HARDSET to 64
    common.m_block_idx = common.coord.x;
  }

  template <bool kWithGroupOffset>
  __device__ int get_global_idx(
      int shape, int block_offset, common_state common,
      int m_block_idx = 0) const { // m_block_idx is always in units of 64
    if constexpr (isGroupedContiguous) {
      const auto offset = kWithGroupOffset
                              ? max(0, __ldg(grouped_layout + m_block_idx * 64))
                              : 0;
      return offset * shape + block_offset;
    } else if constexpr (isGroupedMasked) {
      const auto offset = kWithGroupOffset ? common.cur_group_idx : 0;
      return offset * shape + block_offset;
    } else {
      // normal gemm
      return block_offset;
    }
  }

  __device__ inline bool is_computation_valid(common_state common) const {
    if constexpr (isGroupedContiguous) {
      // we need to make sure to clairfy the m_block_idx * BLOCK_M used between
      // deepgemm and tk, since TK seems to always use BM
      auto group_index = __ldg(grouped_layout + common.m_block_idx * 64);
      return group_index >= 0;
    } else if constexpr (isGroupedMasked) {
      auto group_limit = __ldg(grouped_layout + common.cur_group_idx);
      return (common.m_block_idx) * 64 < group_limit;
    }

    return true;
  }
};
template <int _M, int _N, int _K, int _BM, int _BN, int _BK, int NUM_GROUPS,
          int NUM_CONSUMER_WARPS_, int NUM_PRODUCER_WARPS_, int NUM_STAGES_,
          int KERNEL_SMEM_SIZE, int GEMM_TYPE, typename c_dtype,
          int _SUPER_N = 12>
__device__ inline void
kernel4(const globals<_M, _N, _K, _BM, _BN, _BK, NUM_GROUPS,
                      NUM_CONSUMER_WARPS_, NUM_PRODUCER_WARPS_, NUM_STAGES_,
                      KERNEL_SMEM_SIZE, GEMM_TYPE, c_dtype, _SUPER_N> &G) {

  using GLOBALS = globals<_M, _N, _K, _BM, _BN, _BK, NUM_GROUPS,
                          NUM_CONSUMER_WARPS_, NUM_PRODUCER_WARPS_, NUM_STAGES_,
                          KERNEL_SMEM_SIZE, GEMM_TYPE, c_dtype, _SUPER_N>;
  using pipeline_inputs = typename GLOBALS::pipeline_inputs;
  using pipeline_outputs = typename GLOBALS::pipeline_outputs;
  using common_state = typename GLOBALS::common_state;
  using consumer_state = typename GLOBALS::consumer_state;
  extern __shared__ int __shm[];
  tma_swizzle_allocator allocator((int *)&__shm[0]);

  pipeline_inputs(&inputs)[NUM_STAGES_] =
      allocator.allocate<pipeline_inputs, NUM_STAGES_>();
  // do some checking first
  constexpr int NUM_CONSUMER_WGS = NUM_CONSUMER_WARPS_ / 4;
  constexpr int shape_k_scales = constexpr_ti_ceil_div(_K, _BK);
  constexpr int shape_n_sfb = constexpr_ti_ceil_div(_N, 128);
  constexpr int stride_n_sfb = shape_k_scales;
  constexpr int stride_k_sfb = 1;
  constexpr int FINISH_BLOCK_OFFSET =
      DYNAMIC_SHARED_MEMORY - sizeof(pipeline_outputs); // single SM persistent
  static_assert(FINISH_BLOCK_OFFSET >= 0, "not enough shared memory");
  constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024;
  // round down number of stages if necessary
  constexpr int SAFE_STAGES_BETWEEN_BLOCKS =
      (NON_FINISH_BLOCK_SPACE / sizeof(pipeline_inputs));
  // will integrate finish_finished signalling later for pingpong schedulign
  static_assert(SAFE_STAGES_BETWEEN_BLOCKS >= NUM_STAGES_);

  pipeline_outputs &outputs = allocator.allocate<pipeline_outputs>();
  __shared__ semaphore inputs_arrived[NUM_CONSUMER_WGS][NUM_STAGES_];
  __shared__ semaphore inputs_finished[NUM_CONSUMER_WGS][NUM_STAGES_];
  __shared__ semaphore consumer_can_start[2];
  uint32_t semaphore_bitfield[NUM_CONSUMER_WGS];
  for (int wg = 0; wg < NUM_CONSUMER_WGS; wg++) {
    semaphore_bitfield[wg] = 0xFFFF0000;
  }
  uint32_t consumer_phasebit =
      0b01; // wg0 can start immediately, while wg 1 needs to wait

  if (threadIdx.x == 0) {
#pragma unroll
    for (int wg = 0; wg < NUM_CONSUMER_WGS; wg++) {
#pragma unroll
      for (int stage = 0; stage < GLOBALS::NUM_STAGES; stage++) {
        init_semaphore(inputs_arrived[wg][stage], 1, 0);
        init_semaphore(inputs_finished[wg][stage],
                       NUM_CONSUMER_WARPS_ / NUM_CONSUMER_WGS, 0);
      }
    }
    init_semaphore(consumer_can_start[0], NUM_CONSUMER_WARPS_ / 2, 0);
    init_semaphore(consumer_can_start[1], NUM_CONSUMER_WARPS_ / 2, 0);
  }
  __syncthreads();

  common_state common;
  // producer path
  if (warpid() >= NUM_CONSUMER_WARPS_) {
    warpgroup::decrease_registers<40>();
    using producers = group<NUM_PRODUCER_WARPS_>;
    for (int task_iter = 0; true; task_iter++) {

      int num_iters = -1;
      G.scheduler(common, task_iter, num_iters);
      if (num_iters < 0) {
        break;
      }

      int current_math_wg = task_iter % NUM_CONSUMER_WGS;
      int input_ring = 0; // tracks, which input block, reset to 0 per iteration

      int load_iter;
      for (load_iter = 0; load_iter < num_iters; load_iter++) {
        wait(inputs_finished[current_math_wg][input_ring],
             get_phasebit<1>(semaphore_bitfield[current_math_wg], input_ring));
        update_phasebit<1>(semaphore_bitfield[current_math_wg], input_ring);
        if (warpgroup::laneid() == 0) {
          // if (blockIdx.x == 0 && warpgroup::laneid() == 0) {
          //   printf("loading common coords : %d, %d, load_iter : %d, gropu id:
          //   "
          //          "%d \n",
          //          common.coord.x, common.coord.y, load_iter,
          //          warpgroup::groupid());
          // }
          const int a_element_row =
              G.template get_global_idx<GLOBALS::isGroupedMasked>(
                  _M, common.coord.x * 64, common);
          const int b_element_row = G.template get_global_idx<true>(
              _N, common.coord.y * _BN, common, common.m_block_idx);
          tma::expect(inputs_arrived[current_math_wg][input_ring],
                      inputs[input_ring].a[current_math_wg],
                      inputs[input_ring].sfa[current_math_wg],
                      inputs[input_ring].gate[current_math_wg],
                      inputs[input_ring].up[current_math_wg]);
          tma::load_async(
              inputs[input_ring].a[current_math_wg], G.A,
              coord<ducks::default_type>{a_element_row, load_iter * _BK},
              inputs_arrived[current_math_wg][input_ring]);
          tma::load_async(inputs[input_ring].sfa[current_math_wg], G.scale_a,
                          coord<ducks::default_type>{load_iter, a_element_row},
                          inputs_arrived[current_math_wg][input_ring]);
          tma::load_async(
              inputs[input_ring].gate[current_math_wg], G.gate,
              coord<ducks::default_type>{b_element_row, load_iter * _BK},
              inputs_arrived[current_math_wg][input_ring]);
          tma::load_async(
              inputs[input_ring].up[current_math_wg], G.up,
              coord<ducks::default_type>{b_element_row, load_iter * _BK},
              inputs_arrived[current_math_wg][input_ring]);
        }
        input_ring = ring_advance<NUM_STAGES_>(input_ring);
      }
    }
  }

  else {
    consumer_state state;
    warpgroup::increase_registers<232>();
    using consumer = group<NUM_CONSUMER_WARPS_>;

    for (int task_iter = warpgroup::groupid(); true; task_iter += 2) {

      int num_iters = -1;
      G.scheduler(common, task_iter, num_iters);
      if (num_iters < 0) {
        break;
      }

      // common.coord.y doesn't change
      int num_former_iters, num_full_iters;
      if constexpr (!GLOBALS::kIsUniformScales) {
        num_former_iters = min(_BN, (_BK - (common.coord.y * _BN) % _BK)) / 8;
        num_full_iters = min(_BN, (_N - common.coord.y * _BN)) / 8;
      } else {
        num_former_iters = num_full_iters = _BN / 8;
      }
      warp::zero(state.up_accum);
      warp::zero(state.gate_accum);

      int input_ring = 0; // tracks, which input block, reset to 0 per iteration

      // update current phasebit (flips every time)

      int current_math_wg = warpgroup::groupid();
      // k-loop

      wait(consumer_can_start[current_math_wg],
           (consumer_phasebit >> (current_math_wg)) & 0b1);
      consumer_phasebit ^= (1 << current_math_wg);

      for (int load_iter = 0; load_iter < num_iters; load_iter++) {
        wait(inputs_arrived[current_math_wg][input_ring],
             get_phasebit<0>(semaphore_bitfield[current_math_wg], input_ring));
        update_phasebit<0>(semaphore_bitfield[current_math_wg], input_ring);

        const int a_element_row =
            G.template get_global_idx<GLOBALS::isGroupedMasked>(
                _M, common.coord.x * 64, common);
        const int b_element_row = G.template get_global_idx<true>(
            _N, common.coord.y * _BN, common, common.m_block_idx);
        const bool computation_valid = G.is_computation_valid(common);
        common.computation_valid = computation_valid;

        if (computation_valid) {
          const auto previous_group_offset = G.template get_global_idx<true>(
              shape_k_scales * shape_n_sfb, 0, common, common.m_block_idx);
          const uint32_t scale_b_offset =
              previous_group_offset +
              (((common.coord.y * _BN) / _BK) * stride_n_sfb) +
              (load_iter * stride_k_sfb);
          float *local_gate_sfb = G.scale_gate.raw_ptr + scale_b_offset;
          float *local_up_sfb = G.scale_up.raw_ptr + scale_b_offset;

          // === Phase 1: issue gate WGMMA ===
          warp::zero(state.per_k_gate_accum);
          warpgroup::mma_ABt(state.per_k_gate_accum,
                             inputs[input_ring].a[current_math_wg],
                             inputs[input_ring].gate[current_math_wg]);

          // === Phase 2: while gate WGMMA runs, load gate scales + scale_a,
          //              and build per-column gate scale vector ===
          float gate_scale_0, gate_scale_1, up_scale_0, up_scale_1;
          move<float>::ldg(gate_scale_0, local_gate_sfb);
          if constexpr (!GLOBALS::kIsUniformScales) {
            if (num_full_iters > num_former_iters)
              move<float>::ldg(gate_scale_1, local_gate_sfb + stride_n_sfb);
          }

          typename decltype(state.per_k_gate_accum)::col_vec scale_a_rv;
          warpgroup::load(scale_a_rv, inputs[input_ring].sfa[current_math_wg]);

          row_vec<rt<float, 16, _BN>> gate_col_scale_b, up_col_scale_b;
          if constexpr (!GLOBALS::kIsUniformScales) {
#pragma unroll
            for (uint32_t i = 0; i < _BN / 16; i++) {
              const uint32_t column_chunk = i * 2;
              float first =
                  column_chunk < num_former_iters ? gate_scale_0 : gate_scale_1;
              float second = column_chunk + 1 < num_former_iters ? gate_scale_0
                                                                 : gate_scale_1;
              gate_col_scale_b[i][0] = make_float2(first, first);
              gate_col_scale_b[i][1] = make_float2(second, second);
            }
          }

          // === Phase 3: wait for gate WGMMA ===
          warpgroup::mma_async_wait();

          // === Phase 4: issue up WGMMA ===
          warp::zero(state.per_k_up_accum);
          warpgroup::mma_ABt(state.per_k_up_accum,
                             inputs[input_ring].a[current_math_wg],
                             inputs[input_ring].up[current_math_wg]);

          // === Phase 5: while up WGMMA runs:
          //   a) apply gate scale promotion and accumulate
          //   b) load up scales and build per-column up scale vector ===
          warp::mul_row(state.per_k_gate_accum, state.per_k_gate_accum,
                        scale_a_rv);
          if constexpr (GLOBALS::kIsUniformScales) {
            state.per_k_gate_accum *= gate_scale_0;
          } else {
            warp::mul_col(state.per_k_gate_accum, state.per_k_gate_accum,
                          gate_col_scale_b);
          }
          state.gate_accum += state.per_k_gate_accum;

          move<float>::ldg(up_scale_0, local_up_sfb);
          if constexpr (!GLOBALS::kIsUniformScales) {
            if (num_full_iters > num_former_iters)
              move<float>::ldg(up_scale_1, local_up_sfb + stride_n_sfb);
#pragma unroll
            for (uint32_t i = 0; i < _BN / 16; i++) {
              const uint32_t column_chunk = i * 2;
              float first =
                  column_chunk < num_former_iters ? up_scale_0 : up_scale_1;
              float second =
                  column_chunk + 1 < num_former_iters ? up_scale_0 : up_scale_1;
              up_col_scale_b[i][0] = make_float2(first, first);
              up_col_scale_b[i][1] = make_float2(second, second);
            }
          }

          // === Phase 6: wait for up WGMMA ===
          warpgroup::mma_async_wait();

          // === Phase 7: apply up scale promotion and accumulate ===
          warp::mul_row(state.per_k_up_accum, state.per_k_up_accum, scale_a_rv);
          if constexpr (GLOBALS::kIsUniformScales) {
            state.per_k_up_accum *= up_scale_0;
          } else {
            warp::mul_col(state.per_k_up_accum, state.per_k_up_accum,
                          up_col_scale_b);
          }
          state.up_accum += state.per_k_up_accum;
        }

        if (laneid() == 0) {
          arrive(inputs_finished[current_math_wg][input_ring]);
        }
        input_ring = ring_advance<NUM_STAGES_>(input_ring);
      }

      // after we're done with the k-loop, signal to the next consumer
      // warpgroup to begin

      if (laneid() == 0) {
        arrive(
            consumer_can_start[current_math_wg ^ 1]); // signal other warpgroup
      }

      // epilogue
      if (common.computation_valid) {
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
        tma::store_async_read_wait(); // wait for previous stores to shared
        warpgroup::store(outputs.c[current_math_wg], fp8_output_tile);
        warpgroup::store(outputs.out_scales[current_math_wg], row_amaxes_rv);
        warpgroup::sync(current_math_wg + 4);
        if (warpgroup::laneid() == 0) {
          tma::store_async(
              G.C, outputs.c[current_math_wg],
              coord<ducks::default_type>{
                  G.template get_global_idx<GLOBALS::isGroupedMasked>(
                      _M, common.coord.x * 64, common),
                  common.coord.y * _BN});
          tma::store_async(
              G.out_scales, outputs.out_scales[current_math_wg],
              coord<ducks::default_type>{
                  common.coord.y,
                  G.template get_global_idx<GLOBALS::isGroupedMasked>(
                      _M, common.coord.x * 64, common, common.m_block_idx)});
        }

        // memory to finish
      }

      // once we're done here, signal to the next consumer warpgroup to begin
      // its own consumer epilogue
    }
  }
}

template <int _M, int _N, int _K, int _BM, int _BN, int _BK, int NUM_GROUPS,
          int NUM_CONSUMER_WARPS_, int NUM_PRODUCER_WARPS_, int NUM_STAGES_,
          int KERNEL_SMEM_SIZE, int GEMM_TYPE, typename c_dtype,
          int _SUPER_N = 12>
__global__ void
__launch_bounds__(((NUM_CONSUMER_WARPS_ + NUM_PRODUCER_WARPS_) * 32), 1)
    global_kernel4(
        const __grid_constant__
            globals<_M, _N, _K, _BM, _BN, _BK, NUM_GROUPS, NUM_CONSUMER_WARPS_,
                    NUM_PRODUCER_WARPS_, NUM_STAGES_, KERNEL_SMEM_SIZE,
                    GEMM_TYPE, c_dtype, _SUPER_N>
                G) {
  kernel4(G);
}
} // namespace kernel4
