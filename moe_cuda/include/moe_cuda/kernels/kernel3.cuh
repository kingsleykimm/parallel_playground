/**
 * @file
 * @brief: Extension of Basic FP8 1d2d grouped gemm with fused silu-mul-quant
 * activation in epilogue
 **/
#pragma once

#include "common/common.cuh"
#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

namespace kernel3 {

// Here, GEMM_TYPE corresponds to MGroupedMasked or MGroupedContiguous

template <int _BM, int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
struct grouped_matmul_layout {
  static constexpr int BM = _BM, BN = _BN, BK = _BK;
  static constexpr bool kIsUniformScales = (BK % BN) == 0;

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

  struct globals {
    a_layout A;
    gate_layout gate;
    up_layout up;
    c_layout C;
    scale_a_layout scale_a;
    scale_gate_layout scale_gate;
    scale_up_layout scale_up;
    out_scale_layout out_scales;
    int *grouped_layout;
  };

  // these are conditional on the number of warps
  // remember that sfa is loaded in MN-major
  struct input_block {
    a_tile a[BM > 64 ? 2 : 1];
    gate_tile gate;
    up_tile up;
    sfa_tile sfa[BM > 64 ? 2 : 1];
  };

  struct finish_block {
    c_tile c[BM > 64 ? 2 : 1];
    sfa_tile out_scales[BM > 64 ? 2 : 1];
  };
  struct scratch_block {};
  struct common_state {
    int2 coord;
    int Rblocks;
    int Cblocks;
    int cur_group_idx = 0;
    int current_m_cumsum = 0;
    int m_block_idx;
    bool computation_valid = false;
  };

  struct consumer_state {
    accum_tile<float> gate_accum, up_accum;
    accum_tile<float> per_k_gate_accum, per_k_up_accum;
    int num_former_iters = 0, num_full_iters = 0;
  };
};

// for combined weight, the N here is actually the intermediate_dim of ONE

template <int _M, int _N, int _K, int _BM, int _BN, int _BK, int NUM_GROUPS,
          int NUM_CONSUMER_WARPS_, int NUM_PRODUCER_WARPS_, int NUM_STAGES_,
          int KERNEL_SMEM_SIZE, int GEMM_TYPE, typename c_dtype,
          int _SUPER_N = 12>
struct matmul_template {
  static constexpr int SUPER_N = _SUPER_N;
  static constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPS_;
  static constexpr int NUM_PRODUCER_WARPS = NUM_PRODUCER_WARPS_;
  static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
  static constexpr int INPUT_PIPE_STAGES = NUM_STAGES_;
  static constexpr int MAX_SHARED_MEMORY = KERNEL_SMEM_SIZE;
  static constexpr int DEBUG = 0;
  static constexpr bool isGroupedMasked = GEMM_TYPE == 0;
  static constexpr bool isGroupedContiguous = GEMM_TYPE == 1;

  static_assert(_BN <= 256, "BN is too large");
  static_assert(_BM <= 128, "BM is too large");

  static constexpr int NUM_TILES = (_BM + 63) / 64;
  using layout = grouped_matmul_layout<_BM, _BN, _BK, GEMM_TYPE, c_dtype>;

  template <bool PERSISTENT_GRID = true>
  __host__ static inline dim3 grid(int M, int N, int K) {
    return dim3(PERSISTENT_GRID
                    ? 132
                    : M * N / (NUM_TILES * layout::c_tile::num_elements));
  }

  // helper function

  template <bool kWithGroupOffset>
  __device__ static int
  get_global_idx(int shape, int block_offset, uniform_args<layout> args,
                 int m_block_idx = 0) { // m_block_idx is always in units of 64
    if (isGroupedContiguous) {
      const auto offset =
          kWithGroupOffset
              ? max(0, __ldg(args.globals.grouped_layout + m_block_idx * 64))
              : 0;
      return offset * shape + block_offset;
    } else if constexpr (isGroupedMasked) {
      const auto offset = kWithGroupOffset ? args.common.cur_group_idx : 0;
      return offset * shape + block_offset;
    } else {
      // normal gemm
      return block_offset;
    }
  }

  __device__ static inline bool
  is_computation_valid(uniform_args<layout> args) {
    if constexpr (isGroupedContiguous) {
      // we need to make sure to clairfy the m_block_idx * BLOCK_M used between
      // deepgemm and tk, since TK seems to always use BM
      auto group_index =
          __ldg(args.globals.grouped_layout + args.common.m_block_idx * 64);
      return group_index >= 0;
    } else if constexpr (isGroupedMasked) {
      auto group_limit =
          __ldg(args.globals.grouped_layout + args.common.cur_group_idx);
      return (args.common.m_block_idx) * 64 < group_limit;
    }
  }

  __device__ static inline void common_setup(common_setup_args<layout> args) {

    // args.common.shape_k_scales = ti_ceil_div(args.globals.A.cols(),
    // layout::BK); args.common.shape_n_sfb = ti_ceil_div(args.globals.C.cols(),
    // layout::BK);

    int Rblocks = constexpr_ti_ceil_div(_M, NUM_TILES * layout::c_tile::rows),
        Cblocks = constexpr_ti_ceil_div(_N, layout::c_tile::cols);
    int task_id = args.task_iter * gridDim.x + blockIdx.x;

    args.common.Rblocks = Rblocks;
    args.common.Cblocks = Cblocks;
    // M grouped masked
    if constexpr (isGroupedMasked) {
      while (true) {
        if (args.common.cur_group_idx == NUM_GROUPS) {
          args.num_iters = -1;
          return;
        }

        Rblocks = ti_ceil_div(
            __ldg(args.globals.grouped_layout + args.common.cur_group_idx),
            layout::BM);

        auto cur_cumsum_m_blocks = args.common.current_m_cumsum + Rblocks;
        if (task_id < cur_cumsum_m_blocks * Cblocks) {
          break;
        }

        ++args.common.cur_group_idx;
        args.common.current_m_cumsum = cur_cumsum_m_blocks;
      }
      // after we find the current group, we just need to make sure the task_id
      // is back in the 'local' mnk shape
      task_id -= args.common.current_m_cumsum * Cblocks;
    }

    // thread-block swizzling based off of Multicast direction

    int super_cols = (Cblocks / SUPER_N) * SUPER_N,
        final_cols = Cblocks - super_cols, super_repeat = SUPER_N * Rblocks;

    if (task_id < super_cols * Rblocks)
      args.common.coord = {
          (task_id % super_repeat) / SUPER_N,
          SUPER_N * (task_id / super_repeat) + task_id % SUPER_N,
      };
    else if (task_id < Rblocks * Cblocks) {
      int remainder_id = task_id - super_cols * Rblocks;
      args.common.coord = {remainder_id / final_cols,
                           super_cols + (remainder_id % final_cols)};
    } else {
      args.num_iters = -1;
      return;
    }
    args.num_iters = args.globals.A.cols() / layout::a_tile::cols;
    int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4
                 ? 0
                 : warpgroup::groupid();
    args.common.m_block_idx = args.common.coord.x * NUM_TILES + id;
    args.common.coord = {args.common.coord.x * NUM_TILES + id,
                         args.common.coord.y};
  }

  struct producer {
    __device__ static void setup(producer_setup_args<layout> args) {
      warpgroup::decrease_registers<40>();
    }
    __device__ static void load(producer_load_args<layout> args) {
      if (warpgroup::laneid() == 0) {
        tma::expect(args.inputs_arrived, args.input);
#pragma unroll
        for (int i = 0; i < NUM_TILES; i++) {
          const int a_element_row = get_global_idx<isGroupedMasked>(
              _M, (args.common.coord.x + i) * 64, args);
          tma::load_async(
              args.input.a[i], args.globals.A,
              coord<ducks::default_type>{a_element_row, args.iter * layout::BK},
              args.inputs_arrived);
          tma::load_async(args.input.sfa[i], args.globals.scale_a,
                          coord<ducks::default_type>{args.iter, a_element_row},
                          args.inputs_arrived);
        }
        const int b_element_row =
            get_global_idx<true>(_N, args.common.coord.y * layout::BN, args,
                                 args.common.m_block_idx);
        tma::load_async(
            args.input.gate, args.globals.gate,
            coord<ducks::default_type>{b_element_row, args.iter * layout::BK},
            args.inputs_arrived);
        tma::load_async(
            args.input.up, args.globals.up,
            coord<ducks::default_type>{b_element_row, args.iter * layout::BK},
            args.inputs_arrived);
      }
    }
  };

  struct consumer {

    // convertor wrapper for fp8e4m3 -> float

    using consumers = group<NUM_CONSUMER_WARPS>;
    static constexpr int shape_k_scales = constexpr_ti_ceil_div(_K, layout::BK);
    static constexpr int shape_n_sfb = constexpr_ti_ceil_div(_N, layout::BK);
    static constexpr uint32_t stride_n_sfb = shape_k_scales;
    static constexpr uint32_t stride_k_sfb = 1;

    __device__ static void setup(consumer_setup_args<layout> args) {
      warpgroup::increase_registers<232>();
      warp::zero(args.state.gate_accum);
      warp::zero(args.state.up_accum);

      // need to load in b scales here
      if constexpr (!layout::kIsUniformScales) {
        args.state.num_former_iters =
            min(layout::BN, (layout::BK -
                             (args.common.coord.y * layout::BN) % layout::BK)) /
            8;
        args.state.num_full_iters =
            min(layout::BN, (_N - args.common.coord.y * layout::BN)) / 8;
      } else {
        args.state.num_former_iters = args.state.num_full_iters =
            layout::BN / 8;
      }
    }

    __device__ static void compute(consumer_compute_args<layout> args) {

      const bool computation_valid = is_computation_valid(args);
      args.common.computation_valid = computation_valid;
      if (computation_valid) {

        // Compute scale pointer offsets once (independent of WGMMA results)
        const auto previous_group_offset = get_global_idx<true>(
            shape_k_scales * shape_n_sfb, 0, args, args.common.m_block_idx);
        const uint32_t scale_b_offset =
            previous_group_offset +
            (((args.common.coord.y * layout::BN) / layout::BK) * stride_n_sfb) +
            (args.iter * stride_k_sfb);
        float *local_gate_sfb =
            args.globals.scale_gate.raw_ptr + scale_b_offset;
        float *local_up_sfb = args.globals.scale_up.raw_ptr + scale_b_offset;

        // === Phase 1: issue gate WGMMA ===
        warp::zero(args.state.per_k_gate_accum);
        warpgroup::mma_ABt(args.state.per_k_gate_accum,
                           args.input.a[warpgroup::groupid()], args.input.gate);

        // === Phase 2: while gate WGMMA runs, load gate scales + scale_a,
        //              and build the per-column gate scale vector ===
        float gate_scale_0, gate_scale_1, up_scale_0, up_scale_1;
        move<float>::ldg(gate_scale_0, local_gate_sfb);
        if constexpr (!layout::kIsUniformScales) {
          if (args.state.num_full_iters > args.state.num_former_iters)
            move<float>::ldg(gate_scale_1, local_gate_sfb + stride_n_sfb);
        }

        typename decltype(args.state.per_k_gate_accum)::col_vec scale_a_rv;
        warpgroup::load(scale_a_rv, args.input.sfa[warpgroup::groupid()]);

        row_vec<rt<float, 16, layout::BN>> gate_col_scale_b, up_col_scale_b;
        if constexpr (!layout::kIsUniformScales) {
#pragma unroll
          for (uint32_t i = 0; i < layout::BN / 16; i++) {
            const uint32_t column_chunk = i * 2;
            float first = column_chunk < args.state.num_former_iters
                              ? gate_scale_0
                              : gate_scale_1;
            float second = column_chunk + 1 < args.state.num_former_iters
                               ? gate_scale_0
                               : gate_scale_1;
            gate_col_scale_b[i][0] = make_float2(first, first);
            gate_col_scale_b[i][1] = make_float2(second, second);
          }
        }

        // === Phase 3: wait for gate WGMMA ===
        warpgroup::mma_async_wait();

        // === Phase 4: issue up WGMMA ===
        warp::zero(args.state.per_k_up_accum);
        warpgroup::mma_ABt(args.state.per_k_up_accum,
                           args.input.a[warpgroup::groupid()], args.input.up);

        // === Phase 5: while up WGMMA runs:
        //   a) apply scale promotion to gate result and accumulate
        //   b) load up scales and build per-column up scale vector ===
        warp::mul_row(args.state.per_k_gate_accum, args.state.per_k_gate_accum,
                      scale_a_rv);
        if constexpr (layout::kIsUniformScales) {
          args.state.per_k_gate_accum *= gate_scale_0;
        } else {
          warp::mul_col(args.state.per_k_gate_accum,
                        args.state.per_k_gate_accum, gate_col_scale_b);
        }
        args.state.gate_accum += args.state.per_k_gate_accum;

        move<float>::ldg(up_scale_0, local_up_sfb);
        if constexpr (!layout::kIsUniformScales) {
          if (args.state.num_full_iters > args.state.num_former_iters)
            move<float>::ldg(up_scale_1, local_up_sfb + stride_n_sfb);
#pragma unroll
          for (uint32_t i = 0; i < layout::BN / 16; i++) {
            const uint32_t column_chunk = i * 2;
            float first = column_chunk < args.state.num_former_iters
                              ? up_scale_0
                              : up_scale_1;
            float second = column_chunk + 1 < args.state.num_former_iters
                               ? up_scale_0
                               : up_scale_1;
            up_col_scale_b[i][0] = make_float2(first, first);
            up_col_scale_b[i][1] = make_float2(second, second);
          }
        }

        // === Phase 6: wait for up WGMMA ===
        warpgroup::mma_async_wait();

        // === Phase 7: apply scale promotion to up result and accumulate ===
        warp::mul_row(args.state.per_k_up_accum, args.state.per_k_up_accum,
                      scale_a_rv);
        if constexpr (layout::kIsUniformScales) {
          args.state.per_k_up_accum *= up_scale_0;
        } else {
          warp::mul_col(args.state.per_k_up_accum, args.state.per_k_up_accum,
                        up_col_scale_b);
        }
        args.state.up_accum += args.state.per_k_up_accum;
      }
      if (laneid() == 0)
        arrive(args.inputs_finished);
    }
    __device__ static void finish(consumer_finish_args<layout> args) {

      if (args.common.computation_valid) {

        args.state.up_accum = (args.state.up_accum * args.state.gate_accum) /
                              (warp::exp(args.state.gate_accum * -1.0f) + 1.0f);

        warp::abs(
            args.state.gate_accum,
            args.state
                .up_accum); // we're reusing the gate_accum to save registers

        col_vec<rt<float, 16, layout::BN>> row_amaxes_rv;

        warp::row_reduce<base_ops::max, decltype(row_amaxes_rv),
                         decltype(args.state.gate_accum), true>(
            row_amaxes_rv, args.state.gate_accum, row_amaxes_rv);

        row_amaxes_rv /= 448.0f; // scale factors
        // need to quantize to FP8, which is args.state.up_accum / row_amaxes_rv
        // across rows
        warp::div_row(args.state.up_accum, args.state.up_accum, row_amaxes_rv);

        rt<fp8e4m3, 16, layout::BN> fp8_output_tile;
        warp::copy(fp8_output_tile, args.state.up_accum);
        tma::store_async_read_wait();
        warpgroup::store(args.finish.c[warpgroup::groupid()], fp8_output_tile);
        warpgroup::store(args.finish.out_scales[warpgroup::groupid()],
                         row_amaxes_rv);
        warpgroup::sync(warpgroup::groupid() + 4);
        if (warpgroup::laneid() == 0) {
          tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                           coord<ducks::default_type>{
                               get_global_idx<isGroupedMasked>(
                                   _M, args.common.coord.x * 64, args),
                               args.common.coord.y * layout::BN});
          tma::store_async(args.globals.out_scales,
                           args.finish.out_scales[warpgroup::groupid()],
                           coord<ducks::default_type>{
                               args.common.coord.y,
                               get_global_idx<isGroupedMasked>(
                                   _M, args.common.coord.x * 64, args)});
        }
      }

      if (laneid() == 0)
        arrive(args.finish_finished);
    }
  };
};

// Default instantiation alias for convenience
// using mmt =
//     matmul_template<-1, -1, -1, 64, 128, 128, 1, 8, 1, 4, 0, 0, float, 12>;
// using tk_globals_t = typename mmt::layout::globals;
} // namespace kernel3
