/**
 * @file
 * @brief: Basic FP8 1d2d grouped gemm
 **/
#pragma once

#include "common/common.cuh"
#include "kittens.cuh"
#include "prototype.cuh"
#include <moe_cuda/error.hpp>
using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

namespace kernel2 {

// Here, GEMM_TYPE corresponds to MGroupedMasked or MGroupedContiguous

template <int _BM, int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
struct grouped_matmul_layout {
  static constexpr int BM = _BM, BN = _BN, BK = _BK;
  static constexpr bool kIsUniformScales = (BK % BN) == 0;
  static_assert(BN >= 32, "TK doesn't support BN < 32 WGMMA");
  // =========== SHARED MEMORY TILES ============== //
  using a_tile =
      st_fp8e4m3<64, BK>; // this is static, since this has to fit the TK WGMMA
                          // requirement, see warpgroup.cuh
  using b_tile = st_fp8e4m3<BN, BK>;
  using c_tile = st<c_dtype, 64, BN>;
  using sfa_tile = sv<float, 64>;
  // assume this is k-major here, with shape (N / 128, K / 128)

  // ================ GLOBAL MEMORY ============= //
  using a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
  using b_layout = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
  using c_layout = gl<c_dtype, 1, 1, -1, -1, c_tile>;

  // only sfa_tile since this is the only one we do TMA load on
  using scale_a_layout = gl<float, 1, 1, -1, -1, sfa_tile>;
  using scale_b_layout = gl<float, 1, 1, -1, -1>;

  template <typename T = float> using accum_tile = rt<T, 16, c_tile::cols>;

  struct globals {
    a_layout A;
    b_layout B;
    c_layout C;
    scale_a_layout scale_a;
    scale_b_layout scale_b;
    int *grouped_layout;
  };

  // these are conditional on the number of warps
  // remember that sfa is loaded in MN-major
  struct input_block {
    a_tile a[BM > 64 ? 2 : 1];
    b_tile b;
    sfa_tile sfa[BM > 64 ? 2 : 1];
  };

  struct finish_block {
    c_tile c[BM > 64 ? 2 : 1];
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
    accum_tile<float> accum;
    accum_tile<float> per_k_accum;
    int num_former_iters = 0, num_full_iters = 0;
  };
};

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
            args.input.b, args.globals.B,
            coord<ducks::default_type>{b_element_row, args.iter * layout::BK},
            args.inputs_arrived);
      }
    }
  };

  struct consumer {

    using consumers = group<NUM_CONSUMER_WARPS>;
    static constexpr int shape_k_scales = constexpr_ti_ceil_div(_K, layout::BK);
    static constexpr int shape_n_sfb = constexpr_ti_ceil_div(_N, layout::BK);
    static constexpr uint32_t stride_n_sfb = shape_k_scales;
    static constexpr uint32_t stride_k_sfb = 1;

    __device__ static void setup(consumer_setup_args<layout> args) {
      warpgroup::increase_registers<232>();
      warp::zero(args.state.accum);

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

        warp::zero(args.state.per_k_accum);

        warpgroup::mma_ABt(args.state.per_k_accum,
                           args.input.a[warpgroup::groupid()], args.input.b);
        const auto previous_group_offset = get_global_idx<true>(
            shape_k_scales * shape_n_sfb, 0, args, args.common.m_block_idx);
        const uint32_t scale_b_offset =
            previous_group_offset +
            (((args.common.coord.y * layout::BN) / layout::BK) * stride_n_sfb) +
            (args.iter * stride_k_sfb);
        float *local_sfb = args.globals.scale_b.raw_ptr + scale_b_offset;
        float b_scale_0;
        float b_scale_1;

        move<float>::ldg(b_scale_0, local_sfb);
        if constexpr (!layout::kIsUniformScales) {
          if (args.state.num_full_iters > args.state.num_former_iters) {
            move<float>::ldg(b_scale_1, local_sfb + stride_n_sfb);
          }
        }

        // once WGMMA is completed, apply scale promotion from FP22 -> FP32

        // col_vec<rt<float, 16, layout::BN>> scale_a;
        // WGMMA instructions always have that the register tile is row major,
        // so the scale_a is orthogonal
        typename decltype(args.state.per_k_accum)::col_vec scale_a_rv;
        warpgroup::load(scale_a_rv, args.input.sfa[warpgroup::groupid()]);
        warpgroup::mma_async_wait();

        // if constexpr (DEBUG) {
        //   if (warpgroup::warpid() == 0 && blockIdx.x == 0 && blockIdx.y == 0
        //   &&
        //       blockIdx.z == 0) {
        //     kittens::print(args.state.per_k_accum);
        //     kittens::print(scale_a_rv);
        //   }
        // }
        warp::mul_row(args.state.per_k_accum, args.state.per_k_accum,
                      scale_a_rv);
        if constexpr (layout::kIsUniformScales) { // single scale case
          args.state.per_k_accum *= b_scale_0;
        } else {
          row_vec<rt<float, 16, layout::BN>> col_scale_b;
#pragma unroll
          for (uint32_t i = 0; i < layout::BN / 16; i++) {

            const uint32_t column_chunk = i * 2;
            float first_scale = column_chunk < args.state.num_former_iters
                                    ? b_scale_0
                                    : b_scale_1;
            float second_scale = column_chunk + 1 < args.state.num_former_iters
                                     ? b_scale_0
                                     : b_scale_1;
            col_scale_b[i][0] = make_float2(first_scale, first_scale);
            col_scale_b[i][1] = make_float2(second_scale, second_scale);
          }
          warp::mul_col(args.state.per_k_accum, args.state.per_k_accum,
                        col_scale_b);
          // if constexpr (DEBUG) {
          //   if (warpgroup::warpid() == 0 && blockIdx.x == 0 &&
          //       blockIdx.y == 0 && blockIdx.z == 0 && args.iter == 0) {
          //     kittens::print(col_scale_b);
          //   }
          //   if (warpgroup::laneid() == 0)
          //     printf("scales : %f, %f", b_scale_0, b_scale_1);
          // }
        }

        args.state.accum += args.state.per_k_accum;
      }
      if (laneid() == 0)
        arrive(args.inputs_finished);
    }
    __device__ static void finish(consumer_finish_args<layout> args) {

      if (args.common.computation_valid) {
        warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
        warpgroup::sync(warpgroup::groupid() + 4);
        if (warpgroup::laneid() == 0) {
          tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                           coord<ducks::default_type>{
                               get_global_idx<isGroupedMasked>(
                                   _M, args.common.coord.x * 64, args),
                               args.common.coord.y * layout::BN});
          tma::store_async_read_wait();
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
} // namespace kernel2
