#pragma once

#include "common/scheduler.cuh"
#include "common/sm90_utils.cuh"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/bfloat16.h"
#include <moe_cuda/kernels/common/common.hpp>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <moe_cuda/types.h>

namespace moe_cuda {
namespace kernels {
namespace sm90_bf16_gemm_impl {

// TODO : implement GemmType::Batched Support as well, and implement the kNumMergeStages (increase BLOCK_K), excessive
// stages can cause slowdowns

template <Major kMajorA, Major kMajorB, uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_, uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleDMode, uint32_t kNumStages_, uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA, uint32_t kNumSMs, GemmType kGemmType,
          bool kWithAccumulation, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
    sm90_bf16_gemm(__grid_constant__ const cute::TmaDescriptor a_tensor_map,
                   __grid_constant__ const cute::TmaDescriptor b_tensor_map,
                   __grid_constant__ const cute::TmaDescriptor d_tensor_map, uint32_t shape_m, uint32_t shape_n,
                   uint32_t shape_k, int *grouped_layout) {

  constexpr uint32_t kMergeStages = kNumStages_ >= 10 && kGemmType == GemmType::Normal && kMajorA == Major::K &&
                                    kMajorB == Major::K && kNumMathThreads == 128;
  shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
  shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
  shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

  using WGMMA = typename BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;

  static_assert(BLOCK_M % WGMMA::M == 0 || BLOCK_M < WGMMA::M);

  const uint32_t num_k_blocks = cute::ceil_div(shape_k, BLOCK_K_);
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x & 0x1f;
  const uint32_t lane_predicate = cute::elect_one_sync();

  extern __align__(1024) __shared__ uint8_t smem[];

  constexpr uint32_t SMEM_A_SIZE_PER_STAGE = sizeof(__nv_bfloat16) * BLOCK_M * BLOCK_K_;
  constexpr uint32_t SMEM_B_SIZE_PER_STAGE = sizeof(__nv_bfloat16) * BLOCK_N * BLOCK_K_;
  constexpr uint32_t SMEM_D_SIZE = constexpr_ti_align(sizeof(cd_dtype_t) * BLOCK_M * BLOCK_N, 1024);

  static constexpr uint32_t WGMMA_M_SIZE_PER_STAGE = WGMMA::M * BLOCK_K_ * sizeof(__nv_bfloat16);

  // for the cases where BLOCK_M << WGMMA::M, what we will actually use could be a lot larger, we just need to make sure
  // the smem size is enough so we don't go out of bounds
  static_assert(WGMMA_M_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages_);

  auto smem_d = reinterpret_cast<cd_dtype_t *>(smem);

  auto smem_a = PatternVisitor([&](const uint32_t &stage_idx) {
    return reinterpret_cast<__nv_bfloat16 *>(smem + SMEM_D_SIZE + SMEM_A_SIZE_PER_STAGE * stage_idx);
  });
  auto smem_b = PatternVisitor([&](const uint32_t &stage_idx) {
    return reinterpret_cast<__nv_bfloat16 *>(smem + SMEM_D_SIZE + SMEM_A_SIZE_PER_STAGE * kNumStages_ +
                                             SMEM_B_SIZE_PER_STAGE * stage_idx);
  });

  uint8_t *smem_barrier_offset = smem + (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) * kNumStages_ + SMEM_D_SIZE;

  using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
  using MathBarrier = cutlass::arch::ClusterBarrier;
  auto math_barriers = PatternVisitor([&](const uint32_t &stage_idx) {
    return reinterpret_cast<MathBarrier *>(smem_barrier_offset + sizeof(MathBarrier) * stage_idx);
  });
  auto tma_barriers = PatternVisitor([&](const uint32_t &stage_idx) {
    return reinterpret_cast<TMABarrier *>(smem_barrier_offset + (kNumStages_ + stage_idx) * sizeof(MathBarrier));
  });

  if (lane_predicate) {
    cute::prefetch_tma_descriptor(&a_tensor_map);
    cute::prefetch_tma_descriptor(&b_tensor_map);
    cute::prefetch_tma_descriptor(&d_tensor_map);
  }
  __syncwarp();
// barrier initialization with care for multicast
#pragma unroll
  for (int stage = 0; stage < kNumStages_; stage++) {
    if (warp_id == 0 && lane_predicate) {
      tma_barriers[stage]->init(1);
      math_barriers[stage]->init(kNumTMAMulticast * kNumMathThreads / 32);
      // arrive count for consumers is set up so that each warp issues one
    }
  }
  cutlass::arch::fence_barrier_init();
  kNumTMAMulticast > 1 ? cute::cluster_sync() : __syncthreads();

  // setting up TMA and Math Warpgroups
  uint32_t m_block_idx, n_block_idx;
  auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(
      shape_m, shape_n, shape_k, grouped_layout);

  int stage_idx = 0, phase = 0;

  auto advance_pipeline = [&](int &k_block_idx) {
    k_block_idx++;
    stage_idx = stage_idx == kNumStages_ - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };

  constexpr uint32_t kNumTMARegisters = 48;
  constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

  if (warp_id >= kNumMathThreads / 32) {
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      if (warp_id == kNumMathThreads / 32 + 2 && lane_predicate) {
        const bool is_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
        int numAMulticast = (kIsTMAMulticastOnA && is_multicast_valid) ? kNumTMAMulticast : 1;
        int numBMulticast = (!kIsTMAMulticastOnA && is_multicast_valid) ? kNumTMAMulticast : 1;
        // int numAMulticast = 1;
        // int numBMulticast = 1;
        for (int k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
          int write_index = stage_idx;
          math_barriers[write_index]->wait(phase ^ 1);
          // when we have MGroupedMasked, we need to use a group offset because
          // there are padding rows that grouped_layout does not contain
          // information on, but with MGroupedContiguous the rows are contiguous
          constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;

          // [perform tma_copies, only through one thread in the cta
          auto &barrier = *tma_barriers[write_index];

          // if (blockIdx.x == 0) {
          //   printf("m_block_idx : %d, n_block_idx : %d\n", m_block_idx, n_block_idx);
          // }

          // if the gemm type is batched or masked, then the A descriptor will be 3D. We don't use 3D descriptors for
          // the B tensor because it is usually a weight tensor. In grouped gemms, the scheduler should be able to
          // handle most edge cases
          auto m_idx = scheduler.template get_global_idx<false>(shape_m, m_block_idx, BLOCK_M);

          // only when kMajorB, should we stride advance. if kMajorB == Major::MN, we're doing Normal GEMM
          // only when it is normal can we have a non k major

          auto k_a_idx =
              scheduler.template get_global_idx<kMajorA == Major::MN>(shape_k, k_block_idx, BLOCK_K_, m_block_idx);
          auto k_b_idx =
              scheduler.template get_global_idx<kMajorB == Major::MN>(shape_k, k_block_idx, BLOCK_K_, m_block_idx);
          static_assert(kGemmType == GemmType::Normal || kMajorB == Major::K);
          if constexpr (kGemmType == GemmType::MGroupedMasked || kGemmType == GemmType::Batched) {
            // only MajorK tma copies for these gemm types
            tma_copy<BLOCK_K_, BLOCK_M, kSwizzleAMode, __nv_bfloat16, 3>(
                &a_tensor_map, &barrier, smem_a[stage_idx], k_a_idx, m_idx, numAMulticast, scheduler.cur_group_idx);
          } else {
            if constexpr (kMajorA == Major::K) {
              tma_copy<BLOCK_K_, BLOCK_M, kSwizzleAMode>(&a_tensor_map, &barrier, smem_a[stage_idx], k_a_idx, m_idx,
                                                         numAMulticast);
            } else if constexpr (kMajorA == Major::MN) {
              tma_copy<BLOCK_M, BLOCK_K_, kSwizzleAMode>(&a_tensor_map, &barrier, smem_a[stage_idx], m_idx, k_a_idx,
                                                         numAMulticast);
            }
          }

          if constexpr (kGemmType == GemmType::MGroupedMasked) {
            // for mgroupedmasked, in some edges the BLOCK_N will not be divisible by the global N, which causes
            // incorrect weight loads
            // only Major::K
            auto n_idx = scheduler.template get_global_idx<false>(shape_n, n_block_idx, BLOCK_N);
            tma_copy<BLOCK_K_, BLOCK_N, kSwizzleBMode, __nv_bfloat16, 3>(
                &b_tensor_map, &barrier, smem_b[stage_idx], k_b_idx, n_idx, numBMulticast, scheduler.cur_group_idx);
          } else {
            auto n_idx = scheduler.template get_global_idx<true>(shape_n, n_block_idx, BLOCK_N, m_block_idx);
            if constexpr (kMajorB == Major::K) {
              tma_copy<BLOCK_K_, BLOCK_N, kSwizzleBMode>(&b_tensor_map, &barrier, smem_b[stage_idx], k_b_idx, n_idx,
                                                         numBMulticast);
            } else if constexpr (kMajorB == Major::MN) {
              tma_copy<BLOCK_N, BLOCK_K_, kSwizzleBMode>(&b_tensor_map, &barrier, smem_b[stage_idx], n_idx, k_b_idx,
                                                         numBMulticast);
            }
          }

          // to break down the k_global_idx a little more (I needed to):
          // per group (or batch) there is an array of

          tma_barriers[write_index]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
        }
      }
    }
    if constexpr (kNumTMAMulticast > 1) {
      if (warp_id == kNumMathThreads / 32 + 2 && lane_predicate) {
        for (int i = 0; i < kNumStages_; advance_pipeline(i)) {
          math_barriers[stage_idx]->wait(phase ^ 1);
        }
      }
    }
  } else {

    auto empty_barrier_arrive = [&]() {
      if constexpr (kNumTMAMulticast == 1) {
        lane_id == 0 ? math_barriers[stage_idx]->arrive() : void();
      } else {
        auto peer_cta = scheduler.is_peer_cta_alive ? lane_id : cute::block_rank_in_cluster();
        // per warp, issue multicast predicated for multicast size
        lane_id < kNumTMAMulticast ? math_barriers[stage_idx]->arrive(peer_cta) : void();
      }
    };
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    const int math_wg_idx = __shfl_sync(uint32_t(-1), threadIdx.x >> 7u, 0);
    const auto r_0 = warp_id * 16 + lane_id / 4, r_1 = r_0 + 8;
    auto desc_a = create_wgmma_desc<Major::K, BLOCK_M, BLOCK_K_, kSwizzleAMode, __nv_bfloat16>(
        smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K_);
    auto desc_b = create_wgmma_desc<Major::K, BLOCK_N, BLOCK_K_, kSwizzleBMode, __nv_bfloat16>(smem_b[0]);

    // again unified register trick, but why do we need the lo?
    const uint32_t a_desc_lo = __shfl_sync(uint32_t(-1), desc_a.reg32_[0], 0);
    const uint32_t b_desc_lo = __shfl_sync(uint32_t(-1), desc_b.reg32_[0], 0);
    constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M <= WGMMA::M) ? BLOCK_M : WGMMA::M * 2;
    constexpr uint32_t NUM_WAVES = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : kSwizzleDMode / sizeof(cd_dtype_t);
    const uint32_t wg_block_offset = math_wg_idx * WGMMA::M;
    // only for MN-major
    constexpr uint32_t TMA_A_BLOCK_M = kSwizzleAMode / sizeof(__nv_bfloat16);
    if constexpr (kMajorA == Major::MN) {
      static_assert(WAVE_BLOCK_M % TMA_A_BLOCK_M == 0);
    }
    constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;

    static_assert(TMA_D_BLOCK_N % 8 == 0, "unaligned tma stores for wgmma accumulator layout");
    static_assert(BLOCK_N / TMA_D_BLOCK_N <= 32, "too many tma stores");

    constexpr uint32_t kNumWGMMAStoreThreads = (BLOCK_M / WAVE_BLOCK_M) * kNumMathThreads;
    bool do_wgmma_store = BLOCK_M >= 64 || (warp_id < kNumWGMMAStoreThreads / 32);
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

      if (scheduler.is_computation_valid(m_block_idx, wg_block_offset)) {

#pragma unroll 8
        for (int k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
          tma_barriers[stage_idx]->wait(phase);

          const auto &smem_a_desc_lo = a_desc_lo + ((stage_idx * SMEM_A_SIZE_PER_STAGE) >> 4u);
          const auto &smem_b_desc_lo = b_desc_lo + ((stage_idx * SMEM_B_SIZE_PER_STAGE) >> 4u);
#pragma unroll
          for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); i++) {
            cutlass::warpgroup_fence_operand(accum[i]);
          }
          cute::warpgroup_arrive();
#pragma unroll
          for (uint32_t wave = 0; wave < NUM_WAVES; wave++) {
            float *shifted_accum = accum + wave * WGMMA::kNumAccum;
            const uint32_t m_offset = wave * WAVE_BLOCK_M;

#pragma unroll
            for (int k = 0; k < BLOCK_K_ / WGMMA::K; k++) {
              if constexpr (kMajorA == Major::MN) {
                const uint32_t swizzle_atom_offset =
                    m_offset / TMA_A_BLOCK_M; // when MN-major, this will tell how many atoms to advance in MN direction
                desc_a.reg32_[0] =
                    smem_a_desc_lo +
                    (((swizzle_atom_offset * 8 * kSwizzleAMode +
                       k * WGMMA::K *
                           get_gmma_desc_stride_k<kMajorA, BLOCK_M, BLOCK_K_, kSwizzleAMode, __nv_bfloat16>() *
                           sizeof(__nv_bfloat16))) >>
                     4u);
              } else {
                desc_a.reg32_[0] =
                    smem_a_desc_lo +
                    (((m_offset * BLOCK_K_ +
                       k * WGMMA::K *
                           get_gmma_desc_stride_k<kMajorA, BLOCK_M, BLOCK_K_, kSwizzleAMode, __nv_bfloat16>()) *
                      sizeof(__nv_bfloat16)) >>
                     4u);
              }
              desc_b.reg32_[0] =
                  smem_b_desc_lo +
                  (((k * WGMMA::K *
                     get_gmma_desc_stride_k<kMajorB, BLOCK_N, BLOCK_K_, kSwizzleBMode, __nv_bfloat16>()) *
                    sizeof(__nv_bfloat16)) >>
                   4u);

              WGMMA::wgmma(desc_a, desc_b, shifted_accum, 1);
            }
          }
          cute::warpgroup_commit_batch();
#pragma unroll
          for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); i++) {
            cutlass::warpgroup_fence_operand(accum[i]);
          }
          cute::warpgroup_wait<0>();
          empty_barrier_arrive();
        }
      } else {
        for (int k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
          tma_barriers[stage_idx]->wait(phase);
          empty_barrier_arrive();
        }
      }

      if (!do_wgmma_store) {
        continue;
      }
      if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
        cute::tma_store_wait<0>();
      }

      if constexpr (cute::is_same_v<cd_dtype_t, __nv_bfloat16>) {
        // stsm packed load here
        for (uint32_t wave = 0; wave < NUM_WAVES; wave++) {
          float *shifted_accum = accum + wave * WGMMA::kNumAccum;
          const uint32_t m_offset = wave * WAVE_BLOCK_M;
          auto smem_ptr = smem_d + m_offset * BLOCK_N;
          cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);
          store_accum_to_swizzled_smem<BLOCK_N, BLOCK_M, kSwizzleDMode, WGMMA_M_PER_WARP>(shifted_accum, smem_ptr,
                                                                                          warp_id, lane_id);
        }

      } else { // float type has no swizzle, we use st_shared
#pragma unroll
        for (uint32_t wave = 0; wave < NUM_WAVES; wave++) {
          float *shifted_accum = accum + wave * WGMMA::kNumAccum;
          const uint32_t offset = wave * WAVE_BLOCK_M;
#pragma unroll
          for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; i++) {

            float2 *smem_0 = reinterpret_cast<float2 *>(
                smem_d + (offset + warp_id * WGMMA_M_PER_WARP + lane_id / 4) * BLOCK_N + i * 8 + ((lane_id % 4) * 2));
            float2 *smem_1 =
                reinterpret_cast<float2 *>(smem_d + (offset + warp_id * WGMMA_M_PER_WARP + lane_id / 4 + 8) * BLOCK_N +
                                           i * 8 + ((lane_id % 4) * 2));

            st_shared(smem_0, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
            st_shared(smem_1, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
          }
        }
        cute::tma_store_fence();
      }

      cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

      if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
        auto n_block_offset = threadIdx.x * TMA_D_BLOCK_N;
        if constexpr (kGemmType == GemmType::Batched || kGemmType == GemmType::MGroupedMasked) {
          using cute_tma_t =
              cute::conditional_t<kWithAccumulation, cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
          cute_tma_t::copy(&d_tensor_map, smem_d + BLOCK_M * n_block_offset, n_block_idx * BLOCK_N + n_block_offset,
                           scheduler.template get_global_idx<false>(shape_m, m_block_idx, BLOCK_M),
                           scheduler.cur_group_idx);
          cute::tma_store_arrive();
        } else {
          using cute_tma_t =
              cute::conditional_t<kWithAccumulation, cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
          cute_tma_t::copy(&d_tensor_map, smem_d + BLOCK_M * n_block_offset, n_block_idx * BLOCK_N + n_block_offset,
                           scheduler.template get_global_idx<false>(shape_m, m_block_idx, BLOCK_M));
          cute::tma_store_arrive();
        }
      }
      __syncwarp();
      // tma store
    }
  }
}

} // namespace sm90_bf16_gemm_impl
} // namespace kernels
} // namespace moe_cuda
