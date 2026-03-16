/*
The kernels in this file were based off of the DeepGEMM FP8 SM90 kernels.
However, in order for me to still learn instead of blindly transferring these
kernels over, I'm going to write them in full CutLASS, which DeepGEMM chooses
not to do.
*/
#pragma once
#include "common/scheduler.cuh"
#include "common/sm90_utils.cuh"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/config.hpp"
#include "cute/pointer.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

namespace moe_cuda {
namespace kernels {
namespace sm90_fp8_gemm_impl {

#ifndef KERNEL2_TRACE_REF
#define KERNEL2_TRACE_REF 0
#endif

#ifndef KERNEL2_TRACE_REF_CTA_LIMIT
#define KERNEL2_TRACE_REF_CTA_LIMIT 8
#endif

#ifndef KERNEL2_TRACE_REF_ITER_LIMIT
#define KERNEL2_TRACE_REF_ITER_LIMIT 2
#endif

using namespace cute;

// template-chained search for the dynamic runtime variable kNumFormerIters,
// when the search space is small enough for compile-time
template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd,
          typename func_t>
void dispatch_num_former_iters(uint32_t num_former_iters, const func_t &func) {
  if (num_former_iters == kNumFormerIters) {
    func(cute::Int<kNumFormerIters>{});
    return;
  }

  if constexpr (kNumFormerIters + kGap <= kEnd)
    dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(
        num_former_iters, func);
}

// template <class SA_layout, class SFA_layout, class SB_layout, class
// SFB_Layout, class SD_layout, uint32_t kStages> struct SharedStorage {

//   cute::array_aligned<__nv_fp8_e4m3, cute::cosize_v<SA_layout>, 128> sA;
//   cute::array_aligned<float, cute::cosize_v<SFA_layout>, 128> sSFA; // (1,
//   bM)
//   // sfB will be put directly into registers since it is much smaller
//   cute::array_aligned<__nv_fp8_e4m3, cute::cosize_v<SB_layout>, 128> sB;
//   cute::array_aligned<float, cute::cosize_v<SFB_Layout>, 128> sSFB;
//   cute::array_aligned<__nv_bfloat16, cute::cosize_v<SD_layout>, 1024> sD;
//   alignas(16) uint64_t tma_barriers[kStages];
//   alignas(16) uint64_t mma_barriers[kStages];
// };

// both fp8, inspired by deepgemm
template <uint32_t kNumGroups, int SHAPE_M, int SHAPE_N, int SHAPE_K,
          uint32_t bM, uint32_t bN, uint32_t bK, Major kSFBMajor,
          uint32_t kNumMMAThreads, uint32_t kNumTMAThreads,
          bool kIsTMAMulticastA, uint32_t kNumTMAMulticast, GemmType kGemmType,
          uint32_t kNumSMs, uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleDMode, uint32_t kNumStages, bool kIsBatched>
__global__
__launch_bounds__(kNumMMAThreads + kNumTMAThreads, 1) void sm90_fp8_gemm_1d2d(
    uint32_t M, uint32_t N, uint32_t K,
    CUTE_GRID_CONSTANT const cute::TmaDescriptor a_tensor_map,
    CUTE_GRID_CONSTANT const cute::TmaDescriptor sfa_tensor_map,
    CUTE_GRID_CONSTANT const cute::TmaDescriptor b_tensor_map,
    CUTE_GRID_CONSTANT const cute::TmaDescriptor d_tensor_map, float *sfb,
    int *grouped_layout) {

  // set MNK to compiled dims if possible - this is good for repetitive linear
  // weights, where N and K are fixed
  M = SHAPE_M != 0 ? SHAPE_M : M;
  N = SHAPE_N != 0 ? SHAPE_N : N;
  K = SHAPE_K != 0 ? SHAPE_K : K;

  // shapes
  using WGMMA = typename FP8MMASelector<bN>::type;
  // important part of the kernel is that it will only support up to two
  // scale_b_* values per tile
  CUTE_STATIC_ASSERT(
      cute::ceil_div(bK, bN) == 1 || cute::ceil_div(bN, bK) <= 2,
      "Current FP8 GEMM kernel only supports up to two different b scales");
  CUTE_STATIC_ASSERT(
      bK == 128, "fp8_gemm: Only per-channel 128 block scaling is supported");
  CUTE_STATIC_ASSERT(bM % WGMMA::M == 0,
                     "fp8_gemm: Invalid bM config for FP8 GEMM");

  CUTE_STATIC_ASSERT(
      kNumTMAMulticast <= 2,
      "Fp8_gemm kernel only supports up to 2 Multicast size for scheduler");

  constexpr bool kIsUniformBScales = (bK % bN == 0);
  const uint32_t shape_k_scales = cute::ceil_div(K, bK);
  const uint32_t &shape_n_sfb = cute::ceil_div(N, 128);
  // (bM, bK) is 1d scaled, (bN, bK) is 2D (128, 128) block scaled
  // write setup
  using ConsumerBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

  // shared memory, aligned to 1024 bytes in order to set base-offset to 0
  // inside matrix-descriptor
  extern __shared__ __align__(1024) char sMem[];

  // we want to ensure 1024, since it is automatically divisible by 512 and 256
  constexpr uint32_t smem_d_size =
      constexpr_ti_align(bM * bN * sizeof(__nv_bfloat16), 1024);
  constexpr uint32_t WGMMA_A_SIZE_PER_STAGE =
      WGMMA::M * bK * sizeof(__nv_fp8_e4m3);
  constexpr uint32_t A_SIZE_PER_STAGE =
      bM * bK * sizeof(__nv_fp8_e4m3); // already aligned to 128
  constexpr uint32_t B_SIZE_PER_STAGE = bN * bK * sizeof(__nv_fp8_e4m3);
  constexpr uint32_t SFA_SIZE_PER_STAGE =
      constexpr_ti_align(bM * sizeof(float), 128); // align to 128
  const uint32_t SFB_SIZE_PER_STAGE =
      ti_align(shape_k_scales * (kIsUniformBScales ? 1 : 2) * sizeof(float),
               16); // need to align for barrier next
  CUTE_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <=
                         A_SIZE_PER_STAGE + B_SIZE_PER_STAGE,
                     "WGMMA A size in shared memory is too large");

  __nv_bfloat16 *sD = reinterpret_cast<__nv_bfloat16 *>(&sMem);
  __nv_fp8_e4m3 *sA = reinterpret_cast<__nv_fp8_e4m3 *>(sMem + smem_d_size);
  __nv_fp8_e4m3 *sB = reinterpret_cast<__nv_fp8_e4m3 *>(
      sMem + smem_d_size + A_SIZE_PER_STAGE * kNumStages);
  constexpr uint32_t SF_OFFSET =
      smem_d_size + (A_SIZE_PER_STAGE + B_SIZE_PER_STAGE) * kNumStages;
  float *sSFA = reinterpret_cast<float *>(sMem + SF_OFFSET);
  float *sSFB = reinterpret_cast<float *>(sMem + SF_OFFSET +
                                          (SFA_SIZE_PER_STAGE)*kNumStages);
  const uint32_t barrier_offset =
      smem_d_size + (A_SIZE_PER_STAGE + B_SIZE_PER_STAGE + SFA_SIZE_PER_STAGE +
                     SFB_SIZE_PER_STAGE) *
                        kNumStages;
  ProducerBarrier *tma_barriers =
      reinterpret_cast<ProducerBarrier *>(sMem + barrier_offset);
  ConsumerBarrier *mma_barriers = reinterpret_cast<ConsumerBarrier *>(
      sMem + barrier_offset + kNumStages * sizeof(ProducerBarrier));
  // barriers
  // configs
  const uint32_t num_k_blocks = cute::ceil_div(K, bK);
  const uint32_t warpIdx = threadIdx.x / 32;
  const uint32_t lane_idx = threadIdx.x & 0x1f;
  const uint32_t lane_predicate = cute::elect_one_sync();

  int stage_idx = 0, phase = 0;

  auto advance_pipeline = [&](int &k_block_idx) {
    k_block_idx++;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };

  if (lane_predicate) {
    cute::prefetch_tma_descriptor(&a_tensor_map);
    cute::prefetch_tma_descriptor(&b_tensor_map);
    cute::prefetch_tma_descriptor(&d_tensor_map);
    cute::prefetch_tma_descriptor(&sfa_tensor_map);
  }
  __syncwarp();
// barrier initialization with care for multicast
#pragma unroll
  for (int stage = 0; stage < kNumStages; stage++) {
    if (warpIdx == 0 && lane_predicate) {
      tma_barriers[stage].init(1);
      mma_barriers[stage].init(kNumTMAMulticast * kNumMMAThreads / 32);
      // arrive count for consumers is set up so that each warp issues one
    }
  }
  cutlass::arch::fence_barrier_init();
  kNumTMAMulticast > 1 ? cute::cluster_sync() : __syncthreads();

  // setting up TMA and Math Warpgroups
  uint32_t m_block_idx, n_block_idx;
  auto scheduler =
      Scheduler<kGemmType, bM, bN, kNumGroups, kNumTMAMulticast,
                kIsTMAMulticastA, kNumSMs>(M, N, K, grouped_layout);
  const uint32_t stride_n_sfb = kSFBMajor == Major::K ? shape_k_scales : 1;
  const uint32_t stride_k_sfb = kSFBMajor == Major::K ? 1 : shape_n_sfb;

  // set up number of registers here:
  constexpr uint32_t kNumTMARegisters = 40;
  constexpr uint32_t kNumMMARegisters = kNumMMAThreads == 128 ? 248 : 232;
  constexpr uint64_t kTransactionBytes =
      sizeof(__nv_fp8_e4m3) * (bM * bK + bN * bK) + sizeof(float) * bM;
  // constexpr uint64_t kTransactionBytes = 0;
  // assign warpgroups here
  auto empty_barrier_arrive = [&]() {
    if constexpr (kNumTMAMulticast == 1) {
      lane_idx == 0 ? mma_barriers[stage_idx].arrive() : void();
    } else {
      auto peer_cta = scheduler.is_peer_cta_alive
                          ? lane_idx
                          : cute::block_rank_in_cluster();
      // per warp, issue multicast predicated for multicast size
      lane_idx < kNumTMAMulticast ? mma_barriers[stage_idx].arrive(peer_cta)
                                  : void();
    }
  };
  // since TMA issues are not a compute intensive operation, they do not need as
  // many warpgroups, maybe 1
  if (threadIdx.x >= kNumMMAThreads) {
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      if (warpIdx == kNumMMAThreads / 32 + 2 && lane_predicate) {
        const bool is_multicast_valid =
            scheduler.is_tma_multicast_valid(m_block_idx);
        int numAMulticast =
            (kIsTMAMulticastA && is_multicast_valid) ? kNumTMAMulticast : 1;
        int numBMulticast =
            (!kIsTMAMulticastA && is_multicast_valid) ? kNumTMAMulticast : 1;
        // int numAMulticast = 1;
        // int numBMulticast = 1;
        for (int k_block_idx = 0; k_block_idx < num_k_blocks;
             advance_pipeline(k_block_idx)) {
          int write_index = stage_idx;
          mma_barriers[write_index].wait(phase ^ 1);
          // when we have MGroupedMasked, we need to use a group offset because
          // there are padding rows that grouped_layout does not contain
          // information on, but with MGroupedContiguous the rows are contiguous
          constexpr bool kWithGroupOffsetA = false;
          // [perform tma_copies, only through one thread in the cta
          __nv_fp8_e4m3 *cur_stage_sA = sA + write_index * bM * bK;
          __nv_fp8_e4m3 *cur_stage_sB = sB + write_index * bN * bK;
          float *cur_stage_sSFA = sSFA + write_index * bM;
          auto &barrier = tma_barriers[write_index];
          const uint32_t b_global_idx =
              kGemmType == GemmType::MGroupedMasked
                  ? scheduler.template get_global_idx<false>(N, n_block_idx, bN)
                  : scheduler.template get_global_idx<true>(N, n_block_idx, bN,
                                                            m_block_idx);
          const uint32_t d_global_idx =
              scheduler.template get_global_idx<false>(M, m_block_idx, bM);
          const uint32_t previous_group_offset =
              scheduler.template get_global_idx<true>(
                  shape_k_scales * shape_n_sfb, 0, 0, m_block_idx);
          const uint32_t scale_b_offset =
              previous_group_offset + (n_block_idx * bN / 128) * stride_n_sfb +
              k_block_idx * stride_k_sfb;

          // if (blockIdx.x == 0) {
          //   printf("m_block_idx : %d, n_block_idx : %d\n", m_block_idx,
          //   n_block_idx);
          // }

          // TODO: write the same 3d TMA descriptor handling that we did for b16
          // temm
          if constexpr (kGemmType == GemmType::MGroupedMasked) {
            tma_copy<bK, bM, kSwizzleAMode, __nv_fp8_e4m3, 3>(
                &a_tensor_map, &barrier, cur_stage_sA, k_block_idx * bK,
                scheduler.template get_global_idx<kWithGroupOffsetA>(
                    M, m_block_idx, bM),
                numAMulticast, scheduler.cur_group_idx);

            // sfa can still be 2d, since it's MN-major
            tma_copy<bM, bK, 0>(&sfa_tensor_map, &barrier, cur_stage_sSFA,
                                m_block_idx * bM,
                                scheduler.template get_global_idx<true>(
                                    shape_k_scales, k_block_idx, 1),
                                numAMulticast);
            // to break down the k_global_idx a little more (I needed to):
            // per group (or batch) there is an array of

            tma_copy<bK, bN, kSwizzleBMode, __nv_fp8_e4m3, 3>(
                &b_tensor_map, &barrier, cur_stage_sB, k_block_idx * bK,
                scheduler.template get_global_idx<false>(N, n_block_idx, bN),
                numBMulticast, scheduler.cur_group_idx);
          } else {
            tma_copy<bK, bM, kSwizzleAMode>(
                &a_tensor_map, &barrier, cur_stage_sA, k_block_idx * bK,
                scheduler.template get_global_idx<kWithGroupOffsetA>(
                    M, m_block_idx, bM),
                numAMulticast);

            tma_copy<bM, bK, 0>(
                &sfa_tensor_map, &barrier, cur_stage_sSFA, m_block_idx * bM,
                scheduler.template get_global_idx<kWithGroupOffsetA>(
                    shape_k_scales, k_block_idx, 1),
                numAMulticast);
            // to break down the k_global_idx a little more (I needed to):
            // per group (or batch) there is an array of

            tma_copy<bK, bN, kSwizzleBMode>(
                &b_tensor_map, &barrier, cur_stage_sB, k_block_idx * bK,
                scheduler.template get_global_idx<true>(N, n_block_idx, bN,
                                                        m_block_idx),
                numBMulticast);
          }
          barrier.arrive_and_expect_tx(kTransactionBytes);
        }
      }
      // after the while loop / TMA issues are finished, we need to drain the
      // pipeline state through catching any later ConsumerBarrier arrives
    }
    if constexpr (kNumTMAMulticast > 1) {
      // Only the TMA leader warp should drain — other TMA warps never
      // called advance_pipeline(), so their stage_idx/phase are stale
      // and would deadlock on barriers in the wrong phase.
      if (warpIdx == kNumMMAThreads / 32 + 2 && lane_predicate) {
        for (int i = 0; i < kNumStages; advance_pipeline(i)) {
          mma_barriers[stage_idx].wait(phase ^ 1);
        }
      }
    }
  }

  else {
    // WGMMA warpgroups path
    cutlass::arch::warpgroup_reg_alloc<kNumMMARegisters>();
    // cool register allocation optimization trick - by performing a
    // __shfl_sync, this tells NVCC to use a unified register (single) per warp
    // group
    const int math_wg_idx = __shfl_sync(uint32_t(-1), threadIdx.x / 128, 0);
    // from deepGEMM: in the normal WGMMA tile of M = 64, each warp in a WG is
    // responsible for 64 / 4 = 16 rows now inside a warp, we need to assign
    // these values threads. A naive way would be lane_idx / 2, so that each
    // group of two threads in the warp takes a scale factor. However this means
    // 16 loads per ld_shared line, clogging smem bandwidth instead, if we first
    // split the warp into 32 / 4 = 8 thread-groups (lane_idx / 4), and overload
    // so that a thread will load in two values, the intended 16 row requirement
    // is matched
    const auto r_0 = warpIdx * 16 + lane_idx / 4, r_1 = r_0 + 8;
    auto desc_a =
        create_wgmma_desc<Major::K, bM, bK, kSwizzleAMode, __nv_fp8_e4m3>(
            sA + math_wg_idx * WGMMA::M * bK);
    auto desc_b =
        create_wgmma_desc<Major::K, bN, bK, kSwizzleBMode, __nv_fp8_e4m3>(sB);

    // again unified register trick, but why do we need the lo?
    const uint32_t a_desc_lo = __shfl_sync(uint32_t(-1), desc_a.reg32_[0], 0);
    const uint32_t b_desc_lo = __shfl_sync(uint32_t(-1), desc_b.reg32_[0], 0);
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      // need to first load in scale b, which needs to know how many scales (1
      // or 2) to load for this iteration

      // if (cute::thread0()) {
      //   printf("m block idx : %d, n block idx : %d, group_idx : %d",
      //   m_block_idx, n_block_idx, scheduler.cur_group_idx);
      // }

      uint32_t num_former_iters = bN / 8;
      uint32_t num_full_iters = num_former_iters;

      if constexpr (!kIsUniformBScales) {
        // ^ remember the above condition checks whether only always one scale
        // factor is needed per BN when this is false, we need to first
        // calculate the number of rows that will use the current scale bK -
        // (cur_n_row % bK) = remaining n-rows in current block that use
        // previous scale
        num_former_iters = min(bN, bK - (n_block_idx * bN) % bK) / 8;
        num_full_iters = min(bN, N - n_block_idx * bN) / 8;
      }
      // if the number of remaining rows is greater than the number of remaining
      // rows for the FIRST scale factor, then we need multiple scales

      // number of scale_factors in a bK * 2 scale or 1
      uint32_t num_sfb =
          shape_k_scales * (num_full_iters > num_former_iters ? 2 : 1);

      // Load in B-scales - DeepGEMM chooses to overlap this with the TMA stores
      // for each warp in MathGroups besides the first one

      if (threadIdx.x >= 32) {
        // use this to offset sfb pointer to current group position
        auto previous_group_offset = scheduler.template get_global_idx<true>(
            shape_k_scales * shape_n_sfb, 0, 0, m_block_idx);
        auto local_sfb = sfb + previous_group_offset +
                         (n_block_idx * bN / 128) * stride_n_sfb;
        // grid-stride loop inside kNumMMAThreads - 32 to load in the
        // scale factors

#pragma unroll
        for (int i = threadIdx.x - 32; i < (int)num_sfb;
             i += kNumMMAThreads - 32) {
          st_shared(sSFB + i,
                    __ldg(i < shape_k_scales
                              ? local_sfb + stride_k_sfb * i
                              : local_sfb + stride_n_sfb +
                                    (i - shape_k_scales) * stride_k_sfb));
        }
        // the i < shape_k_scales is simply checking for if 2 or 1 b scales are
        // being loaded in, and then in the case with two, hop to the next row
      }
      cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);

      // CUDA core wave accumulations unit
      constexpr uint32_t WAVE_BLOCK_M = bM <= WGMMA::M ? bM : WGMMA::M * 2;

      // they force another loop inside the k-loop to account for intermediary
      // promotions, if the bM isn't large enough they keep as the entire
      // current CTA - when it's larger than the atom's M dim, they increase it
      // to M * 2, probably for more coverage - the ternary is more for the
      // second case, to enforce some accumulation on really large tiles

      CUTE_STATIC_ASSERT(bM % WGMMA::M == 0,
                         "fp8_gemm.cu: bM should be divisble by WGMMA::M for "
                         "accumulations, in order to proper wave loops");
      CUTE_STATIC_ASSERT(bM >= 64 || kNumMMAThreads == 128,
                         "fp8_gemm.cu: if bM <= 64 (WGMMA::M) atom size, only "
                         "one math WG should be "
                         "used.");
      float final_accum[WGMMA::kNumAccum * bM / WAVE_BLOCK_M] = {0};

      // out of the MATH_WG * 128 threads used, describes which ones shold be
      // storing this is used when bM < WGMMA::M, since not all the
      // threads should be storing then like when bM = 32, WGMMA::M = 64, then
      // the below value is 64
      static_assert(bM >= 64 || kNumMMAThreads == 128,
                    "Only one math warp group for `BLOCK_M < 64`");
      constexpr uint32_t kNumWGMMAStoreThreads =
          WAVE_BLOCK_M * (128 / WGMMA::M);
      const bool do_wgmma_store =
          bM >= WGMMA::M || warpIdx < kNumWGMMAStoreThreads / 32;
      const uint32_t wg_block_offset = math_wg_idx * WGMMA::M;
      const bool is_valid =
          scheduler.is_computation_valid(m_block_idx, wg_block_offset);
      const uint32_t b_global_idx =
          kGemmType == GemmType::MGroupedMasked
              ? scheduler.template get_global_idx<false>(N, n_block_idx, bN)
              : scheduler.template get_global_idx<true>(N, n_block_idx, bN,
                                                        m_block_idx);
      const uint32_t previous_group_offset =
          scheduler.template get_global_idx<true>(shape_k_scales * shape_n_sfb,
                                                  0, 0, m_block_idx);
      const uint32_t scale_b_offset =
          previous_group_offset + (n_block_idx * bN / 128) * stride_n_sfb;
      const uint32_t d_global_idx =
          scheduler.template get_global_idx<false>(M, m_block_idx, bM);

      // send an intiial mmabarrier signal to begin tma copies

      // block_offset - in the math_wg_idx >= 1 problem size, what row
      // usually matters when bM < WGMMA::M, but this just says if the current
      // WGMMA operation
      if (is_valid) {
        // first check is to see if num_former_iters even != num_full_iters
        // then it checks the cycle length of (n_idx * bN) % bK = gcd * (n_idx *
        // bN') % bK' so the cycle length goes from bK -> bK', where bK' = bK /
        // gcd. If it is less than 4, it is acceptable to find in compile-time
        const bool kShouldOptimize =
            !kIsUniformBScales && bK / cute::gcd(bK, bN) <= 4;
        const uint32_t kGap = cute::gcd(bK, bN) / 8;
        const uint32_t kEnd = kShouldOptimize ? bK / 8 : 0;

        float accum[WGMMA::kNumAccum];
        dispatch_num_former_iters<0, kGap, kEnd>(
            kShouldOptimize ? num_former_iters : 0, [&](int _) {
#pragma unroll 8
              for (int k_block_idx = 0; k_block_idx < num_k_blocks;
                   advance_pipeline(k_block_idx)) {

                int read_index = stage_idx;

                // advance smem_addr of a_desc, b_desc to the current stage
                // index
                const auto &a_desc_base_lo =
                    a_desc_lo +
                    read_index *
                        ((bM * bK) / 16); // gmma_desc requires shift by 4
                const auto &b_desc_base_lo =
                    b_desc_lo + read_index * ((bN * bK) / 16);

                // read b scales into rMem, we assume the sSFB is row-major,
                // with no swizzling.
                float b_scale_0 = ld_shared(sSFB + k_block_idx);
                float b_scale_1;
                if constexpr (!kIsUniformBScales) {
                  b_scale_1 = ld_shared(sSFB + k_block_idx + shape_k_scales);
                }

                tma_barriers[read_index].wait(phase);

            // when bM is small, this is one hot loop. when it is large, it is a
            // lot more.
#pragma unroll
                for (uint32_t local_idx = 0; local_idx < bM / WAVE_BLOCK_M;
                     ++local_idx) {
                  auto m_offset = local_idx * WAVE_BLOCK_M;
                  // load in A_scales, but only threads that are storing post
                  // accum should take scale values
                  // sSFA is stored in MN-major, so to advance a k-block, we
                  // have to add the number of sclaes in this block
                  auto cur_k_block_sfa = sSFA + bM * read_index; // shape: (bM)
                  const auto &a_scale_0 =
                      do_wgmma_store
                          ? ld_shared(cur_k_block_sfa + r_0 + m_offset)
                          : 0;
                  const auto &a_scale_1 =
                      do_wgmma_store
                          ? ld_shared(cur_k_block_sfa + r_1 + m_offset)
                          : 0;
              // begin wgmma by committing
              // kNumAccum is the number of times the WG_SIZE divides
              // WGMMA::M*N, so how many times each thread should accumulate
              // before writing out
#pragma unroll
                  for (uint32_t i = 0; i < WGMMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(accum[i]);
                    // asm volatile ("" : "+f(reg)" :: "memory") tells compiler
                    // not to reorder memory around this register, which is read
                    // and written to
                  }
                  cute::warpgroup_arrive();
                  for (int wgmma_k_idx = 0; wgmma_k_idx < bK / WGMMA::K;
                       wgmma_k_idx++) {
                    // move the start_addr of the gmma_desc
                    desc_a.reg32_[0] =
                        a_desc_base_lo +
                        ((wgmma_k_idx * WGMMA::K + m_offset * bK) >> 4);
                    desc_b.reg32_[0] =
                        b_desc_base_lo + ((WGMMA::K * wgmma_k_idx) >> 4);
                    WGMMA::wgmma(desc_a, desc_b, accum, wgmma_k_idx);
                  }
                  cute::warpgroup_commit_batch();
#pragma unroll
                  for (uint32_t i = 0; i < WGMMA::kNumAccum; i++) {
                    cute::warpgroup_fence_operand(accum[i]);
                  }

                  cute::warpgroup_wait<0>();

                  if (local_idx == bM / WAVE_BLOCK_M - 1) {
                    empty_barrier_arrive();
                  }
                  // check if this is actually a valid row / section of shared
                  // memory to write out
                  if (!do_wgmma_store)
                    continue;
                  // grid of (2,1) for uniform scales, (2, 2) for uniform scales
                  float scale_0_0 = a_scale_0 * b_scale_0,
                        scale_1_0 = a_scale_1 * b_scale_0;
                  float scale_0_1, scale_1_1;
                  if constexpr (!kIsUniformBScales) {
                    scale_0_1 = a_scale_0 * b_scale_1,
                    scale_1_1 = a_scale_1 * b_scale_1;
                  }
                  auto shifted_accum =
                      final_accum + local_idx * WGMMA::kNumAccum;
              // loop by 4 since the assembly instruction groups values into 4s,
              // kNumAccum = N / 2

#pragma unroll
                  for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; i++) {
                    const bool &predicate =
                        kIsUniformBScales || i < num_former_iters;
                    // predicate means use the first few scales here
                    shifted_accum[i * 4 + 0] +=
                        (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                    shifted_accum[i * 4 + 1] +=
                        (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                    shifted_accum[i * 4 + 2] +=
                        (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                    shifted_accum[i * 4 + 3] +=
                        (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                  }

                  // if (blockIdx.x == 0 && threadIdx.x == 0) {
                  //   printf(" k block idx : %d, a_scale_0 : %f, a_scale_1 :
                  //   %f, b_scale_1 : %f \n", k_block_idx, a_scale_0,
                  //          a_scale_1, b_scale_0);
                  //   printf("accum 0 : %f, accum 1 : %f", shifted_accum[0],
                  //   shifted_accum[1]);
                  // }
                }
              }
            });
      } else {
#pragma unroll
        for (int k_block_idx = 0; k_block_idx < num_k_blocks;
             advance_pipeline(k_block_idx)) {
          tma_barriers[stage_idx].wait(phase);
          empty_barrier_arrive();
        }
      }

      constexpr uint32_t kNumElemBytes = sizeof(__nv_bfloat16);
      constexpr uint32_t TMA_D_BLOCK_N =
          kSwizzleDMode == 0
              ? bN
              : (kSwizzleDMode /
                 kNumElemBytes); // size in the N-direction of D swizzle atom
      constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;

      CUTE_STATIC_ASSERT(bM % 8 == 0, "Invalid bM size - not divisible by 8 so "
                                      "invalid swizzling atom for K-major");
      CUTE_STATIC_ASSERT(bN % TMA_D_BLOCK_N == 0 && bN / TMA_D_BLOCK_N <= 32,
                         "Unaligned TMA store (bN not aligned with atom size) "
                         "or too many TMA store instructions");
      CUTE_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0,
                         "Invalid TMA Block N"); // need this for the 8 x 8

      if (!do_wgmma_store)
        continue;
      if (threadIdx.x < bN / TMA_D_BLOCK_N)
        cute::tma_store_wait<0>();
      cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

      // this check does nothing for sm90, since atom N size is always divisible
      // by 4
      CUTE_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0,
                         "Invalid STSM x2 vectorization");

#pragma unroll
      for (uint32_t local_idx = 0; local_idx < bM / WAVE_BLOCK_M; local_idx++) {
        auto m_offset = local_idx * WAVE_BLOCK_M;
        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
        __nv_bfloat16 *shifted_smem_ptr = reinterpret_cast<__nv_bfloat16 *>(
            reinterpret_cast<uint8_t *>(sD) + m_offset * kSwizzleDMode);
        store_accum_to_swizzled_smem<bN, bM, kSwizzleDMode, WGMMA_M_PER_WARP>(
            shifted_accum, shifted_smem_ptr, warpIdx, lane_idx);
      }

      // cute::tma_store_fence();
      cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

      constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
      CUTE_STATIC_ASSERT(kNumWGMMAStoreThreads >= bN / TMA_D_BLOCK_N,
                         "Too many TMA blocks");

      if (threadIdx.x < bN / TMA_D_BLOCK_N) {
        auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
        auto smem_ptr = sD + in_block_n_offset * bM;
        if constexpr (kGemmType == GemmType::MGroupedMasked) {
          cute::SM90_TMA_STORE_3D::copy(
              &d_tensor_map, smem_ptr, n_block_idx * bN + in_block_n_offset,
              scheduler.template get_global_idx<false>(M, m_block_idx, bM),
              scheduler.cur_group_idx);
        } else {
          cute::SM90_TMA_STORE_2D::copy(
              &d_tensor_map, smem_ptr, n_block_idx * bN + in_block_n_offset,
              scheduler.template get_global_idx<false>(M, m_block_idx, bM));
        }
        cute::tma_store_arrive();
      }
      __syncwarp();
    }
  }
}

} // namespace sm90_fp8_gemm_impl
} // namespace kernels
} // namespace moe_cuda
