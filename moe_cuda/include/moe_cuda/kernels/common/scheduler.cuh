/*
Based off of DeepGEMM Scheduler
*/
#include "cute/config.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <moe_cuda/types.h>

// DeepGEMM seems to only do 1D clusters, for simplicity, and then perform a 1D
// multicast Yet we still need to perform swizzling on the non-multicast
// direction in order to ensure maximal use of the loaded A/B tile, this method
// performs an extremely small search to find the best swizzle
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {

  uint32_t num_best_blocks = 0, min_usage = UINT32_MAX;
  // usage, min_usage here describe the number of elements needed to load into
  // L2 cache for a single swizzled group
  for (const auto &candidate : {8u, 16u}) {

    uint32_t usage =
        kIsMulticastOnA ? (BLOCK_N * candidate +
                           constexpr_ti_ceil_div(kNumSMs, candidate) *
                               BLOCK_M) // candidate * num_m_blocks <= kNumSMs
                        : (BLOCK_M * candidate +
                           constexpr_ti_ceil_div(kNumSMs, candidate) * BLOCK_N);
    if (usage < min_usage) {
      min_usage = usage;
      num_best_blocks = candidate;
    }
  }
  return num_best_blocks;
}

// General GEMM block idx scheduler
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 128u, // 128 for sm90
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<
              kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {

  int current_iter = -1;
  uint32_t num_blocks;
  uint32_t num_m_blocks;
  uint32_t num_n_blocks;

  uint32_t num_blocks_in_group_swizzle_dir;
  bool is_peer_cta_alive = true;

  // grouped gemm
  int *grouped_layout;
  uint32_t cur_group_idx = 0;

  // masked group
  uint32_t current_m_cumsum = 0;

  __device__ __forceinline__ explicit Scheduler(const uint32_t &global_m,
                                                const uint32_t &global_n,
                                                const uint32_t &global_k,
                                                int *grouped_layout = nullptr) {

    num_m_blocks = ti_ceil_div(global_m, BLOCK_M);
    num_n_blocks = ti_ceil_div(global_n, BLOCK_N);

    if constexpr (kGemmType == GemmType::Normal ||
                  kGemmType == GemmType::Batched) {
      num_blocks = num_m_blocks * num_n_blocks;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      num_blocks = num_m_blocks * num_n_blocks;
      this->grouped_layout = grouped_layout;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
      this->grouped_layout = grouped_layout;
    }
  }

  // reorders the given m_block_idx, n_block_idx based off the direction of the
  // multicast, since this will determine the grouping as well. by restricting
  // the length of the non-multicast loads, it allows for better locality, since
  // it converts the row-major/column-major into a block, allowing more L2 cache
  // hits and reuse With swizzling, groups of CTAs share the same
  // kNum1DBlocksPerGroup A/B rows/columns
  // this also reorders the m_block_idx, n_block_idx so that it is compatible
  // with the multicast direction passed in
  CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t &block_idx,
                                             uint32_t &m_block_idx,
                                             uint32_t &n_block_idx) {

    CUTE_STATIC_ASSERT(
        kNum1DBlocksPerGroup % kNumMulticast == 0,
        "Invalid number of blocks, should be divisible by the multicast case");

    const auto &primary_num_blocks =
        kIsMulticastOnA ? num_n_blocks : num_m_blocks;
    const auto &secondary_num_blocks =
        kIsMulticastOnA ? num_m_blocks : num_n_blocks;
    const auto &group_size = secondary_num_blocks * kNum1DBlocksPerGroup;

    const auto &group_idx = block_idx / group_size;
    auto first_block_idx =
        group_idx *
        kNum1DBlocksPerGroup; // this the index of the first_block in the
                              // primary direction of the group

    auto in_group_idx = block_idx % group_size;
    num_blocks_in_group_swizzle_dir =
        min(kNum1DBlocksPerGroup,
            primary_num_blocks -
                first_block_idx); // for ragged tile cases at the end

    if (kNumMulticast > 1 && num_blocks_in_group_swizzle_dir % 2 !=
                                 0) { // odd sized on the ragged condition
      if (in_group_idx <
          (num_blocks_in_group_swizzle_dir ^ 1) * secondary_num_blocks) {
        num_blocks_in_group_swizzle_dir =
            num_blocks_in_group_swizzle_dir ^
            1; // truncate for blocks that fit in previously
      } else {
        // subtract by the slice of blocks before it
        in_group_idx = in_group_idx - (num_blocks_in_group_swizzle_dir ^ 1) *
                                          secondary_num_blocks;
        first_block_idx +=
            (num_blocks_in_group_swizzle_dir ^
             1); // advance the first_block_idx appropriately by the offset
        num_blocks_in_group_swizzle_dir = 1;
      }
    }

    if constexpr (kIsMulticastOnA) { // N-major, when multicasting A across N
                                     // dimension
      m_block_idx = in_group_idx / num_blocks_in_group_swizzle_dir;
      n_block_idx =
          first_block_idx + in_group_idx % num_blocks_in_group_swizzle_dir;
    } else { // M-major
      m_block_idx =
          first_block_idx + in_group_idx % num_blocks_in_group_swizzle_dir;
      n_block_idx = in_group_idx / num_blocks_in_group_swizzle_dir;
    }
  }

  __device__ __forceinline__ bool get_next_block(uint32_t &m_block_idx,
                                                 uint32_t &n_block_idx) {
    // persistent scheduler pattern
    const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;

    if constexpr (kGemmType == GemmType::MGroupedMasked) {
      while (true) {
        // use a while loop to locate which 'm_tile' of work next_block_idx is
        // given
        if (this->cur_group_idx == kNumGroups)
          return false;

        // check current group is where next_block_idx belongs
        num_m_blocks = ti_ceil_div(
            (uint32_t)__ldg(grouped_layout + cur_group_idx), BLOCK_M);
        auto cur_cumsum_m_blocks = current_m_cumsum + num_m_blocks;
        if (next_block_idx < cur_cumsum_m_blocks * num_n_blocks)
          break; // we break here since num_m_blocks has been updated for the
                 // new slab of experts

        // if blockIdx.x >, we need to keep searching for the next group
        ++cur_group_idx;
        current_m_cumsum = cur_cumsum_m_blocks;
      }
      get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                             m_block_idx, n_block_idx);
    } else if constexpr (kGemmType == GemmType::Batched) {
      if (next_block_idx >= num_blocks * kNumGroups) {
        return false;
      }
      cur_group_idx = next_block_idx / num_blocks;
      const auto &block_idx = next_block_idx - cur_group_idx * num_blocks;
      if constexpr (kIsMulticastOnA) { // determine the m_bloc
        // if multicast is turned on, we are treating N as row major since it
        // broadcasts across
        m_block_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx % num_n_blocks;
      } else {
        m_block_idx = block_idx % num_m_blocks;
        n_block_idx = block_idx / num_m_blocks;
      }
    }

    else { // MGroupedContiguous, Normla
      if (next_block_idx >= num_blocks) {
        return false;
      }

      // used during multicast checks for WGMMA
      is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 ||
                          num_m_blocks % kNumMulticast == 0 ||
                          (next_block_idx ^ 1) < num_blocks;
      // this doesn't have any checks because the expert index / padding check
      // will come after the TMA copyies
      get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    }
    return true;
  }

  // returns the index on the M/N dimension relative to the entire global
  // problem size
  template <bool kWithGroupOffset>
  CUTLASS_DEVICE uint32_t get_global_idx(uint32_t shape, uint32_t block_idx,
                                         uint32_t block_size,
                                         uint32_t m_block_idx = 0) {
    if constexpr (kGemmType == GemmType::Normal) {
      return block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      // the offset is only used in this case when identify which index to use
      // for the experts
      const auto offset =
          kWithGroupOffset
              ? max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M))
              : 0;
      // expert_index * expert_out + local_indexing
      return offset * shape + block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
      // when the kWithGroupOffset is on here, it's for the m dimension, since
      // we're going to 'index' into the slab we're currently working out of,
      // and shape here is the MAX_M rows that any expert will take
      const auto offset = kWithGroupOffset ? cur_group_idx : 0;
      return offset * shape + block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::Batched) {
      const auto offset = cur_group_idx;
      return offset * shape + block_idx * block_size;
    }
  }

  CUTLASS_DEVICE bool is_tma_multicast_valid(uint32_t &m_block_idx) {
    if (num_blocks_in_group_swizzle_dir == 1)
      return false;

    if constexpr (kGemmType == GemmType::Batched ||
                  kGemmType == GemmType::MGroupedMasked ||
                  kGemmType == GemmType::Normal) {
      return true;
    } else { // GemmType == MGroupedContiguous
      CUTE_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous,
                         "Invalid Gemm Type");
      // Bounds check: peer block might be out of range for odd num_m_blocks
      if constexpr (kIsMulticastOnA)
        return true; // threadblock swizzling already handled this
      if ((m_block_idx ^ 1) >= num_m_blocks)
        return false;
      int group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
      int peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
      // Both must be valid (not -1 padding) and same group for multicast
      return group_idx >= 0 && peer_group_idx >= 0 &&
             peer_group_idx == group_idx;
    }
  }

  CUTLASS_DEVICE bool is_computation_valid(uint32_t &block_idx,
                                           const uint32_t &block_offset) const {
    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      auto group_index =
          __ldg(grouped_layout + block_idx * BLOCK_M + block_offset);
      return group_index >= 0;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
      auto group_limit = __ldg(grouped_layout + this->cur_group_idx);
      return block_idx * BLOCK_M + block_offset < group_limit;
    }
    return true;
  }
};
