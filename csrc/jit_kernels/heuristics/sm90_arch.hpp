#pragma once
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/stride.hpp>
#include <moe_cuda/dtype.h>
#include <moe_cuda/types.h>
#include <runtime/tensor.h>
#include <jit/utils/common.hpp>
#include <cstdint>
#include <fcntl.h>
#include <numeric>
#include <vector>


struct SM90Arch {
  static constexpr uint32_t kMaxSharedMemoryPerBlock = 232448; // 232 KB max shared memory per block
  static constexpr uint32_t kMaxSharedMemoryPerSM = 232448;    // 232 KB max shared memory per SM
  static constexpr uint32_t kMaxSMs = 132;                          // Max SMs for H100
  static constexpr uint32_t kWarpSize = 32;
  static constexpr uint32_t kMaxThreadsPerBlock = 1024;
  static constexpr uint32_t kMaxRegistersPerThread = 255;
  static constexpr uint32_t kMaxThreadsPerSM = 2048;
  static constexpr uint32_t kMaxMulticast = 8;

  static std::vector<int> get_block_m_candidates(uint32_t& m, Major major) {

    static std::vector<int> candidates = {64, 128, 256};

    if (major == Major::K) {
        if (m <= 32) candidates.push_back(32);
        if (m <= 16) candidates.push_back(16);
    }
    return candidates;
  }

  static std::vector<int> get_block_n_candidates(uint32_t & n, Major major) {
      // the n direction has more flexibility - the range of values is from 8 - 256, but there's almost no reason to begin at 8
    std::vector<int> candidates;
    for (uint32_t block_n = 16; block_n <= 256; block_n += 16) {
        candidates.push_back(block_n);
    }
    return candidates;
  }

  static std::pair<int, int> get_num_threads(uint32_t & block_m) {
    return std::make_pair(
        128, (block_m <= 64 ? 1 : 2) * 128
    ); // returns (numTMAThreads, numWGMMAThreads)
  }

  static bool is_block_legal(
    Major major_a, Major major_b,
    c10::ScalarType AB_type, c10::ScalarType CD_type,
    const uint32_t& block_m, const uint32_t& block_n, const uint32_t& block_k,
    const uint32_t& m, const uint32_t& n, const uint32_t & k
  ) {

    // avoid too many scaling factors in a single block:
    // when BLOCK_N > BLOCK_K, gcd(BLOCK_N, BLOCK_K) = BLOCK_N - BLOCK_K
    // for kernel type Kernel::1D2D, when block_n > 128, the only values that work are
    // 128 + 16, 128 + 32, 128 + 64
    if (block_n > 128 && (block_n != 144 && block_n != 160 && block_n != 192)) return false;

    // cap overall block sizes for the given number of registers.
    return block_m <= 128 || block_n <= 128;
  }

  // for Kernel pattern of 1D2D
  static bool is_num_stages_legal(c10::ScalarType& AB_type,
    const uint32_t & num_stages, const uint32_t & block_n, const uint32_t & block_k) {
    const uint32_t search_space = block_k / std::gcd(block_n, block_k);
    // first, we want to check if the compiler is going to unroll the dispatch_num_former_iters
    // when it does we check that num_stages <= 4 is to manage code size
    if (search_space <= 4 && block_k % block_n != 0 && AB_type == c10::ScalarType::Float8_e4m3fn) {
      return num_stages <= 4;
    }
    return true;
  }

  // returns legal of multicasting on A or multicasting on B
  static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type,
    const uint32_t& num_groups, const uint32_t& m, const uint32_t & n, const uint32_t& block_m,
    const uint32_t& block_n, const uint32_t & num_sms
  ) {
    // no multicast when batched to avoid mixing between batches
    if (gemm_type == GemmType::Batched) return {false, false};

    return {
      // to check if multicast is legal on A, where we are broadcasting A across N dimension
      // When performing masked grouped GEMM, we need the N blocks to be even to avoid weight mixing
      is_multicast_legal(n, block_n, 2, num_sms, gemm_type == GemmType::MGroupedMasked),
      // when multicasting B, for masked grouped gemms we always ensure that the grid size % 2 == 0.
      // this is mGroupedMasked defines a variable size grid, instead of the fixed (bM, bN), since it stores variable chunks of different group lengths
      // if num_m_blocks, num_n_blocks are both odd
      is_multicast_legal(m, block_m, 2, num_sms, false)
        && (gemm_type != GemmType::MGroupedMasked || is_multicast_legal(n, block_n, 2, num_sms, true))
    };
  }

  static bool should_minimize_sms() { return true; }

  static bool should_cd_swizzle(c10::ScalarType cd_type) {
    // multiple elements do not live in the same bank when FP32
    return cd_type != c10::ScalarType::Float;
  }

  // simple for now - in the future it can be changed with sharding across cluster dim
  static int get_a_load_m(const uint32_t & block_m) {
    return block_m;
  }

  static int get_b_load_n(const uint32_t & block_n) {
    return block_n;
  }

  static int get_sf_smem_size(const uint32_t & block_m, const uint32_t & block_n, const uint32_t & block_k) {
    // block_k is set to 128 for 1D2D scaled kernels
    // also must be aligned to 128 bytes for TMA
    const int num_b_scales = (block_k % block_n != 0) ? 2 : 1;
    return ti_align((block_m + num_b_scales) * sizeof(float), 128);
  }

  // uint64_t * producer and consumer -> 8 bytes * 2
  static int get_barrier_size() {
    return 8 * 2;
  }

  // just checks for multiple warpgroups when storing, pretty useless ternary
  static int get_cd_store_m(const bool is_single_warpgroup, const uint32_t & block_m) {
    return is_single_warpgroup ? 64 : block_m;
  }

  static int get_smem_cd_size(const uint32_t & block_m, const uint32_t & block_n,
  c10::ScalarType cd_type) {
    // for TMA / GMMA swizzling : max swizzle mode is 128, with 8 row / column atoms
    return ti_align(
      block_m * block_n * get_type_size(cd_type), 1024
    );
  }

};
