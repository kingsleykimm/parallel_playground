#pragma once
#include <jit_kernels/heuristics/sm90_arch.hpp>
#include <runtime/device.hpp>
#include <string>
#include <vector>

// Computes the shared memory size required by the TK LCF kernel for a given
// tile config.
//
// TK LCF smem layout (from lcf.cuh):
//   [0, scratch_alloc_size)                           : scratch_alloc_block
//   [scratch_alloc_size, scratch + stages*input)      :
//   input_alloc_block[INPUT_PIPE_STAGES]
//   [(MAX_SHARED_MEMORY-1024) - sizeof(finish_block)) : finish_block  (fixed
//   high-water offset)
//
// matmul_layout<BM, BN, BK, c_dtype>:
//   scratch_block    = {}  (empty) -> padder<{}, 1024> = 1024 bytes
//   input_block      = { st_fp8e4m3<64,BK> a[tiles]; st_fp8e4m3<BN,BK> b;
//   sv<float,64> sfa[tiles]; } finish_block     = { st<c_dtype, 64, BN>
//   c[tiles]; } where tiles = (BM > 64) ? 2 : 1,  a_tile height is always 64
//   (not BM)
//
// Setting MAX_SHARED_MEMORY = smem_size and allocating exactly that many bytes
// ensures the finish_block lands at (smem_size - 1024 - sizeof(finish_block)),
// non-overlapping with the input pipeline stages, and
// SAFE_STAGES_BETWEEN_BLOCKS == INPUT_PIPE_STAGES.
inline int get_tk_lcf_smem_size(const int block_m, const int block_n,
                                const int block_k, const int num_stages,
                                const size_t cd_size) {
  const int num_tiles = (block_m > 64) ? 2 : 1;

  // scratch_alloc_block: padder<empty, 1024> — 1024 bytes regardless of content
  const int scratch_alloc_size = 1024;

  // input_block raw bytes: a_tile[num_tiles] + b_tile + sfa_tile[num_tiles]
  //   a_tile = st_fp8e4m3<64, BK>  : 64 * BK bytes  (fp8 = 1 byte)
  //   b_tile = st_fp8e4m3<BN, BK>  : BN * BK bytes
  //   sfa_tile = sv<float, 64>     : 64 * 4 bytes
  const int input_block_bytes = num_tiles * 64 * block_k + block_n * block_k +
                                num_tiles * 64 * (int)sizeof(float);
  const int input_alloc_size = host_align(input_block_bytes, 1024);

  // finish_block: c_tile[num_tiles] where c_tile = st<c_dtype, 64, BN>
  const int finish_block_size = num_tiles * 64 * block_n * (int)cd_size;

  // Total: scratch | stages*input | gap | finish — +2*1024 ensures
  // SAFE_STAGES_BETWEEN_BLOCKS == num_stages (producer never stalls on
  // finish_finished).
  return scratch_alloc_size + num_stages * input_alloc_size +
         finish_block_size + 1024;
}

// Computes the shared memory size required by the TK LCF kernel3 (fused
// silu-mul-quant).
//
// kernel3 smem layout differences vs kernel2:
//   input_block  : a[tiles] + gate_tile + up_tile + sfa[tiles]
//                  (two B tiles instead of one — gate and up projections)
//   finish_block : c[tiles] + out_scales[tiles]
//                  (extra sfa_tile per warpgroup for quantization output
//                  scales)
//   scratch_block: {} (empty, same as kernel2)
//
// Tile sizes:
//   a_tile    = st_fp8e4m3<64, BK>   : 64 * BK bytes
//   gate_tile = st_fp8e4m3<BN, BK>   : BN * BK bytes
//   up_tile   = st_fp8e4m3<BN, BK>   : BN * BK bytes
//   c_tile    = st<c_dtype, 64, BN>  : 64 * BN * cd_size bytes
//   sfa_tile  = sv<float, 64>        : 64 * 4 bytes
inline int get_tk_lcf_kernel3_smem_size(const int block_m, const int block_n,
                                        const int block_k, const int num_stages,
                                        const size_t cd_size) {
  const int num_tiles = (block_m > 64) ? 2 : 1;

  // scratch_alloc_block: padder<empty, 1024> — 1024 bytes regardless of content
  const int scratch_alloc_size = 1024;

  // input_block raw bytes: a_tile[num_tiles] + gate_tile + up_tile +
  // sfa_tile[num_tiles]
  //   a_tile    = st_fp8e4m3<64, BK>  : 64 * BK bytes (fp8 = 1 byte)
  //   gate_tile = st_fp8e4m3<BN, BK>  : BN * BK bytes
  //   up_tile   = st_fp8e4m3<BN, BK>  : BN * BK bytes
  //   sfa_tile  = sv<float, 64>       : 64 * 4 bytes
  const int input_block_bytes = num_tiles * 64 * block_k +
                                2 * block_n * block_k // gate + up
                                + num_tiles * 64 * (int)sizeof(float);
  const int input_alloc_size = host_align(input_block_bytes, 1024);

  // finish_block: c_tile[num_tiles] + out_scales[num_tiles]
  //   c_tile     = st<c_dtype, 64, BN> : 64 * BN * cd_size bytes
  //   out_scales = sv<float, 64>       : 64 * 4 bytes
  const int finish_block_size = num_tiles * 64 * block_n * (int)cd_size +
                                num_tiles * 64 * (int)sizeof(float);

  // Total: scratch | stages*input | gap | finish — +2*1024 ensures
  // SAFE_STAGES_BETWEEN_BLOCKS == num_stages (producer never stalls on
  // finish_finished).
  return scratch_alloc_size + num_stages * input_alloc_size +
         finish_block_size + 2 * 1024;
}

inline int get_tk_lcf_kernel4_smem_size(const int block_m, const int block_n,
                                        const int block_k, const int num_stages,
                                        const size_t cd_size) {
  const int num_tiles = 2;

  // scratch_alloc_block: padder<empty, 1024> — 1024 bytes regardless of content
  const int scratch_alloc_size = 1024;

  // input_block raw bytes: a_tile[num_tiles] + gate_tile + up_tile +
  // sfa_tile[num_tiles]
  //   a_tile    = st_fp8e4m3<64, BK>  : 64 * BK bytes (fp8 = 1 byte)
  //   gate_tile = st_fp8e4m3<BN, BK>  : BN * BK bytes
  //   up_tile   = st_fp8e4m3<BN, BK>  : BN * BK bytes
  //   sfa_tile  = sv<float, 64>       : 64 * 4 bytes
  const int input_block_bytes =
      num_tiles * 64 * block_k +
      num_tiles * 2 * block_n * block_k // (gate + up) * 2
      + num_tiles * 64 * (int)sizeof(float);
  const int input_alloc_size = host_align(input_block_bytes, 1024);

  // finish_block: c_tile[num_tiles] + out_scales[num_tiles]
  //   c_tile     = st<c_dtype, 64, BN> : 64 * BN * cd_size bytes
  //   out_scales = sv<float, 64>       : 64 * 4 bytes
  const int finish_block_size = num_tiles * 64 * block_n * (int)cd_size +
                                num_tiles * 64 * (int)sizeof(float);

  // Total: scratch | stages*input | gap | finish — +2*1024 ensures
  // SAFE_STAGES_BETWEEN_BLOCKS == num_stages (producer never stalls on
  // finish_finished).
  return scratch_alloc_size + num_stages * input_alloc_size +
         host_align(finish_block_size * 2, 1024) + 2 * 1024;
}

inline GemmConfig get_kernel3_config(GemmType gemm_type, uint32_t M, uint32_t N,
                                     uint32_t K, uint32_t num_groups,
                                     Major AMajor, Major BMajor, Major CMajor,
                                     c10::ScalarType AB_type,
                                     c10::ScalarType CD_type,
                                     const uint32_t &num_sms,
                                     bool is_consumer_pp = false) {
  const uint32_t block_k = 128 / get_type_size(AB_type);

  auto get_num_blocks = [&](const uint32_t block_m,
                            const uint32_t block_n) -> uint32_t {
    return host_ceil_div(M, block_m) * host_ceil_div(N, block_n) * num_groups;
  };
  auto get_num_sm_waves = [&](const uint32_t block_m, const uint32_t block_n,
                              const uint32_t num_sms_arg) -> uint32_t {
    return host_ceil_div(get_num_blocks(block_m, block_n), num_sms_arg);
  };
  auto get_last_wave_util = [&](const uint32_t block_m,
                                const uint32_t block_n) -> uint32_t {
    auto last_wave_blocks = get_num_blocks(block_m, block_n) % num_sms;
    return last_wave_blocks == 0 ? num_sms : last_wave_blocks;
  };

  std::vector<int> block_m_candidates =
      SM90Arch::get_block_m_candidates(M, AMajor);
  if (gemm_type == GemmType::MGroupedContiguous) {
    block_m_candidates = {128};
  } else if (gemm_type == GemmType::MGroupedMasked) {
    block_m_candidates = {64, 128};
  }

  if (is_consumer_pp) {
    block_m_candidates = {64};
  }

  // kernel3 requires BN % 128 == 0 (grouped_matmul_layout static_assert)
  std::vector<int> block_n_candidates;
  for (int bn : SM90Arch::get_block_n_candidates(N, BMajor)) {
    if (bn % 128 == 0)
      block_n_candidates.push_back(bn);
  }

  uint32_t best_block_m = 0, best_block_n = 0;
  int best_num_waves = 0, best_last_util = 0;

  for (auto block_m : block_m_candidates) {
    for (auto block_n : block_n_candidates) {
      uint32_t num_waves =
          get_num_sm_waves(block_m, block_n, SM90Arch::kMaxSMs);
      uint32_t last_wave_util = get_last_wave_util(block_m, block_n);

      if (!SM90Arch::is_block_legal(AMajor, BMajor, AB_type, CD_type, block_m,
                                    block_n, block_k, M, N, K))
        continue;

      bool success = false;
      if (best_block_m == 0 || best_block_n == 0 || num_waves < best_num_waves)
        success = true;
      else if (num_waves == best_num_waves) {
        success = last_wave_util > best_last_util;
        if (last_wave_util == best_last_util) {
          success |= block_m == best_block_m && block_n < best_block_n;
          success |= block_m < best_block_m && block_n == best_block_n;
          success |= block_m != best_block_m && block_n > best_block_n &&
                     block_n <= N && block_m <= M;
        }
      }

      if (success) {
        best_block_m = block_m;
        best_block_n = block_n;
        best_num_waves = num_waves;
        best_last_util = last_wave_util;
      }
    }
  }
  HOST_ASSERT(best_block_m != 0 && best_block_n != 0,
              "Error: BLOCK_M, BLOCK_N search yielded no results for kernel3");

  bool tma_multicast_a = false;
  uint32_t num_tma_multicast = 1;
  const auto &[a_legal_multicast, b_legal_multicast] =
      SM90Arch::get_multicast_legality(gemm_type, num_groups, M, N,
                                       best_block_m, best_block_n, num_sms);
  const bool is_legal[2] = {b_legal_multicast, a_legal_multicast};
  bool order[2] = {false, true};
  if (best_block_m > best_block_n) {
    std::swap(order[0], order[1]);
  }
  for (const bool &is_multicast_on_a : order) {
    if (M >= 512 && is_legal[static_cast<int>(is_multicast_on_a)]) {
      tma_multicast_a = is_multicast_on_a;
      num_tma_multicast = 2;
      break;
    }
  }

  auto [num_tma_threads, num_math_threads] =
      SM90Arch::get_num_threads(best_block_m);
  if (is_consumer_pp) {
    num_math_threads = 256; // two consumer warpgroups
  }

  constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
  SharedMemoryConfig smem_config;
  int best_num_stages = 0;
  const size_t cd_size = get_type_size(CD_type);
  for (int num_stages = 16; num_stages > 0; num_stages--) {
    int smem_size;
    if (is_consumer_pp) {
      smem_size = get_tk_lcf_kernel4_smem_size(best_block_m, best_block_n,
                                               block_k, num_stages, cd_size);

    } else {
      smem_size = get_tk_lcf_kernel3_smem_size(best_block_m, best_block_n,
                                               block_k, num_stages, cd_size);
    }
    if (smem_size <= smem_capacity) {
      best_num_stages = num_stages;
      smem_config.smem_size = smem_size;
      break;
    }
  }

  int min_sms = num_sms;
  if (SM90Arch::should_minimize_sms()) {
    min_sms = host_ceil_div(host_ceil_div(M, best_block_m) *
                                host_ceil_div(N, best_block_n) * num_groups,
                            best_num_waves);
    min_sms = host_align(min_sms, num_tma_multicast);
    if (min_sms > num_sms) {
      HOST_ERROR("While trying to minimize SMs in kernel3 FP8 Heuristic");
    }
  }
  return GemmConfig{gemm_type,
                    best_block_m,
                    best_block_n,
                    block_k,
                    smem_config,
                    num_tma_multicast,
                    tma_multicast_a,
                    static_cast<uint32_t>(num_tma_threads),
                    static_cast<uint32_t>(num_math_threads),
                    static_cast<uint32_t>(min_sms),
                    best_num_stages};
}

/*
Because the silu-mul-quant epilogue is fused in, we start to get more
restrictive shapes. block_n = block_k = 128 are fixed, and the
num_consumer_warps is either 8 or 4, which determines BLOCK_M (128 or 64
respectively). No multicast. We determine num_stages based on smem usage
matching kernel5's compute_path layout: SAFE_STAGES = (DYNAMIC_SHARED_MEMORY -
sizeof(pipeline_outputs) - 1024) / sizeof(pipeline_inputs)
*/
inline int get_tk_lcf_kernel5_smem_size(const int block_m,
                                        const int num_stages) {
  const int num_tiles = (block_m > 64) ? 2 : 1;

  // scratch_alloc_block: padder<empty, 1024> — 1024 bytes
  const int scratch_alloc_size = 1024;

  // pipeline_inputs: a_tile[num_tiles] + gate_tile + up_tile +
  // sfa_tile[num_tiles]
  //   a_tile    = st_fp8e4m3<64, 128>  : 64 * 128 = 8192 bytes
  //   gate_tile = st_fp8e4m3<128, 128> : 128 * 128 = 16384 bytes
  //   up_tile   = st_fp8e4m3<128, 128> : 128 * 128 = 16384 bytes
  //   sfa_tile  = sv<float, 64>        : 64 * 4 = 256 bytes
  const int input_block_bytes = num_tiles * (64 * 128) // a_tile
                                + 128 * 128            // gate_tile
                                + 128 * 128            // up_tile
                                + num_tiles * 256;     // sfa_tile
  const int input_alloc_size = host_align(input_block_bytes, 1024);

  // pipeline_outputs: c_tile[num_tiles] + out_scales[num_tiles]
  //   c_tile     = st<fp8e4m3, 64, 128> : 64 * 128 = 8192 bytes
  //   out_scales = sv<float, 64>        : 256 bytes
  const int finish_block_size = num_tiles * (64 * 128) // c_tile
                                + num_tiles * 256;     // out_scales

  return scratch_alloc_size + num_stages * input_alloc_size +
         finish_block_size + 2 * 1024;
}

inline int get_kernel5_max_stages(const int block_m) {
  constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
  for (int num_stages = 16; num_stages > 0; num_stages--) {
    if (get_tk_lcf_kernel5_smem_size(block_m, num_stages) <= smem_capacity)
      return num_stages;
  }
  return 0;
}

inline GemmConfig get_kernel5_1_config(uint32_t M, uint32_t I,
                                       uint32_t num_experts,
                                       const uint32_t &num_sms) {
  constexpr uint32_t block_n = 128;
  constexpr uint32_t block_k = 128;

  // candidate configs: {num_consumer_warps, num_producer_warps, block_m}
  // MAIN_THREADS = 384 = 12 warps total
  //   8 consumer (2 C_WG) + 4 producer (1 P_WG) -> BM=128
  //   4 consumer (1 C_WG) + 8 producer (2 P_WG) -> BM=64
  struct Candidate {
    uint32_t block_m;
    uint32_t num_consumer_warps;
    uint32_t num_producer_warps;
  };
  const Candidate candidates[] = {
      {128, 8, 4},
      {64, 4, 4},
  };

  const uint32_t num_c_blocks = I / block_n;

  uint32_t best_block_m = 0;
  int best_num_stages = 0;
  uint32_t best_consumer_warps = 0;
  uint32_t best_producer_warps = 0;
  int best_num_waves = 0;
  int best_last_util = 0;

  for (const auto &c : candidates) {
    int max_stages = get_kernel5_max_stages(c.block_m);
    if (max_stages <= 0)
      continue;

    // total blocks = ceil(M / BM) * (I / 128) across all experts
    uint32_t num_blocks = host_ceil_div(M, c.block_m) * num_c_blocks;
    int num_waves = host_ceil_div(num_blocks, num_sms);
    int last_wave_blocks = num_blocks % num_sms;
    int last_wave_util =
        (last_wave_blocks == 0) ? (int)num_sms : last_wave_blocks;

    bool success = false;
    if (best_block_m == 0) {
      success = true;
    } else if (num_waves < best_num_waves) {
      success = true;
    } else if (num_waves == best_num_waves) {
      if (last_wave_util > best_last_util) {
        success = true;
      } else if (last_wave_util == best_last_util) {
        // prefer larger BM for better throughput per block
        success = c.block_m > best_block_m;
      }
    }

    if (success) {
      best_block_m = c.block_m;
      best_num_stages = max_stages;
      best_consumer_warps = c.num_consumer_warps;
      best_producer_warps = c.num_producer_warps;
      best_num_waves = num_waves;
      best_last_util = last_wave_util;
    }
  }
  HOST_ASSERT(best_block_m != 0,
              "Error: kernel5 config search yielded no results");

  return GemmConfig{
      GemmType::MGroupedContiguous,
      best_block_m,
      block_n,
      block_k,
      SharedMemoryConfig{
          get_tk_lcf_kernel5_smem_size(best_block_m, best_num_stages), 0, 0, 0},
      1,                        // num_tma_multicast (disabled)
      false,                    // tma_multicast_a
      best_producer_warps * 32, // num_tma_threads
      best_consumer_warps * 32, // num_math_threads
      num_sms,
      best_num_stages};
}

// compared to kernel5_1, which requires a much more rigid block structure, we
// can have more flexibility with BN, since we aren't requiring ourselves to
// quantize

inline GemmConfig search_configs(GemmType gemm_type, uint32_t M, uint32_t N,
                                 uint32_t K, uint32_t num_groups, Major AMajor,
                                 Major BMajor, Major CMajor,
                                 c10::ScalarType AB_type,
                                 c10::ScalarType CD_type,
                                 const uint32_t &num_sms, bool fused_combine = false) {
  // we need to determine smem and multicast config
  const uint32_t block_k = 128 / get_type_size(AB_type);

  // first determine the best block_m and block_n
  auto get_num_blocks = [&](const uint32_t block_m,
                            const uint32_t block_n) -> uint32_t {
    return host_ceil_div(M, block_m) * host_ceil_div(N, block_n) * num_groups;
  };
  // number of sm iterations in persistent kernel
  auto get_num_sm_waves = [&](const uint32_t block_m, const uint32_t block_n,
                              const uint32_t num_sms_arg) -> uint32_t {
    return host_ceil_div(get_num_blocks(block_m, block_n), num_sms_arg);
  };

  auto get_last_wave_util = [&](const uint32_t block_m,
                                const uint32_t block_n) -> uint32_t {
    auto last_wave_blocks = get_num_blocks(block_m, block_n) % num_sms;
    return last_wave_blocks == 0 ? num_sms : last_wave_blocks;
  };

  std::vector<int> block_m_candidates =
      SM90Arch::get_block_m_candidates(M, AMajor);
  if (gemm_type == GemmType::MGroupedContiguous) {
    block_m_candidates = {128};
  } else if (gemm_type == GemmType::MGroupedMasked) {
    block_m_candidates = {64, 128};
  }
  std::vector<int> block_n_candidates =
      SM90Arch::get_block_n_candidates(N, BMajor);

  uint32_t best_block_m = 0, best_block_n = 0;
  int best_num_waves = 0, best_last_util = 0;

  for (auto block_m : block_m_candidates) {
    for (auto block_n : block_n_candidates) {
      uint32_t num_waves =
          get_num_sm_waves(block_m, block_n, SM90Arch::kMaxSMs);
      uint32_t last_wave_util = get_last_wave_util(block_m, block_n);

      if (!SM90Arch::is_block_legal(AMajor, BMajor, AB_type, CD_type, block_m,
                                    block_n, block_k, M, N, K))
        continue;
      // first iteration, immediately assign
      bool success = false;
      if (best_block_m == 0 || best_block_n == 0 || num_waves < best_num_waves)
        success = true;
      else if (num_waves == best_num_waves) {
        // prioritize last wave utilization
        success = last_wave_util > best_last_util;
        // if equal, then we need to check three cases
        if (last_wave_util == best_last_util) {
          // check three cases for the same number of waves and last wave
          // utilization Case 1 : same 'block_m', but candidate block_n is less
          // than, so more efficient
          success |= block_m == best_block_m && block_n < best_block_n;
          // Case 2: same block_n, smaller block_m - again tile sizes are wasted
          success |= block_m < best_block_m && block_n == best_block_n;
          // case 3 - when both are different than the current best, a larger
          // block n is preferred to utilize more of the space
          success |= block_m != best_block_m && block_n > best_block_n &&
                     block_n <= N && block_m <= M;
        }
      }
      // we don't consider configs where num_waves > best_num_waves

      if (success) {
        best_block_m = block_m;
        best_block_n = block_n;
        best_num_waves = num_waves;
        best_last_util = last_wave_util;
      }
    }
  }
  HOST_ASSERT(best_block_m != 0 && best_block_n != 0,
              "Error: BLOCK_M, BLOCK_N search yielded no results");
  bool tma_multicast_a = false;
  uint32_t num_tma_multicast = 1;
  const auto &[a_legal_multicast, b_legal_multicast] =
      SM90Arch::get_multicast_legality(gemm_type, num_groups, M, N,
                                       best_block_m, best_block_n, num_sms);
  const bool is_legal[2] = {b_legal_multicast, a_legal_multicast};
  // {down m dimension, down n dimension}
  bool order[2] = {false, true};
  if (best_block_m >
      best_block_n) { // if the m block is larger, priotize broadcasting across
                      // N dimension, since more N blocks
    std::swap(order[0], order[1]);
  }
  // because of the break; order matters
  for (const bool &is_multicast_on_a : order) {
    // true, false -> {1, 0}
    // looks like this is a heuristic boundary to check that M is large enough
    // to see if multicasting is worth it
    if (M >= 512 && is_legal[static_cast<int>(is_multicast_on_a)]) {
      // so setting tma_multicast_a to true means we're doing it ACROSS the N
      // dimension
      tma_multicast_a = is_multicast_on_a;
      num_tma_multicast = 2; // hard set to 2
      break;
    }
  }

  // multicast config is complete, move to shared
  const auto &[num_tma_threads, num_math_threads] =
      SM90Arch::get_num_threads(best_block_m);

  // Shared Memory Config — use TK LCF layout (not the generic heuristic model)
  constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
  SharedMemoryConfig smem_config;
  int best_num_stages = 0;
  const size_t cd_size = get_type_size(CD_type);
  for (int num_stages = 16; num_stages > 0; num_stages--) {
    int smem_size = get_tk_lcf_smem_size(best_block_m, best_block_n, block_k,
                                         num_stages, cd_size);
    if (fused_combine) {
      smem_size += host_align(num_groups, 16) * sizeof(int);
    }
    if (smem_size <= smem_capacity) {
      best_num_stages = num_stages;
      smem_config.smem_size = smem_size;
      break;
    }
  }
  HOST_ASSERT(best_num_stages != 0,
              "Error: kernel5 config search yielded no results");

  int min_sms = num_sms;
  if (SM90Arch::should_minimize_sms()) {
    min_sms = host_ceil_div(host_ceil_div(M, best_block_m) *
                                host_ceil_div(N, best_block_n) * num_groups,
                            best_num_waves);
    min_sms = host_align(min_sms, num_tma_multicast);
    if (min_sms > num_sms) {
      HOST_ERROR("While trying to minimize SMs in FP8 Heuristic");
    }
  }
  return GemmConfig{gemm_type,
                    best_block_m,
                    best_block_n,
                    block_k,
                    smem_config,
                    num_tma_multicast,
                    tma_multicast_a,
                    static_cast<uint32_t>(num_tma_threads),
                    static_cast<uint32_t>(num_math_threads),
                    static_cast<uint32_t>(min_sms),
                    best_num_stages};
}

// what do we need to account for in transpose? the two variables are the number
// of threads and the mn size block_mn is kind of decided by the mn size - if mn
// is small, we can use a small block_n
inline std::tuple<int, int, int>
get_transpose_config(int mn, int sf_k,
                     c10::ScalarType dtype = c10::ScalarType::Float) {
  const std::vector<int> block_mn_candidates = {128, 64, 32, 16, 8};
  const std::vector<int> num_threads_candidates = {512, 256, 128, 64, 32};
  int best_sm_occupancy = 0;
  int best_block_mn = 0;
  int best_threads = 0;
  int best_smem_size = 0;
  for (const auto block_mn : block_mn_candidates) {
    // Potential issue #1: Need to account for padding (PADDED_SF_K in
    // transpose_fp32)
    int padded_sf_k = sf_k + (sf_k + 1) % 2;
    int smem_size = block_mn * padded_sf_k * get_type_size(dtype);

    int usage =
        host_ceil_div(mn, block_mn); // this is the number of blocks used
    // we want high intra-sm occupancy,
    // but also high grid occupancy

    bool valid = smem_size < device_prop->get_smem_size();
    if (valid) {
      int num_blocks_per_sm =
          device_prop->get_prop()->sharedMemPerMultiprocessor / smem_size;

      // Potential issue #3: num_blocks_per_sm could be 0 if smem_size is very
      // large
      if (num_blocks_per_sm == 0)
        continue;

      for (const auto thread : num_threads_candidates) {
        // Potential issue #4: Need to check thread count doesn't exceed max
        // threads per block
        if (thread > device_prop->get_prop()->maxThreadsPerBlock)
          continue;

        int actual_blocks_per_sm = std::min(
            num_blocks_per_sm,
            device_prop->get_prop()->maxThreadsPerMultiProcessor / thread);
        float occupancy = (float)actual_blocks_per_sm * thread /
                          device_prop->get_prop()->maxThreadsPerMultiProcessor;
        if (occupancy > best_sm_occupancy) {
          best_sm_occupancy = occupancy;
          best_block_mn = block_mn;
          best_threads = thread;
          best_smem_size = smem_size;
        } else if (occupancy == best_sm_occupancy) {
          if (best_threads < thread) {
            best_block_mn = block_mn;
            best_threads = thread;
            best_smem_size = smem_size;
          }
        }
      }
    }
  }
  HOST_ASSERT(best_block_mn > 0 && best_threads > 0,
              "Error in heuristic search");
  return std::make_tuple(best_block_mn, best_threads, best_smem_size);
}
