#pragma once
#include <runtime/device.hpp>
#include <vector>
#include <string>
#include <jit_kernels/heuristics/sm90_arch.hpp>


inline SharedMemoryConfig get_smem_config(const GemmType& gemm_type, c10::ScalarType AB_type,
c10::ScalarType CD_type, const int & m, const int & n, const int &k,
const int& block_m, const int& block_n, const int & block_k,
Major major_a, Major major_b, const int & num_stages) {

    const size_t& ab_size = get_type_size(AB_type);
    const size_t& cd_size = get_type_size(CD_type);

    const int& load_block_m = SM90Arch::get_a_load_m(block_m);
    const int& load_block_n = SM90Arch::get_b_load_n(block_n);
    const int& load_block_k = 128;

    // swizzle modes
    const int& swizzle_a_mode = get_swizzle_mode(major_a == Major::K ? block_k : block_m, ab_size);
    const int& swizzle_b_mode = get_swizzle_mode(major_b == Major::K ? block_k : block_n, cd_size);
    const int& swizzle_cd_mode = SM90Arch::should_cd_swizzle(CD_type) ? get_swizzle_mode(block_n, cd_size) : 0;

    const int& smem_cd_size = SM90Arch::get_smem_cd_size(block_m, block_n, CD_type);
    const int& smem_a_size_per_stage = load_block_m * load_block_k * ab_size;
    const int& smem_b_size_per_stage = load_block_n * load_block_k * ab_size;

    const int& shape_k_scales = ti_ceil_div(k, block_k);
    const int& smem_sf_size_per_stage = ti_align(block_m * sizeof(float) + shape_k_scales * sizeof(float) * (block_k % block_n != 0 ? 2 : 1), 16);

    const int& smem_barrier_size_per_stage = SM90Arch::get_barrier_size();

    int smem_size = 0;
    smem_size += smem_cd_size;
    smem_size += (smem_a_size_per_stage +
                smem_b_size_per_stage +
                smem_sf_size_per_stage +
                smem_barrier_size_per_stage) * num_stages;

    return SharedMemoryConfig {
        .smem_size = smem_size,
        .swizzle_a_mode = swizzle_a_mode,
        .swizzle_b_mode = swizzle_b_mode,
        .swizzle_cd_mode = swizzle_cd_mode
    };
}

inline GemmConfig search_configs(
    GemmType gemm_type,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t num_groups,
    Major AMajor,
    Major BMajor,
    Major CMajor,
    c10::ScalarType AB_type, c10::ScalarType CD_type,
    const uint32_t & num_sms
) {
    // we need to determine smem and multicast config
    const uint32_t block_k = 128 /  get_type_size(AB_type);

    // first determine the best block_m and block_n
    auto get_num_blocks = [&](const uint32_t block_m, const uint32_t block_n) -> uint32_t {
        return ti_ceil_div(M, block_m) * ti_ceil_div(N, block_n) * num_groups;
    };
    // number of sm iterations in persistent kernel
    auto get_num_sm_waves = [&](const uint32_t block_m, const uint32_t block_n, const uint32_t num_sms_arg) -> uint32_t {
        return ti_ceil_div(get_num_blocks(block_m, block_n), num_sms_arg);
    };

    auto get_last_wave_util = [&](const uint32_t block_m, const uint32_t block_n) -> uint32_t {
        auto last_wave_blocks = get_num_blocks(block_m, block_n) % num_sms;
        return last_wave_blocks == 0 ? num_sms : last_wave_blocks;
    };

    std::vector<int> block_m_candidates = SM90Arch::get_block_m_candidates(M, AMajor);
    if (gemm_type == GemmType::MGroupedContiguous) {
        block_m_candidates = {128};
    }
    else if (gemm_type == GemmType::MGroupedMasked) {
        block_m_candidates = {64, 128};
    }
    std::vector<int> block_n_candidates = SM90Arch::get_block_n_candidates(N, BMajor);

    uint32_t best_block_m = 0, best_block_n = 0;
    int best_num_waves = 0, best_last_util = 0;

    for (auto block_m : block_m_candidates) {
        for (auto block_n : block_n_candidates) {
            uint32_t num_waves = get_num_sm_waves(block_m, block_n, SM90Arch::kMaxSMs);
            uint32_t last_wave_util = get_last_wave_util(block_m, block_n);

            if (!SM90Arch::is_block_legal(
                AMajor, BMajor, AB_type, CD_type,
                block_m, block_n, block_k, M, N, K
              )) continue;
            // first iteration, immediately assign
            bool success = false;
            if (best_block_m == 0 || best_block_n == 0 || num_waves < best_num_waves)
              success = true;
            else if (num_waves == best_num_waves) {
                // prioritize last wave utilization
                success = last_wave_util > best_last_util;
                // if equal, then we need to check three cases
                if (last_wave_util == best_last_util) {
                    // check three cases for the same number of waves and last wave utilization
                    // Case 1 : same 'block_m', but candidate block_n is less than, so more efficient
                    success |= block_m == best_block_m && block_n < best_block_n;
                    // Case 2: same block_n, smaller block_m - again tile sizes are wasted
                    success |= block_m < best_block_m && block_n == best_block_n;
                    // case 3 - when both are different than the current best, a larger block n is preferred to utilize more of the space
                    success |= block_m != best_block_m && block_n > best_block_n && block_n <= N && block_m <= M;
                }
            }
            // we don't consider configs where num_waves > best_num_waves

            if (success) {
                best_block_m = block_m; best_block_n = block_n;
                best_num_waves = num_waves; best_last_util = last_wave_util;
            }
        }
    }
    HOST_ASSERT(best_block_m != 0 && best_block_n != 0, "Error: BLOCK_M, BLOCK_N search yielded no results");
    bool tma_multicast_a = false;
    uint32_t num_tma_multicast = 1;
    const auto& [a_legal_multicast, b_legal_multicast] = SM90Arch::get_multicast_legality(gemm_type, num_groups, M, N, best_block_m, best_block_n, num_sms);
    const bool is_legal[2] = {b_legal_multicast, a_legal_multicast};
    // {down m dimension, down n dimension}
    bool order[2] = {false, true};
    if (best_block_m > best_block_n) { // if the m block is larger, priotize broadcasting across N dimension, since more N blocks
        std::swap(order[0], order[1]);
    }
    // because of the break; order matters
    for (const bool& is_multicast_on_a : order) {
        // true, false -> {1, 0}
        // looks like this is a heuristic boundary to check that M is large enough to see if multicasting is worth it
        if (M >= 512 && is_legal[static_cast<int>(is_multicast_on_a)]) {
            // so setting tma_multicast_a to true means we're doing it ACROSS the N dimension
            tma_multicast_a = is_multicast_on_a;
            num_tma_multicast = 2; // hard set to 2
            break;
        }
    }

    // multicast config is complete, move to shared
    const auto& [num_tma_threads, num_math_threads] = SM90Arch::get_num_threads(best_block_m);

    // Shared Memory Config
    constexpr int smem_capacity = SM90Arch::kMaxSharedMemoryPerBlock;
    SharedMemoryConfig smem_config;
    int best_num_stages = 0;
    for (int num_stages = 32; num_stages > 0; num_stages--) {
        if (!SM90Arch::is_num_stages_legal(AB_type, num_stages, best_block_n, 128))
            continue;

        smem_config = get_smem_config(gemm_type, AB_type, CD_type, M, N,
            K, best_block_m, best_block_n, block_k, AMajor, BMajor, num_stages);

        // use the largest stage possible that fits in smem capacity
        if (smem_config.smem_size <= smem_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }

    int min_sms = num_sms;
    if (SM90Arch::should_minimize_sms()) {
        min_sms = ti_ceil_div(
            ti_ceil_div(M, best_block_m) * ti_ceil_div(N, best_block_n) * num_groups, best_num_waves
        );
        min_sms = ti_align(min_sms, num_tma_multicast);
        if (min_sms > num_sms) {
            HOST_ERROR("While trying to minimize SMs in FP8 Heuristic");
        }
    }
    return GemmConfig {
        gemm_type,
        best_block_m,
        best_block_n,
        block_k,
        smem_config,
        num_tma_multicast,
        tma_multicast_a,
        static_cast<uint32_t>(num_tma_threads),
        static_cast<uint32_t>(num_math_threads),
        static_cast<uint32_t>(min_sms),
        best_num_stages
    };

}


// what do we need to account for in transpose? the two variables are the number of threads and the mn size
// block_mn is kind of decided by the mn size - if mn is small, we can use a small block_n
inline std::tuple<int, int, int> get_transpose_config(
    int mn,
    int sf_k,
    c10::ScalarType dtype = c10::ScalarType::Float
) {
    const std::vector<int> block_mn_candidates = {128, 64, 32, 16, 8};
    const std::vector<int> num_threads_candidates = {512, 256, 128, 64, 32};
    int best_sm_occupancy = 0;
    int best_block_mn = 0;
    int best_threads = 0;
    int best_smem_size = 0;
    for (const auto block_mn : block_mn_candidates) {
        // Potential issue #1: Need to account for padding (PADDED_SF_K in transpose_fp32)
        int padded_sf_k = sf_k + (sf_k + 1) % 2;
        int smem_size = block_mn * padded_sf_k * get_type_size(dtype);
        
        int usage = ti_ceil_div(mn, block_mn); // this is the number of blocks used
        // we want high intra-sm occupancy,
        // but also high grid occupancy
        
        bool valid = smem_size < device_prop->get_smem_size();
        if (valid) {
            int num_blocks_per_sm = device_prop->get_prop()->sharedMemPerMultiprocessor / smem_size;
            
            // Potential issue #3: num_blocks_per_sm could be 0 if smem_size is very large
            if (num_blocks_per_sm == 0) continue;
            
            for (const auto thread : num_threads_candidates) {
                // Potential issue #4: Need to check thread count doesn't exceed max threads per block
                if (thread > device_prop->get_prop()->maxThreadsPerBlock) continue;
                
                int actual_blocks_per_sm = std::min(num_blocks_per_sm, device_prop->get_prop()->maxThreadsPerMultiProcessor / thread);
                float occupancy = (float) actual_blocks_per_sm * thread / device_prop->get_prop()->maxThreadsPerMultiProcessor;
                if (occupancy > best_sm_occupancy) {
                    best_sm_occupancy = occupancy;
                    best_block_mn = block_mn;
                    best_threads = thread;
                    best_smem_size = smem_size;
                }
                else if (occupancy == best_sm_occupancy) {
                    if (best_threads < thread) {
                        best_block_mn = block_mn;
                        best_threads = thread;
                        best_smem_size = smem_size;
                    }
                }
            }
        }
    }
    HOST_ASSERT(best_block_mn > 0 && best_threads > 0, "Error in heuristic search");
    return std::make_tuple(best_block_mn, best_threads, best_smem_size);
}

