#pragma once
#include <cuda.h>
#include <jit/utils/files.hpp>
#include <moe_cuda/dtype.h>
#include <moe_cuda/error.hpp>
#include <moe_cuda/types.h>
#include <runtime/utils.h>

template <typename TA, typename TB> __host__ TA host_ceil_div(TA a, TB b) {
  return (a + b - 1) / b;
}

// align a to b
template <typename TA, typename TB> __host__ TA host_align(TA a, TB b) {
  return host_ceil_div(a, b) * b;
}

// mirrors kittens dtypes in common/base_types.cuh
static std::string to_string(const at::ScalarType &dtype) {
  switch (dtype) {
  case torch::kInt:
    return "int";
  case torch::kFloat:
    return "float";
  case torch::kBFloat16:
    return "bf16";
  case torch::kHalf:
    return "half";
  case torch::kFloat8_e4m3fn:
    return "fp8e4m3";
  case torch::kFloat8_e5m2:
    return "fp8e5m2";
  default:
    HOST_ERROR("Unsupported dtype");
  }
  return "";
}

struct SharedMemoryConfig {
  int smem_size;
  int swizzle_a_mode;
  int swizzle_b_mode;
  int swizzle_cd_mode;
};
// contains most of the items needed for the FP8 gemm call

struct GemmConfig {
  GemmType gemm_type;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t block_k;
  SharedMemoryConfig smem_config;
  uint32_t num_tma_multicast;
  bool tma_multicast_a;
  uint32_t num_tma_threads;
  uint32_t num_math_threads;
  uint32_t num_sms;
  int num_stages;

  inline void to_str() {
    printf("GemmConfig:\n");
    printf("  gemm_type: %d\n", static_cast<int>(gemm_type));
    printf("  block_m: %u, block_n: %u, block_k: %u\n", block_m, block_n,
           block_k);
    printf("  smem_size: %d\n", smem_config.smem_size);
    printf("  swizzle_a_mode: %d, swizzle_b_mode: %d, swizzle_cd_mode: %d\n",
           smem_config.swizzle_a_mode, smem_config.swizzle_b_mode,
           smem_config.swizzle_cd_mode);
    printf("  num_tma_multicast: %u, tma_multicast_a: %d\n", num_tma_multicast,
           tma_multicast_a);
    printf("  num_tma_threads: %u, num_math_threads: %u\n", num_tma_threads,
           num_math_threads);
    printf("  num_sms: %u, num_stages: %d\n", num_sms, num_stages);
  }
};

struct LaunchConfig {

  dim3 blockDim;
  dim3 gridDim;
  cudaStream_t stream;
  int smem_size;
  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = Attr::4
  uint32_t num_multicast;
  // CU_LAUNCH_ATTRIBUTE_COOPERATIVE = Attr::2
  bool cooperative = false;
};

// divisibility conditions for multicast check
inline bool is_multicast_legal(const uint32_t &shape_dim,
                               const uint32_t &block_dim,
                               const uint32_t &num_multicast,
                               const uint32_t &num_sms,
                               bool require_divisible) {
  bool divisible = !require_divisible ||
                   host_ceil_div(shape_dim, block_dim) % num_multicast == 0;
  return divisible && num_sms % num_multicast == 0;
}

inline int get_compiled_dim(const std::string &compiled_dims, char dim,
                            int dim_value) {
  for (const auto c : compiled_dims) {
    if (c == dim)
      return dim_value;
  }
  return 0;
}

// [DEPRECATED] Host-side TMA descriptor helpers removed — all active kernels
// now use the ThunderKittens gl<>/pgl<> constructors via the pre-compiled
// globals factory (fp8_tk_globals_factory.cu). The following were removed:
//   convert_to_cudtype, getCuSwizzle, getCuL2PromotionSize,
//   get_inner_outer_dims, make_tma_2d_desc, make_tma_3d_desc,
//   make_tma_a_desc, make_tma_b_desc, make_tma_d_desc,
//   make_tma_a_desc_3d, make_tma_b_desc_3d, make_tma_d_desc_3d,
//   make_tma_sf_desc

