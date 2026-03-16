#pragma once
#include <cuda.h>
#include <moe_cuda/dtype.h>
#include <moe_cuda/types.h>
#include <runtime/tensor.h>
#include <moe_cuda/error.hpp>
#include <runtime/utils.h>
#include <jit/utils/files.hpp>


template <typename TA, typename TB> __host__  TA host_ceil_div(TA a, TB b) { return (a + b - 1) / b; }

// align a to b
template <typename TA, typename TB> __host__ TA host_align(TA a, TB b) { return host_ceil_div(a, b) * b; }

template <typename TA, typename TB> __host__  constexpr TA constexpr_host_ceil_div(TA a, TB b) {
  return (a + b - 1) / b;
}

// align a to b
template <typename TA, typename TB> __host__ constexpr TA constexpr_host_align(TA a, TB b) {
  return constexpr_host_ceil_div(a, b) * b;
}

// mirrors kittens dtypes in common/base_types.cuh
static std::string to_string(const at::ScalarType& dtype) {
  switch (dtype) {
      case torch::kInt:           return "int";
      case torch::kFloat:         return "float";
      case torch::kBFloat16:      return "bf16";
      case torch::kHalf:          return "half";
      case torch::kFloat8_e4m3fn: return "fp8e4m3";
      case torch::kFloat8_e5m2:   return "fp8e5m2";
      default: HOST_ERROR("Unsupported dtype");
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
        printf("  block_m: %u, block_n: %u, block_k: %u\n", block_m, block_n, block_k);
        printf("  smem_size: %d\n", smem_config.smem_size);
        printf("  swizzle_a_mode: %d, swizzle_b_mode: %d, swizzle_cd_mode: %d\n", 
               smem_config.swizzle_a_mode, smem_config.swizzle_b_mode, smem_config.swizzle_cd_mode);
        printf("  num_tma_multicast: %u, tma_multicast_a: %d\n", num_tma_multicast, tma_multicast_a);
        printf("  num_tma_threads: %u, num_math_threads: %u\n", num_tma_threads, num_math_threads);
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
};




// divisibility conditions for multicast check
inline bool is_multicast_legal(
    const uint32_t & shape_dim,
    const uint32_t & block_dim,
    const uint32_t & num_multicast,
    const uint32_t & num_sms,
    bool require_divisible
) {
    bool divisible = !require_divisible ||
        host_ceil_div(shape_dim, block_dim) % num_multicast == 0;
    return divisible && num_sms % num_multicast == 0;
}

inline int get_swizzle_mode(const uint32_t & block_size, const uint32_t & elem_size) {
    const uint32_t & num_elements_bytes = block_size * elem_size;
    // prefer larger swizzling modes if possible
    for (const auto mode : {128, 64, 32, 16}) {
        if (num_elements_bytes % mode == 0)
            return mode;
    }
    std::string msg = "There does not exist a compatible swizzle mode for the current block size " +
    std::to_string(block_size) + " and elem_size " + std::to_string(elem_size);
    HOST_ERROR(msg.c_str());
    return 0; // unreachable
}



// checks for 
inline int get_compiled_dim(const std::string& compiled_dims, char dim, int dim_value) {
    for (const auto c : compiled_dims) {
        if (c == dim)
            return dim_value;
    }
    return 0;
}


static CUtensorMapDataType convert_to_cudtype(c10::ScalarType dtype) {

    switch (dtype) {
    case c10::ScalarType::Float:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case c10::ScalarType::Half:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case c10::ScalarType::BFloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case c10::ScalarType::Int:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case c10::ScalarType::Float8_e4m3fn:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default:
      HOST_ERROR("Unsupported dtype => CuTensorMap conversion");
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32; // unreachable
    }
  }
  
  static CUtensorMapSwizzle getCuSwizzle(const int &swizzle_size) {
    switch (swizzle_size) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      HOST_ERROR("sm90_utils.cuh, getCuSwizzle: unsupported swizzle mode!");
      return CU_TENSOR_MAP_SWIZZLE_NONE; // unreachable
    }
  }

  static CUtensorMapL2promotion getCuL2PromotionSize(const int& l2_promotion_size) {
    switch (l2_promotion_size) {
      case 0 : 
        return CU_TENSOR_MAP_L2_PROMOTION_NONE;
      case 64 : 
        return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
      case 128 :
        return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
      default:
        return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    }
  }
  
  inline std::pair<int, int> get_inner_outer_dims(Major major, int mn, int k) {
    int inner = (major == Major::MN) ? mn : k;
    int outer = (major == Major::MN) ? k : mn;
    return std::make_pair(inner, outer);
  }
  
  static CUtensorMap make_tma_2d_desc(at::Tensor &t,
                                      size_t  gmem_inner_dim, // faster dimension
                                      size_t gmem_outer_dim, size_t gmem_outer_stride,
                                      size_t smem_inner_dim, // faster dimension
                                      size_t smem_outer_dim, const int &swizzle_size, const int& l2_promotion_size, CUtensorMapFloatOOBfill_enum fill_mode = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    CUtensorMap map;
    size_t type_size = get_type_size(dtype_of(t));
    if (swizzle_size > 0) { // nonzero swizzle side
      // if there is a nonzero swizzling chunk, then TMA's box chunk (smem dims)
      // must match the size of the swizzling atom, since TMA applies the
      // swizzling pattern within the box load
      smem_inner_dim = swizzle_size / type_size;
      // ex) type_size = 8, swizzle atom size is 64B, then inner_box_dim = 8 to
      // match
    }
    // construct strides
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    // GMemStrides is always of length rank - 1, and the strides of the actual
    // atom are calculated column-major wise (L to R)
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * type_size)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
        printf("Making TMA desc: global memory: %zu %zu, shared memory: %zu %zu, outer stride: %zu, swizzle: %d elem size: %zu\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride * type_size, swizzle_size, type_size);
    }
    // set element strides to (1, 1) for dense striding
    const cuuint32_t elem_strides[2] = {1, 1};
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map, convert_to_cudtype(dtype_of(t)), 2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type, getCuL2PromotionSize(l2_promotion_size), fill_mode));
    return map;
  }
  
  static CUtensorMap make_tma_3d_desc(at::Tensor &t, size_t gmem_dim_0, size_t gmem_dim_1, size_t gmem_dim_2,
                                      size_t gmem_stride_1, size_t gmem_stride_2, size_t smem_dim_0, size_t smem_dim_1,
                                      size_t smem_dim_2, const int &swizzle_size, const int& l2_promotion_size) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    size_t type_size = get_type_size(dtype_of(t));
    if (swizzle_size > 0) {
      smem_dim_0 = swizzle_size / type_size;
    }
    CUtensorMap map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1),
                                     static_cast<cuuint64_t>(gmem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_1 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_2 * type_size)};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1),
                                     static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
      printf("Making TMA desc: global memory: %zu %zu %zu, shared memory: %zu %zu %zu, outer stride: %zu %zu, swizzle: %d elem size: %zu \n",
             gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2,
             gmem_stride_1, gmem_stride_2, swizzle_size, type_size);
  }
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map, convert_to_cudtype(dtype_of(t)), 3, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type, getCuL2PromotionSize(l2_promotion_size), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
  }
  static CUtensorMap make_tma_4d_desc(at::Tensor &t, size_t gmem_dim_0, size_t gmem_dim_1, size_t gmem_dim_2, size_t gmem_dim_3,
                                      size_t gmem_stride_1, size_t gmem_stride_2, size_t gmem_stride_3, size_t smem_dim_0, size_t smem_dim_1,
                                      size_t smem_dim_2, size_t smem_dim_3,const int &swizzle_size, const int& l2_promotion_size) {
    HOST_ASSERT(t.data_ptr() != nullptr, "Tensor data is null");
    size_t type_size = get_type_size(dtype_of(t));
    if (swizzle_size > 0) {
      smem_dim_0 = swizzle_size / type_size;
    }
    CUtensorMap map;
    const cuuint64_t gmem_dims[4] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1),
                                     static_cast<cuuint64_t>(gmem_dim_2), static_cast<cuuint64_t>(gmem_dim_3)};
    const cuuint64_t gmem_strides[3] = {static_cast<cuuint64_t>(gmem_stride_1 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_2 * type_size),
                                        static_cast<cuuint64_t>(gmem_stride_3 * type_size)};
    const cuuint32_t smem_dims[4] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1),
                                     static_cast<cuuint32_t>(smem_dim_2), static_cast<cuuint32_t>(smem_dim_3)};
    const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
    CUtensorMapSwizzle swizzle_type = getCuSwizzle(swizzle_size);
    if (get_env<int>("JIT_DEBUG")) {
      printf("Making TMA desc: global memory: %zu %zu %zu %zu, shared memory: %zu %zu %zu %zu, outer stride: %zu %zu %zu, swizzle: %d elem size: %zu \n",
             gmem_dim_0, gmem_dim_1, gmem_dim_2, gmem_dim_3, smem_dim_0, smem_dim_1, smem_dim_2, smem_dim_3,
             gmem_stride_1, gmem_stride_2, gmem_stride_3, swizzle_size, type_size);
  }
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map, convert_to_cudtype(dtype_of(t)), 4, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type, getCuL2PromotionSize(l2_promotion_size), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
  }
  
  static CUtensorMap make_tma_a_desc(at::Tensor &t, Major major, uint32_t num_groups, uint32_t block_m, uint32_t block_k,
                                     uint32_t outer_stride, const int &swizzle_size, const int& l2_promotion_size) {
    uint32_t global_m = t.size(-2);
    uint32_t global_k = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_m * num_groups, global_k);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_m, block_k);
    return make_tma_2d_desc(t, gmem_inner_dim, gmem_outer_dim, outer_stride, smem_inner_dim, smem_outer_dim,
                            swizzle_size, l2_promotion_size);
  }

  static CUtensorMap make_tma_a_desc_3d(at::Tensor& t, Major major, uint32_t num_groups, uint32_t block_m, uint32_t block_k,
                        const int& swizzle_size, const int l2_promotion_size) {
    uint32_t global_m = t.size(-2);
    uint32_t global_k = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_m, global_k);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_m, block_k);
    auto batch_stride = t.stride(0);
    auto outer_stride = major == Major::K ? t.stride(-2) : t.stride(-1);
    return make_tma_3d_desc(t, gmem_inner_dim, gmem_outer_dim, num_groups, 
    outer_stride, batch_stride, smem_inner_dim, smem_outer_dim, 1, swizzle_size, l2_promotion_size);
  }

  static CUtensorMap make_tma_b_desc_3d(at::Tensor& t, Major major, uint32_t num_groups, uint32_t block_n, uint32_t block_k,
                        const int& swizzle_size, const int l2_promotion_size) {
    uint32_t global_n = t.size(-2);
    uint32_t global_k = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_n, global_k);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_n, block_k);
    auto batch_stride = t.stride(0);
    auto outer_stride = major == Major::K ? t.stride(-2) : t.stride(-1);
    return make_tma_3d_desc(t, gmem_inner_dim, gmem_outer_dim, num_groups, 
    outer_stride, batch_stride, smem_inner_dim, smem_outer_dim, 1, swizzle_size, l2_promotion_size);
  }

  static CUtensorMap make_tma_d_desc_3d(at::Tensor& t, Major major, uint32_t num_groups, uint32_t block_m, uint32_t block_n,
                        const int& swizzle_size, const int l2_promotion_size) {
    uint32_t global_m = t.size(-2);
    uint32_t global_n = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_m, global_n);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_m, block_n);
    auto batch_stride = t.stride(0);
    auto outer_stride = major == Major::K ? t.stride(-2) : t.stride(-1);
    return make_tma_3d_desc(t, gmem_inner_dim, gmem_outer_dim, num_groups, 
    outer_stride, batch_stride, smem_inner_dim, smem_outer_dim, 1, swizzle_size, l2_promotion_size);
    }

  // static CUtensorMap make_tma_batch_a_desc(at::Tensor& t, Major major, int batch_size, int num_groups, int block_m, int block_k, int outer_stride, const int& swizzle_size) {
  //     // outer stride is always the outermost stride, for batched we can assume that the stride_1 is fixed by majorness
  //     int global_m = t.size(1);
  //     int global_k = t.size(2);
  //     auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_m, global_k);
  //     auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_m, block_k);
  //     return make_tma_3d_desc(t, gmem_inner_dim, gmem_outer_dim * num_groups, batch_size, 
  //       gmem_inner_dim, outer_stride, smem_inner_dim, smem_outer_dim, smem_inner_dim * smem_outer_dim, swizzle_size);
  // }
  
  static CUtensorMap make_tma_b_desc(at::Tensor &t, Major major, uint32_t num_groups, uint32_t block_n, uint32_t block_k,
                                     uint32_t outer_stride, const int &swizzle_size,const int& l2_promotion_size) {
    uint32_t global_n = t.size(-2);
    uint32_t global_k = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_n, global_k);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_n, block_k);
    return make_tma_2d_desc(t, gmem_inner_dim, gmem_outer_dim * num_groups, outer_stride, smem_inner_dim, smem_outer_dim,
                            swizzle_size, l2_promotion_size);
  }

  
  
  static CUtensorMap make_tma_d_desc(at::Tensor &t, Major major, uint32_t num_groups, uint32_t block_m, uint32_t block_n,
                                     uint32_t outer_stride, const int &swizzle_size, const int& l2_promotion_size) {
    uint32_t global_m = t.size(-2);
    uint32_t global_n = t.size(-1);
    auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, global_m, global_n);
    auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_m, block_n);
    return make_tma_2d_desc(t, gmem_inner_dim, gmem_outer_dim * num_groups, outer_stride, smem_inner_dim, smem_outer_dim,
                            swizzle_size, l2_promotion_size);
  }

  
  
  // scale factor tma descriptor (mostly for A)
  static CUtensorMap make_tma_sf_desc(at::Tensor &t, Major major, int num_groups, int global_mn, int global_k, int block_mn, int block_k) {
    if (major != Major::MN) {
      HOST_ERROR("Tma_sf_desc should be always in MN major");
    }

  
    // for the scale factor tensors, really for the activations, we should make mn
    // the contiguous dimension, since in any GEMM the k dimension is what is
    // looped over in blocks, while the MN dimension is accessed contiguously
    // across the rows/cols in the problem
  
    // this method makes the assumption that only one scale factor across K is
    // ever needed per k_loop, which is obvious the small ternary on the
    // gmem_outer_dim is for FP8-FP32 scale factor, vs NVFP4-U8M0 scale factor
  
    // since global_mn is the inner_dimension, and TMA requires Gmem addresses to
    // be 16 byte aligned furthermore the size of the data transfer must also be a
    // multiple of 16 bytes
    uint32_t num_elements = 16 / get_type_size(dtype_of(t));
    uint32_t aligned_global_mn = host_align(global_mn, (int)num_elements);
    // global_k is already in scale-block units (K/block_k for FP32, K/(block_k/4) for NVFP4)
    // so we should NOT divide by block_k again
    return make_tma_2d_desc(t, aligned_global_mn,
                            host_ceil_div(global_k, block_k) * (dtype_of(t) == c10::ScalarType::Float ? 1 : 4) * num_groups, // already in k_blocks, just multiply by num_groups
                            aligned_global_mn, block_mn, 1, 0, 0);
  }

// prepare chunk indices for chunked varlen kernels
inline std::vector<int> prepare_chunk_indices(
  const std::vector<int>& cu_seqlens,
  int chunk_size = 64
) {
    // find diffs first
    alignas(8) std::vector<int> chunks(cu_seqlens.size() - 1);
    for (int i = 1; i < cu_seqlens.size(); i++) {
        int length = cu_seqlens[i] - cu_seqlens[i-1];
        chunks[i - 1] = host_ceil_div(length, chunk_size);
    }
    std::vector<int> chunk_indices;

    for (int i = 0; i < chunks.size(); i++) {
        int num_chunks = chunks[i];
        for (int j = 0; j < num_chunks; j++) {
            chunk_indices.push_back(i); // batch_index
            chunk_indices.push_back(j); // chunk_index
        }
    }
    return chunk_indices;
}


inline std::vector<int> prepare_cu_chunks(
  const std::vector<int>& cu_seqlens,
  int chunk_size = 64,
  bool output_final_state = false
) {
  alignas(8) std::vector<int> chunks(cu_seqlens.size() - 1); // batch_size
    for (int i = 1; i < cu_seqlens.size(); i++) {
        int length = cu_seqlens[i] - cu_seqlens[i-1];
        chunks[i - 1] = host_ceil_div(length, chunk_size);
    }
  std::vector<int> cu_chunks; // (batch_size + 1)
  cu_chunks.push_back(0);
  int chunk_sum = 0;
  for (int i = 1; i < chunks.size() + 1; i++) {
    chunk_sum += chunks[i-1];
    cu_chunks.push_back(chunk_sum);
  }
  
  return cu_chunks;
}
