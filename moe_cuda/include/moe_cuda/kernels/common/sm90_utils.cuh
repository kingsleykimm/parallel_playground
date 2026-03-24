/*
Based off of
https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/common/sm90_utils.cuh
 */
#pragma once

#include <kittens.cuh>
#include <moe_cuda/kernels/common/common.cuh>

template <uint32_t kSwizzleMode> struct getSwizzleEnum {
  static int get_value() {
    if constexpr (kSwizzleMode == 0) {
      return 0;
    } else if constexpr (kSwizzleMode == 128) {
      return 1;
    } else if constexpr (kSwizzleMode == 64) {
      return 2;
    } else if constexpr (kSwizzleMode == 32) {
      return 3;
    }
  }
};

// inline ptx

__forceinline__ __device__ uint32_t pack_float_2(float a, float b) {
  uint32_t val;
  asm volatile("{ cvt.rn.bf16x2.f32 %0, %1, %2; }"
               : "=r"(val)
               : "f"(a), "f"(b)
               :);
  return val;
}

// store into shared, "l" = long, "f" = float
__forceinline__ __device__ void st_shared(const __nv_bfloat16 *ptr,
                                          __nv_bfloat16 val) {
  uint16_t val16 = __bfloat16_as_ushort(val);
  asm volatile("{ st.shared.u16 [%0], %1; }"
               :
               : "l"(__cvta_generic_to_shared(ptr)), "h"(val16)
               :);
}

__forceinline__ __device__ void st_shared(const float *ptr, float val) {
  asm volatile("{ st.shared.f32 [%0], %1; }"
               :
               : "l"(__cvta_generic_to_shared(ptr)), "f"(val)
               :);
}

__forceinline__ __device__ void st_shared(const float2 *ptr, float2 val) {
  asm volatile("{ st.shared.v2.f32 [%0], {%1, %2}; }"
               :
               : "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y)
               :);
}

__forceinline__ __device__ void st_shared(const uint32_t *ptr, uint32_t val) {
  asm volatile(
      "{ st.shared.u32 [%0], %1; }" ::"l"(__cvta_generic_to_shared(ptr)),
      "r"(val));
}

__forceinline__ __device__ void st_shared(const uint2 *ptr, uint2 val) {
  asm volatile("{ st.shared.v2.u32 [%0], {%1, %2}; }" ::"l"(
                   __cvta_generic_to_shared(ptr)),
               "r"(val.x), "r"(val.y));
}

__forceinline__ __device__ void st_shared(const uint4 *ptr, uint4 val) {
  asm volatile("{st.shared.v4.u32 [%0], {%1, %2, %3, %4};}" ::"l"(
                   __cvta_generic_to_shared(ptr)),
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__forceinline__ __device__ float ld_shared(float *ptr) {
  float ret;
  asm volatile("{ ld.shared.f32 %0, [%1]; }"
               : "=f"(ret)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return ret;
}

__forceinline__ __device__ float2 ld_shared(float2 *ptr) {
  float ret1, ret2;
  asm volatile("{ ld.shared.v2.f32 {%0, %1}, [%2]; }"
               : "=f"(ret1), "=f"(ret2)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return make_float2(ret1, ret2);
}

__forceinline__ __device__ float4 ld_shared(float4 *ptr) {
  float ret1, ret2, ret3, ret4;
  asm volatile("{ ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; }"
               : "=f"(ret1), "=f"(ret2), "=f"(ret3), "=f"(ret4)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return make_float4(ret1, ret2, ret3, ret4);
}

__forceinline__ __device__ __nv_bfloat162 ld_sharedbf16x2(__nv_bfloat16 *ptr) {
  uint32_t ret;
  asm volatile("{ ld.shared.u32 %0, [%1]; }"
               : "=r"(ret)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return *reinterpret_cast<__nv_bfloat162 *>(&ret);
}

__forceinline__ __device__ __nv_bfloat16 ld_shared(__nv_bfloat16 *ptr) {
  uint16_t ret;
  asm volatile("{ ld.shared.u16 %0, [%1]; }"
               : "=h"(ret)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return __short_as_bfloat16(ret);
  ;
}

__forceinline__ __device__ uint32_t ld_shared(uint32_t *ptr) {
  uint32_t ret;
  asm volatile("{ ld.shared.u32 %0, [%1]; }"
               : "=r"(ret)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return ret;
}

__forceinline__ __device__ uint2 ld_shared(uint2 *ptr) {
  uint32_t ret1, ret2;
  asm volatile("{ ld.shared.v2.u32 {%0, %1}, [%2]; }"
               : "=r"(ret1), "=r"(ret2)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return make_uint2(ret1, ret2);
}

__forceinline__ __device__ uint4 ld_shared(uint4 *ptr) {
  uint32_t ret1, ret2, ret3, ret4;
  asm volatile("{ ld.shared.v4.u32 {%0, %1, %2, %3}, [%4]; }"
               : "=r"(ret1), "=r"(ret2), "=r"(ret3), "=r"(ret4)
               : "l"(__cvta_generic_to_shared(ptr))
               : "memory");
  return make_uint4(ret1, ret2, ret3, ret4);
}

// special STSM matrix instruction (CC 9.0 <=) that loads from rMem to Smem
template <typename dtype_t> struct custom_SM90_U32x2_STSM_N {
  __device__ static void copy(dtype_t src_0, dtype_t src_1, void *smem_dst) {
    const uint32_t src[2] = {*reinterpret_cast<uint32_t *>(&src_0),
                             *reinterpret_cast<uint32_t *>(&src_1)};
    asm volatile(
        "{ stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2}; }\n" ::"l"(
            __cvta_generic_to_shared(smem_dst)),
        "r"(src[0]), "r"(src[1]));
  }
};

struct custom_SM90_U32x2_STLM_N {
  __device__ static uint2 load(void *smem_dst) {
    uint32_t src_0, src_1;
    asm volatile("{ ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];}\n"
                 : "=r"(src_0), "=r"(src_1)
                 : "l"(__cvta_generic_to_shared(smem_dst))
                 : "memory");
    return make_uint2(src_0, src_1);
  }
};

__forceinline__ __host__ __device__ uint32_t ld_volatile_u32(uint32_t *ptr) {
  uint32_t x;
  asm volatile("{ ld.volatile.u32 %0, [%1]; }\n"
               : "=r"(x)
               : "l"(ptr)
               : "memory");
  return x;
}

__forceinline__ __host__ __device__ void st_volatile_u32(uint32_t *ptr,
                                                         uint32_t val) {
  asm volatile("{ st.volatile.u32 [%1], %0; }\n" ::"r"(val), "l"(ptr) :);
}

__forceinline__ __host__ __device__ uint32_t ld_acquire_u32(uint32_t *ptr) {
  uint32_t x;
  asm volatile("{ ld.acquire.sys.global.u32 %0, [%1]; }\n"
               : "=r"(x)
               : "l"(ptr)
               :);
  return x;
}

__forceinline__ __host__ __device__ void st_release_u32(uint32_t *ptr,
                                                        uint32_t val) {
  asm volatile("{ st.release.sys.global.u32 [%1], %0; }\n" ::"r"(val), "l"(ptr)
               :);
}

__forceinline__ __host__ __device__ uint32_t ti_elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "     mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(laneid), "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

// this will use the noncoherent (texture) cache, which doesn't require
// maintaining coherency with global memory. only
// unintended for read-only data, much faster

__forceinline__ __host__ __device__ uint4 ld_global_nc_uint4(const uint4 *ptr) {
  uint4 val;

  // L1 no allocate
  // L2 - cache prefetch size of 256B, since we're almost always fetching
  // contiguous chunks of uint4
  asm volatile("{ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, "
               "%3}, [%4]; }\n"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(ptr)
               :);
  return val;
}

__forceinline__ __device__ void st_global_nc_uint4(uint4 val,
                                                   const uint4 *ptr) {
  asm volatile(
      "{st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};\n}" ::"l"(ptr),
      "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
      :);
}

__forceinline__ __device__ uint16_t ld_global_uint16_dispatch(uint16_t *ptr) {
  uint16_t result;
  asm volatile("{ ld.global.u16 %0, [%1]; \n}" : "=h"(result) : "l"(ptr) :);
  return result;
}

__forceinline__ __device__ uint32_t ld_global_uint32_dispatch(uint32_t *ptr) {
  uint32_t result;
  asm volatile("{ ld.global.u32 %0, [%1]; \n}" : "=r"(result) : "l"(ptr) :);
  return result;
}

__forceinline__ __device__ uint2 ld_global_uint2_dispatch(uint2 *ptr) {
  uint2 result;
  asm volatile("{ ld.global.v2.u32 {%0, %1}, [%2]; \n}"
               : "=r"(result.x), "=r"(result.y)
               : "l"(ptr)
               :);
  return result;
}

__forceinline__ __device__ uint4 ld_global_uint4_dispatch(uint4 *ptr) {
  uint4 result;
  asm volatile("{ ld.global.v4.u32 {%0, %1, %2, %3}, [%4]; \n}"
               : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
               : "l"(ptr)
               :);
  return result;
}

template <typename uint_t>
__forceinline__ __device__ uint_t ld_global_uint_dispatch(uint_t *ptr) {
  if constexpr (std::is_same_v<uint_t, uint16_t>) {
    return ld_global_uint16_dispatch(ptr);
  } else if constexpr (std::is_same_v<uint_t, uint32_t>) {
    return ld_global_uint32_dispatch(ptr);
  } else if constexpr (std::is_same_v<uint_t, uint2>) {
    return ld_global_uint2_dispatch(ptr);
  } else if constexpr (std::is_same_v<uint_t, uint4>) {
    return ld_global_uint4_dispatch(ptr);
  } else {
    DEVICE_ASSERT(false);
  }
}

__forceinline__ __device__ void st_global_uint16_dispatch(uint16_t val,
                                                          uint16_t *ptr) {
  asm volatile("{ st.global.u16 [%0], %1; \n}" ::"l"(ptr), "h"(val) :);
}

__forceinline__ __device__ void st_global_uint32_dispatch(uint32_t val,
                                                          uint32_t *ptr) {
  asm volatile("{ st.global.u32 [%0], %1; \n}" ::"l"(ptr), "r"(val) :);
}

__forceinline__ __device__ void st_global_uint2_dispatch(uint2 val,
                                                         uint2 *ptr) {
  asm volatile("{ st.global.v2.u32 [%0], {%1, %2}; \n}" ::"l"(ptr), "r"(val.x),
               "r"(val.y)
               :);
}

__forceinline__ __device__ void st_global_uint4_dispatch(uint4 val,
                                                         uint4 *ptr) {
  asm volatile("{ st.global.v4.u32 [%0], {%1, %2, %3, %4}; \n}" ::"l"(ptr),
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               :);
}

template <typename uint_t>
__forceinline__ __device__ void st_global_uint_dispatch(uint_t val,
                                                        uint_t *ptr) {
  if constexpr (std::is_same_v<uint_t, uint16_t>) {
    st_global_uint16_dispatch(val, ptr);
  } else if constexpr (std::is_same_v<uint_t, uint32_t>) {
    st_global_uint32_dispatch(val, ptr);
  } else if constexpr (std::is_same_v<uint_t, uint2>) {
    st_global_uint2_dispatch(val, ptr);
  } else if constexpr (std::is_same_v<uint_t, uint4>) {
    st_global_uint4_dispatch(val, ptr);
  } else {
    DEVICE_ASSERT(false);
  }
}

// =============== Cluster Operations ================= //

__forceinline__ __device__ uint32_t
mapa_shared_addr_cluster(uint32_t *smem_addr, uint32_t peer_rank) {
  uint32_t result_u32;
  uint32_t smem_u32 =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
  asm volatile("{mapa.shared::cluster.u32 %0, %1, %2; } \n"
               : "=r"(result_u32)
               : "r"(smem_u32), "r"(peer_rank)
               :);
  return result_u32;
}

__forceinline__ __device__ uint32_t atom_add_shared_cluster(uint32_t smem_addr,
                                                            uint32_t val) {
  uint32_t result;
  asm volatile("{atom.add.shared::cluster.u32 %0, [%1], %2; \n}"
               : "=r"(result)
               : "r"(smem_addr), "r"(val)
               :);
  return result;
}

__forceinline__ __device__ uint32_t ld_shared_cluster_u32(uint32_t smem_addr) {
  uint32_t result;
  asm volatile("{ld.shared::cluster.u32 %0, [%1]; \n}"
               : "=r"(result)
               : "r"(smem_addr)
               :);
  return result;
}

// ===== Multi-GPU Synchronization ==== //

namespace node_sync {
template <class barrier_t> // pgl
__device__ static inline void
signal(const barrier_t &barrier,
       const kittens::coord<kittens::ducks::default_type> &idx,
       const int dst_dev_idx, const int val) {
  asm volatile(
      "{st.release.sys.global.s32 [%0], %1;}" ::"l"(&barrier[dst_dev_idx][idx]),
      "r"(val)
      : "memory");
}

template <class barrier_t>
__device__ static inline void
wait(const barrier_t &barrier,
     const kittens::coord<kittens::ducks::default_type> &idx, const int dev_idx,
     const int expected) {
  int val;
  do {
    asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}"
                 : "=r"(val)
                 : "l"(&barrier[dev_idx][idx])
                 : "memory");
  } while (val != expected);
}
} // namespace node_sync
// vectorized loads
template <size_t PADDED_SF_K> struct vec_load_size;

template <> struct vec_load_size<2> {
  using type = float2;
};

template <> struct vec_load_size<4> {
  using type = float4;
};
