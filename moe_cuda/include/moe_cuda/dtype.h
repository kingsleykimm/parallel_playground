#pragma once
// utilities for cuda / cpu dtypes

#include "cuda_common.h"
#include <c10/core/ScalarType.h>
#include <cassert>
#include <cstdio>
#include <library_types.h>
#include <string>
#include <type_traits>

inline size_t get_type_size(c10::ScalarType type) {
    switch (type) {
        case c10::ScalarType::Long:
            return 8;
        case c10::ScalarType::Float:
        case c10::ScalarType::Int:
            return 4;
        case c10::ScalarType::Half:
        case c10::ScalarType::BFloat16:
        case c10::ScalarType::Short:
            return 2;
        case c10::ScalarType::Char:
        case c10::ScalarType::Float8_e4m3fn:
        case c10::ScalarType::Bool:
            return 1;
    }
    return 0;
}

inline std::string type_to_string(c10::ScalarType type) {
    switch (type) {
        case c10::ScalarType::Long: return "INT64";
        case c10::ScalarType::Float: return "FP32";
        case c10::ScalarType::Int: return "INT32";
        case c10::ScalarType::Half: return "FP16";
        case c10::ScalarType::BFloat16: return "BF16";
        case c10::ScalarType::Short: return "INT16";
        case c10::ScalarType::Char: return "INT8";
        case c10::ScalarType::Bool: return "BOOL";
        case c10::ScalarType::Float8_e4m3fn: return "FP8";
    }
    return "";
}

// Torch type conversion functions are in dtype_torch.h to avoid c10 dependency
// Table of finfo attributes for FP32 and BF16
// These are the same as torch.finfo(torch.float32) and torch.finfo(torch.bfloat16)
struct DTypeFInfo {
    float eps;
    float min;
    float max;
};

constexpr DTypeFInfo finfo_table[] = {

    // this can be expanded later, mainly for testing
    // FP32
    {1.1920928955078125e-07f, -3.4028234663852886e+38f, 3.4028234663852886e+38f},
    // FP16 (not used here, but could be added)
    {6.103515625e-05f, -65504.0f, 65504.0f},
    // BF16
    {0.0078125f, -3.38953139e+38f, 3.38953139e+38f},
    // INT32
    {1.0f, -2147483648.0f, 2147483647.0f},
    // INT16
    {1.0f, -32768, 32767},
    // INT8
    {1.0f, -128, 127},
    // BOOL
    {1.0f, 0, 1},
    // FP8 (E4M3)
    {0.0625f, -448.0, 448.0}
};



// Helper to get finfo for a dtype
inline const DTypeFInfo& get_finfo_from_TDType(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Float:
            return finfo_table[0];
        case c10::ScalarType::BFloat16:
            return finfo_table[2];
        default:
            // For unsupported types, fallback to FP32
            return finfo_table[0];
    }
}
template<typename T>
inline const DTypeFInfo& get_finfo_from_typename() {
    if (std::is_same<T, float>::value) {
        return finfo_table[0];
    }
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return finfo_table[2];
    }
    else if (std::is_same<T, __nv_fp8_e4m3>::value) {
        return finfo_table[7];
    }
    else {
        printf("DType is not supported yet");
        assert (false);
        return finfo_table[0];
    }
}
template<typename T>
inline bool tensordtype_match(c10::ScalarType dtype) {
    if (dtype == c10::ScalarType::Float) {
        return std::is_same<T, float>::value;
    }
    else if (dtype == c10::ScalarType::BFloat16) {
        return std::is_same<T, __nv_bfloat16>::value;
    }
    else if (dtype == c10::ScalarType::Half) {
        return std::is_same<T, __half>::value;
    }
    else if (dtype == c10::ScalarType::Bool) {
        return std::is_same<T, bool>::value;
    }
    else if (dtype == c10::ScalarType::Int) {
        return std::is_same<T, int32_t>::value;
    }
    else if (dtype == c10::ScalarType::Short) {
        return std::is_same<T, int16_t>::value;
    }
    else if (dtype == c10::ScalarType::Char) {
        return std::is_same<T, int8_t>::value;
    }
    else if (dtype == c10::ScalarType::Float8_e4m3fn) {
        return std::is_same<T, __nv_fp8_e4m3>::value;
    }
    else {
        return false;
    }
}

inline void * dtype_cast_ptr(c10::ScalarType dtype, void * data) {
    switch (dtype) {
        case c10::ScalarType::Float:
            return reinterpret_cast<float *>(data);
        case c10::ScalarType::BFloat16:
            return reinterpret_cast<__nv_bfloat16 *>(data);
        case c10::ScalarType::Half:
            return reinterpret_cast<__half *>(data);
        case c10::ScalarType::Long:
            return reinterpret_cast<int64_t *>(data);
        case c10::ScalarType::Int:
            return reinterpret_cast<int32_t *>(data);
        case c10::ScalarType::Short:
            return reinterpret_cast<int16_t *>(data);
        case c10::ScalarType::Char:
            return reinterpret_cast<int8_t *>(data);
        case c10::ScalarType::Bool:
            return reinterpret_cast<bool *>(data);
        case c10::ScalarType::Float8_e4m3fn:
            return reinterpret_cast<__nv_fp8_e4m3 *>(data);
        default:
            return reinterpret_cast<float *>(data);
    }
}

inline cudaDataType_t tensorDType_to_cudaDType(
    c10::ScalarType type
) {
    switch (type) {
        case c10::ScalarType::Float:
            return CUDA_R_32F;
        case c10::ScalarType::Half:
            return CUDA_R_16F;
        case c10::ScalarType::BFloat16:
            return CUDA_R_16BF;
        case c10::ScalarType::Int:
            return CUDA_R_32I;
        default:
            // You may want to handle error cases more gracefully
            return CUDA_R_32F;
    }
}

template<typename T>
inline cudaDataType_t toCudaDType() {

    if constexpr (std::is_same<T, float>::value) return CUDA_R_32F;
    else if constexpr (std::is_same<T, __half>::value) return CUDA_R_16F;
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) return CUDA_R_16BF;
    else if constexpr (std::is_same<T, int32_t>::value) return CUDA_R_32I;
    return CUDA_R_32F;
}

// cuda compute mappings, TO
// default trait
template<typename T> struct compute_t { using type = T; };
// bfloat16 trait
template<> struct compute_t<__nv_bfloat16> { using type = float; };
template<> struct compute_t<__half> {using type = float; };
template <typename T> using compute_t_t = typename compute_t<T>::type;

// helper function for casting
template<typename T>
inline __device__ compute_t_t<T> to_compute(T val) { 
    if constexpr (std::is_same<T, __nv_bfloat16>::value ) {
        return  __bfloat162float(val);
    }
    else if constexpr (std::is_same<T, __half>::value) {
        return __half2float(val);
    }
    else if constexpr (std::is_same<T, float>::value) {
        return val;
    }
    else {
        static_assert(sizeof(T) == 0, "dtype is not supported yet");
    }
 }
template <typename T>
inline __device__ T from_compute(compute_t_t<T> val) {
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return __float2bfloat16(val);
    }
    else if constexpr (std::is_same<T, __half>::value) {
        return __float2half(val);
    }
    else if constexpr (std::is_same<T, float>::value) {
        return val;
    }
    else {
        static_assert(sizeof(T) == 0, "dtype is not supported yet");
    }
}

template <c10::ScalarType dtype> struct tDType_t { };
template<> struct tDType_t<c10::ScalarType::Float> {using type = float; };
template<> struct tDType_t<c10::ScalarType::BFloat16> {using type = __nv_bfloat16; };
template<> struct tDType_t<c10::ScalarType::Half> {using type = __half; };

template<c10::ScalarType T>
inline typename tDType_t<T>::type from_tensordtype() {
    return typename tDType_t<T>::type{};
}
