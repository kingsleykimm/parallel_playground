// based off of
// https://github.com/perplexityai/pplx-garden/blob/main/p2p-all-to-all/a2a-kernels/src/core/launch_utils.cuh
#pragma once
#include <cassert>
#include <moe_cuda/dtype.h>

// thin wrapper to indicate static variable
template <size_t V> class Fixed {
public:
  __device__ Fixed(size_t value) {}

  __device__ operator size_t() { return V; }

  static constexpr size_t Value = V;
};

class NotFixed {
public:
  __device__ NotFixed(size_t value) : value_(value) {}
  __device__ operator size_t() const { return value_; }

private:
  size_t value_;
};

#define _LAUNCH_TYPE(kind, var, value, ...)                                                                            \
  case kind: {                                                                                                         \
    using var = value;                                                                                                 \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
    break;                                                                                                             \
  }

#define _LAUNCH_VALUE(kind, var, value, ...)                                                                           \
  case kind: {                                                                                                         \
    static constexpr decltype(value) var = value;                                                                      \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
    break;                                                                                                             \
  }

// macro specialization for model hidden dim
#ifndef LAUNCH_TOKEN_DIM
#define LAUNCH_TOKEN_DIM(dtype, var, ...)                                                                              \
  switch (dtype) {                                                                                                     \
    _LAUNCH_TYPE(2048, var, Fixed<2048>, __VA_ARGS__)                                                                  \
    _LAUNCH_TYPE(4096, var, Fixed<4096>, __VA_ARGS__)                                                                  \
    _LAUNCH_TYPE(7168, var, Fixed<7168>, __VA_ARGS__)                                                                  \
  default: {                                                                                                           \
    using var = NotFixed;                                                                                              \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
  }; break;                                                                                                            \
  }

#endif
#ifndef LAUNCH_HIDDEN_DIM_SCALE
#define LAUNCH_HIDDEN_DIM_SCALE(dtype, var, ...)                                                                       \
  switch (dtype) {                                                                                                     \
    _LAUNCH_TYPE(16, var, Fixed<16>, __VA_ARGS__)                                                                      \
    _LAUNCH_TYPE(32, var, Fixed<32>, __VA_ARGS__)                                                                      \
    _LAUNCH_TYPE(56, var, Fixed<56>, __VA_ARGS__)                                                                      \
  default: {                                                                                                           \
    using var = NotFixed;                                                                                              \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
  }; break;                                                                                                            \
  }

#endif

#ifndef LAUNCH_NUM_EXPERTS_PER_TOKEN
#define LAUNCH_NUM_EXPERTS_PER_TOKEN(dtype, var, ...)                                                                  \
  switch (dtype) {                                                                                                     \
    _LAUNCH_TYPE(6, var, Fixed<6>, __VA_ARGS__)                                                                        \
    _LAUNCH_TYPE(8, var, Fixed<8>, __VA_ARGS__)                                                                        \
    _LAUNCH_TYPE(10, var, Fixed<10>, __VA_ARGS__)                                                                      \
  default: {                                                                                                           \
    using var = NotFixed;                                                                                              \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
    break;                                                                                                             \
  }                                                                                                                    \
  }
#endif

#ifndef LAUNCH_WORLD_SIZE
#define LAUNCH_WORLD_SIZE(dtype, var, ...)                                                                             \
  switch (dtype) {                                                                                                     \
    _LAUNCH_VALUE(1, var, 1, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(2, var, 2, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(4, var, 4, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(8, var, 8, __VA_ARGS__)                                                                              \
  default: {                                                                                                           \
    assert(false && "Unsupported world size (must be 1, 2, 4, or 8)");                                                 \
    break;                                                                                                             \
  }                                                                                                                    \
  }
#endif

#ifndef LAUNCH_VEC_SIZE
#define LAUNCH_VEC_SIZE(dtype, var, ...)                                                                               \
  switch (dtype) {                                                                                                     \
    _LAUNCH_VALUE(1, var, 1, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(2, var, 2, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(4, var, 4, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(8, var, 8, __VA_ARGS__)                                                                              \
    _LAUNCH_VALUE(16, var, 16, __VA_ARGS__)                                                                            \
  default: {                                                                                                           \
    assert(false && "Unsupported vec load size (must be 1, 2, 4, 8, 16)");                                             \
    break;                                                                                                             \
  }                                                                                                                    \
  }
#endif

#ifndef LAUNCH_ACT_TYPE
#define LAUNCH_ACT_TYPE(dtype, var, ...)                                                                               \
  switch (dtype) {                                                                                                     \
    _LAUNCH_TYPE(c10::ScalarType::BFloat16, var, __nv_bfloat16, __VA_ARGS__)                                                   \
    _LAUNCH_TYPE(c10::ScalarType::Float, var, float, __VA_ARGS__)                                                           \
    _LAUNCH_TYPE(c10::ScalarType::Half, var, __half, __VA_ARGS__)                                                          \
  default: {                                                                                                           \
    assert(false && "Unsupported dtype");                                                                              \
    break;                                                                                                             \
  }                                                                                                                    \
  }
#endif

#ifndef LAUNCH_CONDITIONAL_TYPE
#define LAUNCH_CONDITIONAL_TYPE(flag, true_dtype, false_dtype, var, ...)                                               \
  if (flag) {                                                                                                          \
    using var = true_dtype;                                                                                            \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
  } else {                                                                                                             \
    using var = false_dtype;                                                                                           \
    {                                                                                                                  \
      __VA_ARGS__;                                                                                                     \
    }                                                                                                                  \
  }
#endif

// macro to map CC -> kNumThreads
#ifndef DISPATCH_COMPUTE_CAP_NUM_THREADS
#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...)                                         \
  if (compute_capacity.first >= 8) {                                                                                   \
    constexpr uint32_t BLOCK_THREADS = 1024;                                                                           \
    __VA_ARGS__                                                                                                        \
  } else {                                                                                                             \
    constexpr uint32_t BLOCK_THREADS = 512;                                                                            \
    __VA_ARGS__                                                                                                        \
  }
#endif

#ifndef DISPATCH_BOOL_FLAG
#define DISPATCH_BOOL_FLAG(flag, var, ...)                                                                             \
  if (flag) {                                                                                                          \
    static constexpr bool var = true;                                                                                  \
    __VA_ARGS__                                                                                                        \
  } else {                                                                                                             \
    static constexpr bool var = false;                                                                                 \
    __VA_ARGS__                                                                                                        \
  }
#endif
