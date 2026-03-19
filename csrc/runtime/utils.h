#pragma once
#include <vector>
#include <queue>
#include <moe_cuda/error.hpp>
#include <string>
#include <nvtx3/nvToolsExt.h>
class NvtxRange {
    public:
        explicit NvtxRange(const char * s) noexcept {
            nvtxRangePush(s);
        }; 
        NvtxRange(const std::string& base_str, int number) {
            std::string range_string = base_str + " " + std::to_string(number);
            nvtxRangePush(range_string.c_str());
        };
        ~NvtxRange() noexcept {
            nvtxRangePop();
        };
};

// pool of streams, round robin allocation

struct StreamPool {
    int num_devices;
    int num_streams;
    int oldest_stream;
    std::vector<cudaStream_t> streams;
    std::queue<int> available;

    StreamPool(int num_streams, int num_devices) noexcept: num_streams(num_streams), available(),
    num_devices(num_devices), streams(num_streams), oldest_stream(0) {
        #pragma unroll
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            available.push(i);
        }
    }

    inline int fetchStream(cudaStream_t& stream_pointer) {
         if (available.empty()) {
            throw std::runtime_error("StreamPool: No available streams");
            return -1;
         }
         else {
            int stream_ind = available.front();
            stream_pointer = streams[stream_ind];
            available.pop();
            return stream_ind;
         }
    };
    
    inline void returnStream(int stream_index) {
        available.push(stream_index);
    }

    ~StreamPool() {
        #pragma unroll
        for (int i = 0; i < streams.size(); i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }
};


template<typename dtype_t>
static dtype_t get_env(const std::string& name, dtype_t default_val = dtype_t()) {
    auto env_var = std::getenv(name.c_str());

    if (env_var == NULL) {
        return default_val;
    }
    if constexpr (std::is_same_v<dtype_t, std::string>) {
        return std::string(env_var);
    }
    else if constexpr (std::is_same_v<dtype_t, int>) {
        return std::atoi(env_var);
    }
    else {
        throw std::runtime_error("Invalid dtype for env variable"); // we need to throw here, because using HOST_ASSERT will create an ambiguous error trace
        return default_val; // unreachable
    }
}

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

#define _LAUNCH_TYPE(kind, var, value, ...)                                    \
  case kind: {                                                                 \
    using var = value;                                                         \
    {                                                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
    break;                                                                     \
  }

#define _LAUNCH_VALUE(kind, var, value, ...)                                   \
  case kind: {                                                                 \
    static constexpr decltype(value) var = value;                              \
    {                                                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
    break;                                                                     \
  }

// macro specialization for model hidden dim
#ifndef LAUNCH_TOKEN_DIM
#define LAUNCH_TOKEN_DIM(dtype, var, ...)                                      \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(2048, var, 2048, __VA_ARGS__)                          \
    _LAUNCH_VALUE(4096, var, 4096, __VA_ARGS__)                          \
    _LAUNCH_VALUE(7168, var, 7168, __VA_ARGS__)                          \
  default: {                                                                   \
    HOST_ERROR("invalid token dim size"); \
  }; break;                                                                    \
  }

#endif
#ifndef LAUNCH_HIDDEN_DIM_SCALE
#define LAUNCH_HIDDEN_DIM_SCALE(dtype, var, ...)                               \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(16, var, 16, __VA_ARGS__)                              \
    _LAUNCH_VALUE(32, var, 32, __VA_ARGS__)                              \
    _LAUNCH_VALUE(56, var, 56, __VA_ARGS__)                              \
  default: {                                                                   \
    HOST_ERROR("invalid hidden dim scale size"); \
  }; break;                                                                    \
  }

#endif

#ifndef LAUNCH_NUM_EXPERTS_PER_TOKEN
#define LAUNCH_NUM_EXPERTS_PER_TOKEN(dtype, var, ...)                          \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(8, var, 8, __VA_ARGS__)                                \
    _LAUNCH_VALUE(10, var, 10, __VA_ARGS__)                              \
    _LAUNCH_VALUE(12, var, 12, __VA_ARGS__) \
  default: {                                                                   \
    HOST_ERROR("invalid num experts per token value"); \
    break;                                                                     \
  }                                                                            \
  }
#endif


#ifndef LAUNCH_NUM_EXPERTS
#define LAUNCH_NUM_EXPERTS(dtype, var, ...)                                     \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(128, var, 128, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(256, var, 256, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(512, var, 512, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(1024, var, 1024, __VA_ARGS__)                                      \
  default: {                                                                   \
    HOST_ERROR("invalid num experts value");         \
    break;                                                                     \
  }                                                                            \
  }
#endif

#ifndef LAUNCH_WORLD_SIZE
#define LAUNCH_WORLD_SIZE(dtype, var, ...)                                     \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(1, var, 1, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(2, var, 2, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(4, var, 4, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(8, var, 8, __VA_ARGS__)                                      \
  default: {                                                                   \
    assert(false && "Unsupported world size (must be 1, 2, 4, or 8)");         \
    break;                                                                     \
  }                                                                            \
  }
#endif

#ifndef LAUNCH_VEC_SIZE
#define LAUNCH_VEC_SIZE(dtype, var, ...)                                       \
  switch (dtype) {                                                             \
    _LAUNCH_VALUE(1, var, 1, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(2, var, 2, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(4, var, 4, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(8, var, 8, __VA_ARGS__)                                      \
    _LAUNCH_VALUE(16, var, 16, __VA_ARGS__)                                    \
  default: {                                                                   \
    assert(false && "Unsupported vec load size (must be 1, 2, 4, 8, 16)");     \
    break;                                                                     \
  }                                                                            \
  }
#endif

#ifndef LAUNCH_ACT_TYPE
#define LAUNCH_ACT_TYPE(dtype, var, ...)                                       \
  switch (dtype) {                                                             \
    _LAUNCH_TYPE(c10::ScalarType::BFloat16, var, __nv_bfloat16, __VA_ARGS__)   \
    _LAUNCH_TYPE(c10::ScalarType::Float, var, float, __VA_ARGS__)              \
    _LAUNCH_TYPE(c10::ScalarType::Half, var, __half, __VA_ARGS__)              \
  default: {                                                                   \
    assert(false && "Unsupported dtype");                                      \
    break;                                                                     \
  }                                                                            \
  }
#endif

#ifndef LAUNCH_CONDITIONAL_TYPE
#define LAUNCH_CONDITIONAL_TYPE(flag, true_dtype, false_dtype, var, ...)       \
  if (flag) {                                                                  \
    using var = true_dtype;                                                    \
    {                                                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } else {                                                                     \
    using var = false_dtype;                                                   \
    {                                                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }
#endif

// macro to map CC -> kNumThreads
#ifndef DISPATCH_COMPUTE_CAP_NUM_THREADS
#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }
#endif

#ifndef DISPATCH_BOOL_FLAG
#define DISPATCH_BOOL_FLAG(flag, var, ...)                                     \
  if (flag) {                                                                  \
    static constexpr bool var = true;                                          \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    static constexpr bool var = false;                                         \
    __VA_ARGS__                                                                \
  }
#endif

