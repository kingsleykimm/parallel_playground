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
