#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cublasLt.h>
#include <moe_cuda/error.hpp>
// Note: device.hpp includes this file, don't include device.hpp here to avoid circular dependency
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#if defined(ENABLE_FP32) 
#define CUBLAS_P CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_P CUDA_R_16F
#else
#define CUBLAS_P CUDA_R_16BF
#endif

// extern size_t workspaceSize; // for UVA servers
// // size_t workSpaceSize = 32 * 1024 * 1024; // for hopper
// extern std::byte * workspace;

const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
const int heuristics_cache_capacity = 8192 * 2;
// cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F_FAST_16BF;

inline void CUBLAS_CHECK_(cublasStatus_t status, const char * file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error at file %s:%d, %s", file, line, _cudaGetErrorEnum(status));
        exit(EXIT_FAILURE);
    }
}

#define CUBLAS_CHECK(status) CUBLAS_CHECK_(status, __FILE__, __LINE__);
