#pragma once
#include <cuda.h>
#include <driver_types.h>
#include <jit/utils/common.hpp>
#include <moe_cuda/error.hpp>
#include <runtime/device.hpp>

using KernelHandle = CUfunction;

using LaunchConfigHandle = CUlaunchConfig;
using LibraryHandle = CUmodule;

template <typename... Args>
CUresult launch_kernel(KernelHandle &kernel,
                       const LaunchConfigHandle &launch_config,
                       Args &&...args) {
  // array of void pointers, args... expands the parameter pack, and takes
  // per-addresses const_cast needed because CUDA API takes void** but doesn't
  // modify the data
  void *kernelParams[] = {
      const_cast<void *>(static_cast<const void *>(&args))...};

  if (get_env<int>("JIT_DEBUG") > 0) {
    printf("  Launching kernel: grid=(%u,%u,%u), block=(%u,%u,%u), smem=%u, "
           "numAttrs=%u\n",
           launch_config.gridDimX, launch_config.gridDimY,
           launch_config.gridDimZ, launch_config.blockDimX,
           launch_config.blockDimY, launch_config.blockDimZ,
           launch_config.sharedMemBytes, launch_config.numAttrs);
    fflush(stdout);
  }

  CUresult result =
      cuLaunchKernelEx(&launch_config, kernel, kernelParams, nullptr);

  if (result != CUDA_SUCCESS) {
    const char *error_str;
    cuGetErrorString(result, &error_str);
    printf("  cuLaunchKernelEx FAILED: %s (code %d)\n", error_str, result);
  }
  return result;
}

inline LaunchConfigHandle
create_launch_config(KernelHandle &kernel, const int &smem_size,
                     const dim3 &blockDim, const dim3 &gridDim,
                     const uint32_t &num_multicast, cudaStream_t stream, bool cooperative = false) {
  if (smem_size > 0) {
    CUDA_CHECK(cuFuncSetAttribute(
        kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));
  }

  LaunchConfigHandle launch_handle;
  std::memset(&launch_handle, 0, sizeof(launch_handle));
  launch_handle.blockDimX = blockDim.x;
  launch_handle.blockDimY = blockDim.y;
  launch_handle.blockDimZ = blockDim.z;
  launch_handle.gridDimX = gridDim.x;
  launch_handle.gridDimY = gridDim.y;
  launch_handle.gridDimZ = gridDim.z;
  launch_handle.hStream = (CUstream)stream;
  launch_handle.sharedMemBytes = smem_size;
  // Static so the attrs array outlives the returned handle (which stores a
  // pointer into it).  Thread-safety note: callers serialise kernel launches
  // on a single stream, so a thread_local static is sufficient.
  static thread_local CUlaunchAttribute attrs[2];
  launch_handle.attrs = attrs;
  launch_handle.numAttrs = 0;

  if (num_multicast > 1) {
    // Verify grid is divisible by cluster dimension
    if (gridDim.x % num_multicast != 0) {
      printf("ERROR: gridDim.x=%u is not divisible by cluster size=%u\n",
             gridDim.x, num_multicast);
    }
    // Required for cluster sizes > 1: allow non-portable cluster configurations
    if (num_multicast > device_prop->get_max_clusters().first) {
      CUDA_CHECK(cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }
    attrs[launch_handle.numAttrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[launch_handle.numAttrs].value.clusterDim.x = num_multicast;
    attrs[launch_handle.numAttrs].value.clusterDim.y = 1;
    attrs[launch_handle.numAttrs].value.clusterDim.z = 1;
    launch_handle.numAttrs++;

    if (get_env<int>("JIT_DEBUG") > 0) {
      printf("  Cluster launch: grid=%u, cluster=%u\n", gridDim.x,
             num_multicast);
    }
  }

  if (cooperative) {
    attrs[launch_handle.numAttrs].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[launch_handle.numAttrs].value.cooperative = 1;
    launch_handle.numAttrs++;
  }

  return launch_handle;
}
