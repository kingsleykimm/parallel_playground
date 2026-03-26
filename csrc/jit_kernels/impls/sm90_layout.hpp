#pragma once
#include <jit/compiler.hpp>
#include <jit/runtime.hpp>
#include <jit/utils/culib.hpp>
#include <jit_kernels/heuristics/heuristics.hpp>
#include <runtime/device.hpp>
#include <runtime/format.hpp>
#include <runtime/tensor.h>

class SM90_Transpose_SF_Runtime : LaunchRuntime<SM90_Transpose_SF_Runtime> {
public:
  struct Args {
    uint32_t num_threads;
    uint32_t block_mn;
    uint32_t sf_k;

    float *sf;
    float *out;
    size_t mn, aligned_mn;

    cudaStream_t stream;
    LaunchConfig launch_config;
  };

  static std::string generate_impl(const Args &args) {
    const std::string code =
        fmt::format(R"(
            #include <moe_cuda/kernels/sm90_layout.cuh>

            using namespace moe_cuda::kernels::sm90_layout_impl;

            static void __instantiate_kernel() {{

            auto kernel_ptr = reinterpret_cast<void *>(&transpose_fp32_sf<
            {}, {}, {}
            >);
            }}
            )",
                    args.num_threads, args.block_mn, args.sf_k);
    return code;
  }

  static void launch_impl(KernelHandle &kernel,
                          const LaunchConfigHandle &launch_config,
                          const Args &args) {
    CUDA_CHECK(launch_kernel(kernel, launch_config, args.sf, args.out, args.mn,
                             args.aligned_mn));
  }
};

class SM90_Transpose_BF16_Runtime : LaunchRuntime<SM90_Transpose_BF16_Runtime> {
public:
  struct Args {
    uint32_t num_threads;
    uint32_t block_mn;
    uint32_t sf_k;

    __nv_bfloat16 *sf;
    __nv_bfloat16 *out;
    size_t mn, aligned_mn;

    cudaStream_t stream;
    LaunchConfig launch_config;
  };

  static std::string generate_impl(const Args &args) {
    const std::string code =
        fmt::format(R"(
            #include <moe_cuda/kernels/sm90_layout.cuh>

            using namespace moe_cuda::kernels::sm90_layout_impl;

            static void __instantiate_kernel() {{

            auto kernel_ptr = reinterpret_cast<void *>(&transpose_generic<
            __nv_bfloat16,
            {}, {}, {}
            >);
            }}
            )",
                    args.num_threads, args.block_mn, args.sf_k);
    return code;
  }

  static void launch_impl(KernelHandle &kernel,
                          const LaunchConfigHandle &launch_config,
                          const Args &args) {
    CUDA_CHECK(launch_kernel(kernel, launch_config, args.sf, args.out, args.mn,
                             args.aligned_mn));
  }
};

// when calling any of these methods, it is assumed sfa and sfb are row-major
inline void sm90_transpose_sf(at::Tensor &sf, at::Tensor &transposed_sf,
                              cudaStream_t &stream) {
  // determine num_threads ,block_mn, sf_k
  // HOST_ASSERT(sf.dim() < 3, "scale factor ndim should be less than 2 for
  // GEMMs");
  HOST_ASSERT(sf.dim() == 2 || sf.dim() == 3,
              "scale factor ndim should be 2 or 3 for GEMMs");
  (sf.dim() == 2) ? custom::unsqueeze(sf, 0) : void();

  size_t num_groups = sf.size(0);
  size_t mn = sf.size(1);
  size_t k = sf.size(2);
  // determine BLOCK_MN with a small heuristic search based off of mn size
  const auto [block_mn, num_threads, smem_size] = get_transpose_config(mn, k);
  const size_t aligned_mn = host_align(mn, 16 / get_type_size(dtype_of(sf)));
  LaunchConfig launch_config = {dim3(num_threads),
                                dim3(host_ceil_div(mn, block_mn), num_groups),
                                stream, smem_size, 1};
  SM90_Transpose_SF_Runtime::Args args = {static_cast<uint32_t>(num_threads),
                                          (uint32_t)block_mn,
                                          static_cast<uint32_t>(k),
                                          sf.data_ptr<float>(),
                                          transposed_sf.data_ptr<float>(),
                                          mn,
                                          aligned_mn,
                                          stream,
                                          launch_config};
  const std::string &code =
      LaunchRuntime<SM90_Transpose_SF_Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("sm90_transpose_sf", code);
  LaunchRuntime<SM90_Transpose_SF_Runtime>::launch(runtime, args);
}

inline void sm90_transpose_bf16(__nv_bfloat16 *input, __nv_bfloat16 *output,
                                size_t mn, size_t k, size_t num_groups,
                                uint32_t alignment, cudaStream_t &stream) {
  const auto [block_mn, num_threads, smem_size] =
      get_transpose_config(mn, k, c10::ScalarType::BFloat16);
  const size_t aligned_mn = host_align(mn, alignment / 2);
  LaunchConfig launch_config = {dim3(num_threads),
                                dim3(host_ceil_div(mn, block_mn), num_groups),
                                stream, smem_size, 1};

  SM90_Transpose_BF16_Runtime::Args args = {static_cast<uint32_t>(num_threads),
                                            (uint32_t)block_mn,
                                            static_cast<uint32_t>(k),
                                            input,
                                            output,
                                            mn,
                                            aligned_mn,
                                            stream,
                                            launch_config};

  const std::string &code =
      LaunchRuntime<SM90_Transpose_BF16_Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("sm90_transpose_bf16", code);
  LaunchRuntime<SM90_Transpose_BF16_Runtime>::launch(runtime, args);
}

inline void sm90_transpose_fp32(float *input, float *output, size_t mn,
                                size_t k, size_t num_groups, uint32_t alignment,
                                cudaStream_t &stream) {
  const auto [block_mn, num_threads, smem_size] = get_transpose_config(mn, k);
  const size_t aligned_mn = host_align(mn, alignment / 4);
  LaunchConfig launch_config = {dim3(num_threads),
                                dim3(host_ceil_div(mn, block_mn), num_groups),
                                stream, smem_size, 1};

  SM90_Transpose_SF_Runtime::Args args = {static_cast<uint32_t>(num_threads),
                                          (uint32_t)block_mn,
                                          static_cast<uint32_t>(k),
                                          input,
                                          output,
                                          mn,
                                          aligned_mn,
                                          stream,
                                          launch_config};

  const std::string &code =
      LaunchRuntime<SM90_Transpose_SF_Runtime>::generate(args);
  std::shared_ptr<KernelRuntime> runtime =
      compiler->build("sm90_transpose_fp32", code);
  LaunchRuntime<SM90_Transpose_SF_Runtime>::launch(runtime, args);
}
