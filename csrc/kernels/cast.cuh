#pragma once

#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <moe_cuda/kernels/common/common.cuh>
#include <moe_cuda/error.hpp>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <optional>
#include <runtime/device.hpp>
#include <runtime/tensor_compat.h>
#include <runtime/utils.h>
#include <utility>

namespace moe_cuda {
namespace kernels {
namespace cast_impl {

// BF16 to F32
__global__ inline void cast_kernel_bf16_f32(__nv_bfloat16 *inp, float *out,
                                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  while (idx < n) {
    out[idx] = __bfloat162float(inp[idx]);
    idx += stride;
  }
}

// F32 to BF16
__global__ inline void cast_kernel_f32_bf16(float *inp, __nv_bfloat16 *out,
                                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  while (idx < n) {
    out[idx] = __float2bfloat16(inp[idx]);
    idx += stride;
  }
}

// BF16 to FP8 with block scaling
__global__ inline void
cast_bf16_to_e4m3_blkscaled_128_kernel(__nv_bfloat16 *inp, __nv_fp8_e4m3 *out,
                                       float *scale, size_t hidden_dim,
                                       size_t num_scales_per_row) {

  int tid = threadIdx.x;
  int warp_id = tid >> 5; // Which warp within this block (tid / 32)
  int lane_id = tid & 31; // Which thread within the warp (tid % 32)

  size_t col_start = blockIdx.y * ((blockDim.x >> 5) * 128);
  size_t warp_col_offset = warp_id * 128; // Each warp processes 128 elements
  size_t global_col = col_start + warp_col_offset;

  __nv_bfloat16 *inp_ptr = inp + blockIdx.x * hidden_dim + global_col;
  __nv_fp8_e4m3 *out_ptr =
      out + blockIdx.x * hidden_dim + global_col + lane_id * 4;
  __nv_bfloat16 bf16_vals[4];
  float f_vals[4];
  uint32_t row_max = 0;

  if (global_col + lane_id * 4 < hidden_dim) {
    uint2 val = ld_global_uint2_dispatch(
        reinterpret_cast<uint2 *>(inp_ptr + lane_id * 4));
    VEC_LOAD_CVT<uint2, __nv_bfloat16>::convert(val, &bf16_vals[0]);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      f_vals[i] = __bfloat162float(bf16_vals[i]);
      float abs_val = fabsf(f_vals[i]);
      uint32_t abs_val_int = *(uint32_t *)&abs_val;
      row_max = max(row_max, abs_val_int);
    }
  }
  unsigned int mask = __ballot_sync(uint32_t(-1), true);

  asm volatile("{redux.sync.max.u32 %0, %1, %2; \n}"
               : "=r"(row_max)
               : "r"(row_max), "r"(mask));

  float row_max_f = *(float *)&row_max;
  // for (int offset = 1; offset < 32; offset <<= 1) {

  //   float shfl_val = __shfl_xor_sync(uint32_t(-1), row_max, offset);
  //   unsigned int shfl_val_int = *(unsigned int *)&shfl_val;
  //   row_max = fmax(row_max, __shfl_xor_sync(uint32_t(-1), row_max, offset));
  // }

  // Avoid division by zero: if row_max is 0, set S to 1.0f
  float S = row_max_f > 0.0f ? row_max_f / 448.0f : 1.0f;

  if (global_col + lane_id * 4 < hidden_dim) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float clamped = fmaxf(-448.0f, fminf(448.0f, f_vals[i] / S));
      out_ptr[i] = __nv_fp8_e4m3(clamped);
    }
  }

  // Each warp's first thread writes its scale
  if (lane_id == 0 && global_col < hidden_dim) {
    size_t scale_idx = blockIdx.x * num_scales_per_row +
                       blockIdx.y * (blockDim.x >> 5) + warp_id;
    scale[scale_idx] = S;
  }
}

// if the intermediate size is K, then the output of the gate + up projection is
// (num_rows, 2 * K)
__global__ inline void fused_silu_mul_quant_kernel(__nv_bfloat16 *gemm_out,
                                                   __nv_fp8_e4m3 *swiglu_out,
                                                   float *scale,
                                                   size_t hidden_dim,
                                                   size_t num_scales_per_row) {
  int tid = threadIdx.x;
  int warp_id = tid >> 5; // Which warp within this block (tid / 32)
  int lane_id = tid & 31; // Which thread within the warp (tid % 32)

  size_t col_start = blockIdx.y * ((blockDim.x >> 5) * 128);
  size_t warp_col_offset = warp_id * 128; // Each warp processes 128 elements
  size_t gate_global_col = col_start + warp_col_offset;
  size_t up_global_col = col_start + warp_col_offset + hidden_dim;

  __nv_bfloat16 *inp_ptr = gemm_out + blockIdx.x * 2 * hidden_dim;
  __nv_fp8_e4m3 *out_ptr =
      swiglu_out + blockIdx.x * hidden_dim + gate_global_col + lane_id * 4;
  __nv_bfloat16 gate_vals[4];
  __nv_bfloat16 up_vals[4];

  uint32_t row_max = 0;
  float f_vals[8] = {0};

  if (gate_global_col + lane_id * 4 < hidden_dim) {
    uint2 gate = ld_global_uint2_dispatch(
        reinterpret_cast<uint2 *>(inp_ptr + gate_global_col + lane_id * 4));
    uint2 up = ld_global_uint2_dispatch(
        reinterpret_cast<uint2 *>(inp_ptr + up_global_col + lane_id * 4));

    VEC_LOAD_CVT<uint2, __nv_bfloat16>::convert(gate, &gate_vals[0]);
    VEC_LOAD_CVT<uint2, __nv_bfloat16>::convert(up, &up_vals[0]);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      f_vals[i] = __bfloat162float(gate_vals[i]);
      f_vals[i + 4] = __bfloat162float(up_vals[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      f_vals[i] = 0.0f;
      f_vals[i + 4] = 0.0f;
    }
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    float silu = f_vals[i] / (1.0f + expf(-f_vals[i]));
    float mul = silu * f_vals[i + 4];
    f_vals[i] = mul; // rewrite into registers, since we don't need the gate
                     // values anymore
  }

  bool pred = gate_global_col + lane_id * 4 < hidden_dim;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float abs_val = fabsf(f_vals[i]);
    uint32_t abs_val_int = *(uint32_t *)&abs_val;
    row_max = max(row_max, abs_val_int);
  }

  uint32_t mask = __ballot_sync(uint32_t(-1), pred);
  asm volatile("{redux.sync.max.u32 %0, %1, %2; \n}"
               : "=r"(row_max)
               : "r"(row_max), "r"(mask));

  float row_max_f = *(float *)&row_max;

  float S = row_max_f > 0.0f ? row_max_f / 448.0f : 1.0f;

  if (pred) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float clamped = fmaxf(-448.0f, fminf(448.0f, f_vals[i] / S));
      out_ptr[i] = __nv_fp8_e4m3(clamped);
    }
  }

  if (lane_id == 0 && gate_global_col < hidden_dim) {
    size_t scale_idx = blockIdx.x * num_scales_per_row +
                       blockIdx.y * (blockDim.x >> 5) + warp_id;
    scale[scale_idx] = S;
  }
}

// Launcher functions
inline void cast_bf16_to_f32(const __nv_bfloat16 *inp, float *out, size_t n,
                             dim3 grid, dim3 block, cudaStream_t stream) {
  nvtxRangePush("cast_bf16_to_f32");
  cast_kernel_bf16_f32<<<grid, block, 0, stream>>>(
      const_cast<__nv_bfloat16 *>(inp), out, n);
  nvtxRangePop();
}

inline void cast_f32_to_bf16(const float *inp, __nv_bfloat16 *out, size_t n,
                             dim3 grid, dim3 block, cudaStream_t stream) {
  nvtxRangePush("cast_f32_to_bf16");
  cast_kernel_f32_bf16<<<grid, block, 0, stream>>>(const_cast<float *>(inp),
                                                   out, n);
  nvtxRangePop();
}

inline void cast_bf16_to_fp8_blkscaled(const __nv_bfloat16 *inp,
                                       __nv_fp8_e4m3 *out, float *scale,
                                       size_t num_rows, size_t hidden_dim,
                                       cudaStream_t stream) {
  constexpr int kBlockSize = 128; // each warp can do one 128 elem chunk
  size_t num_scales_per_row = hidden_dim >> 7;
  HOST_ASSERT(uintptr_t(inp) % 8 == 0,
              "inp must be aligned to 8 bytes, because of bf16x4 loads");
  HOST_ASSERT(hidden_dim % kBlockSize == 0,
              "hidden_dim must be divisible by kBlockSize");

  const int num_threads = std::min(
      num_scales_per_row * 32,
      static_cast<size_t>(device_prop->get_prop()->maxThreadsPerBlock));
  const int num_col_ctas = ti_ceil_div(hidden_dim, (num_threads >> 5) * 128);

  dim3 dimGrid(num_rows, num_col_ctas);
  dim3 dimBlock(num_threads);

  if (get_env<int>("KERNEL_DEBUG", 0)) {
    printf("cast_bf16_to_fp8_blkscaled: num_rows: %zu, num_col_ctas: %d, "
           "hidden_dim: %zu, num_scales_per_row: %zu\n",
           num_rows, num_col_ctas, hidden_dim, num_scales_per_row);
  }
  auto kernel = (void *)cast_bf16_to_e4m3_blkscaled_128_kernel;
  void *args[] = {
      &inp, &out, &scale, &hidden_dim, &num_scales_per_row,
  };

  nvtxRangePush("cast_bf16_to_fp8_blkscaled");
  CUDA_CHECK(cudaLaunchKernel(kernel, dimGrid, dimBlock, args, 0, stream));
  nvtxRangePop();
  CUDA_SYNC_DEBUG();
}

inline void fused_silu_mul_quant(__nv_bfloat16 *gemm_out,
                                 __nv_fp8_e4m3 *swiglu_out, float *scale,
                                 size_t num_rows, size_t hidden_dim,
                                 cudaStream_t stream) {
  constexpr int kBlockSize = 128; // each warp can do one 128 elem chunk
  size_t num_scales_per_row = hidden_dim >> 7;
  HOST_ASSERT(uintptr_t(gemm_out) % 8 == 0,
              "gemm_out must be aligned to 8 bytes, because of bf16x4 loads");
  HOST_ASSERT(hidden_dim % kBlockSize == 0,
              "hidden_dim must be divisible by kBlockSize");

  const int num_threads = std::min(
      num_scales_per_row * 32,
      static_cast<size_t>(device_prop->get_prop()->maxThreadsPerBlock));
  const int num_col_ctas = ti_ceil_div(hidden_dim, (num_threads >> 5) * 128);

  dim3 dimGrid(num_rows, num_col_ctas);
  dim3 dimBlock(num_threads);
  if (get_env<int>("KERNEL_DEBUG", 0)) {
    printf("fused_silu_mul_quant: num_rows: %zu, num_col_ctas: %d, hidden_dim: "
           "%zu, num_scales_per_row: %zu\n",
           num_rows, num_col_ctas, hidden_dim, num_scales_per_row);
  }
  auto kernel = (void *)fused_silu_mul_quant_kernel;
  void *args[] = {
      &gemm_out, &swiglu_out, &scale, &hidden_dim, &num_scales_per_row,
  };
  nvtxRangePush("fused_silu_mul_quant");
  CUDA_CHECK(cudaLaunchKernel(kernel, dimGrid, dimBlock, args, 0, stream));
  nvtxRangePop();
  CUDA_SYNC_DEBUG();
}

inline void cast_(at::Tensor &inp, at::Tensor &out,
                  std::optional<at::Tensor> &scale, cudaStream_t stream) {
  NvtxRange range("cast");
  if (dtype_of(inp) == dtype_of(out)) {
    printf("inp and out are the same dtype\n");
    out = std::move(inp);
    return;
  }
  HOST_ASSERT(inp.dim() == out.dim(),
              "cast.cu: cast input and output must have matching dimensions");
  for (int i = 0; i < inp.dim(); ++i) {
    HOST_ASSERT(inp.size(i) == out.size(i),
                "cast.cu: cast input and output must have matching sizes");
  }

  const int block_size = 128;
  size_t nelem = inp.numel();

  // Get number of SMs for grid limiting
  int num_sms = device_prop->get_num_sms();
  int maxThreadsPerSM = device_prop->get_prop()->maxThreadsPerMultiProcessor;

  // Calculate grid size, capped to avoid launching too many blocks
  int max_blocks = num_sms * maxThreadsPerSM / block_size; // 512
  int needed_blocks = (nelem + block_size - 1) / block_size;
  int grid_size = (needed_blocks > max_blocks) ? max_blocks : needed_blocks;

  dim3 grid(grid_size);
  dim3 block(block_size);

  if (dtype_of(inp) == c10::ScalarType::BFloat16 &&
      dtype_of(out) == c10::ScalarType::Float) {
    cast_bf16_to_f32(
        reinterpret_cast<const __nv_bfloat16 *>(
            inp.data_ptr<c10::BFloat16>()),
        out.data_ptr<float>(), nelem, grid, block, stream);
    CUDA_SYNC_DEBUG();
  } else if (dtype_of(inp) == c10::ScalarType::Float &&
             dtype_of(out) == c10::ScalarType::BFloat16) {
    cast_f32_to_bf16(
        inp.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<c10::BFloat16>()),
        nelem, grid, block, stream);
    CUDA_SYNC_DEBUG();
  } else if (dtype_of(inp) == c10::ScalarType::BFloat16 &&
             dtype_of(out) == c10::ScalarType::Float8_e4m3fn) {
    HOST_ASSERT(scale.has_value(), "cast.cu, bf16 to fp8 requires an "
                                   "additional scale tensor to be passed in");
    HOST_ASSERT(dtype_of(*scale) == c10::ScalarType::Float,
                "cast.cu: scale tensor for fp8 must be fp32");
    size_t num_rows = 1;
    for (int i = 0; i < inp.dim() - 1; i++) {
      num_rows *= inp.size(i);
    }
    cast_bf16_to_fp8_blkscaled(
        reinterpret_cast<const __nv_bfloat16 *>(
            inp.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_fp8_e4m3 *>(
            out.data_ptr<c10::Float8_e4m3fn>()),
        scale->data_ptr<float>(), num_rows, inp.size(-1), stream);
    CUDA_SYNC_DEBUG();
  } else {
    printf("inp dtype: %d, out dtype: %d\n", static_cast<int>(dtype_of(inp)),
           static_cast<int>(dtype_of(out)));
    HOST_ERROR("cast.cu: Unsupported cast operation");
  }
  CUDA_CHECK(cudaGetLastError());
}

} // namespace cast_impl
} // namespace kernels
} // namespace moe_cuda
