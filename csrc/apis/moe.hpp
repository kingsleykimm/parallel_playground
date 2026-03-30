#pragma once

#include "moe_cuda/types.h"
#include <jit_kernels/heuristics/common.hpp>
#include <jit_kernels/impls/kernel2.hpp>
#include <jit_kernels/impls/kernel3.hpp>
#include <jit_kernels/impls/kernel4.hpp>
#include <jit_kernels/impls/kernel5_1.hpp>
#include <jit_kernels/impls/kernel5_2.hpp>
#include <jit_kernels/impls/sm90_bf16_gemm.hpp>
#include <jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp>
#include <runtime/device.hpp>
#include <runtime/tensor_compat.h>

namespace moe_cuda {
namespace api {

// we always assume SFA is K-major
inline void fp8_gemm_nt(at::Tensor &act, at::Tensor &act_scale,
                        at::Tensor &weight, at::Tensor &weight_scale,
                        at::Tensor &output, GemmType gemm_type,
                        const std::string &compiled_dims, int *grouped_layout,
                        cudaStream_t &stream) {
  if (gemm_type != GemmType::MGroupedMasked) {
    HOST_ASSERT(act.dim() < 3,
                "A tensor for FP8 GEMM must have less than three dims");
  }

  auto sfa_mn_major = act_scale.transpose(-1, -2).contiguous();

  if (gemm_type == GemmType::Normal) {
    sm90_fp8_gemm_1d2d_nt(act, weight, sfa_mn_major, weight_scale, output,
                          stream);
  }
  //  else if (gemm_type == GemmType::MGroupedContiguous) {
  //     HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null
  //     for grouped FP8 GEMM"); sm90_fp8_grouped_gemm_1d2d_contiguous(
  //         act.first, weight.first, sfa_mn_major, weight.second, output,
  //         compiled_dims, grouped_layout, stream);
  // } else if (gemm_type == GemmType::MGroupedMasked) {
  //     HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null
  //     for grouped FP8 GEMM"); sm90_fp8_grouped_gemm_1d2d_masked(
  //         act.first, weight.first, sfa_mn_major, weight.second, output,
  //         compiled_dims, grouped_layout, stream);
  // }
  else {
    HOST_ASSERT(false, "Batched FP8 GEMM is not implemented");
  }
}

inline void fp8_grouped_gemm(at::Tensor &act, at::Tensor &act_scale,
                             at::Tensor &weight, at::Tensor &weight_scale,
                             at::Tensor &output, GemmType gemm_type,
                             int *grouped_layout, cudaStream_t &stream) {
  if (gemm_type == GemmType::MGroupedContiguous) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped FP8 GEMM");
    sm90_fp8_grouped_gemm_contiguous(act, weight, act_scale, weight_scale,
                                     output, grouped_layout, stream);
  } else if (gemm_type == GemmType::MGroupedMasked) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped FP8 GEMM");
    sm90_fp8_grouped_gemm_masked(act, weight, act_scale, weight_scale, output,
                                 grouped_layout, stream);
  } else {
    HOST_ASSERT(
        false,
        "fp8_grouped_gemm only supports MGroupedContiguous and MGroupedMasked");
  }
}

inline void bf16_gemm(at::Tensor &A, at::Tensor &B,
                      std::optional<at::Tensor> &C, at::Tensor &D,
                      GemmType gemm_type, const std::string &compiled_dims,
                      int *grouped_layout, cudaStream_t &stream) {
  if (gemm_type != GemmType::MGroupedMasked) {
    HOST_ASSERT(A.dim() < 3,
                "A tensor for BF16 GEMM must have less than three dims");
  }

  if (gemm_type == GemmType::Normal) {
    sm90_bf16_gemm(A, B, C, D, compiled_dims, stream);
  } else if (gemm_type == GemmType::MGroupedContiguous) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped BF16 GEMM");
    sm90_bf16_grouped_gemm_contiguous(A, B, D, compiled_dims, grouped_layout,
                                      stream);
  } else if (gemm_type == GemmType::MGroupedMasked) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped BF16 GEMM");
    sm90_bf16_grouped_gemm_masked(A, B, D, compiled_dims, grouped_layout,
                                  stream);
  } else if (gemm_type == GemmType::Batched) {
    HOST_ASSERT(A.dim() == 3,
                "A tensor for BF16 GEMM must have three dims for batched mode");
    sm90_bf16_batched_gemm(A, B, D, compiled_dims, stream);
  }
}

inline void fp8_grouped_gemm_swiglu(at::Tensor &A, at::Tensor &gate_weight,
                                    at::Tensor &up_weight, at::Tensor &scale_a,
                                    at::Tensor &scale_gate,
                                    at::Tensor &scale_up, at::Tensor &scale_d,
                                    at::Tensor &D, GemmType gemm_type,
                                    int *grouped_layout, cudaStream_t &stream) {
  if (get_env<int>("MOE_CUDA_DEBUG") != 0) {
    printf("FP8 Grouped GEMM Swiglu launching in moe.hpp \n");
  }
  HOST_ASSERT(grouped_layout != nullptr,
              "grouped_layout cannot be null for grouped FP8 swiglu GEMM");
  if (gemm_type == GemmType::MGroupedMasked) {
    kernel3_masked(A, up_weight, gate_weight, scale_a, scale_up, scale_gate,
                   scale_d, D, grouped_layout, stream);
  } else {
    kernel3_contiguous(A, up_weight, gate_weight, scale_a, scale_up, scale_gate,
                       scale_d, D, grouped_layout, stream);
  }
}

inline void fp8_grouped_gemm_swiglu_sub(at::Tensor &A, at::Tensor &gate_weight,
                                    at::Tensor &up_weight, at::Tensor &scale_a,
                                    at::Tensor &scale_gate,
                                    at::Tensor &scale_up, at::Tensor &scale_d,
                                    at::Tensor &D, GemmType gemm_type,
                                    int *grouped_layout, cudaStream_t &stream) {
  HOST_ASSERT(grouped_layout != nullptr,
              "grouped_layout cannot be null for grouped FP8 swiglu GEMM (sub)");
  if (gemm_type == GemmType::MGroupedMasked) {
    kernel3_sub_masked(A, up_weight, gate_weight, scale_a, scale_up, scale_gate,
                       scale_d, D, grouped_layout, stream);
  } else {
    kernel3_sub_contiguous(A, up_weight, gate_weight, scale_a, scale_up,
                           scale_gate, scale_d, D, grouped_layout, stream);
  }
}

void fp8_grouped_gemm_swiglu_consumer_pp(
    at::Tensor &A, at::Tensor &gate_weight, at::Tensor &up_weight,
    at::Tensor &scale_a, at::Tensor &scale_gate, at::Tensor &scale_up,
    at::Tensor &scale_d, at::Tensor &D, GemmType gemm_type, int *grouped_layout,
    cudaStream_t &stream) {

  HOST_ASSERT(grouped_layout != nullptr,
              "grouped_layout cannot be null for masked grouped FP8 swiglu "
              "GEMM consumer PP");
  if (gemm_type == GemmType::MGroupedMasked) {
    kernel4_masked(A, gate_weight, up_weight, scale_a, scale_gate, scale_up,
                   scale_d, D, grouped_layout, stream);
  } else {
    kernel4_contiguous(A, gate_weight, up_weight, scale_a, scale_gate, scale_up,
                       scale_d, D, grouped_layout, stream);
  }
}

} // namespace api
} // namespace moe_cuda
