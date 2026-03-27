#pragma once

#include "moe_cuda/types.h"
#include <jit_kernels/heuristics/common.hpp>
#include <jit_kernels/impls/kernel2.hpp>
#include <jit_kernels/impls/kernel3.hpp>
#include <jit_kernels/impls/kernel4.hpp>
#include <jit_kernels/impls/kernel5_1.hpp>
// #include <jit_kernels/impls/sm90_bf16_gemm.hpp>
#include <jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp>
#include <runtime/device.hpp>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
namespace moe_cuda {
namespace api {

// we always assume SFA is K-major; normal (non-grouped) FP8 GEMM NT only
inline void fp8_gemm_nt(torch::stable::Tensor &act,
                        torch::stable::Tensor &act_scale,
                        torch::stable::Tensor &weight,
                        torch::stable::Tensor &weight_scale,
                        torch::stable::Tensor &output,
                        [[maybe_unused]] const std::string &compiled_dims,
                        cudaStream_t &stream) {
  HOST_ASSERT(act.dim() < 3,
              "A tensor for FP8 GEMM must have less than three dims");
  auto sfa_mn_major = contiguous(transpose(act_scale, -1, -2));
  sm90_fp8_gemm_1d2d_nt(act, weight, sfa_mn_major, weight_scale, output,
                        stream);
}

inline void fp8_grouped_gemm(torch::stable::Tensor &act,
                             torch::stable::Tensor &act_scale,
                             torch::stable::Tensor &weight,
                             torch::stable::Tensor &weight_scale,
                             torch::stable::Tensor &output, GemmType gemm_type,
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

// inline void bf16_gemm(torch::stable::Tensor &A, torch::stable::Tensor &B,
//                       std::optional<torch::stable::Tensor> &C,
//                       torch::stable::Tensor &D, GemmType gemm_type, const
//                       std::string &compiled_dims, int *grouped_layout,
//                       cudaStream_t &stream) {
//   if (gemm_type != GemmType::MGroupedMasked) {
//     HOST_ASSERT(A.dim() < 3,
//                 "A tensor for BF16 GEMM must have less than three dims");
//   }

//   if (gemm_type == GemmType::Normal) {
//     sm90_bf16_gemm(A, B, C, D, compiled_dims, stream);
//   } else if (gemm_type == GemmType::MGroupedContiguous) {
//     HOST_ASSERT(grouped_layout != nullptr,
//                 "grouped_layout cannot be null for grouped BF16 GEMM");
//     sm90_bf16_grouped_gemm_contiguous(A, B, D, compiled_dims, grouped_layout,
//                                       stream);
//   } else if (gemm_type == GemmType::MGroupedMasked) {
//     HOST_ASSERT(grouped_layout != nullptr,
//                 "grouped_layout cannot be null for grouped BF16 GEMM");
//     sm90_bf16_grouped_gemm_masked(A, B, D, compiled_dims, grouped_layout,
//                                   stream);
//   } else if (gemm_type == GemmType::Batched) {
//     HOST_ASSERT(A.dim() == 3,
//                 "A tensor for BF16 GEMM must have three dims for batched
//                 mode");
//     sm90_bf16_batched_gemm(A, B, D, compiled_dims, stream);
//   }
// }

inline void fp8_grouped_gemm_swiglu(
    torch::stable::Tensor &A, torch::stable::Tensor &gate_weight,
    torch::stable::Tensor &up_weight, torch::stable::Tensor &scale_a,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
    GemmType gemm_type, int *grouped_layout, cudaStream_t &stream) {
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

void fp8_grouped_gemm_swiglu_consumer_pp(
    torch::stable::Tensor &A, torch::stable::Tensor &gate_weight,
    torch::stable::Tensor &up_weight, torch::stable::Tensor &scale_a,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &scale_d, torch::stable::Tensor &D,
    GemmType gemm_type, int *grouped_layout, cudaStream_t &stream) {

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
