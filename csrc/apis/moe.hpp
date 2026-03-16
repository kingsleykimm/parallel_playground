#pragma once

#include <jit_kernels/impls/sm90_bf16_gemm.hpp>
#include <jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp>
#include <jit_kernels/impls/kernel2.hpp>
#include <jit_kernels/impls/kernel3.hpp>
#include <jit_kernels/heuristics/common.hpp>
#include <runtime/device.hpp>
#include <runtime/tensor_compat.h>

namespace moe_cuda {
namespace api {


// we always assume SFA is K-major
inline void fp8_gemm_nt(std::pair<at::Tensor&, at::Tensor&> act,
                     std::pair<at::Tensor&, at::Tensor&> weight,
                     at::Tensor& output,
                     GemmType gemm_type,
                     const std::string& compiled_dims,
                     int* grouped_layout,
                     cudaStream_t& stream) {
    if (gemm_type != GemmType::MGroupedMasked) {
        HOST_ASSERT(act.first.dim() < 3, "A tensor for FP8 GEMM must have less than three dims");
    }



    auto sfa_mn_major = act.second.transpose(-1, -2).contiguous();

    if (gemm_type == GemmType::Normal) {
        sm90_fp8_gemm_1d2d_nt(act.first, weight.first, sfa_mn_major, weight.second, output, stream);
    }
    //  else if (gemm_type == GemmType::MGroupedContiguous) {
    //     HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped FP8 GEMM");
    //     sm90_fp8_grouped_gemm_1d2d_contiguous(
    //         act.first, weight.first, sfa_mn_major, weight.second, output, compiled_dims, grouped_layout, stream);
    // } else if (gemm_type == GemmType::MGroupedMasked) {
    //     HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped FP8 GEMM");
    //     sm90_fp8_grouped_gemm_1d2d_masked(
    //         act.first, weight.first, sfa_mn_major, weight.second, output, compiled_dims, grouped_layout, stream);
    // }
    else {
        HOST_ASSERT(false, "Batched FP8 GEMM is not implemented");
    }
}

inline void fp8_grouped_gemm(std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream) {
    // HOST_ASSERT(major_of(act.second) == Major::MN,
    //             "grouped FP8 GEMM expects scale_a in MN-major layout");

    if (gemm_type == GemmType::MGroupedContiguous) {
        HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped FP8 GEMM");
        sm90_fp8_grouped_gemm_contiguous(
            act.first, weight.first, act.second, weight.second, output, grouped_layout, stream);
    } else if (gemm_type == GemmType::MGroupedMasked) {
        HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped FP8 GEMM");
        sm90_fp8_grouped_gemm_masked(
            act.first, weight.first, act.second, weight.second, output, grouped_layout, stream);
    } else {
        HOST_ASSERT(false, "fp8_grouped_gemm only supports MGroupedContiguous and MGroupedMasked");
    }
}

inline void bf16_gemm(at::Tensor& A,
                      at::Tensor& B,
                      std::optional<at::Tensor>& C,
                      at::Tensor& D,
                      GemmType gemm_type,
                      const std::string& compiled_dims,
                      int* grouped_layout,
                      cudaStream_t& stream) {
    if (gemm_type != GemmType::MGroupedMasked) {
        HOST_ASSERT(A.dim() < 3, "A tensor for BF16 GEMM must have less than three dims");
    }

    if (gemm_type == GemmType::Normal) {
        sm90_bf16_gemm(A, B, C, D, compiled_dims, stream);
    } else if (gemm_type == GemmType::MGroupedContiguous) {
        HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped BF16 GEMM");
        sm90_bf16_grouped_gemm_contiguous(A, B, D, compiled_dims, grouped_layout, stream);
    } else if (gemm_type == GemmType::MGroupedMasked) {
        HOST_ASSERT(grouped_layout != nullptr, "grouped_layout cannot be null for grouped BF16 GEMM");
        sm90_bf16_grouped_gemm_masked(A, B, D, compiled_dims, grouped_layout, stream);
    } else if (gemm_type == GemmType::Batched) {
        HOST_ASSERT(A.dim() == 3, "A tensor for BF16 GEMM must have three dims for batched mode");
        sm90_bf16_batched_gemm(A, B, D, compiled_dims, stream);
    }
}

inline void fp8_grouped_gemm_swiglu_contiguous(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    int* grouped_layout,
    cudaStream_t& stream) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for grouped FP8 swiglu GEMM");
    kernel3_contiguous(A, up_weight, gate_weight, scale_a, scale_up, scale_gate,
                       scale_d, D, grouped_layout, stream);
}

inline void fp8_grouped_gemm_swiglu_masked(
    at::Tensor& A,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& scale_a,
    at::Tensor& scale_gate,
    at::Tensor& scale_up,
    at::Tensor& scale_d,
    at::Tensor& D,
    int* grouped_layout,
    cudaStream_t& stream) {
    HOST_ASSERT(grouped_layout != nullptr,
                "grouped_layout cannot be null for masked grouped FP8 swiglu GEMM");
    kernel3_masked(A, up_weight, gate_weight, scale_a, scale_up, scale_gate,
                   scale_d, D, grouped_layout, stream);
}

}  // namespace api
}  // namespace moe_cuda
