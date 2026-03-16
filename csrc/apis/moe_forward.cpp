#include <apis/moe_forward.hpp>

#include <jit/compiler.hpp>
#include <kernels/internal_api.hpp>

namespace moe_cuda {

void init(const std::string& library_root, const std::string& cuda_home) {
    Compiler::init_static_vars(library_root, cuda_home);
}

void fp8_grouped_gemm_swiglu_contiguous(
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
    kernels::fp8_grouped_gemm_swiglu_contiguous(
        A, gate_weight, up_weight, scale_a, scale_gate, scale_up,
        scale_d, D, grouped_layout, stream);
}

void fp8_grouped_gemm_swiglu_masked(
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
    kernels::fp8_grouped_gemm_swiglu_masked(
        A, gate_weight, up_weight, scale_a, scale_gate, scale_up,
        scale_d, D, grouped_layout, stream);
}

void bf16_gemm(
    at::Tensor& A,
    at::Tensor& B,
    std::optional<at::Tensor>& C,
    at::Tensor& D,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream) {
    kernels::bf16_gemm(A, B, C, D, gemm_type, compiled_dims, grouped_layout, stream);
}

void fp8_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    const std::string& compiled_dims,
    int* grouped_layout,
    cudaStream_t& stream) {
    kernels::fp8_gemm_nt(act, weight, output, gemm_type, compiled_dims, grouped_layout, stream);
}

void fp8_grouped_gemm_nt(
    std::pair<at::Tensor&, at::Tensor&> act,
    std::pair<at::Tensor&, at::Tensor&> weight,
    at::Tensor& output,
    GemmType gemm_type,
    int* grouped_layout,
    cudaStream_t& stream) {
    kernels::fp8_grouped_gemm_nt(act, weight, output, gemm_type, grouped_layout, stream);
}

}  // namespace moe_cuda
