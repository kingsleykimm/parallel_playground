#include <apis/moe_forward.hpp>

#include <jit/compiler.hpp>
#include <kernels/internal_api.hpp>

namespace moe_cuda {

void init(const std::string& library_root, const std::string& cuda_home) {
    Compiler::init_static_vars(library_root, cuda_home);
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

#ifdef MOE_CUDA_USE_MPI
void a2a_dispatch(
    All2All& all2all,
    at::Tensor& out_expert_num_tokens,
    at::Tensor& out_expert_x,
    std::optional<at::Tensor>& out_expert_x_scale,
    at::Tensor& dp_x,
    std::optional<at::Tensor>& dp_x_scale,
    at::Tensor& indices,
    at::Tensor& weights,
    std::optional<at::Tensor>& bound_m,
    bool do_send,
    bool do_recv,
    cudaStream_t stream) {
    kernels::a2a_dispatch(
        all2all,
        out_expert_num_tokens,
        out_expert_x,
        out_expert_x_scale,
        dp_x,
        dp_x_scale,
        indices,
        weights,
        bound_m,
        do_send,
        do_recv,
        stream);
}

void a2a_combine(
    All2All& all2all,
    at::Tensor& out_tokens,
    at::Tensor& indices,
    at::Tensor& weights,
    at::Tensor& expert_y,
    std::optional<at::Tensor>& bound_m,
    bool do_send,
    bool do_recv,
    bool accumulate,
    cudaStream_t stream) {
    kernels::a2a_combine(
        all2all,
        out_tokens,
        indices,
        weights,
        expert_y,
        bound_m,
        do_send,
        do_recv,
        accumulate,
        stream);
}
#endif

}  // namespace moe_cuda
