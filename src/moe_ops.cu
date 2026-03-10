#include <apis/moe.hpp>
#include <kernels/internal_api.hpp>
#ifdef MOE_CUDA_USE_MPI
#include <kernels/all2all/a2a_combine_recv.cuh>
#include <kernels/all2all/a2a_combine_send.cuh>
#include <kernels/all2all/a2a_dispatch_recv.cuh>
#include <kernels/all2all/a2a_dispatch_send.cuh>
#endif

namespace moe_cuda {
namespace kernels {

void fp8_gemm_nt(std::pair<at::Tensor &, at::Tensor &> act,
                 std::pair<at::Tensor &, at::Tensor &> weight,
                 at::Tensor &output, GemmType gemm_type,
                 const std::string &compiled_dims, int *grouped_layout,
                 cudaStream_t &stream) {
  api::fp8_gemm_nt(act, weight, output, gemm_type, compiled_dims,
                   grouped_layout, stream);
}

void fp8_grouped_gemm_nt(std::pair<at::Tensor &, at::Tensor &> act,
                          std::pair<at::Tensor &, at::Tensor &> weight,
                          at::Tensor &output, GemmType gemm_type,
                          int *grouped_layout, cudaStream_t &stream) {
  api::fp8_grouped_gemm(act, weight, output, gemm_type, grouped_layout, stream);
}

void bf16_gemm(at::Tensor &A, at::Tensor &B, std::optional<at::Tensor> &C,
               at::Tensor &D, GemmType gemm_type,
               const std::string &compiled_dims, int *grouped_layout,
               cudaStream_t &stream) {
  api::bf16_gemm(A, B, C, D, gemm_type, compiled_dims, grouped_layout, stream);
}

#ifdef MOE_CUDA_USE_MPI
void a2a_dispatch(All2All &all2all, at::Tensor &out_expert_num_tokens,
                  at::Tensor &out_expert_x,
                  std::optional<at::Tensor> &out_expert_x_scale,
                  at::Tensor &dp_x, std::optional<at::Tensor> &dp_x_scale,
                  at::Tensor &indices, at::Tensor &weights,
                  std::optional<at::Tensor> &bound_m, bool do_send,
                  bool do_recv, cudaStream_t stream) {
  all2all.dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale,
                   dp_x, dp_x_scale, indices, weights, bound_m, do_send,
                   do_recv, stream);
}

void a2a_combine(All2All &all2all, at::Tensor &out_tokens, at::Tensor &indices,
                 at::Tensor &weights, at::Tensor &expert_y,
                 std::optional<at::Tensor> &bound_m, bool do_send, bool do_recv,
                 bool accumulate, cudaStream_t stream) {
  all2all.combine(out_tokens, indices, weights, expert_y, bound_m, do_send,
                  do_recv, accumulate, stream);
}
#endif

} // namespace kernels
} // namespace moe_cuda
