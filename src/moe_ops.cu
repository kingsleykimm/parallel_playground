/**
   @brief: CUDA source file to link gcc defined apis with kernel calls
 */

// ================================================
// JIT launcher (gcc) APIs
// ================================================
#include <apis/moe.hpp>

// ================================================
// Kernel APIs
// ================================================
#include <all2all/all2all_tk.hpp>
#include <kernels/cast.cuh>
#include <kernels/internal_api.hpp>

#include <kernels/all2all/a2a_combine_recv.cuh>
#include <kernels/all2all/a2a_combine_send.cuh>
#include <kernels/all2all/a2a_dispatch_recv.cuh>
#include <kernels/all2all/a2a_dispatch_send.cuh>

#include <ATen/core/TensorBase.h>

namespace moe_cuda {
namespace kernels {

void cast(at::Tensor &inp, at::Tensor &out, std::optional<at::Tensor> &scale,
          cudaStream_t stream) {
  cast_impl::cast_(inp, out, scale, stream);
}

void cast(at::Tensor &inp, at::Tensor &out, cudaStream_t stream) {
  std::optional<at::Tensor> scale = std::nullopt;
  cast_impl::cast_(inp, out, scale, stream);
}

void fused_silu_mul_quant(at::Tensor &gemm_out, at::Tensor &swiglu_out,
                          at::Tensor &scale, cudaStream_t stream) {
  size_t num_rows = 1;
  for (int i = 0; i < gemm_out.dim() - 1; ++i) {
    num_rows *= gemm_out.size(i);
  }
  const size_t hidden_dim = gemm_out.size(-1) / 2;
  cast_impl::fused_silu_mul_quant(
      reinterpret_cast<__nv_bfloat16 *>(gemm_out.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_fp8_e4m3 *>(
          swiglu_out.data_ptr<c10::Float8_e4m3fn>()),
      scale.data_ptr<float>(), num_rows, hidden_dim, stream);
}

void fp8_grouped_gemm_swiglu_contiguous(
    at::Tensor& A, at::Tensor& gate_weight, at::Tensor& up_weight,
    at::Tensor& scale_a, at::Tensor& scale_gate, at::Tensor& scale_up,
    at::Tensor& scale_d, at::Tensor& D,
    int* grouped_layout, cudaStream_t& stream) {
  api::fp8_grouped_gemm_swiglu_contiguous(A, gate_weight, up_weight, scale_a,
                                          scale_gate, scale_up, scale_d, D,
                                          grouped_layout, stream);
}

void fp8_grouped_gemm_swiglu_masked(
    at::Tensor& A, at::Tensor& gate_weight, at::Tensor& up_weight,
    at::Tensor& scale_a, at::Tensor& scale_gate, at::Tensor& scale_up,
    at::Tensor& scale_d, at::Tensor& D,
    int* grouped_layout, cudaStream_t& stream) {
  api::fp8_grouped_gemm_swiglu_masked(A, gate_weight, up_weight, scale_a,
                                      scale_gate, scale_up, scale_d, D,
                                      grouped_layout, stream);
}

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

// template <typename All2AllT>
// void a2a_dispatch(All2AllT &all2all, at::Tensor &out_expert_num_tokens,
//                   at::Tensor &out_expert_x,
//                   std::optional<at::Tensor> &out_expert_x_scale,
//                   at::Tensor &dp_x, std::optional<at::Tensor> &dp_x_scale,
//                   at::Tensor &indices, at::Tensor &weights,
//                   std::optional<at::Tensor> &bound_m, bool do_send,
//                   bool do_recv, cudaStream_t stream) {
//   all2all.dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale,
//                    dp_x, dp_x_scale, indices, weights, bound_m, do_send,
//                    do_recv, stream);
// }

// template <typename All2AllT>
// void a2a_combine(All2AllT &all2all, at::Tensor &out_tokens, at::Tensor
// &indices,
//                  at::Tensor &weights, at::Tensor &expert_y,
//                  std::optional<at::Tensor> &bound_m, bool do_send, bool
//                  do_recv, bool accumulate, cudaStream_t stream) {
//   all2all.combine(out_tokens, indices, weights, expert_y, bound_m, do_send,
//                   do_recv, accumulate, stream);
// }

} // namespace kernels
} // namespace moe_cuda
