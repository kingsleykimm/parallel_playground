/**
   @brief: CUDA source file to link gcc defined apis with kernel calls
 */

// ================================================
// JIT launcher (gcc) APIs
// ================================================
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include <apis/moe.hpp>

// ================================================
// Kernel APIs
// ================================================
#include <all2all/all2all_base.hpp>
#include <all2all/all2all_tk.hpp>
#include <kernels/cast.cuh>
#include <kernels/internal_api.hpp>

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

void fp8_grouped_gemm_swiglu(at::Tensor &A, at::Tensor &gate_weight,
                             at::Tensor &up_weight, at::Tensor &scale_a,
                             at::Tensor &scale_gate, at::Tensor &scale_up,
                             at::Tensor &scale_d, at::Tensor &D,
                             GemmType gemm_type, int *grouped_layout,
                             cudaStream_t &stream) {
  api::fp8_grouped_gemm_swiglu(A, gate_weight, up_weight, scale_a, scale_gate,
                               scale_up, scale_d, D, gemm_type, grouped_layout,
                               stream);
}

void fp8_grouped_gemm_swiglu_consumer_pp(
    at::Tensor &A, at::Tensor &gate_weight, at::Tensor &up_weight,
    at::Tensor &scale_a, at::Tensor &scale_gate, at::Tensor &scale_up,
    at::Tensor &scale_d, at::Tensor &D, GemmType gemm_type, int *grouped_layout,
    cudaStream_t &stream) {
  api::fp8_grouped_gemm_swiglu_consumer_pp(A, gate_weight, up_weight, scale_a,
                                           scale_gate, scale_up, scale_d, D,
                                           gemm_type, grouped_layout, stream);
};

void fp8_gemm_nt(at::Tensor &act, at::Tensor &act_scale, at::Tensor &weight,
                 at::Tensor &weight_scale, at::Tensor &output,
                 GemmType gemm_type, const std::string &compiled_dims,
                 int *grouped_layout, cudaStream_t &stream) {
  api::fp8_gemm_nt(act, act_scale, weight, weight_scale, output, gemm_type,
                   compiled_dims, grouped_layout, stream);
}

void fp8_grouped_gemm_nt(at::Tensor &act, at::Tensor &act_scale,
                         at::Tensor &weight, at::Tensor &weight_scale,
                         at::Tensor &output, GemmType gemm_type,
                         int *grouped_layout, cudaStream_t &stream) {
  api::fp8_grouped_gemm(act, act_scale, weight, weight_scale, output, gemm_type,
                        grouped_layout, stream);
}

void bf16_gemm(at::Tensor &A, at::Tensor &B, std::optional<at::Tensor> &C,
               at::Tensor &D, GemmType gemm_type,
               const std::string &compiled_dims, int *grouped_layout,
               cudaStream_t &stream) {
  api::bf16_gemm(A, B, C, D, gemm_type, compiled_dims, grouped_layout, stream);
}

// this uses the All2All struct, which manages all the a2a information and
// kernel launches, and connects it with the fused swiglu matmuls
// this does not perform any fusing between the dispatch + swiglu (up) and
// combine + down, and it performs routing calculation in the host side, so
// expectedly the performance is worse
template <int EXPERTS_PER_TOKEN, int NUM_EXPERTS, int TOKEN_DIM>
void naive_moe_forward(All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &a2a,
                       GemmType gemm_type, at::Tensor &input,
                       at::Tensor &input_scales, at::Tensor &gate,
                       at::Tensor &gate_scales, at::Tensor &up,
                       at::Tensor &up_scales, at::Tensor &down,
                       at::Tensor &down_scales, at::Tensor &indices,
                       at::Tensor &weights, cudaStream_t stream) {

  uint32_t max_recv_tokens = a2a.max_recv_tokens();
  at::TensorOptions options = at::TensorOptions().device(torch::kCUDA);
  at::Tensor out_expert_num_tokens =
      at::empty(std::vector<int64_t>{a2a.num_experts_per_rank()},
                options.dtype(torch::kInt32));

  int H = input.size(-1);
  int I = input.size(-2);

  // to maintain consistency with opt api
  std::optional<at::Tensor> input_scales_opt = input_scales;

  at::Tensor expert_x = at::empty(std::vector<int64_t>(max_recv_tokens, H),
                                  options.dtype(torch::kFloat8_e4m3fn));
  std::optional<at::Tensor> expert_x_scales =
      at::empty(std::vector<int64_t>(H / 128, max_recv_tokens),
                options.dtype(torch::kFloat32));
  at::Tensor inter_y = at::empty(std::vector<int64_t>(max_recv_tokens, I),
                                 options.dtype(torch::kFloat8_e4m3fn));
  at::Tensor inter_y_scales =
      at::empty(std::vector<int64_t>(I / 128, max_recv_tokens),
                options.dtype(torch::kFloat32));
  at::Tensor expert_y = at::empty(std::vector<int64_t>(max_recv_tokens, H),
                                  options.dtype(torch::kBFloat16));
  std::optional<at::Tensor> bound_m = std::nullopt;
  a2a.dispatch(out_expert_num_tokens, expert_x, expert_x_scales, input,
               input_scales_opt, indices, weights, bound_m, true, true, stream);

  int *grouped_layout =
      gemm_type == GemmType::MGroupedMasked
          ? (int *)(a2a.context().worker.padded_offsets)
          : (int *)(a2a.context().worker.tokens_per_expert_host.data());
  fp8_grouped_gemm_swiglu(expert_x, gate, up, expert_x_scales.value(),
                          gate_scales, up_scales, inter_y_scales, inter_y,
                          gemm_type, grouped_layout, stream);
  fp8_grouped_gemm_nt(inter_y, inter_y_scales, down, down_scales, expert_y,
                      gemm_type, grouped_layout, stream);
  a2a.combine(input, indices, weights, expert_y, bound_m, true, true, stream);
}

} // namespace kernels

std::shared_ptr<All2AllBase>
make_all2all(uint32_t max_num_tokens, uint32_t num_experts,
             uint32_t experts_per_token, uint32_t expert_padding,
             uint32_t hidden_dim, std::optional<uint32_t> hidden_dim_scale,
             c10::ScalarType in_dtype, c10::ScalarType out_dtype,
             std::optional<c10::ScalarType> scale_dtype,
             std::optional<uint32_t> max_private_tokens_opt, int local_rank,
             ParallelConfig parallel_config, cudaStream_t stream) {
  LAUNCH_NUM_EXPERTS(num_experts, NUM_EXPERTS, {
    LAUNCH_NUM_EXPERTS_PER_TOKEN(
        experts_per_token, EXPERTS_PER_TOKEN,
        {LAUNCH_TOKEN_DIM(hidden_dim, TOKEN_DIM, {
          return std::make_shared<
              All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM>>(
              max_num_tokens, num_experts, expert_padding, hidden_dim,
              hidden_dim_scale, in_dtype, out_dtype, scale_dtype,
              experts_per_token, max_private_tokens_opt, local_rank,
              parallel_config, stream);
        })});
  });
}
void naive_moe_forward_dispatch(All2AllBase &a2a_base, uint32_t num_experts,
                                uint32_t experts_per_token, uint32_t hidden_dim,
                                GemmType gemm_type, at::Tensor &input,
                                at::Tensor &input_scales, at::Tensor &gate,
                                at::Tensor &gate_scales, at::Tensor &up,
                                at::Tensor &up_scales, at::Tensor &down,
                                at::Tensor &down_scales, at::Tensor &indices,
                                at::Tensor &weights, cudaStream_t stream) {
  LAUNCH_NUM_EXPERTS(num_experts, NUM_EXPERTS, {
    LAUNCH_NUM_EXPERTS_PER_TOKEN(experts_per_token, EXPERTS_PER_TOKEN, {
      LAUNCH_TOKEN_DIM(hidden_dim, TOKEN_DIM, {
        auto &a2a =
            static_cast<All2All<EXPERTS_PER_TOKEN, NUM_EXPERTS, TOKEN_DIM> &>(
                a2a_base);
        moe_cuda::kernels::naive_moe_forward(
            a2a, gemm_type, input, input_scales, gate, gate_scales, up,
            up_scales, down, down_scales, indices, weights, stream);
      });
    });
  });
}

} // namespace moe_cuda
