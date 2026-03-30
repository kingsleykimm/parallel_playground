#include <ATen/cuda/CUDAContext.h>
#include <kernels/internal_api.hpp>
#include <pybind11/pybind11.h>
#include <pyutils/parallel_tensor.cuh>

namespace {
cudaStream_t current_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}
} // namespace

void bind_tk_parallel_tensor(pybind11::module_ &m) {
  BIND_TK_PARALLEL_TENSOR(m);
}

void bind_fused_dispatch_grouped_gemm_swiglu(pybind11::module_ &m) {
  m.def(
      "fused_dispatch_grouped_gemm_swiglu",
      [](kittens::py::TKParallelTensor &in_tokens,
         kittens::py::TKParallelTensor &in_tokens_scales,
         at::Tensor &expert_x_tokens, at::Tensor &expert_x_tokens_scale,
         at::Tensor &gate, at::Tensor &up, at::Tensor &C,
         at::Tensor &scale_gate, at::Tensor &scale_up, at::Tensor &out_scales,
         at::Tensor &indices, kittens::py::TKParallelTensor &global_num_routed,
         kittens::py::TKParallelTensor &expert_to_token_map,
         kittens::py::TKParallelTensor &expert_to_slot_map,
         at::Tensor &padded_expert_counts, at::Tensor &src_token_idx,
         at::Tensor &src_dev_idx, at::Tensor &src_slot_idx,
         kittens::py::TKParallelTensor &barrier, int num_tokens,
         at::Tensor &num_recv_tokens, int dp_rank, int rank, int dp_size,
         int cur_dp_group, int num_dp_groups, int world_size, int num_experts,
         int experts_per_token, int num_comm_sms, int num_comp_sms) {
        auto stream = current_stream();
        moe_cuda::kernels::fused_dispatch_grouped_gemm_swiglu(
            in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
            gate, up, C, scale_gate, scale_up, out_scales, indices,
            global_num_routed, expert_to_token_map, expert_to_slot_map,
            padded_expert_counts, src_token_idx, src_dev_idx, src_slot_idx,
            barrier, num_tokens, num_recv_tokens.data_ptr<int>(), dp_rank, rank,
            dp_size, cur_dp_group, num_dp_groups, world_size, num_experts,
            experts_per_token, num_comm_sms, num_comp_sms, stream);
      },
      pybind11::arg("in_tokens"), pybind11::arg("in_tokens_scales"),
      pybind11::arg("expert_x_tokens"), pybind11::arg("expert_x_tokens_scale"),
      pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("C"),
      pybind11::arg("scale_gate"), pybind11::arg("scale_up"),
      pybind11::arg("out_scales"), pybind11::arg("indices"),
      pybind11::arg("global_num_routed"), pybind11::arg("expert_to_token_map"),
      pybind11::arg("expert_to_slot_map"),
      pybind11::arg("padded_expert_counts"), pybind11::arg("src_token_idx"),
      pybind11::arg("src_dev_idx"), pybind11::arg("src_slot_idx"),
      pybind11::arg("barrier"), pybind11::arg("num_tokens"),
      pybind11::arg("num_recv_tokens"), pybind11::arg("dp_rank"),
      pybind11::arg("rank"), pybind11::arg("dp_size"),
      pybind11::arg("cur_dp_group"), pybind11::arg("num_dp_groups"),
      pybind11::arg("world_size"), pybind11::arg("num_experts"),
      pybind11::arg("experts_per_token"), pybind11::arg("num_comm_sms"),
      pybind11::arg("num_comp_sms"));
}

void bind_fused_grouped_gemm_combine(pybind11::module_ &m) {
  m.def(
      "fused_grouped_gemm_combine",
      [](kittens::py::TKParallelTensor &out_tokens, at::Tensor &expert_y_tokens,
         at::Tensor &expert_y_tokens_scale, at::Tensor &down,
         at::Tensor &scale_down, at::Tensor &C, at::Tensor &weights,
         at::Tensor &padded_expert_counts,
         at::Tensor &src_token_idx, at::Tensor &src_dev_idx,
         at::Tensor &src_slot_idx, int num_experts, int experts_per_token,
         at::Tensor &num_recv_tokens, int dp_rank, int rank, int dp_size,
         int cur_dp_group, int num_dp_groups, int num_comm_sms,
         int num_comp_sms) {
        auto stream = current_stream();
        moe_cuda::kernels::fused_grouped_gemm_combine(
            out_tokens, expert_y_tokens, expert_y_tokens_scale, down,
            scale_down, C, weights, padded_expert_counts, src_token_idx,
            src_dev_idx, src_slot_idx, num_experts, experts_per_token,
            num_recv_tokens.data_ptr<int>(), dp_rank, rank, dp_size,
            cur_dp_group, num_dp_groups, num_comm_sms, num_comp_sms, stream);
      },
      pybind11::arg("out_tokens"), pybind11::arg("expert_y_tokens"),
      pybind11::arg("expert_y_tokens_scale"), pybind11::arg("down"),
      pybind11::arg("scale_down"), pybind11::arg("C"), pybind11::arg("weights"),
      pybind11::arg("padded_expert_counts"),
      pybind11::arg("src_token_idx"), pybind11::arg("src_dev_idx"),
      pybind11::arg("src_slot_idx"), pybind11::arg("num_experts"),
      pybind11::arg("experts_per_token"), pybind11::arg("num_recv_tokens"),
      pybind11::arg("dp_rank"), pybind11::arg("rank"), pybind11::arg("dp_size"),
      pybind11::arg("cur_dp_group"), pybind11::arg("num_dp_groups"),
      pybind11::arg("num_comm_sms"), pybind11::arg("num_comp_sms"));
}
