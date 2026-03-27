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
         torch::stable::Tensor &expert_x_tokens,
         torch::stable::Tensor &expert_x_tokens_scale,
         torch::stable::Tensor &gate, torch::stable::Tensor &up,
         torch::stable::Tensor &C, torch::stable::Tensor &scale_gate,
         torch::stable::Tensor &scale_up, torch::stable::Tensor &out_scales,
         torch::stable::Tensor &indices,
         kittens::py::TKParallelTensor &global_num_routed,
         kittens::py::TKParallelTensor &expert_to_token_map,
         torch::stable::Tensor &padded_expert_counts,
         torch::stable::Tensor &src_token_idx,
         torch::stable::Tensor &src_dev_idx,
         kittens::py::TKParallelTensor &barrier, int num_tokens,
         torch::stable::Tensor &num_recv_tokens, int dp_rank, int rank,
         int dp_size, int cur_dp_group, int num_dp_groups, int world_size,
         int num_experts, int experts_per_token, int num_comm_sms,
         int num_comp_sms) {
        auto stream = current_stream();
        moe_cuda::kernels::fused_dispatch_grouped_gemm_swiglu(
            in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
            gate, up, C, scale_gate, scale_up, out_scales, indices,
            global_num_routed, expert_to_token_map, padded_expert_counts,
            src_token_idx, src_dev_idx, barrier, num_tokens,
            (int *)num_recv_tokens.data_ptr(), dp_rank, rank, dp_size,
            cur_dp_group, num_dp_groups, world_size, num_experts,
            experts_per_token, num_comm_sms, num_comp_sms, stream);
      },
      pybind11::arg("in_tokens"), pybind11::arg("in_tokens_scales"),
      pybind11::arg("expert_x_tokens"), pybind11::arg("expert_x_tokens_scale"),
      pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("C"),
      pybind11::arg("scale_gate"), pybind11::arg("scale_up"),
      pybind11::arg("out_scales"), pybind11::arg("indices"),
      pybind11::arg("global_num_routed"), pybind11::arg("expert_to_token_map"),
      pybind11::arg("padded_expert_counts"), pybind11::arg("src_token_idx"),
      pybind11::arg("src_dev_idx"), pybind11::arg("barrier"),
      pybind11::arg("num_tokens"), pybind11::arg("num_recv_tokens"),
      pybind11::arg("dp_rank"), pybind11::arg("rank"), pybind11::arg("dp_size"),
      pybind11::arg("cur_dp_group"), pybind11::arg("num_dp_groups"),
      pybind11::arg("world_size"), pybind11::arg("num_experts"),
      pybind11::arg("experts_per_token"), pybind11::arg("num_comm_sms"),
      pybind11::arg("num_comp_sms"));
}
