// Python bindings for moe_cuda via pybind11.

#include "moe_cuda/types.h"
#include "tk_bindings.h"
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <jit/compiler.hpp>
#include <kernels/internal_api.hpp>
#include <runtime/parallel.h>
#include <runtime/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/macros/Macros.h>

#include <all2all/all2all_base.hpp>

namespace {

cudaStream_t current_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

std::string normalize_dtype_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return name;
}

c10::ScalarType parse_scalar_type(const pybind11::handle &value,
                                  const char *arg_name) {
  if (pybind11::isinstance<pybind11::int_>(value)) {
    return static_cast<c10::ScalarType>(value.cast<int>());
  }

  std::string name =
      normalize_dtype_name(pybind11::str(value).cast<std::string>());
  if (name == "torch.float32" || name == "float32" || name == "float" ||
      name == "fp32") {
    return c10::ScalarType::Float;
  }
  if (name == "torch.bfloat16" || name == "bfloat16" || name == "bf16") {
    return c10::ScalarType::BFloat16;
  }
  if (name == "torch.float16" || name == "float16" || name == "half" ||
      name == "fp16") {
    return c10::ScalarType::Half;
  }
  if (name == "torch.int32" || name == "int32" || name == "int" ||
      name == "torch.int") {
    return c10::ScalarType::Int;
  }
  if (name == "torch.int64" || name == "int64" || name == "long" ||
      name == "torch.long") {
    return c10::ScalarType::Long;
  }
  if (name == "torch.bool" || name == "bool") {
    return c10::ScalarType::Bool;
  }
  if (name == "torch.float8_e4m3fn" || name == "float8_e4m3fn" ||
      name == "fp8_e4m3fn") {
    return c10::ScalarType::Float8_e4m3fn;
  }

  throw std::invalid_argument(std::string("Unsupported dtype for ") + arg_name +
                              ": " + pybind11::str(value).cast<std::string>());
}

std::optional<c10::ScalarType>
parse_optional_scalar_type(const pybind11::object &value,
                           const char *arg_name) {
  if (value.is_none()) {
    return std::nullopt;
  }
  return parse_scalar_type(value, arg_name);
}

class PyAll2All {
public:
  PyAll2All(uint32_t max_num_tokens, uint32_t num_experts,
            uint32_t experts_per_token, uint32_t expert_padding,
            uint32_t hidden_dim, std::optional<uint32_t> hidden_dim_scale,
            at::ScalarType &in_dtype, at::ScalarType &out_dtype,
            std::optional<at::ScalarType> &scale_dtype,
            std::optional<uint32_t> max_private_tokens,
            std::optional<uint32_t> dp_group_size,
            std::optional<uint32_t> node_group_size,
            std::optional<uint32_t> ep_group_size, int device,
            std::optional<at::cuda::CUDAStream> stream)
      : rank_(device), world_size_(node_group_size.value_or(1)),
        device_(device) {

    rank_ = device;

    const uint32_t node_size =
        node_group_size.value_or(static_cast<uint32_t>(world_size_));
    const uint32_t dp_size = dp_group_size.value_or(1);
    const uint32_t ep_size = ep_group_size.value_or(1);
    if (node_size == 0 || world_size_ % node_size != 0) {
      throw std::invalid_argument("node_group_size must divide world size");
    }
    if (dp_size == 0 || world_size_ % dp_size != 0) {
      throw std::invalid_argument("dp_group_size must divide world size");
    }
    if (ep_size == 0 || world_size_ % ep_size != 0) {
      throw std::invalid_argument("ep_group_size must divide world size");
    }

    ParallelConfig parallel_config = {
        .tp_size = 1, // not set for now
        .dp_size = dp_size,
        .ep_size = ep_size,
        .node_size = node_size,
        .world_size = node_size, // for single node for now
    };

    cudaStream_t input_stream;
    if (stream.has_value()) {
      input_stream = stream->stream();
    } else {
      input_stream = current_stream();
    }

    handle_ = moe_cuda::make_all2all(
        max_num_tokens, num_experts, experts_per_token, expert_padding,
        hidden_dim, hidden_dim_scale, in_dtype, out_dtype, scale_dtype,
        max_private_tokens, rank_, parallel_config, input_stream);
  }

  void dispatch(torch::stable::Tensor &out_expert_num_tokens,
                torch::stable::Tensor &out_expert_x,
                std::optional<torch::stable::Tensor> &out_expert_x_scale,
                torch::stable::Tensor &dp_x,
                std::optional<torch::stable::Tensor> &dp_x_scale,
                torch::stable::Tensor &indices, torch::stable::Tensor &weights,
                std::optional<torch::stable::Tensor> bound_m, bool do_send,
                bool do_recv) {
    handle_->dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale,
                      dp_x, dp_x_scale, indices, weights, bound_m, do_send,
                      do_recv, current_stream());
  }

  void combine(torch::stable::Tensor &out_tokens,
               torch::stable::Tensor &indices, torch::stable::Tensor &weights,
               torch::stable::Tensor &expert_y,
               std::optional<torch::stable::Tensor> bound_m, bool do_send,
               bool do_recv, bool accumulate) {
    handle_->combine(out_tokens, indices, weights, expert_y, bound_m, do_send,
                     do_recv, accumulate, current_stream());
  }

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int device() const { return device_; }
  uint32_t max_recv_tokens() const { return handle_->max_recv_tokens(); }
  std::shared_ptr<moe_cuda::All2AllBase> handle() const { return handle_; }

private:
  int rank_;
  int world_size_;
  int device_;
  ParallelConfig parallel_config;
  std::shared_ptr<moe_cuda::All2AllBase> handle_;
};

} // namespace

PYBIND11_MODULE(moe_cuda, m) {
  m.doc() = "MoE CUDA kernels for H100 (SM90a)";

  bind_tk_parallel_tensor(m);

  // wrapper around the parallel tensor in thunderkittens

  pybind11::enum_<GemmType>(m, "GemmType")
      .value("Normal", GemmType::Normal)
      .value("MGroupedContiguous", GemmType::MGroupedContiguous)
      .value("MGroupedMasked", GemmType::MGroupedMasked)
      .value("Batched", GemmType::Batched)
      .export_values();

  m.def(
      "init",
      [](const std::string &library_root, const std::string &cuda_home) {
        Compiler::init_static_vars(library_root, cuda_home);
      },
      pybind11::arg("library_root"), pybind11::arg("cuda_home"),
      "Initialize moe_cuda JIT runtime paths.");
  bind_fused_dispatch_grouped_gemm_swiglu(m);

  pybind11::class_<PyAll2All>(m, "All2All")
      .def(pybind11::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          std::optional<uint32_t>, at::ScalarType &,
                          at::ScalarType &, std::optional<at::ScalarType> &,
                          std::optional<uint32_t>, std::optional<uint32_t>,
                          std::optional<uint32_t>, std::optional<uint32_t>, int,
                          std::optional<at::cuda::CUDAStream>>(),
           pybind11::arg("max_num_tokens"), pybind11::arg("num_experts"),
           pybind11::arg("experts_per_token"), pybind11::arg("expert_padding"),
           pybind11::arg("hidden_dim"),
           pybind11::arg("hidden_dim_scale") = pybind11::none(),
           pybind11::arg("in_dtype"), pybind11::arg("out_dtype"),
           pybind11::arg("scale_dtype") = pybind11::none(),
           pybind11::arg("max_private_tokens") = pybind11::none(),
           pybind11::arg("dp_group_size") = pybind11::none(),
           pybind11::arg("node_group_size") = pybind11::none(),
           pybind11::arg("ep_group_size") = pybind11::none(),
           pybind11::arg("device") = -1,
           pybind11::arg("stream") = pybind11::none())
      .def(
          "dispatch", &PyAll2All::dispatch,
          pybind11::arg("out_expert_num_tokens"), pybind11::arg("out_expert_x"),
          pybind11::arg("out_expert_x_scale") = pybind11::none(),
          pybind11::arg("dp_x"), pybind11::arg("dp_x_scale") = pybind11::none(),
          pybind11::arg("indices"), pybind11::arg("weights"),
          pybind11::arg("bound_m") = pybind11::none(),
          pybind11::arg("do_send") = true, pybind11::arg("do_recv") = true)
      .def("combine", &PyAll2All::combine, pybind11::arg("out_tokens"),
           pybind11::arg("indices"), pybind11::arg("weights"),
           pybind11::arg("expert_y"),
           pybind11::arg("bound_m") = pybind11::none(),
           pybind11::arg("do_send") = true, pybind11::arg("do_recv") = true,
           pybind11::arg("accumulate") = false)
      .def_property_readonly("rank", &PyAll2All::rank)
      .def_property_readonly("world_size", &PyAll2All::world_size)
      .def_property_readonly("device", &PyAll2All::device)
      .def_property_readonly("max_recv_tokens", &PyAll2All::max_recv_tokens)
      .def_property_readonly("handle", &PyAll2All::handle);

  m.def("naive_moe_forward_dispatch",
        [](PyAll2All &all2all, uint32_t num_experts, uint32_t experts_per_token,
           uint32_t hidden_dim, GemmType gemm_type,
           torch::stable::Tensor &input, torch::stable::Tensor &input_scales,
           torch::stable::Tensor &gate, torch::stable::Tensor &gate_scales,
           torch::stable::Tensor &up, torch::stable::Tensor &up_scales,
           torch::stable::Tensor &down, torch::stable::Tensor &down_scales,
           torch::stable::Tensor &indices, torch::stable::Tensor &weights,
           torch::stable::Tensor &out_tokens, torch::stable::Tensor &expert_x,
           torch::stable::Tensor &expert_x_scales,
           torch::stable::Tensor &inter_y,
           torch::stable::Tensor &inter_y_scales,
           torch::stable::Tensor &expert_y) {
          auto stream = current_stream();
          moe_cuda::All2AllBase &a2a = *all2all.handle().get();
          naive_moe_forward_dispatch(
              a2a, num_experts, experts_per_token, hidden_dim, gemm_type, input,
              input_scales, gate, gate_scales, up, up_scales, down, down_scales,
              indices, weights, out_tokens, expert_x, expert_x_scales, inter_y,
              inter_y_scales, expert_y, stream);
        });
}
