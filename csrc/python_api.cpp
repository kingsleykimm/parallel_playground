// Python bindings for moe_cuda via pybind11.

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

#include <apis/moe_forward.hpp>
#include <runtime/parallel.h>

#ifdef MOE_CUDA_USE_MPI
#include <all2all/all2all.hpp>
#include <mpi.h>
#endif

namespace py = pybind11;

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

c10::ScalarType parse_scalar_type(const py::handle& value, const char* arg_name) {
    if (py::isinstance<py::int_>(value)) {
        return static_cast<c10::ScalarType>(value.cast<int>());
    }

    std::string name = normalize_dtype_name(py::str(value).cast<std::string>());
    if (name == "torch.float32" || name == "float32" || name == "float" || name == "fp32") {
        return c10::ScalarType::Float;
    }
    if (name == "torch.bfloat16" || name == "bfloat16" || name == "bf16") {
        return c10::ScalarType::BFloat16;
    }
    if (name == "torch.float16" || name == "float16" || name == "half" || name == "fp16") {
        return c10::ScalarType::Half;
    }
    if (name == "torch.int32" || name == "int32" || name == "int" || name == "torch.int") {
        return c10::ScalarType::Int;
    }
    if (name == "torch.int64" || name == "int64" || name == "long" || name == "torch.long") {
        return c10::ScalarType::Long;
    }
    if (name == "torch.bool" || name == "bool") {
        return c10::ScalarType::Bool;
    }
    if (name == "torch.float8_e4m3fn" || name == "float8_e4m3fn" || name == "fp8_e4m3fn") {
        return c10::ScalarType::Float8_e4m3fn;
    }

    throw std::invalid_argument(std::string("Unsupported dtype for ") + arg_name + ": " +
                                py::str(value).cast<std::string>());
}

std::optional<c10::ScalarType> parse_optional_scalar_type(const py::object& value,
                                                          const char* arg_name) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return parse_scalar_type(value, arg_name);
}

#ifdef MOE_CUDA_USE_MPI
void ensure_mpi_initialized() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }
}

class PyAll2All {
public:
    PyAll2All(uint32_t max_num_tokens,
              uint32_t num_experts,
              uint32_t expert_padding,
              uint32_t hidden_dim,
              std::optional<uint32_t> hidden_dim_scale,
              const py::object& in_dtype,
              const py::object& out_dtype,
              const py::object& scale_dtype,
              uint32_t num_experts_per_token,
              std::optional<uint32_t> max_private_tokens,
              std::optional<uint32_t> dp_group_size,
              std::optional<uint32_t> node_group_size,
              int device)
        : rank_(0), world_size_(1), device_(device), global_group_(), node_group_() {
        ensure_mpi_initialized();

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        rank_ = rank;
        world_size_ = world_size;

        if (device_ < 0) {
            device_ = rank_;
        }

        const uint32_t node_size = node_group_size.value_or(static_cast<uint32_t>(world_size_));
        if (node_size == 0 || world_size_ % static_cast<int>(node_size) != 0) {
            throw std::invalid_argument("node_group_size must divide MPI world size");
        }
        if (dp_group_size.has_value()) {
            if (*dp_group_size == 0 || world_size_ % static_cast<int>(*dp_group_size) != 0) {
                throw std::invalid_argument("dp_group_size must divide MPI world size");
            }
            dp_group_.emplace(rank_, *dp_group_size);
        }

        global_group_ = ParallelGroup(rank_, world_size_);
        node_group_ = ParallelGroup(rank_, node_size);

        handle_ = std::make_unique<moe_cuda::All2All>(
            max_num_tokens,
            num_experts,
            expert_padding,
            hidden_dim,
            hidden_dim_scale,
            parse_scalar_type(in_dtype, "in_dtype"),
            parse_scalar_type(out_dtype, "out_dtype"),
            parse_optional_scalar_type(scale_dtype, "scale_dtype"),
            num_experts_per_token,
            max_private_tokens,
            dp_group_,
            node_group_,
            device_,
            global_group_,
            current_stream());
    }

    void dispatch(at::Tensor& out_expert_num_tokens,
                  at::Tensor& out_expert_x,
                  std::optional<at::Tensor> out_expert_x_scale,
                  at::Tensor& dp_x,
                  std::optional<at::Tensor> dp_x_scale,
                  at::Tensor& indices,
                  at::Tensor& weights,
                  std::optional<at::Tensor> bound_m,
                  bool do_send,
                  bool do_recv) {
        handle_->dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale, dp_x,
                          dp_x_scale, indices, weights, bound_m, do_send, do_recv,
                          current_stream());
    }

    void combine(at::Tensor& out_tokens,
                 at::Tensor& indices,
                 at::Tensor& weights,
                 at::Tensor& expert_y,
                 std::optional<at::Tensor> bound_m,
                 bool do_send,
                 bool do_recv,
                 bool accumulate) {
        handle_->combine(out_tokens, indices, weights, expert_y, bound_m, do_send, do_recv,
                         accumulate, current_stream());
    }

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    int device() const { return device_; }

private:
    int rank_;
    int world_size_;
    int device_;
    std::optional<ParallelGroup> dp_group_;
    ParallelGroup global_group_;
    ParallelGroup node_group_;
    std::unique_ptr<moe_cuda::All2All> handle_;
};
#endif

}  // namespace

PYBIND11_MODULE(moe_cuda, m) {
    m.doc() = "MoE CUDA kernels for H100 (SM90a)";

    py::enum_<GemmType>(m, "GemmType")
        .value("Normal", GemmType::Normal)
        .value("MGroupedContiguous", GemmType::MGroupedContiguous)
        .value("MGroupedMasked", GemmType::MGroupedMasked)
        .value("Batched", GemmType::Batched)
        .export_values();

    m.def("init", &moe_cuda::init, py::arg("library_root"), py::arg("cuda_home"),
          "Initialize moe_cuda JIT runtime paths.");

    m.def(
        "bf16_gemm",
        [](at::Tensor& a, at::Tensor& b, at::Tensor& d, std::optional<at::Tensor> c,
           GemmType gemm_type, const std::string& compiled_dims,
           std::optional<at::Tensor> grouped_layout) {
            auto stream = current_stream();
            int* grouped_layout_ptr =
                grouped_layout.has_value() ? grouped_layout->data_ptr<int>() : nullptr;
            moe_cuda::bf16_gemm(a, b, c, d, gemm_type, compiled_dims, grouped_layout_ptr,
                                stream);
        },
        py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = py::none(),
        py::arg("gemm_type") = GemmType::Normal, py::arg("compiled_dims") = "",
        py::arg("grouped_layout") = py::none(),
        "Launch the BF16 GEMM entrypoint on the current CUDA stream.");

    m.def(
        "fp8_gemm_nt",
        [](at::Tensor& act, at::Tensor& act_scale, at::Tensor& weight, at::Tensor& weight_scale,
           at::Tensor& output, GemmType gemm_type, const std::string& compiled_dims,
           std::optional<at::Tensor> grouped_layout) {
            auto stream = current_stream();
            int* grouped_layout_ptr =
                grouped_layout.has_value() ? grouped_layout->data_ptr<int>() : nullptr;
            moe_cuda::fp8_gemm_nt({act, act_scale}, {weight, weight_scale}, output, gemm_type,
                                  compiled_dims, grouped_layout_ptr, stream);
        },
        py::arg("act"), py::arg("act_scale"), py::arg("weight"), py::arg("weight_scale"),
        py::arg("output"), py::arg("gemm_type") = GemmType::Normal,
        py::arg("compiled_dims") = "", py::arg("grouped_layout") = py::none(),
        "Launch the FP8 GEMM NT entrypoint on the current CUDA stream.");

    m.def(
        "fp8_grouped_gemm_nt",
        [](at::Tensor& act, at::Tensor& act_scale, at::Tensor& weight, at::Tensor& weight_scale,
           at::Tensor& output, GemmType gemm_type, std::optional<at::Tensor> grouped_layout) {
            auto stream = current_stream();
            int* grouped_layout_ptr =
                grouped_layout.has_value() ? grouped_layout->data_ptr<int>() : nullptr;
            moe_cuda::fp8_grouped_gemm_nt({act, act_scale}, {weight, weight_scale}, output,
                                          gemm_type, grouped_layout_ptr, stream);
        },
        py::arg("act"), py::arg("act_scale"), py::arg("weight"), py::arg("weight_scale"),
        py::arg("output"), py::arg("gemm_type") = GemmType::MGroupedContiguous,
        py::arg("grouped_layout") = py::none(),
        "Launch the grouped FP8 GEMM entrypoint on the current CUDA stream.");

#ifdef MOE_CUDA_USE_MPI
    m.def("mpi_is_initialized", []() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    });

    m.def("mpi_init", []() { ensure_mpi_initialized(); });

    m.def("mpi_world_info", []() {
        ensure_mpi_initialized();
        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return py::make_tuple(rank, world_size);
    });

    py::class_<PyAll2All>(m, "All2All")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, std::optional<uint32_t>,
                      const py::object&, const py::object&, const py::object&, uint32_t,
                      std::optional<uint32_t>, std::optional<uint32_t>, std::optional<uint32_t>,
                      int>(),
             py::arg("max_num_tokens"), py::arg("num_experts"), py::arg("expert_padding"),
             py::arg("hidden_dim"), py::arg("hidden_dim_scale") = py::none(),
             py::arg("in_dtype") = py::str("bfloat16"),
             py::arg("out_dtype") = py::str("bfloat16"),
             py::arg("scale_dtype") = py::none(), py::arg("num_experts_per_token") = 1,
             py::arg("max_private_tokens") = py::none(), py::arg("dp_group_size") = py::none(),
             py::arg("node_group_size") = py::none(), py::arg("device") = -1)
        .def(
            "dispatch",
            [](PyAll2All& self, at::Tensor& out_expert_num_tokens, at::Tensor& out_expert_x,
               at::Tensor& dp_x, at::Tensor& indices, at::Tensor& weights,
               std::optional<at::Tensor> out_expert_x_scale,
               std::optional<at::Tensor> dp_x_scale, std::optional<at::Tensor> bound_m,
               bool do_send, bool do_recv) {
                self.dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale, dp_x,
                              dp_x_scale, indices, weights, bound_m, do_send, do_recv);
            },
            py::arg("out_expert_num_tokens"), py::arg("out_expert_x"), py::arg("dp_x"),
            py::arg("indices"), py::arg("weights"), py::arg("out_expert_x_scale") = py::none(),
            py::arg("dp_x_scale") = py::none(), py::arg("bound_m") = py::none(),
            py::arg("do_send") = true, py::arg("do_recv") = true)
        .def("combine", &PyAll2All::combine, py::arg("out_tokens"), py::arg("indices"),
             py::arg("weights"), py::arg("expert_y"), py::arg("bound_m") = py::none(),
             py::arg("do_send") = true, py::arg("do_recv") = true,
             py::arg("accumulate") = false)
        .def_property_readonly("rank", &PyAll2All::rank)
        .def_property_readonly("world_size", &PyAll2All::world_size)
        .def_property_readonly("device", &PyAll2All::device);
#endif
}
