#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <kernels/internal_api.hpp>
#include <torch/all.h>

// dummy empty _C module
extern "C" {
    PyObject* PyInit__C (void) {
  
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL
        };
  
        return PyModule_Create(&module_def);
    }
  }




STABLE_TORCH_LIBRARY(moe_cuda, m) {
  
  m.def(
      "fp8_nt_gemm(Tensor act, Tensor act_scale, Tensor weight, Tensor weight_scale, Tensor output, str compiled_dims) -> ()");
      m.def("fp8_grouped_gemm_nt(Tensor act, Tensor act_scale, Tensor weight, Tensor weight_scale, Tensor output, int gemm_type, optional<Tensor> grouped_layout) -> ()");
    m.def("fp8_grouped_gemm_swiglu(Tensor act, Tensor gate_weight, Tensor up_weight, Tensor scale_a, Tensor gate_scale, Tensor up_scale, Tensor scale_d, Tensor output) -> ()");    
    m.def("fp8_grouped_gemm_swiglu_contiguous(Tensor A, Tensor scale_a, Tensor gate_weight, Tensor gate_scale, Tensor up_weight, Tensor up_scale, Tensor D, Tensor scale_d) -> ()");
    m.def("fp8_grouped_gemm_swiglu_masked(Tensor A, Tensor scale_a, Tensor gate_weight, Tensor gate_scale, Tensor up_weight, Tensor up_scale, Tensor D, Tensor scale_d) -> ()");
    m.def("fp8_grouped_gemm_swiglu_consumer_pp(Tensor A, Tensor gate_weight, Tensor up_weight, Tensor scale_a, Tensor gate_scale, Tensor up_scale, Tensor scale_d, Tensor output, int gemm_type, optional<Tensor> grouped_layout) -> ()");
    m.def("cast(Tensor inp, Tensor out, optional<Tensor> scale) -> ()");
    m.def("fused_silu_mul_quant(Tensor gemm_out, Tensor swiglu_out, Tensor scale) -> ()");
}


STABLE_TORCH_LIBRARY_IMPL(moe_cuda, CUDA, m) {
    m.impl("fp8_nt_gemm",
        TORCH_BOX(&moe_cuda::kernels::fp8_gemm_nt));
    m.impl("fp8_grouped_gemm_nt",
        TORCH_BOX(&moe_cuda::kernels::fp8_grouped_gemm_nt));
    m.impl("fp8_grouped_gemm_swiglu",
        TORCH_BOX(&moe_cuda::kernels::fp8_grouped_gemm_swiglu));
    m.impl("fp8_grouped_gemm_swiglu_contiguous",
        TORCH_BOX(&moe_cuda::kernels::fp8_grouped_gemm_swiglu_consumer_pp));
    m.impl("cast",
        TORCH_BOX(&moe_cuda::kernels::cast_dispatch));
    m.impl("fused_silu_mul_quant",
        TORCH_BOX(&moe_cuda::kernels::fused_silu_mul_quant));
}