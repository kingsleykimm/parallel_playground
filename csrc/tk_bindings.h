#pragma once
#include <pybind11/pybind11.h>

void bind_tk_parallel_tensor(pybind11::module_& m);
void bind_fused_dispatch_grouped_gemm_swiglu(pybind11::module_& m);
