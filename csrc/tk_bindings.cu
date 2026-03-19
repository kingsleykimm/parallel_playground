#include <pybind11/pybind11.h>
#include <pyutils/parallel_tensor.cuh>

void bind_tk_parallel_tensor(pybind11::module_& m) {
    BIND_TK_PARALLEL_TENSOR(m);
}
