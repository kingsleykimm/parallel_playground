#pragma once
#include <cstddef>
#include <c10/core/ScalarType.h>

extern "C" size_t tk_globals_size(int bm, int bn, int bk,
    c10::ScalarType c_dtype);
extern "C" void tk_build_globals(int bm, int bn, int bk,
    c10::ScalarType c_dtype, void* out,
    void* A, void* B, void* C, void* scale_a, void* scale_b,
    size_t M, size_t N, size_t K);

// Grouped GEMM globals (kernel2::grouped_matmul_layout)
// gemm_type: 0 = MGroupedMasked, 1 = MGroupedContiguous
// total_M and total_N already incorporate num_groups
extern "C" size_t tk_grouped_globals_size(int bm, int bn, int bk,
    int gemm_type, c10::ScalarType c_dtype);
extern "C" void tk_build_grouped_globals(int bm, int bn, int bk,
    int gemm_type, int num_groups, c10::ScalarType c_dtype, void* out,
    void* A, void* B, void* C, void* scale_a, void* scale_b,
    void* grouped_layout, size_t total_M, size_t total_N, size_t K);
