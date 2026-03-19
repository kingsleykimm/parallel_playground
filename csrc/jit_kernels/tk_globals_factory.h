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
extern "C" void tk_dump_grouped_globals(int bm, int bn, int bk,
    int gemm_type, c10::ScalarType c_dtype, const void* globals_ptr);

// kernel3 globals (kernel3::grouped_matmul_layout) — gate+up fused silu-mul-quant
// gate, up : separate weight tensors (each covers half the combined N)
// out_scales: per-row quantization scale output
extern "C" size_t tk_kernel3_globals_size(int bm, int bn, int bk,
    int gemm_type, c10::ScalarType c_dtype);
extern "C" void tk_build_kernel3_globals(int bm, int bn, int bk,
    int gemm_type, int num_groups, c10::ScalarType c_dtype, void* out,
    void* A, void* gate, void* up, void* C,
    void* scale_a, void* scale_gate, void* scale_up, void* out_scales,
    void* grouped_layout, size_t total_M, size_t total_N, size_t K);

// kernel4 globals (kernel4::globals) — ping-pong consumer scheduled FP8 GEMM
// BM is fixed at 64; only BN and BK are needed for config dispatch.
extern "C" size_t tk_kernel4_globals_size(int bn, int bk,
    int gemm_type, c10::ScalarType c_dtype);
extern "C" void tk_build_kernel4_globals(int bn, int bk,
    int gemm_type, int num_groups, c10::ScalarType c_dtype, void* out,
    void* A, void* gate, void* up, void* D,
    void* scale_a, void* scale_gate, void* scale_up, void* scale_d,
    void* grouped_layout, size_t total_M, size_t total_N, size_t K);
