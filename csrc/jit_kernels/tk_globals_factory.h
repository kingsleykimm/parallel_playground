#pragma once
#include <cstddef>
#include <pyutils/parallel_tensor.cuh>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

extern "C" size_t tk_globals_size(int bm, int bn, int bk,
                                  c10::ScalarType c_dtype);
extern "C" void tk_build_globals(int bm, int bn, int bk,
                                 c10::ScalarType c_dtype, void *out, void *A,
                                 void *B, void *C, void *scale_a, void *scale_b,
                                 size_t M, size_t N, size_t K);

// Grouped GEMM globals (kernel2::grouped_matmul_layout)
// gemm_type: 0 = MGroupedMasked, 1 = MGroupedContiguous
// total_M and total_N already incorporate num_groups
extern "C" size_t tk_grouped_globals_size(int bm, int bn, int bk, int gemm_type,
                                          c10::ScalarType c_dtype);
extern "C" void tk_build_grouped_globals(int bm, int bn, int bk, int gemm_type,
                                         int num_groups,
                                         c10::ScalarType c_dtype, void *out,
                                         void *A, void *B, void *C,
                                         void *scale_a, void *scale_b,
                                         void *grouped_layout, size_t total_M,
                                         size_t total_N, size_t K);
extern "C" void tk_dump_grouped_globals(int bm, int bn, int bk, int gemm_type,
                                        c10::ScalarType c_dtype,
                                        const void *globals_ptr);

// kernel3 globals (kernel3::grouped_matmul_layout) — gate+up fused
// silu-mul-quant gate, up : separate weight tensors (each covers half the
// combined N) out_scales: per-row quantization scale output
extern "C" size_t tk_kernel3_globals_size(int bm, int bn, int bk, int gemm_type,
                                          c10::ScalarType c_dtype);
extern "C" void tk_build_kernel3_globals(
    int bm, int bn, int bk, int gemm_type, int num_groups,
    c10::ScalarType c_dtype, void *out, void *A, void *gate, void *up, void *C,
    void *scale_a, void *scale_gate, void *scale_up, void *out_scales,
    void *grouped_layout, size_t total_M, size_t total_N, size_t K);

// kernel4 globals (kernel4::globals) — ping-pong consumer scheduled FP8 GEMM
// BM is fixed at 64; only BN and BK are needed for config dispatch.
extern "C" size_t tk_kernel4_globals_size(int bn, int bk, int gemm_type,
                                          c10::ScalarType c_dtype);
extern "C" void tk_build_kernel4_globals(
    int bn, int bk, int gemm_type, int num_groups, c10::ScalarType c_dtype,
    void *out, void *A, void *gate, void *up, void *D, void *scale_a,
    void *scale_gate, void *scale_up, void *scale_d, void *grouped_layout,
    size_t total_M, size_t total_N, size_t K);

// kernel5_1 globals (kernel5_1::globals) — Fused Dispatch + FC1 of SwiGLU MLP
// H is dispatched at runtime because it affects TMA descriptor tile sizes
// (token_vec_tile = sv_fp8e4m3<H>).
size_t tk_kernel5_1_globals_size(int H);
void tk_build_kernel5_1_globals(
    int H, void *out, kittens::py::TKParallelTensor &in_tokens,
    kittens::py::TKParallelTensor &in_tokens_scales,
    torch::stable::Tensor &expert_x_tokens,
    torch::stable::Tensor &expert_x_tokens_scale,
    torch::stable::Tensor &comm_comp_barrier, torch::stable::Tensor &gate,
    torch::stable::Tensor &up, torch::stable::Tensor &C,
    torch::stable::Tensor &scale_gate, torch::stable::Tensor &scale_up,
    torch::stable::Tensor &out_scales, torch::stable::Tensor &indices,
    kittens::py::TKParallelTensor &global_num_routed,
    kittens::py::TKParallelTensor &expert_to_token_map,
    torch::stable::Tensor &padded_expert_counts,
    torch::stable::Tensor &src_token_idx, torch::stable::Tensor &src_dev_idx,
    kittens::py::TKParallelTensor &barrier, int num_tokens,
    int *num_recv_tokens, int dp_rank, int rank, int dp_size, int cur_dp_group,
    int num_dp_groups, int num_comm_sms, int num_comp_sms);
