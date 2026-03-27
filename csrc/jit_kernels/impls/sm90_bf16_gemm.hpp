#pragma once
// [DEPRECATED] All BF16 GEMM launchers removed — they used host-side TMA
// descriptor creation (make_tma_*_desc) which has been superseded by the TK
// globals factory path. The SM90_BF16_GEMM_Runtime class and the following
// functions were removed:
//   - sm90_bf16_gemm
//   - sm90_bf16_grouped_gemm_contiguous
//   - sm90_bf16_grouped_gemm_masked
//   - sm90_bf16_batched_gemm
