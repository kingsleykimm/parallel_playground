// Pre-compiled globals factory for ThunderKittens FP8 GEMM kernels.
// Compiled once with nvcc at build time. Constructs globals structs
// (containing TMA descriptors via gl<> constructors) for each supported
// tile configuration.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <kittens.cuh>
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>

// Use matmul_layout directly from the .cuh to guarantee ABI match with NVRTC
// kernel.
template <int _BM, int _BN, int _BK> struct fp8_tk_factory {
  using globals_t = typename matmul_layout<_BM, _BN, _BK>::globals;

  static size_t size() { return sizeof(globals_t); }

  static void build(void *out, void *A, void *B, void *C, void *scale_a,
                    void *scale_b, size_t M, size_t N, size_t K) {
    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, M, K},
        {(fp8e4m3 *)B, nullptr, nullptr, N, K},
        {(float *)C, nullptr, nullptr, M, N},
        {(float *)scale_a, nullptr, nullptr, M, K / 128},
        {(float *)scale_b, nullptr, nullptr, N / 128, K / 128},
    };
    memcpy(out, &G, sizeof(G));
  }
};

// Dispatch macros
#define TK_SIZE_CASE(bm, bn, bk)                                               \
  if (bm_ == bm && bn_ == bn && bk_ == bk)                                     \
    return fp8_tk_factory<bm, bn, bk>::size();

#define TK_BUILD_CASE(bm, bn, bk)                                              \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    fp8_tk_factory<bm, bn, bk>::build(out, A, B, C, scale_a, scale_b, M, N,    \
                                      K);                                      \
    return;                                                                    \
  }

// Supported configs: {64,128} x {64,128,256} x {64,128}
#define TK_ALL_CONFIGS(MACRO)                                                  \
  MACRO(64, 64, 128)                                                           \
  MACRO(64, 128, 128)                                                          \
  MACRO(64, 256, 128)                                                          \
  MACRO(128, 64, 128)                                                          \
  MACRO(128, 128, 128)                                                         \
  MACRO(128, 256, 128)

extern "C" size_t tk_globals_size(int m, int n, int k, int bm_, int bn_,
                                  int bk_) {
  TK_ALL_CONFIGS(TK_SIZE_CASE)
  fprintf(stderr, "tk_globals_size: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

extern "C" void tk_build_globals(int bm_, int bn_, int bk_, void *out, void *A,
                                 void *B, void *C, void *scale_a, void *scale_b,
                                 size_t M, size_t N, size_t K) {
  TK_ALL_CONFIGS(TK_BUILD_CASE)
  fprintf(stderr, "tk_build_globals: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

#undef TK_SIZE_CASE
#undef TK_BUILD_CASE
#undef TK_ALL_CONFIGS
