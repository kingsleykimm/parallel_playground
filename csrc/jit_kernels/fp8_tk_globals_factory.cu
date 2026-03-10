// Pre-compiled globals factory for ThunderKittens FP8 GEMM kernels.
// Compiled once with nvcc at build time. Constructs globals structs
// (containing TMA descriptors via gl<> constructors) for each supported
// tile configuration.

#include <algorithm>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <kittens.cuh>
#include <moe_cuda/kernels/kernel2.cuh>
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>
#include <runtime/utils.h>
#include <type_traits>

// Use matmul_layout directly from the .cuh to guarantee ABI match with NVRTC
// kernel.
template <int _BM, int _BN, int _BK, typename c_dtype> struct fp8_tk_factory {
  using globals_t = typename matmul_layout<_BM, _BN, _BK, c_dtype>::globals;

  static size_t size() { return sizeof(globals_t); }

  static void build(void *out, void *A, void *B, void *C, void *scale_a,
                    void *scale_b, size_t M, size_t N, size_t K) {
    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, M, K},
        {(fp8e4m3 *)B, nullptr, nullptr, N, K},
        {(c_dtype *)C, nullptr, nullptr, M, N},
        {(float *)scale_a, nullptr, nullptr, K / 128, M},
        {(float *)scale_b, nullptr, nullptr, N / 128, K / 128},
    };
    memcpy(out, &G, sizeof(G));
  }
};

// Dispatch macros
#define TK_SIZE_CASE(bm, bn, bk, c_dtype_t)                                    \
  if (bm_ == bm && bn_ == bn && bk_ == bk)                                     \
    return fp8_tk_factory<bm, bn, bk, c_dtype_t>::size();

#define TK_BUILD_CASE(bm, bn, bk, c_dtype_t)                                   \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    fp8_tk_factory<bm, bn, bk, c_dtype_t>::build(out, A, B, C, scale_a,        \
                                                 scale_b, M, N, K);            \
    return;                                                                    \
  }

// Supported configs: {64,128} x {64,128,256} x {64,128}
#define TK_ALL_CONFIGS(MACRO, c_dtype_t)                                       \
  MACRO(64, 32, 128, c_dtype_t)                                                \
  MACRO(64, 48, 128, c_dtype_t)                                                \
  MACRO(64, 64, 128, c_dtype_t)                                                \
  MACRO(64, 80, 128, c_dtype_t)                                                \
  MACRO(64, 96, 128, c_dtype_t)                                                \
  MACRO(64, 112, 128, c_dtype_t)                                               \
  MACRO(64, 128, 128, c_dtype_t)                                               \
  MACRO(64, 144, 128, c_dtype_t)                                               \
  MACRO(64, 160, 128, c_dtype_t)                                               \
  MACRO(64, 176, 128, c_dtype_t)                                               \
  MACRO(64, 192, 128, c_dtype_t)                                               \
  MACRO(64, 208, 128, c_dtype_t)                                               \
  MACRO(64, 224, 128, c_dtype_t)                                               \
  MACRO(64, 240, 128, c_dtype_t)                                               \
  MACRO(64, 256, 128, c_dtype_t)                                               \
  MACRO(128, 32, 128, c_dtype_t)                                               \
  MACRO(128, 48, 128, c_dtype_t)                                               \
  MACRO(128, 64, 128, c_dtype_t)                                               \
  MACRO(128, 80, 128, c_dtype_t)                                               \
  MACRO(128, 96, 128, c_dtype_t)                                               \
  MACRO(128, 112, 128, c_dtype_t)                                              \
  MACRO(128, 128, 128, c_dtype_t)                                              \
  MACRO(128, 144, 128, c_dtype_t)                                              \
  MACRO(128, 160, 128, c_dtype_t)                                              \
  MACRO(128, 176, 128, c_dtype_t)                                              \
  MACRO(128, 192, 128, c_dtype_t)                                              \
  MACRO(128, 208, 128, c_dtype_t)                                              \
  MACRO(128, 224, 128, c_dtype_t)                                              \
  MACRO(128, 240, 128, c_dtype_t)                                              \
  MACRO(128, 256, 128, c_dtype_t)

template <typename c_dtype>
static size_t tk_globals_size_impl(int bm_, int bn_, int bk_) {
  if (get_env<int>("JIT_DEBUG") > 0) {
    printf("tk_globals_size: BM=%d BN=%d BK=%d c_dtype=%s\n", bm_, bn_, bk_,
           std::is_same_v<c_dtype, float> ? "float" : "__nv_bfloat16");
  }
  TK_ALL_CONFIGS(TK_SIZE_CASE, c_dtype)
  fprintf(stderr, "tk_globals_size: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

template <typename c_dtype>
static void tk_build_globals_impl(int bm_, int bn_, int bk_, void *out, void *A,
                                  void *B, void *C, void *scale_a,
                                  void *scale_b, size_t M, size_t N, size_t K) {
  TK_ALL_CONFIGS(TK_BUILD_CASE, c_dtype)
  fprintf(stderr, "tk_build_globals: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

extern "C" size_t tk_globals_size(int bm_, int bn_, int bk_,
                                  c10::ScalarType c_dtype_) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    return tk_globals_size_impl<float>(bm_, bn_, bk_);
  case c10::ScalarType::BFloat16:
    return tk_globals_size_impl<__nv_bfloat16>(bm_, bn_, bk_);
  default:
    fprintf(
        stderr,
        "tk_globals_size: unsupported output dtype=%d for BM=%d BN=%d BK=%d\n",
        static_cast<int>(c_dtype_), bm_, bn_, bk_);
    abort();
  }
}

extern "C" void tk_build_globals(int bm_, int bn_, int bk_,
                                 c10::ScalarType c_dtype_, void *out, void *A,
                                 void *B, void *C, void *scale_a, void *scale_b,
                                 size_t M, size_t N, size_t K) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    tk_build_globals_impl<float>(bm_, bn_, bk_, out, A, B, C, scale_a, scale_b,
                                 M, N, K);
    return;
  case c10::ScalarType::BFloat16:
    tk_build_globals_impl<__nv_bfloat16>(bm_, bn_, bk_, out, A, B, C, scale_a,
                                         scale_b, M, N, K);
    return;
  default:
    fprintf(
        stderr,
        "tk_build_globals: unsupported output dtype=%d for BM=%d BN=%d BK=%d\n",
        static_cast<int>(c_dtype_), bm_, bn_, bk_);
    abort();
  }
}

#undef TK_SIZE_CASE
#undef TK_BUILD_CASE
#undef TK_ALL_CONFIGS

// =========== Grouped GEMM factory (kernel2::grouped_matmul_layout) ===========

// NUM_GROUPS is NOT a template param here — grouped_matmul_layout doesn't
// depend on it, so globals_t size is the same for all group counts.
// num_groups is passed at runtime and used only to compute C.cols.
template <int _BM, int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
struct fp8_grouped_tk_factory {
  using globals_t =
      typename kernel2::grouped_matmul_layout<_BM, _BN, _BK, GEMM_TYPE,
                                              c_dtype>::globals;

  static size_t size() { return sizeof(globals_t); }

  // A / B / scale_b TMA descriptors span the full concatenated tensor.
  // C cols = total_N / num_groups:
  //   contiguous (num_groups=1): C.cols = total_N   (all groups' output cols)
  //   masked     (num_groups>1): C.cols = N_per_group (each group's own N cols)
  static void build(void *out, void *A, void *B, void *C, void *scale_a,
                    void *scale_b, void *grouped_layout, size_t total_M,
                    size_t total_N, size_t K, int num_groups) {
    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, total_M, K},
        {(fp8e4m3 *)B, nullptr, nullptr, total_N, K},
        {(c_dtype *)C, nullptr, nullptr, total_M, total_N / num_groups},
        {(float *)scale_a, nullptr, nullptr, K / 128, total_M},
        {(float *)scale_b, nullptr, nullptr, total_N / 128, K / 128},
        (float *)grouped_layout,
        /*cur_group_idx=*/0,
        /*current_m_cumsum=*/0,
    };
    memcpy(out, &G, sizeof(G));
  }
};

#define TK_GROUPED_SIZE_CASE(bm, bn, bk, c_dtype_t)                            \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0)                                                       \
      return fp8_grouped_tk_factory<bm, bn, bk, 0, c_dtype_t>::size();        \
    if (gemm_type_ == 1)                                                       \
      return fp8_grouped_tk_factory<bm, bn, bk, 1, c_dtype_t>::size();        \
  }

#define TK_GROUPED_BUILD_CASE(bm, bn, bk, c_dtype_t)                           \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0) {                                                     \
      fp8_grouped_tk_factory<bm, bn, bk, 0, c_dtype_t>::build(                \
          out, A, B, C, scale_a, scale_b, grouped_layout, total_M, total_N,   \
          K, num_groups_);                                                     \
      return;                                                                  \
    }                                                                          \
    if (gemm_type_ == 1) {                                                     \
      fp8_grouped_tk_factory<bm, bn, bk, 1, c_dtype_t>::build(                \
          out, A, B, C, scale_a, scale_b, grouped_layout, total_M, total_N,   \
          K, num_groups_);                                                     \
      return;                                                                  \
    }                                                                          \
  }

#define TK_ALL_GROUPED_CONFIGS(MACRO, c_dtype_t)                               \
  MACRO(64, 32, 128, c_dtype_t)                                                \
  MACRO(64, 48, 128, c_dtype_t)                                                \
  MACRO(64, 64, 128, c_dtype_t)                                                \
  MACRO(64, 80, 128, c_dtype_t)                                                \
  MACRO(64, 96, 128, c_dtype_t)                                                \
  MACRO(64, 112, 128, c_dtype_t)                                               \
  MACRO(64, 128, 128, c_dtype_t)                                               \
  MACRO(64, 144, 128, c_dtype_t)                                               \
  MACRO(64, 160, 128, c_dtype_t)                                               \
  MACRO(64, 192, 128, c_dtype_t)                                               \
  MACRO(64, 256, 128, c_dtype_t)                                               \
  MACRO(128, 32, 128, c_dtype_t)                                               \
  MACRO(128, 64, 128, c_dtype_t)                                               \
  MACRO(128, 96, 128, c_dtype_t)                                               \
  MACRO(128, 128, 128, c_dtype_t)                                              \
  MACRO(128, 160, 128, c_dtype_t)                                              \
  MACRO(128, 192, 128, c_dtype_t)                                              \
  MACRO(128, 256, 128, c_dtype_t)

template <typename c_dtype>
static size_t tk_grouped_globals_size_impl(int bm_, int bn_, int bk_,
                                           int gemm_type_) {
  TK_ALL_GROUPED_CONFIGS(TK_GROUPED_SIZE_CASE, c_dtype)
  fprintf(stderr,
          "tk_grouped_globals_size: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

template <typename c_dtype>
static void tk_build_grouped_globals_impl(int bm_, int bn_, int bk_,
                                          int gemm_type_, int num_groups_,
                                          void *out, void *A, void *B, void *C,
                                          void *scale_a, void *scale_b,
                                          void *grouped_layout, size_t total_M,
                                          size_t total_N, size_t K) {
  TK_ALL_GROUPED_CONFIGS(TK_GROUPED_BUILD_CASE, c_dtype)
  fprintf(stderr,
          "tk_build_grouped_globals: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

extern "C" size_t tk_grouped_globals_size(int bm_, int bn_, int bk_,
                                          int gemm_type_,
                                          c10::ScalarType c_dtype_) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    return tk_grouped_globals_size_impl<float>(bm_, bn_, bk_, gemm_type_);
  case c10::ScalarType::BFloat16:
    return tk_grouped_globals_size_impl<__nv_bfloat16>(bm_, bn_, bk_,
                                                       gemm_type_);
  default:
    fprintf(stderr, "tk_grouped_globals_size: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

extern "C" void tk_build_grouped_globals(
    int bm_, int bn_, int bk_, int gemm_type_, int num_groups_,
    c10::ScalarType c_dtype_, void *out, void *A, void *B, void *C,
    void *scale_a, void *scale_b, void *grouped_layout, size_t total_M,
    size_t total_N, size_t K) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    tk_build_grouped_globals_impl<float>(bm_, bn_, bk_, gemm_type_, num_groups_,
                                         out, A, B, C, scale_a, scale_b,
                                         grouped_layout, total_M, total_N, K);
    return;
  case c10::ScalarType::BFloat16:
    tk_build_grouped_globals_impl<__nv_bfloat16>(
        bm_, bn_, bk_, gemm_type_, num_groups_, out, A, B, C, scale_a, scale_b,
        grouped_layout, total_M, total_N, K);
    return;
  default:
    fprintf(stderr, "tk_build_grouped_globals: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

#undef TK_GROUPED_SIZE_CASE
#undef TK_GROUPED_BUILD_CASE
#undef TK_ALL_GROUPED_CONFIGS
