// Pre-compiled globals factory for ThunderKittens FP8 GEMM kernels.
// Compiled once with nvcc at build time. Constructs globals structs
// (containing TMA descriptors via gl<> constructors) for each supported
// tile configuration.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <kittens.cuh>
#include <moe_cuda/kernels/kernel2.cuh>
#include <moe_cuda/kernels/sm90_fp8_gemm_1d2d_tk.cuh>
#include <pyutils/parallel_tensor.cuh>
#include <pyutils/torchutils.cuh>
#include <runtime/utils.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <type_traits>

#include <torch/csrc/inductor/aoti_torch/utils.h>
inline at::Tensor &stable_to_aten(torch::stable::Tensor &t) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(t.get());
}
inline const at::Tensor &stable_to_aten(const torch::stable::Tensor &t) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(t.get());
}

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
  MACRO(64, 64, 128, c_dtype_t)                                                \
  MACRO(64, 96, 128, c_dtype_t)                                                \
  MACRO(64, 128, 128, c_dtype_t)                                               \
  MACRO(64, 160, 128, c_dtype_t)                                               \
  MACRO(64, 192, 128, c_dtype_t)                                               \
  MACRO(64, 224, 128, c_dtype_t)                                               \
  MACRO(64, 256, 128, c_dtype_t)                                               \
  MACRO(128, 32, 128, c_dtype_t)                                               \
  MACRO(128, 64, 128, c_dtype_t)                                               \
  MACRO(128, 96, 128, c_dtype_t)                                               \
  MACRO(128, 128, 128, c_dtype_t)                                              \
  MACRO(128, 160, 128, c_dtype_t)                                              \
  MACRO(128, 192, 128, c_dtype_t)                                              \
  MACRO(128, 224, 128, c_dtype_t)                                              \
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
  case c10::ScalarType::Float8_e4m3fn:
    return tk_globals_size_impl<__nv_fp8_e4m3>(bm_, bn_, bk_);
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
  //   contiguous (num_groups=1): C.cols = N_per_group   (all groups' output
  //   cols) masked     (num_groups>1): C.cols = N_per_group (each group's own N
  //   cols)
  static void build(void *out, void *A, void *B, void *C, void *scale_a,
                    void *scale_b, void *grouped_layout, size_t total_M,
                    size_t total_N, size_t K, int num_groups) {
    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, total_M, K},
        {(fp8e4m3 *)B, nullptr, nullptr, total_N, K},
        {(c_dtype *)C, nullptr, nullptr, total_M, total_N / num_groups},
        {(float *)scale_a, nullptr, nullptr, K / 128, total_M},
        {(float *)scale_b, nullptr, nullptr, total_N / 128, K / 128},
        (int *)grouped_layout,
    };
    memcpy(out, &G, sizeof(G));
  }
};

#define TK_GROUPED_SIZE_CASE(bm, bn, bk, c_dtype_t)                            \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0)                                                       \
      return fp8_grouped_tk_factory<bm, bn, bk, 0, c_dtype_t>::size();         \
    if (gemm_type_ == 1)                                                       \
      return fp8_grouped_tk_factory<bm, bn, bk, 1, c_dtype_t>::size();         \
  }

#define TK_GROUPED_BUILD_CASE(bm, bn, bk, c_dtype_t)                           \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0) {                                                     \
      fp8_grouped_tk_factory<bm, bn, bk, 0, c_dtype_t>::build(                 \
          out, A, B, C, scale_a, scale_b, grouped_layout, total_M, total_N, K, \
          num_groups_);                                                        \
      return;                                                                  \
    }                                                                          \
    if (gemm_type_ == 1) {                                                     \
      fp8_grouped_tk_factory<bm, bn, bk, 1, c_dtype_t>::build(                 \
          out, A, B, C, scale_a, scale_b, grouped_layout, total_M, total_N, K, \
          num_groups_);                                                        \
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

template <typename GL>
static void dump_gl_layout(const char *name, const GL &gl) {
  printf("  %s raw_ptr=%p batch=%d depth=%d rows=%d cols=%d numel=%zu\n", name,
         static_cast<const void *>(gl.raw_ptr), gl.batch(), gl.depth(),
         gl.rows(), gl.cols(), gl.numel());
}

template <int _BM, int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
static void tk_dump_grouped_globals_impl_typed(const void *globals_ptr) {
  using globals_t =
      typename kernel2::grouped_matmul_layout<_BM, _BN, _BK, GEMM_TYPE,
                                              c_dtype>::globals;
  const auto &G = *reinterpret_cast<const globals_t *>(globals_ptr);
  printf("grouped globals size=%zu\n", sizeof(globals_t));
  dump_gl_layout("A", G.A);
  dump_gl_layout("B", G.B);
  dump_gl_layout("C", G.C);
  dump_gl_layout("scale_a", G.scale_a);
  dump_gl_layout("scale_b", G.scale_b);
  printf("  grouped_layout raw_ptr=%p\n",
         static_cast<const void *>(G.grouped_layout));
}

#define TK_GROUPED_DUMP_CASE(bm, bn, bk, c_dtype_t)                            \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0) {                                                     \
      tk_dump_grouped_globals_impl_typed<bm, bn, bk, 0, c_dtype_t>(            \
          globals_ptr);                                                        \
      return;                                                                  \
    }                                                                          \
    if (gemm_type_ == 1) {                                                     \
      tk_dump_grouped_globals_impl_typed<bm, bn, bk, 1, c_dtype_t>(            \
          globals_ptr);                                                        \
      return;                                                                  \
    }                                                                          \
  }

template <typename c_dtype>
static void tk_dump_grouped_globals_dispatch(int bm_, int bn_, int bk_,
                                             int gemm_type_,
                                             const void *globals_ptr) {
  TK_ALL_GROUPED_CONFIGS(TK_GROUPED_DUMP_CASE, c_dtype)
  fprintf(stderr,
          "tk_dump_grouped_globals: unsupported config BM=%d BN=%d BK=%d\n",
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

extern "C" void tk_build_grouped_globals(int bm_, int bn_, int bk_,
                                         int gemm_type_, int num_groups_,
                                         c10::ScalarType c_dtype_, void *out,
                                         void *A, void *B, void *C,
                                         void *scale_a, void *scale_b,
                                         void *grouped_layout, size_t total_M,
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

extern "C" void tk_dump_grouped_globals(int bm_, int bn_, int bk_,
                                        int gemm_type_,
                                        c10::ScalarType c_dtype_,
                                        const void *globals_ptr) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    tk_dump_grouped_globals_dispatch<float>(bm_, bn_, bk_, gemm_type_,
                                            globals_ptr);
    return;
  case c10::ScalarType::BFloat16:
    tk_dump_grouped_globals_dispatch<__nv_bfloat16>(bm_, bn_, bk_, gemm_type_,
                                                    globals_ptr);
    return;
  default:
    fprintf(stderr, "tk_dump_grouped_globals: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

#undef TK_GROUPED_SIZE_CASE
#undef TK_GROUPED_BUILD_CASE
#undef TK_ALL_GROUPED_CONFIGS

// =========== kernel3 factory (kernel3::grouped_matmul_layout) ===========
// kernel3 globals differ from kernel2 in:
//   - Two separate B tiles (gate, up) instead of one B
//   - An out_scales layout for the quantized epilogue output
//   - C.cols = N_per_group / 2  (silu-mul output is half the combined N)
//   - scale_gate / scale_up cover total_N/2 rows (not total_N)
//   - out_scales rows = N_per_group / (2 * _BN)  (N column-block count)
#include <moe_cuda/kernels/kernel3.cuh>

template <int _BM, int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
struct fp8_kernel3_factory {
  using globals_t =
      typename kernel3::grouped_matmul_layout<_BM, _BN, _BK, GEMM_TYPE,
                                              c_dtype>::globals;

  static size_t size() { return sizeof(globals_t); }

  static void build(void *out, void *A, void *gate, void *up, void *C,
                    void *scale_a, void *scale_gate, void *scale_up,
                    void *out_scales, void *grouped_layout, size_t total_M,
                    size_t total_N, size_t K, int num_groups) {
    // gate and up each cover half of the combined N dimension per group
    size_t N_per_group = total_N / (size_t)num_groups;

    // out_scales rows = number of BN-wide N-blocks per group
    // accessed with coord{coord_y, M_offset} where coord_y in [0,
    // N_per_group/(2*_BN))
    size_t out_scales_rows = N_per_group / 128; // per-token 128-wide scaling

    if (get_env<int>("JIT_DEBUG") > 0) {
      printf("out_Scales_rows : %zu", out_scales_rows);
    }

    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, total_M, K},
        {(fp8e4m3 *)gate, nullptr, nullptr, total_N, K},
        {(fp8e4m3 *)up, nullptr, nullptr, total_N, K},
        {(c_dtype *)C, nullptr, nullptr, total_M, N_per_group},
        {(float *)scale_a, nullptr, nullptr, K / 128, total_M},
        {(float *)scale_gate, nullptr, nullptr, total_N / 128, K / 128},
        {(float *)scale_up, nullptr, nullptr, total_N / 128, K / 128},
        {(float *)out_scales, nullptr, nullptr, out_scales_rows, total_M},
        (int *)grouped_layout,
    };
    memcpy(out, &G, sizeof(G));
  }
};

#define TK_KERNEL3_SIZE_CASE(bm, bn, bk, c_dtype_t)                            \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0)                                                       \
      return fp8_kernel3_factory<bm, bn, bk, 0, c_dtype_t>::size();            \
    if (gemm_type_ == 1)                                                       \
      return fp8_kernel3_factory<bm, bn, bk, 1, c_dtype_t>::size();            \
  }

#define TK_KERNEL3_BUILD_CASE(bm, bn, bk, c_dtype_t)                           \
  if (bm_ == bm && bn_ == bn && bk_ == bk) {                                   \
    if (gemm_type_ == 0) {                                                     \
      fp8_kernel3_factory<bm, bn, bk, 0, c_dtype_t>::build(                    \
          out, A, gate, up, C, scale_a, scale_gate, scale_up, out_scales,      \
          grouped_layout, total_M, total_N, K, num_groups_);                   \
      return;                                                                  \
    }                                                                          \
    if (gemm_type_ == 1) {                                                     \
      fp8_kernel3_factory<bm, bn, bk, 1, c_dtype_t>::build(                    \
          out, A, gate, up, C, scale_a, scale_gate, scale_up, out_scales,      \
          grouped_layout, total_M, total_N, K, num_groups_);                   \
      return;                                                                  \
    }                                                                          \
  }

// kernel3 only supports BN that is a multiple of 128
#define TK_ALL_KERNEL3_CONFIGS(MACRO, c_dtype_t)                               \
  MACRO(64, 128, 128, c_dtype_t)                                               \
  MACRO(128, 128, 128, c_dtype_t)

template <typename c_dtype>
static size_t tk_kernel3_globals_size_impl(int bm_, int bn_, int bk_,
                                           int gemm_type_) {
  TK_ALL_KERNEL3_CONFIGS(TK_KERNEL3_SIZE_CASE, c_dtype)
  fprintf(stderr,
          "tk_kernel3_globals_size: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

template <typename c_dtype>
static void tk_build_kernel3_globals_impl(
    int bm_, int bn_, int bk_, int gemm_type_, int num_groups_, void *out,
    void *A, void *gate, void *up, void *C, void *scale_a, void *scale_gate,
    void *scale_up, void *out_scales, void *grouped_layout, size_t total_M,
    size_t total_N, size_t K) {
  TK_ALL_KERNEL3_CONFIGS(TK_KERNEL3_BUILD_CASE, c_dtype)
  fprintf(stderr,
          "tk_build_kernel3_globals: unsupported config BM=%d BN=%d BK=%d\n",
          bm_, bn_, bk_);
  abort();
}

extern "C" size_t tk_kernel3_globals_size(int bm_, int bn_, int bk_,
                                          int gemm_type_,
                                          c10::ScalarType c_dtype_) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    return tk_kernel3_globals_size_impl<float>(bm_, bn_, bk_, gemm_type_);
  case c10::ScalarType::BFloat16:
    return tk_kernel3_globals_size_impl<__nv_bfloat16>(bm_, bn_, bk_,
                                                       gemm_type_);
  case c10::ScalarType::Float8_e4m3fn:
    return tk_kernel3_globals_size_impl<__nv_fp8_e4m3>(bm_, bn_, bk_,
                                                       gemm_type_);
  default:
    fprintf(stderr, "tk_kernel3_globals_size: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

extern "C" void tk_build_kernel3_globals(
    int bm_, int bn_, int bk_, int gemm_type_, int num_groups_,
    c10::ScalarType c_dtype_, void *out, void *A, void *gate, void *up, void *C,
    void *scale_a, void *scale_gate, void *scale_up, void *out_scales,
    void *grouped_layout, size_t total_M, size_t total_N, size_t K) {
  switch (c_dtype_) {
  case c10::ScalarType::Float:
    tk_build_kernel3_globals_impl<float>(
        bm_, bn_, bk_, gemm_type_, num_groups_, out, A, gate, up, C, scale_a,
        scale_gate, scale_up, out_scales, grouped_layout, total_M, total_N, K);
    return;
  case c10::ScalarType::BFloat16:
    tk_build_kernel3_globals_impl<__nv_bfloat16>(
        bm_, bn_, bk_, gemm_type_, num_groups_, out, A, gate, up, C, scale_a,
        scale_gate, scale_up, out_scales, grouped_layout, total_M, total_N, K);
    return;
  case c10::ScalarType::Float8_e4m3fn:
    tk_build_kernel3_globals_impl<__nv_fp8_e4m3>(
        bm_, bn_, bk_, gemm_type_, num_groups_, out, A, gate, up, C, scale_a,
        scale_gate, scale_up, out_scales, grouped_layout, total_M, total_N, K);
    return;
  default:
    fprintf(stderr, "tk_build_kernel3_globals: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

#undef TK_KERNEL3_SIZE_CASE
#undef TK_KERNEL3_BUILD_CASE
#undef TK_ALL_KERNEL3_CONFIGS

// =========== kernel4 factory (kernel4::globals) ===========
// kernel4 adds ping-pong consumer scheduling on top of kernel3.
// globals<> has M, N, K, NUM_GROUPS, NCW, NPW, NS, SS, SN as template params
// but none of these affect the data member types — only BM(=64), BN, BK, and
// c_dtype determine the gl<> field types. We instantiate with fixed dummy
// values for M/N/K/etc. and pass runtime values to the gl<> constructors.
#include <moe_cuda/kernels/kernel4.cuh>

// Dummy template params that satisfy kernel4::globals static_asserts:
//   BM == 64  (always)
//   BN % 128 == 0  (guaranteed by config list below)
static constexpr int K4_DUMMY_M = 128;
static constexpr int K4_DUMMY_NCW = 8;
static constexpr int K4_DUMMY_NPW = 2;
static constexpr int K4_DUMMY_NS = 2;
static constexpr int K4_DUMMY_SS = 227 * 1024 - 1024;
static constexpr int K4_DUMMY_SN = 8;

template <int _BN, int _BK, int GEMM_TYPE, typename c_dtype>
struct fp8_kernel4_factory {
  // M/N/K/NUM_GROUPS dummy values — only BM, BN, BK, c_dtype affect layout
  using globals_t =
      kernel4::globals<K4_DUMMY_M, _BN, _BK, 64, _BN, _BK, 1, K4_DUMMY_NCW,
                       K4_DUMMY_NPW, K4_DUMMY_NS, K4_DUMMY_SS, GEMM_TYPE,
                       c_dtype, K4_DUMMY_SN>;

  static size_t size() { return sizeof(globals_t); }

  static void build(void *out, void *A, void *gate, void *up, void *D,
                    void *scale_a, void *scale_gate, void *scale_up,
                    void *scale_d, void *grouped_layout, size_t total_M,
                    size_t total_N, size_t K, int num_groups) {
    size_t out_scales_rows = total_N / _BN;

    globals_t G{
        {(fp8e4m3 *)A, nullptr, nullptr, total_M, K},
        {(fp8e4m3 *)gate, nullptr, nullptr, total_N, K},
        {(fp8e4m3 *)up, nullptr, nullptr, total_N, K},
        {(c_dtype *)D, nullptr, nullptr, total_M, total_N / num_groups},
        {(float *)scale_a, nullptr, nullptr, K / 128, total_M},
        {(float *)scale_gate, nullptr, nullptr, total_N / 128, K / 128},
        {(float *)scale_up, nullptr, nullptr, total_N / 128, K / 128},
        {(float *)scale_d, nullptr, nullptr, out_scales_rows, total_M},
        (int *)grouped_layout,
    };
    memcpy(out, &G, sizeof(G));
  }
};

#define TK_KERNEL4_SIZE_CASE(bn, bk, c_dtype_t)                                \
  if (bn_ == bn && bk_ == bk) {                                                \
    if (gemm_type_ == 0)                                                       \
      return fp8_kernel4_factory<bn, bk, 0, c_dtype_t>::size();                \
    if (gemm_type_ == 1)                                                       \
      return fp8_kernel4_factory<bn, bk, 1, c_dtype_t>::size();                \
  }

#define TK_KERNEL4_BUILD_CASE(bn, bk, c_dtype_t)                               \
  if (bn_ == bn && bk_ == bk) {                                                \
    if (gemm_type_ == 0) {                                                     \
      fp8_kernel4_factory<bn, bk, 0, c_dtype_t>::build(                        \
          out, A, gate, up, D, scale_a, scale_gate, scale_up, scale_d,         \
          grouped_layout, total_M, total_N, K, num_groups_);                   \
      return;                                                                  \
    }                                                                          \
    if (gemm_type_ == 1) {                                                     \
      fp8_kernel4_factory<bn, bk, 1, c_dtype_t>::build(                        \
          out, A, gate, up, D, scale_a, scale_gate, scale_up, scale_d,         \
          grouped_layout, total_M, total_N, K, num_groups_);                   \
      return;                                                                  \
    }                                                                          \
  }

// BM is fixed at 64 by kernel4 static_assert; BN must be a multiple of 128
#define TK_ALL_KERNEL4_CONFIGS(MACRO, c_dtype_t)                               \
  MACRO(128, 128, c_dtype_t)                                                   \
  MACRO(256, 128, c_dtype_t)

template <typename c_dtype>
static size_t tk_kernel4_globals_size_impl(int bn_, int bk_, int gemm_type_) {
  TK_ALL_KERNEL4_CONFIGS(TK_KERNEL4_SIZE_CASE, c_dtype)
  fprintf(stderr, "tk_kernel4_globals_size: unsupported config BN=%d BK=%d\n",
          bn_, bk_);
  abort();
}

template <typename c_dtype>
static void
tk_build_kernel4_globals_impl(int bn_, int bk_, int gemm_type_, int num_groups_,
                              void *out, void *A, void *gate, void *up, void *D,
                              void *scale_a, void *scale_gate, void *scale_up,
                              void *scale_d, void *grouped_layout,
                              size_t total_M, size_t total_N, size_t K) {
  TK_ALL_KERNEL4_CONFIGS(TK_KERNEL4_BUILD_CASE, c_dtype)
  fprintf(stderr, "tk_build_kernel4_globals: unsupported config BN=%d BK=%d\n",
          bn_, bk_);
  abort();
}

extern "C" size_t tk_kernel4_globals_size(int bn_, int bk_, int gemm_type_,
                                          c10::ScalarType c_dtype_) {
  switch (c_dtype_) {
  case c10::ScalarType::Float8_e4m3fn:
    return tk_kernel4_globals_size_impl<__nv_fp8_e4m3>(bn_, bk_, gemm_type_);
  default:
    fprintf(stderr, "tk_kernel4_globals_size: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

extern "C" void tk_build_kernel4_globals(
    int bn_, int bk_, int gemm_type_, int num_groups_, c10::ScalarType c_dtype_,
    void *out, void *A, void *gate, void *up, void *D, void *scale_a,
    void *scale_gate, void *scale_up, void *scale_d, void *grouped_layout,
    size_t total_M, size_t total_N, size_t K) {
  switch (c_dtype_) {
  case c10::ScalarType::Float8_e4m3fn:
    tk_build_kernel4_globals_impl<__nv_fp8_e4m3>(
        bn_, bk_, gemm_type_, num_groups_, out, A, gate, up, D, scale_a,
        scale_gate, scale_up, scale_d, grouped_layout, total_M, total_N, K);
    return;
  default:
    fprintf(stderr, "tk_build_kernel4_globals: unsupported output dtype=%d\n",
            static_cast<int>(c_dtype_));
    abort();
  }
}

#undef TK_KERNEL4_SIZE_CASE
#undef TK_KERNEL4_BUILD_CASE
#undef TK_ALL_KERNEL4_CONFIGS

// ============== kernel5_1 factory (kernel5_1::globals) ======= /
//
// H must be dispatched at runtime because it determines token_vec_tile =
// sv_fp8e4m3<H>, which is a TMA type on in_tokens and expert_x_tokens GLs.
// The TMA descriptor's smem box size is baked in at GL construction time,
// so the factory must use the same H the JIT kernel is instantiated with.
// All other template params either don't affect data member types (M, I,
// num_warps, stages, smem_size) or are fixed (BM=64, BN=128).

#include <moe_cuda/kernels/kernel5_1.cuh>

// Fixed constants — these don't affect TMA descriptor tile sizes
static constexpr int K5_BM = 64;
static constexpr int K5_BN = 128;
static constexpr int K5_NUM_CONSUMER_WARPS = 8;
static constexpr int K5_NUM_PRODUCER_WARPS = 4;
static constexpr int K5_NUM_STAGES = 4;
static constexpr int K5_KERNEL_SMEM_SIZE = 227 * 1024;
static constexpr int K5_NUM_EXPERTS = 32;
static constexpr int K5_EXPERTS_PER_TOKEN = 8;
static constexpr int K5_SUPER_M = 12;
// M and I only affect static constexprs, not types — use small dummies
static constexpr int K5_M = 128;
static constexpr int K5_I = 256;

template <int _H> struct fp8_kernel5_1_factory {
  using globals_t =
      kernel5_1::globals<K5_M, K5_I, _H, K5_BM, K5_BN, K5_NUM_CONSUMER_WARPS,
                         K5_NUM_PRODUCER_WARPS, K5_NUM_STAGES,
                         K5_KERNEL_SMEM_SIZE, K5_NUM_EXPERTS,
                         K5_EXPERTS_PER_TOKEN, K5_SUPER_M>;

  static size_t size() { return sizeof(globals_t); }

  static void build(
      void *out, kittens::py::TKParallelTensor &in_tokens,
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
      int *num_recv_tokens, int dp_rank, int rank, int dp_size,
      int cur_dp_group, int num_dp_groups, int num_comm_sms, int num_comp_sms) {

    globals_t G{
        .in_tokens = kittens::py::parallel_tensor_to_pgl<
            typename globals_t::in_tokens_layout>(in_tokens),
        .in_tokens_scales = kittens::py::parallel_tensor_to_pgl<
            typename globals_t::in_tokens_scales_layout>(in_tokens_scales),
        .expert_x_tokens = kittens::py::tensor_to_gl<
            typename globals_t::expert_x_tokens_layout>(
            stable_to_aten(expert_x_tokens)),
        .expert_x_tokens_scale = kittens::py::tensor_to_gl<
            typename globals_t::expert_x_tokens_scale_layout>(
            stable_to_aten(expert_x_tokens_scale)),
        .comm_comp_barrier = kittens::py::tensor_to_gl<
            typename globals_t::comm_comp_barrier_layout>(
            stable_to_aten(comm_comp_barrier)),
        .gate = kittens::py::tensor_to_gl<typename globals_t::gate_layout>(
            stable_to_aten(gate)),
        .up = kittens::py::tensor_to_gl<typename globals_t::up_layout>(
            stable_to_aten(up)),
        .C = kittens::py::tensor_to_gl<typename globals_t::c_layout>(
            stable_to_aten(C)),
        .scale_gate =
            kittens::py::tensor_to_gl<typename globals_t::scale_gate_layout>(
                stable_to_aten(scale_gate)),
        .scale_up =
            kittens::py::tensor_to_gl<typename globals_t::scale_up_layout>(
                stable_to_aten(scale_up)),
        .out_scales =
            kittens::py::tensor_to_gl<typename globals_t::out_scales_layout>(
                stable_to_aten(out_scales)),
        .indices =
            kittens::py::tensor_to_gl<typename globals_t::indices_layout>(
                stable_to_aten(indices)),
        .global_num_routed = kittens::py::parallel_tensor_to_pgl<
            typename globals_t::global_num_routed_layout>(global_num_routed),
        .expert_to_token_map = kittens::py::parallel_tensor_to_pgl<
            typename globals_t::expert_to_token_map_layout>(
            expert_to_token_map),
        .padded_expert_counts = kittens::py::tensor_to_gl<
            typename globals_t::padded_expert_counts_layout>(
            stable_to_aten(padded_expert_counts)),
        .src_token_idx =
            kittens::py::tensor_to_gl<typename globals_t::src_token_idx_layout>(
                stable_to_aten(src_token_idx)),
        .src_dev_idx =
            kittens::py::tensor_to_gl<typename globals_t::src_dev_idx_layout>(
                stable_to_aten(src_dev_idx)),
        .barrier = kittens::py::parallel_tensor_to_pgl<
            typename globals_t::barrier_layout>(barrier),
        .num_tokens = num_tokens,
        .num_recv_tokens = num_recv_tokens,
        .dp_rank = dp_rank,
        .rank = rank,
        .dp_size = dp_size,
        .cur_dp_group = cur_dp_group,
        .num_dp_groups = num_dp_groups,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = num_comp_sms};

    memcpy(out, &G, sizeof(G));
  }
};

// Dispatch macros for kernel5_1 — keyed on H (hidden size)
#define TK_KERNEL5_1_SIZE_CASE(h)                                              \
  if (H_ == h)                                                                 \
    return fp8_kernel5_1_factory<h>::size();

#define TK_KERNEL5_1_BUILD_CASE(h)                                             \
  if (H_ == h) {                                                               \
    fp8_kernel5_1_factory<h>::build(                                           \
        out, in_tokens, in_tokens_scales, expert_x_tokens,                     \
        expert_x_tokens_scale, comm_comp_barrier, gate, up, C, scale_gate,     \
        scale_up, out_scales, indices, global_num_routed, expert_to_token_map, \
        padded_expert_counts, src_token_idx, src_dev_idx, barrier, num_tokens, \
        num_recv_tokens, dp_rank, rank, dp_size, cur_dp_group, num_dp_groups,  \
        num_comm_sms, num_comp_sms);                                           \
    return;                                                                    \
  }

// Common hidden sizes in transformer models
#define TK_ALL_KERNEL5_1_H_CONFIGS(MACRO)                                      \
  MACRO(512)                                                                   \
  MACRO(1024)                                                                  \
  MACRO(2048)                                                                  \
  MACRO(3072)                                                                  \
  MACRO(4096)                                                                  \
  MACRO(5120)                                                                  \
  MACRO(6144)                                                                  \
  MACRO(7168)                                                                  \
  MACRO(8192)                                                                  \
  HOST_ERROR("Unsupported Hidden Dimension Type");

size_t tk_kernel5_1_globals_size(int H_) {
  TK_ALL_KERNEL5_1_H_CONFIGS(TK_KERNEL5_1_SIZE_CASE)
  fprintf(stderr,
          "tk_kernel5_1_globals_size: unsupported H=%d (add to "
          "TK_ALL_KERNEL5_1_H_CONFIGS)\n",
          H_);
  abort();
}

void tk_build_kernel5_1_globals(
    int H_, void *out, kittens::py::TKParallelTensor &in_tokens,
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
    int num_dp_groups, int num_comm_sms, int num_comp_sms) {
  TK_ALL_KERNEL5_1_H_CONFIGS(TK_KERNEL5_1_BUILD_CASE)
  fprintf(stderr,
          "tk_build_kernel5_1_globals: unsupported H=%d (add to "
          "TK_ALL_KERNEL5_1_H_CONFIGS)\n",
          H_);
  abort();
}
