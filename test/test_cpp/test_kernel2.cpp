#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "test_utils.h"
#include <jit/compiler.hpp>
#include <jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp>
#include <kernels/internal_api.hpp>
#include <moe_cuda/types.h>
#include <runtime/utils.h>

using test_utils::calc_diff;
using test_utils::shape_to_string;

constexpr int MAX_M = 2048;

// Output layout produced by the kernel:
//
//   Contiguous:
//     grouped_layout: (total_M,)  — group index per row, -1 for padding
//     D shape:        (total_M, N)
//     row r with layout[r]=g writes A[r] @ B[g].T to D[r, :]
//     padding rows (-1) produce no output
//
//   Masked:
//     grouped_layout: (num_groups,)  — actual M count per group
//     D shape:        (groups*max_M, N)
//     group g writes to D[g*max_M : g*max_M+actual_m[g], 0:N]

struct TestShape {
  int64_t M; // expected_per_group_M (contiguous) or max_M (masked)
  int64_t N; // N per group — must be divisible by 128
  int64_t K; // must be divisible by 128
  int groups;
  int64_t expected_m_per_group; // masked only; contiguous keeps this equal to M
  const char *desc;
};

static const std::vector<TestShape> contiguous_shapes = {
    {256, 512, 128, 1, 256, "contig-4g"},
    {256, 1024, 512, 8, 256, "contig-8g"},
    {128, 512, 256, 2, 128, "contig-2g"},
    {512, 1024, 512, 4, 512, "contig-4g-large"},
};

static const std::vector<TestShape> masked_shapes = {
    {4096, 4096, 7168, 1, 1024, "masked-1g-4096x7168"},
    {4096, 7168, 2048, 1, 1024, "masked-1g-7168x2048"},
    {4096, 4096, 7168, 2, 512, "masked-2g-4096x7168"},
    {4096, 7168, 2048, 2, 512, "masked-2g-7168x2048"},
    {4096, 4096, 7168, 4, 256, "masked-4g-4096x7168"},
    {4096, 7168, 2048, 4, 256, "masked-4g-7168x2048"},
};

struct Config {
  enum class Impl {
    Kernel2,
    Ref,
  };
  enum class LayoutMode {
    Dense,
    Padded,
  };

  GemmType type = GemmType::MGroupedContiguous;
  bool all_shapes = true;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int groups = 1;
  Impl impl = Impl::Kernel2;
  LayoutMode layout = LayoutMode::Padded;
  uint64_t seed = 1234;
  bool single_shape = false;
  bool verbose = false;
};

static const char *type_to_string(GemmType t) {
  switch (t) {
  case GemmType::MGroupedContiguous:
    return "contiguous";
  case GemmType::MGroupedMasked:
    return "masked";
  default:
    return "unknown";
  }
}

static const char *dtype_str(c10::ScalarType dt) {
  switch (dt) {
  case c10::ScalarType::Float:
    return "float";
  case c10::ScalarType::BFloat16:
    return "bf16";
  default:
    return "unknown";
  }
}

static const char *impl_to_string(Config::Impl impl) {
  switch (impl) {
  case Config::Impl::Kernel2:
    return "kernel2";
  case Config::Impl::Ref:
    return "ref";
  default:
    return "unknown";
  }
}

static const char *layout_to_string(Config::LayoutMode layout) {
  switch (layout) {
  case Config::LayoutMode::Dense:
    return "dense";
  case Config::LayoutMode::Padded:
    return "padded";
  default:
    return "unknown";
  }
}

static bool parse_args(int argc, char **argv, Config &cfg) {
  bool type_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "--type") && i + 1 < argc) {
      std::string t = argv[++i];
      if (t == "contiguous")
        cfg.type = GemmType::MGroupedContiguous;
      else if (t == "masked")
        cfg.type = GemmType::MGroupedMasked;
      else
        return false;
      type_set = true;
    } else if ((a == "--impl") && i + 1 < argc) {
      std::string impl = argv[++i];
      if (impl == "kernel2")
        cfg.impl = Config::Impl::Kernel2;
      else if (impl == "ref")
        cfg.impl = Config::Impl::Ref;
      else
        return false;
    } else if ((a == "--layout") && i + 1 < argc) {
      std::string layout = argv[++i];
      if (layout == "dense")
        cfg.layout = Config::LayoutMode::Dense;
      else if (layout == "padded")
        cfg.layout = Config::LayoutMode::Padded;
      else
        return false;
    } else if ((a == "--m") && i + 1 < argc) {
      cfg.M = std::stoll(argv[++i]);
      cfg.all_shapes = false;
    } else if ((a == "--n") && i + 1 < argc) {
      cfg.N = std::stoll(argv[++i]);
      cfg.all_shapes = false;
    } else if ((a == "--k") && i + 1 < argc) {
      cfg.K = std::stoll(argv[++i]);
      cfg.all_shapes = false;
    } else if ((a == "--groups") && i + 1 < argc) {
      cfg.groups = std::stoi(argv[++i]);
    } else if ((a == "--seed") && i + 1 < argc) {
      cfg.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
    } else if (a == "--single-shape") {
      cfg.single_shape = true;
      cfg.all_shapes = false;
    } else if (a == "--verbose") {
      cfg.verbose = true;
    } else {
      return false;
    }
  }
  return type_set;
}

// ========================= Contiguous =========================
//
// grouped_layout: (total_M,) int32, value = group index (0..groups-1) or -1
//   Rows are partitioned into [groups] sections; each section is
//   ceil(actual_m / 128)*128 rows, with actual_m valid rows followed by
//   -1 padding rows to the next 128-alignment boundary.
//
// D output: (total_M, N)
//   Rows belonging to group g write the routed expert output A[r] @ B[g].T.
//   Padding rows (-1) produce no output (reference is zero there).

static void launch_contiguous_impl(Config::Impl impl, at::Tensor &act,
                                   at::Tensor &act_scale, at::Tensor &weight,
                                   at::Tensor &weight_scale, at::Tensor &out,
                                   int *grouped_layout, cudaStream_t &stream) {
  if (impl == Config::Impl::Kernel2) {
    moe_cuda::kernels::fp8_grouped_gemm_nt(act, act_scale, weight, weight_scale,
                                           out, GemmType::MGroupedContiguous,
                                           grouped_layout, stream);
    return;
  }

  sm90_fp8_grouped_gemm_1d2d_contiguous_ref(
      act, weight, act_scale, weight_scale, out, grouped_layout, stream);
}

static bool run_contiguous(int64_t expected_per_group_M, int64_t N, int64_t K,
                           int groups, Config::Impl impl,
                           Config::LayoutMode layout_mode, uint64_t seed,
                           c10::ScalarType output_dtype, bool verbose) {
  auto dev = torch::Device(torch::kCUDA);
  auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  auto out_opts = torch::TensorOptions().dtype(output_dtype).device(dev);

  // Build grouped_layout: each group's M is randomly near expected_per_group_M
  // but aligned up to a multiple of 128. total_rows = sum of aligned_ms.
  int total_rows = 0;
  auto [layout_gpu, actual_ms, aligned_ms] =
      test_utils::generate_contiguous_grouped_layout(
          total_rows, groups, static_cast<int>(expected_per_group_M), dev,
          layout_mode == Config::LayoutMode::Padded, seed);
  int64_t total_M = total_rows;

  auto layout_cpu = layout_gpu.cpu();
  auto layout_acc = layout_cpu.accessor<int32_t, 1>();

  float s = 1.0f / std::sqrt(static_cast<float>(K));
  torch::Tensor A = torch::randn({total_M, K}, bf16) * s;
  // B is (groups, N, K); quantize flat as (groups*N, K) for 2D block scales
  torch::Tensor B = torch::randn({(int64_t)groups, N, K}, bf16) * s;

  // Reference: D[r, :] = A[r] @ B[g].T for rows where layout[r]=g
  auto ref = torch::zeros({total_M, N}, out_opts);
  {
    auto A_ref = A.to(output_dtype);
    auto B_ref = B.to(output_dtype);
    using S = torch::indexing::Slice;
    for (int g = 0; g < groups; ++g) {
      // collect valid row indices for this group
      std::vector<int64_t> rows_g;
      for (int64_t r = 0; r < total_M; ++r)
        if (layout_acc[r] == g)
          rows_g.push_back(r);
      if (rows_g.empty())
        continue;
      auto idx = torch::tensor(rows_g, torch::kLong).to(dev);
      auto out_block = torch::mm(A_ref.index_select(0, idx), B_ref[g].t());
      ref.index_put_({idx, S()}, out_block);
    }
  }

  // Quantize A (total_M, K) with 1D per-row block scales
  auto [A_fp8, sfa] = test_utils::quantize_fp8_1d_block(A, Major::K, dev);
  sfa = sfa.reshape({total_M, K / 128}).transpose(-1, -2).contiguous();
  // Quantize B: flatten to (groups*N, K), get 2D block scales (groups*N/128,
  // K/128)
  auto B_flat = B.reshape({(int64_t)groups * N, K});
  auto [B_fp8_flat, sfb] = test_utils::quantize_fp8_2d_block(B_flat, dev);
  torch::Tensor B_fp8 = B_fp8_flat.reshape({(int64_t)groups, N, K});

  torch::Tensor out = torch::zeros({total_M, N}, out_opts);
  at::Tensor A_t = A_fp8, B_t = B_fp8, sfa_t = sfa, sfb_t = sfb, out_t = out;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  auto t0 = std::chrono::high_resolution_clock::now();
  launch_contiguous_impl(impl, A_t, sfa_t, B_t, sfb_t, out_t,
                         layout_gpu.data_ptr<int>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (verbose) {
    test_utils::inspect_tensor(ref, 10);
    test_utils::inspect_tensor(out, 10);
  }

  double diff = 0.0;
  if (layout_mode == Config::LayoutMode::Padded) {
    std::vector<int64_t> valid_rows;
    valid_rows.reserve(total_M);
    for (int64_t r = 0; r < total_M; ++r) {
      if (layout_acc[r] >= 0)
        valid_rows.push_back(r);
    }
    if (valid_rows.empty()) {
      diff = 0.0;
    } else {
      auto idx =
          torch::tensor(valid_rows, torch::TensorOptions().dtype(torch::kLong))
              .to(dev);
      diff = calc_diff(ref.index_select(0, idx).to(torch::kFloat32),
                       out.index_select(0, idx).to(torch::kFloat32));
      test_utils::check_tensor_close(
          ref.index_select(0, idx).to(torch::kFloat32),
          out.index_select(0, idx).to(torch::kFloat32), 0.1, 0.1);
    }
  } else {
    diff = calc_diff(ref.to(torch::kFloat32), out.to(torch::kFloat32));
  }
  if (verbose) {

    std::cout << "contiguous total_M=" << total_M << " N=" << N << " K=" << K
              << " groups=" << groups << " impl=" << impl_to_string(impl)
              << " layout=" << layout_to_string(layout_mode) << " seed=" << seed
              << " out_dtype=" << dtype_str(output_dtype) << " kernel_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                     .count()
              << " diff=" << diff << "\n";
  }
  return diff < 0.001;
}

// ========================= Masked =========================
//
// grouped_layout: (num_groups,) int32, value = actual M for that group
//   generate_masked_grouped_layout fills this with varying counts ≤ max_M.
//
// D output: (groups*max_M, N)
//   group g writes to rows [g*max_M, g*max_M + actual_m[g]) and cols [0, N).
//   (C.cols = N_per_group in the factory since num_groups > 1.)

static bool run_masked(int64_t max_M, int64_t expected_per_group_M, int64_t N,
                       int64_t K, int groups, Config::Impl impl, uint64_t seed,
                       c10::ScalarType output_dtype, bool verbose) {
  HOST_ASSERT(impl == Config::Impl::Kernel2,
              "masked reference tracing is not implemented");
  auto dev = torch::Device(torch::kCUDA);
  auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  auto out_opts = torch::TensorOptions().dtype(output_dtype).device(dev);

  // grouped_layout[g] = actual number of valid rows for group g, sampled
  // around expected_per_group_M like DeepGEMM.
  torch::Tensor layout_gpu = test_utils::generate_masked_grouped_layout(
      max_M, expected_per_group_M, groups, dev, seed);
  auto layout_cpu = layout_gpu.cpu();
  auto layout_acc = layout_cpu.accessor<int32_t, 1>();

  int64_t total_M = (int64_t)groups * max_M;

  float s = 1.0f / std::sqrt(static_cast<float>(K));
  torch::Tensor A = torch::randn({(int64_t)groups, max_M, K}, bf16) * s;
  torch::Tensor B = torch::randn({(int64_t)groups, N, K}, bf16) * s;

  // Reference: D[g*max_M : g*max_M+actual_m[g], 0:N] = A[g,:actual_m] @ B[g].T
  torch::Tensor ref = torch::zeros({groups, max_M, N}, out_opts);
  ref = torch::einsum("gmk,gnk->gmn", {A, B})
            .to(output_dtype)
            .reshape({total_M, N});
  auto [A_fp8, sfa] = test_utils::quantize_fp8_1d_block(A, Major::K, dev);
  sfa = sfa.reshape({total_M, K / 128}).transpose(-1, -2).contiguous();

  // Quantize B: flatten (groups, N, K) → (groups*N, K)
  auto B_flat = B.reshape({(int64_t)groups * N, K});
  auto [B_fp8_flat, sfb] = test_utils::quantize_fp8_2d_block(B_flat, dev);
  torch::Tensor B_fp8 = B_fp8_flat.reshape({(int64_t)groups, N, K});

  // D: (total_M, N) — factory uses C.cols = total_N / num_groups = N
  torch::Tensor out = torch::zeros({total_M, N}, out_opts);
  at::Tensor A_t = A_fp8, B_t = B_fp8, sfa_t = sfa, sfb_t = sfb, out_t = out;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  auto t0 = std::chrono::high_resolution_clock::now();
  moe_cuda::kernels::fp8_grouped_gemm_nt(A_t, sfa_t, B_t, sfb_t, out_t,
                                         GemmType::MGroupedMasked,
                                         layout_gpu.data_ptr<int>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (verbose) {
    test_utils::inspect_tensor(ref, max_M * N - 10, max_M * N);
    test_utils::inspect_tensor(out, max_M * N - 10, max_M * N);
  }

  // Compare only the valid rows per group
  std::vector<torch::Tensor> ref_parts, out_parts;
  auto out_f = out.to(torch::kFloat32);
  for (int g = 0; g < groups; ++g) {
    int64_t am = layout_acc[g];
    using S = torch::indexing::Slice;
    ref_parts.push_back(ref.index({S(g * max_M, g * max_M + am)}));
    out_parts.push_back(out_f.index({S(g * max_M, g * max_M + am)}));
  }
  double diff = calc_diff(torch::cat(ref_parts, 0).to(torch::kFloat32),
                          torch::cat(out_parts, 0));

  if (verbose) {
    // Find index of maximum absolute diff across cat(ref_parts, 0) and
    // cat(out_parts, 0)
    auto ref_cat = torch::cat(ref_parts, 0).to(torch::kFloat32);
    auto out_cat = torch::cat(out_parts, 0);
    auto abs_diff = (ref_cat - out_cat).abs();
    std::cout << "masked max_M=" << max_M << " N=" << N << " K=" << K
              << " groups=" << groups
              << " out_dtype=" << dtype_str(output_dtype) << " kernel_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                     .count()
              << " diff=" << diff << "\n";
    test_utils::check_tensor_close(ref_cat, out_cat, 0.1, 0.1);
  }
  return diff < 0.001;
}

// ========================= main =========================

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    std::cerr << "Usage: " << argv[0] << " --type <contiguous|masked>"
              << " [--impl kernel2|ref --layout dense|padded"
              << " --m M --n N --k K --groups G --seed SEED"
              << " --single-shape --verbose]\n";
    return 1;
  }

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is unavailable\n";
    return 1;
  }

  Compiler::init_static_vars(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                             get_env<std::string>("CUDA_HOME_PATH", ""));

  int passed = 0, failed = 0;

  auto run_one = [&](const TestShape &s) {
    for (c10::ScalarType output_dtype : {c10::ScalarType::BFloat16}) {
      bool ok = false;
      if (cfg.type == GemmType::MGroupedContiguous)
        ok = run_contiguous(s.M, s.N, s.K, s.groups, cfg.impl, cfg.layout,
                            cfg.seed, output_dtype, cfg.verbose);
      else if (cfg.type == GemmType::MGroupedMasked)
        ok = run_masked(s.M, s.expected_m_per_group, s.N, s.K, s.groups,
                        cfg.impl, cfg.seed, output_dtype, cfg.verbose);

      if (ok)
        ++passed;
      else
        ++failed;

      // #region agent log
      {
        FILE *f = fopen("/u/bjb3az/.cursor/debug-95615d.log", "a");
        if (f) {
          fprintf(f,
                  "{\"sessionId\":\"95615d\",\"hypothesisId\":\"H1_test\","
                  "\"location\":\"test_kernel2.cpp:run_one\","
                  "\"message\":\"test_result\","
                  "\"data\":{\"passed\":%s,\"M\":%ld,\"N\":%ld,\"K\":%ld,"
                  "\"groups\":%d,\"desc\":\"%s\",\"type\":\"%s\"},"
                  "\"timestamp\":%ld}\n",
                  ok ? "true" : "false", (long)s.M, (long)s.N, (long)s.K,
                  s.groups, s.desc, type_to_string(cfg.type),
                  (long)time(nullptr));
          fclose(f);
        }
      }
      // #endregion

      std::cout << (ok ? "[PASSED] " : "[FAILED] ") << type_to_string(cfg.type)
                << " impl=" << impl_to_string(cfg.impl)
                << " layout=" << layout_to_string(cfg.layout) << " M=" << s.M
                << " N=" << s.N << " K=" << s.K << " G=" << s.groups
                << " C=" << dtype_str(output_dtype) << " (" << s.desc << ")\n";
    }
  };

  if (!cfg.all_shapes) {
    if (cfg.M == 0)
      cfg.M = 256;
    if (cfg.N == 0)
      cfg.N = 512;
    if (cfg.K == 0)
      cfg.K = 128;
    run_one({cfg.M, cfg.N, cfg.K, cfg.groups, cfg.M, "custom"});
  } else {
    const auto &shapes = (cfg.type == GemmType::MGroupedMasked)
                             ? masked_shapes
                             : contiguous_shapes;
    for (const auto &s : shapes)
      run_one(s);
  }

  std::cout << "Summary: passed=" << passed << " failed=" << failed << "\n";
  return failed == 0 ? 0 : 1;
}
