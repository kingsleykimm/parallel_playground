/*
 * BF16 GEMM Testing Harness for sm90_bf16_gemm kernels
 *
 * Tests the BF16 GEMM kernel for:
 * - Normal: standard A @ B^T GEMM
 * - Batched: batched GEMM where each batch is a group
 * - MGroupedContiguous: MoE-style grouped GEMM with contiguous token layout
 * - MGroupedMasked: MoE-style grouped GEMM with masked layout (for CUDA graphs)
 *
 * Usage:
 *   ./test_bf16_gemm --type <normal|batched|contiguous|masked> [options]
 *
 * Examples:
 *   ./test_bf16_gemm --type normal
 *   ./test_bf16_gemm --type normal --m 512 --n 1024 --k 256
 *   ./test_bf16_gemm --type batched --m 256 --n 512 --k 128 --groups 4
 *   ./test_bf16_gemm --type contiguous --m 1024 --n 512 --k 256 --groups 8
 *   ./test_bf16_gemm --type masked --m 256 --n 512 --k 128 --groups 4
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>

#include "test_utils.h"
#include <apis/moe_forward.hpp>
#include <jit/compiler.hpp>
#include <moe_cuda/dtype.h>
#include <runtime/tensor.h>
#include <moe_cuda/types.h>
#include <runtime/utils.h>

using test_utils::check_tensor_close;
using test_utils::inspect_tensor;
using test_utils::shape_to_string;

// ============================================================================
// Test shape definitions
// ============================================================================

struct TestShape {
  int64_t M, N, K;
  int num_groups;
  const char *description;
};

// Normal GEMM shapes (no grouping)
static const std::vector<TestShape> normal_shapes = {
    {128, 128, 128, 1, "Small square (128x128x128)"},
    {256, 256, 128, 1, "Medium (256x256x128)"},
    {256, 256, 256, 1, "Medium square (256x256x256)"},
    {512, 512, 512, 1, "Large square (512x512x512)"},
    {128, 4096, 1024, 1, "LLM-like narrow M (128x4096x1024)"},
    {256, 4096, 1024, 1, "LLM-like medium M (256x4096x1024)"},
    {512, 4096, 1024, 1, "LLM-like wide M (512x4096x1024)"},
    {128, 8192, 1024, 1, "Wide N (128x8192x1024)"},
    {256, 1024, 256, 1, "Small K (256x1024x256)"},
    {256, 1024, 512, 1, "Medium K (256x1024x512)"},
    {256, 1024, 1024, 1, "Large K (256x1024x1024)"},
    // Ragged M shapes (non-power-of-2)
    {96, 256, 128, 1, "Ragged M=96"},
    {192, 512, 256, 1, "Ragged M=192"},
    {320, 512, 256, 1, "Ragged M=320"},
    {384, 1024, 256, 1, "Ragged M=384"},
    {640, 512, 512, 1, "Ragged M=640"},
    {768, 1024, 512, 1, "Ragged M=768"},
    {160, 4096, 1024, 1, "Ragged LLM-like M=160"},
};

// Batched GEMM shapes (each batch is a separate GEMM)
static const std::vector<TestShape> batched_shapes = {
    {128, 256, 128, 2, "2 batches (128x256x128)"},
    {256, 512, 256, 4, "4 batches (256x512x256)"},
    {128, 1024, 256, 8, "8 batches (128x1024x256)"},
    {256, 512, 512, 4, "4 batches large K (256x512x512)"},
    {192, 512, 256, 2, "2 batches ragged M=192"},
    {320, 256, 128, 4, "4 batches ragged M=320"},
};

// Grouped contiguous GEMM shapes (MoE with contiguous token layout)
static const std::vector<TestShape> contiguous_shapes = {
    {512, 512, 128, 4, "4 groups (512x512x128)"},
    {2048, 512, 256, 8, "8 groups (2048x512x256)"},
    {4096, 1024, 512, 16, "16 groups (4096x1024x512)"},
    {1536, 512, 256, 4, "4 groups ragged total_M=1536"},
    {2560, 1024, 256, 8, "8 groups ragged total_M=2560"},
    {768, 256, 128, 4, "4 groups small ragged total_M=768"},
};

// Grouped masked GEMM shapes (MoE with fixed-size masked layout for CUDA graphs)
static const std::vector<TestShape> masked_shapes = {
    {256, 512, 256, 4, "4 groups masked (256x512x256)"},
    {256, 512, 256, 8, "8 groups masked (256x512x256)"},
    {512, 1024, 512, 4, "4 groups large masked (512x1024x512)"},
    {192, 256, 128, 4, "4 groups ragged masked M=192"},
    {384, 512, 256, 2, "2 groups ragged masked M=384"},
    {320, 1024, 256, 8, "8 groups ragged masked M=320"},
};

// ============================================================================
// Test configuration
// ============================================================================

struct TestConfig {
  GemmType gemm_type = GemmType::Normal;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int num_groups = 1;
  bool run_all_shapes = true;
  bool verbose = false;
  float atol = 1e-2f; // BF16 precision
  float rtol = 1e-2f;
  bool test_fp32_output = true;  // also test FP32 output path
};

// ============================================================================
// Argument parsing
// ============================================================================

void print_usage(const char *program_name) {
  std::cout << "BF16 GEMM Testing Harness for sm90_bf16_gemm kernels\n\n";
  std::cout << "Usage: " << program_name << " --type <normal|batched|contiguous|masked> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --type <type>     Test type: normal, batched, contiguous, or masked (required)\n";
  std::cout << "  --m <M>           M dimension (default: run all default shapes)\n";
  std::cout << "  --n <N>           N dimension\n";
  std::cout << "  --k <K>           K dimension\n";
  std::cout << "  --groups <G>      Number of groups/batches (default: 1)\n";
  std::cout << "  --atol <tol>      Absolute tolerance (default: 0.01)\n";
  std::cout << "  --rtol <tol>      Relative tolerance (default: 0.01)\n";
  std::cout << "  --verbose         Print detailed output\n";
  std::cout << "  --help            Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << program_name << " --type normal\n";
  std::cout << "  " << program_name << " --type normal --m 512 --n 1024 --k 256\n";
  std::cout << "  " << program_name << " --type batched --m 256 --n 512 --k 128 --groups 4\n";
  std::cout << "  " << program_name << " --type contiguous --m 2048 --n 512 --k 256 --groups 8\n";
  std::cout << "  " << program_name << " --type masked --m 256 --n 512 --k 128 --groups 4\n";
}

bool parse_args(int argc, char **argv, TestConfig &config) {
  bool type_specified = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--type" && i + 1 < argc) {
      std::string type_str = argv[++i];
      if (type_str == "normal") {
        config.gemm_type = GemmType::Normal;
      } else if (type_str == "batched") {
        config.gemm_type = GemmType::Batched;
      } else if (type_str == "contiguous") {
        config.gemm_type = GemmType::MGroupedContiguous;
      } else if (type_str == "masked") {
        config.gemm_type = GemmType::MGroupedMasked;
      } else {
        std::cerr << "Error: Invalid type '" << type_str << "'. Use normal, batched, contiguous, or masked.\n";
        return false;
      }
      type_specified = true;
    } else if (arg == "--m" && i + 1 < argc) {
      config.M = std::stoll(argv[++i]);
      config.run_all_shapes = false;
    } else if (arg == "--n" && i + 1 < argc) {
      config.N = std::stoll(argv[++i]);
      config.run_all_shapes = false;
    } else if (arg == "--k" && i + 1 < argc) {
      config.K = std::stoll(argv[++i]);
      config.run_all_shapes = false;
    } else if (arg == "--groups" && i + 1 < argc) {
      config.num_groups = std::stoi(argv[++i]);
    } else if (arg == "--atol" && i + 1 < argc) {
      config.atol = std::stof(argv[++i]);
    } else if (arg == "--rtol" && i + 1 < argc) {
      config.rtol = std::stof(argv[++i]);
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return false;
    }
  }

  if (!type_specified) {
    std::cerr << "Error: --type is required\n";
    print_usage(argv[0]);
    return false;
  }

  if (!config.run_all_shapes) {
    if (config.M <= 0 || config.N <= 0 || config.K <= 0) {
      std::cerr << "Error: M, N, K must all be specified and positive\n";
      return false;
    }
  }

  return true;
}

const char *gemm_type_to_string(GemmType type) {
  switch (type) {
  case GemmType::Normal:
    return "Normal";
  case GemmType::Batched:
    return "Batched";
  case GemmType::MGroupedMasked:
    return "MGroupedMasked";
  case GemmType::MGroupedContiguous:
    return "MGroupedContiguous";
  default:
    return "Unknown";
  }
}

// ============================================================================
// Test functions for each GEMM type
// ============================================================================

bool test_bf16_normal(int64_t M, int64_t N, int64_t K, float atol, float rtol, bool verbose,
                      c10::ScalarType output_dtype = c10::ScalarType::BFloat16) {
  const char *dtype_str = (output_dtype == c10::ScalarType::Float) ? "FP32" : "BF16";
  std::cout << "\n=== Testing BF16 Normal GEMM (output=" << dtype_str << "): M=" << M << ", N=" << N << ", K=" << K
            << " ===\n";

  try {
    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    bool fp32_output = (output_dtype == c10::ScalarType::Float);
    torch::TensorOptions d_options =
        fp32_output ? torch::TensorOptions().dtype(torch::kFloat32).device(device) : bf16_options;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float init_scale = 1.0f / sqrt(K);
    torch::Tensor A_bf16 = torch::randn({M, K}, bf16_options) * init_scale;
    torch::Tensor B_bf16 = torch::randn({N, K}, bf16_options) * init_scale;

    torch::Tensor reference = torch::mm(A_bf16, B_bf16.t());
    if (fp32_output)
      reference = reference.to(torch::kFloat32);

    if (verbose) {
      std::cout << "A_bf16 shape: " << shape_to_string(A_bf16.sizes().vec()) << "\n";
      std::cout << "B_bf16 shape: " << shape_to_string(B_bf16.sizes().vec()) << "\n";
    }

    at::Tensor A_custom = (A_bf16);

    at::Tensor B_custom = (B_bf16);

    torch::Tensor D_torch = torch::empty({M, N}, d_options);
    at::Tensor D_custom = (D_torch);

    std::optional<at::Tensor> C_opt = std::nullopt;
    GemmType type = GemmType::Normal;
    std::string compiled_dims = "";

    auto start = std::chrono::high_resolution_clock::now();
    moe_cuda::bf16_gemm(A_custom, B_custom, C_opt, D_custom, type, compiled_dims, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " us\n";

    bool passed = fp32_output ? check_tensor_close<float>(reference, D_custom, atol, rtol)
                              : check_tensor_close<__nv_bfloat16>(reference, D_custom, atol, rtol);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed;

  } catch (const std::exception &e) {
    std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
    return false;
  }
}

bool test_bf16_batched(int64_t M, int64_t N, int64_t K, int num_groups, float atol, float rtol, bool verbose,
                       c10::ScalarType output_dtype = c10::ScalarType::BFloat16) {
  const char *dtype_str = (output_dtype == c10::ScalarType::Float) ? "FP32" : "BF16";
  std::cout << "\n=== Testing BF16 Batched GEMM (output=" << dtype_str << "): M=" << M << ", N=" << N << ", K=" << K
            << ", Batches=" << num_groups << " ===\n";

  try {
    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    bool fp32_output = (output_dtype == c10::ScalarType::Float);
    torch::TensorOptions d_options =
        fp32_output ? torch::TensorOptions().dtype(torch::kFloat32).device(device) : bf16_options;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float init_scale = 1.0f / sqrt(K);
    torch::Tensor A_bf16 = torch::randn({num_groups, M, K}, bf16_options) * init_scale;
    torch::Tensor B_bf16 = torch::randn({num_groups, N, K}, bf16_options) * init_scale;

    torch::Tensor reference = torch::einsum("gmk,gnk->gmn", {A_bf16, B_bf16});
    if (fp32_output)
      reference = reference.to(torch::kFloat32);

    if (verbose) {
      std::cout << "A_bf16 shape: " << shape_to_string(A_bf16.sizes().vec()) << "\n";
      std::cout << "B_bf16 shape: " << shape_to_string(B_bf16.sizes().vec()) << "\n";
    }

    at::Tensor A_custom = (A_bf16);

    at::Tensor B_custom = (B_bf16);

    torch::Tensor D_torch = torch::empty({num_groups, M, N}, d_options);
    at::Tensor D_custom = (D_torch);

    std::optional<at::Tensor> C_opt = std::nullopt;
    GemmType type = GemmType::Batched;
    std::string compiled_dims = "";

    auto start = std::chrono::high_resolution_clock::now();
    moe_cuda::bf16_gemm(A_custom, B_custom, C_opt, D_custom, type, compiled_dims, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " us\n";

    bool passed = fp32_output ? check_tensor_close<float>(reference, D_custom, atol, rtol)
                              : check_tensor_close<__nv_bfloat16>(reference, D_custom, atol, rtol);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed;

  } catch (const std::exception &e) {
    std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
    return false;
  }
}

bool test_bf16_grouped_contiguous(int64_t M, int64_t N, int64_t K, int num_groups, float atol, float rtol, bool verbose,
                                  c10::ScalarType output_dtype = c10::ScalarType::BFloat16) {
  const char *dtype_str = (output_dtype == c10::ScalarType::Float) ? "FP32" : "BF16";
  std::cout << "\n=== Testing BF16 Grouped Contiguous GEMM (output=" << dtype_str << "): M=" << M << ", N=" << N
            << ", K=" << K << ", Groups=" << num_groups << " ===\n";

  try {
    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    bool fp32_output = (output_dtype == c10::ScalarType::Float);
    torch::TensorOptions d_options =
        fp32_output ? torch::TensorOptions().dtype(torch::kFloat32).device(device) : bf16_options;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t per_group_M = M / num_groups;

    float init_scale = 1.0f / sqrt(K);
    int total_M;
    auto tup = test_utils::generate_contiguous_grouped_layout(total_M, num_groups, per_group_M, device);

    torch::Tensor grouped_layout_tensor = std::get<0>(tup);
    std::vector<size_t> actual_ms = std::get<1>(tup);
    std::vector<size_t> aligned_ms = std::get<2>(tup);

    torch::Tensor A_bf16 = torch::randn({total_M, K}, bf16_options) * init_scale;
    torch::Tensor B_bf16 = torch::randn({num_groups, N, K}, bf16_options) * init_scale;

    torch::Tensor reference = torch::zeros({total_M, N}, bf16_options);
    torch::Tensor layout_cpu = grouped_layout_tensor.cpu();
    auto layout_acc = layout_cpu.accessor<int32_t, 1>();

    int row_start = 0;
    for (int g = 0; g < num_groups; g++) {
      auto valid_slice = A_bf16.index({torch::indexing::Slice(row_start, row_start + aligned_ms[g])});
      auto group_slice = B_bf16[g];
      auto ref_slice = torch::mm(valid_slice, group_slice.t());
      reference.index_put_({torch::indexing::Slice(row_start, row_start + aligned_ms[g])}, ref_slice);
      row_start += aligned_ms[g];
    }

    if (fp32_output)
      reference = reference.to(torch::kFloat32);

    if (verbose) {
      std::cout << "A_bf16 shape: " << shape_to_string(A_bf16.sizes().vec()) << "\n";
      std::cout << "B_bf16 shape: " << shape_to_string(B_bf16.sizes().vec()) << "\n";
      std::cout << "Total M: " << total_M << "\n";
    }

    at::Tensor A_custom = (A_bf16);

    at::Tensor B_custom = (B_bf16);

    torch::Tensor D_torch = torch::empty({total_M, N}, d_options);
    at::Tensor D_custom = (D_torch);

    int *grouped_layout = grouped_layout_tensor.data_ptr<int>();

    std::optional<at::Tensor> C_opt = std::nullopt;
    GemmType type = GemmType::MGroupedContiguous;
    std::string compiled_dims = "";

    auto start = std::chrono::high_resolution_clock::now();
    moe_cuda::bf16_gemm(A_custom, B_custom, C_opt, D_custom, type, compiled_dims, grouped_layout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " us\n";

    std::vector<torch::Tensor> ref_list;
    std::vector<torch::Tensor> D_list;
    row_start = 0;
    for (int g = 0; g < num_groups; g++) {
      uint32_t num_rows = actual_ms[g];
      auto ref_group = reference.index({torch::indexing::Slice(row_start, row_start + num_rows)});
      auto out_group = D_torch.index({torch::indexing::Slice(row_start, row_start + num_rows)});
      ref_list.push_back(ref_group);
      D_list.push_back(out_group);
      row_start += aligned_ms[g];
    }

    auto ref = torch::cat(ref_list, 0);
    auto out = torch::cat(D_list, 0);

    bool passed = fp32_output ? check_tensor_close(ref, out, atol, rtol)
                              : check_tensor_close(ref, out, atol, rtol);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed;

  } catch (const std::exception &e) {
    std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
    return false;
  }
}

bool test_bf16_grouped_masked(int64_t M, int64_t N, int64_t K, int num_groups, float atol, float rtol, bool verbose,
                              c10::ScalarType output_dtype = c10::ScalarType::BFloat16) {
  const char *dtype_str = (output_dtype == c10::ScalarType::Float) ? "FP32" : "BF16";
  std::cout << "\n=== Testing BF16 Grouped Masked GEMM (output=" << dtype_str << "): M=" << M << ", N=" << N
            << ", K=" << K << ", Groups=" << num_groups << " ===\n";

  try {
    torch::Device device = torch::kCUDA;
    torch::TensorOptions bf16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    bool fp32_output = (output_dtype == c10::ScalarType::Float);
    torch::TensorOptions d_options =
        fp32_output ? torch::TensorOptions().dtype(torch::kFloat32).device(device) : bf16_options;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t max_M = M;

    float init_scale = 1.0f / sqrt(K);
    torch::Tensor A_bf16 = torch::randn({num_groups, max_M, K}, bf16_options) * init_scale;
    torch::Tensor B_bf16 = torch::randn({num_groups, N, K}, bf16_options) * init_scale;

    torch::Tensor grouped_layout_tensor = test_utils::generate_masked_grouped_layout(max_M, num_groups, device);
    auto grouped_layout_cpu_tensor = grouped_layout_tensor.cpu();
    int *grouped_layout = grouped_layout_tensor.data_ptr<int>();
    auto layout_acc = grouped_layout_cpu_tensor.accessor<int32_t, 1>();

    torch::Tensor reference = torch::einsum("gmk,gnk->gmn", {A_bf16, B_bf16});
    if (fp32_output)
      reference = reference.to(torch::kFloat32);

    if (verbose) {
      std::cout << "A_bf16 shape: " << shape_to_string(A_bf16.sizes().vec()) << "\n";
      std::cout << "B_bf16 shape: " << shape_to_string(B_bf16.sizes().vec()) << "\n";
      std::cout << "grouped_layout shape: " << shape_to_string(grouped_layout_tensor.sizes().vec()) << "\n";
    }

    at::Tensor A_custom = (A_bf16);

    at::Tensor B_custom = (B_bf16);

    torch::Tensor D_torch = torch::empty({num_groups, max_M, N}, d_options);
    at::Tensor D_custom = (D_torch);

    std::optional<at::Tensor> C_opt = std::nullopt;
    GemmType type = GemmType::MGroupedMasked;
    std::string compiled_dims = "";

    
    auto start = std::chrono::high_resolution_clock::now();
    moe_cuda::bf16_gemm(A_custom, B_custom, C_opt, D_custom, type, compiled_dims, grouped_layout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " us\n";
    
    test_utils::inspect_tensor(D_custom, 10);
    test_utils::inspect_tensor(reference, 10);
    std::vector<torch::Tensor> ref_list;
    std::vector<torch::Tensor> D_list;
    for (int g = 0; g < num_groups; g++) {
      uint32_t num_rows = layout_acc[g];
      auto ref_group = reference.index({g, torch::indexing::Slice(0, num_rows)});
      auto out_group = D_torch.index({g, torch::indexing::Slice(0, num_rows)});
      ref_list.push_back(ref_group);
      D_list.push_back(out_group);
    }

    auto ref = torch::cat(ref_list, 0);
    auto out = torch::cat(D_list, 0);

    if (verbose) {
      std::cout << "Final ref shape: " << shape_to_string(ref.sizes().vec()) << "\n";
      std::cout << "Final out shape: " << shape_to_string(out.sizes().vec()) << "\n";
    }

    at::Tensor out_custom = (out);

    bool passed = fp32_output ? check_tensor_close<float>(ref, out_custom, atol, rtol)
                              : check_tensor_close<__nv_bfloat16>(ref, out_custom, atol, rtol);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed;

  } catch (const std::exception &e) {
    std::cerr << "\033[0;31mError: " << e.what() << "\033[0m\n";
    return false;
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  TestConfig config;

  if (!parse_args(argc, argv, config)) {
    return 1;
  }

  if (!torch::cuda::is_available()) {
    std::cerr << "Error: CUDA is not available\n";
    return 1;
  }

  std::cout << "============================================\n";
  std::cout << "BF16 GEMM Testing Harness\n";
  std::cout << "Test Type: " << gemm_type_to_string(config.gemm_type) << "\n";
  std::cout << "============================================\n";

  int passed = 0, failed = 0;
  Compiler::init_static_vars(get_env<std::string>("LIBRARY_ROOT_PATH", ""), get_env<std::string>("CUDA_HOME_PATH"));

  auto run_test = [&](const TestShape &shape, c10::ScalarType output_dtype) {
    bool result = false;

    switch (config.gemm_type) {
    case GemmType::Normal:
      result = test_bf16_normal(shape.M, shape.N, shape.K, config.atol, config.rtol, config.verbose, output_dtype);
      break;
    case GemmType::Batched:
      result = test_bf16_batched(shape.M, shape.N, shape.K, shape.num_groups, config.atol, config.rtol, config.verbose,
                                 output_dtype);
      break;
    case GemmType::MGroupedContiguous:
      result = test_bf16_grouped_contiguous(shape.M, shape.N, shape.K, shape.num_groups, config.atol, config.rtol,
                                            config.verbose, output_dtype);
      break;
    case GemmType::MGroupedMasked:
      result = test_bf16_grouped_masked(shape.M, shape.N, shape.K, shape.num_groups, config.atol, config.rtol,
                                        config.verbose, output_dtype);
      break;
    }

    if (result) {
      passed++;
    } else {
      failed++;
    }
  };

  if (config.run_all_shapes) {
    const std::vector<TestShape> *shapes = nullptr;

    switch (config.gemm_type) {
    case GemmType::Normal:
      shapes = &normal_shapes;
      break;
    case GemmType::Batched:
      shapes = &batched_shapes;
      break;
    case GemmType::MGroupedContiguous:
      shapes = &contiguous_shapes;
      break;
    case GemmType::MGroupedMasked:
      shapes = &masked_shapes;
      break;
    }

    for (const auto &shape : *shapes) {
      run_test(shape, c10::ScalarType::BFloat16);
      if (config.test_fp32_output) {
        run_test(shape, c10::ScalarType::Float);
      }
    }
  } else {
    TestShape shape = {config.M, config.N, config.K, config.num_groups, "Custom shape"};
    run_test(shape, c10::ScalarType::BFloat16);
    if (config.test_fp32_output) {
      run_test(shape, c10::ScalarType::Float);
    }
  }

  std::cout << "\n============================================\n";
  std::cout << "Test Summary\n";
  std::cout << "============================================\n";
  std::cout << "Passed: " << passed << "\n";
  std::cout << "Failed: " << failed << "\n";

  if (failed > 0) {
    std::cout << "\033[0;31mSome tests FAILED\033[0m\n";
    return 1;
  } else {
    std::cout << "\033[0;32mAll tests PASSED\033[0m\n";
    return 0;
  }
}
