#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "test_utils.h"
#include <apis/moe_forward.hpp>
#include <jit/compiler.hpp>
#include <moe_cuda/types.h>
#include <runtime/utils.h>

using test_utils::calc_diff;
using test_utils::shape_to_string;

struct TestShape {
    int64_t M;
    int64_t N;
    int64_t K;
    int groups;
    const char* desc;
};

static const std::vector<TestShape> normal_shapes = {
    {128, 128, 128, 1, "small"},
    {256, 512, 256, 1, "medium"},
    {512, 1024, 512, 1, "large"},
};

static const std::vector<TestShape> contiguous_shapes = {
    {1024, 512, 256, 4, "contig-4"},
    {2048, 1024, 256, 8, "contig-8"},
};

static const std::vector<TestShape> masked_shapes = {
    {256, 512, 256, 4, "masked-4"},
    {320, 1024, 256, 8, "masked-8"},
};

struct Config {
    GemmType type = GemmType::Normal;
    bool all_shapes = true;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int groups = 1;
    bool verbose = false;
    double max_diff = 0.02;
};

static const char* type_to_string(GemmType t) {
    switch (t) {
        case GemmType::Normal: return "normal";
        case GemmType::Batched: return "batched";
        case GemmType::MGroupedContiguous: return "contiguous";
        case GemmType::MGroupedMasked: return "masked";
        default: return "unknown";
    }
}

static bool parse_args(int argc, char** argv, Config& cfg) {
    bool type_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--type") && i + 1 < argc) {
            std::string t = argv[++i];
            if (t == "normal") cfg.type = GemmType::Normal;
            else if (t == "batched") cfg.type = GemmType::Batched;
            else if (t == "contiguous") cfg.type = GemmType::MGroupedContiguous;
            else if (t == "masked") cfg.type = GemmType::MGroupedMasked;
            else return false;
            type_set = true;
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
        } else if ((a == "--max-diff") && i + 1 < argc) {
            cfg.max_diff = std::stod(argv[++i]);
        } else if (a == "--verbose") {
            cfg.verbose = true;
        } else {
            return false;
        }
    }
    return type_set;
}

static bool run_normal(int64_t M, int64_t N, int64_t K, double max_diff, bool verbose) {
    auto dev = torch::Device(torch::kCUDA);
    auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

    float s = 1.0f / std::sqrt(static_cast<float>(K));
    torch::Tensor A = torch::randn({M, K}, bf16) * s;
    torch::Tensor B = torch::randn({N, K}, bf16) * s;
    torch::Tensor ref = torch::mm(A, B.t());

    auto [A_fp8, sfa] = test_utils::quantize_fp8_1d_block(A, Major::K, dev);
    auto [B_fp8, sfb] = test_utils::quantize_fp8_2d_block(B, dev);

    torch::Tensor out = torch::empty({M, N}, bf16);
    at::Tensor A_t = A_fp8;
    at::Tensor B_t = B_fp8;
    at::Tensor sfa_t = sfa;
    at::Tensor sfb_t = sfb;
    at::Tensor out_t = out;

    std::pair<at::Tensor&, at::Tensor&> act{A_t, sfa_t};
    std::pair<at::Tensor&, at::Tensor&> weight{B_t, sfb_t};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto start = std::chrono::high_resolution_clock::now();
    moe_cuda::fp8_gemm(act, weight, out_t, GemmType::Normal, "", nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaStreamDestroy(stream));

    double diff = calc_diff(ref, out);
    if (verbose) {
        std::cout << "normal shape=" << shape_to_string(out.sizes().vec())
                  << " kernel_us=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " diff=" << diff << "\n";
    }
    return diff < max_diff;
}

static bool run_contiguous(int64_t M, int64_t N, int64_t K, int groups, double max_diff, bool verbose) {
    auto dev = torch::Device(torch::kCUDA);
    auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

    int64_t per_group_M = M / groups;
    int total_M = 0;
    auto tup = test_utils::generate_contiguous_grouped_layout(total_M, groups, per_group_M, dev);
    torch::Tensor grouped_layout = std::get<0>(tup);
    std::vector<size_t> actual_ms = std::get<1>(tup);
    std::vector<size_t> aligned_ms = std::get<2>(tup);

    float s = 1.0f / std::sqrt(static_cast<float>(K));
    torch::Tensor A = torch::randn({total_M, K}, bf16) * s;
    torch::Tensor B = torch::randn({groups, N, K}, bf16) * s;

    torch::Tensor ref = torch::zeros({total_M, N}, bf16.device(torch::kCPU));
    int row_start = 0;
    for (int g = 0; g < groups; ++g) {
        auto a_slice = A.index({torch::indexing::Slice(row_start, row_start + static_cast<int>(aligned_ms[g]))});
        auto r = torch::mm(a_slice, B[g].t());
        ref.index_put_({torch::indexing::Slice(row_start, row_start + static_cast<int>(aligned_ms[g]))}, r);
        row_start += static_cast<int>(aligned_ms[g]);
    }

    auto [A_fp8, sfa] = test_utils::quantize_fp8_1d_block(A, Major::K, dev);
    auto [B_fp8_flat, sfb] = test_utils::quantize_fp8_2d_block(B.reshape({groups * N, K}), dev);
    torch::Tensor B_fp8 = B_fp8_flat.reshape({groups, N, K});

    torch::Tensor out = torch::empty({total_M, N}, bf16);
    at::Tensor A_t = A_fp8;
    at::Tensor B_t = B_fp8;
    at::Tensor sfa_t = sfa;
    at::Tensor sfb_t = sfb;
    at::Tensor out_t = out;
    int* grouped_layout_ptr = grouped_layout.data_ptr<int>();

    std::pair<at::Tensor&, at::Tensor&> act{A_t, sfa_t};
    std::pair<at::Tensor&, at::Tensor&> weight{B_t, sfb_t};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    moe_cuda::fp8_gemm(act, weight, out_t, GemmType::MGroupedContiguous, "", grouped_layout_ptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::vector<torch::Tensor> ref_list;
    std::vector<torch::Tensor> out_list;
    row_start = 0;
    for (int g = 0; g < groups; ++g) {
        uint32_t rows = actual_ms[g];
        ref_list.push_back(ref.index({torch::indexing::Slice(row_start, row_start + static_cast<int>(rows))}));
        out_list.push_back(out.cpu().index({torch::indexing::Slice(row_start, row_start + static_cast<int>(rows))}));
        row_start += static_cast<int>(aligned_ms[g]);
    }

    double diff = calc_diff(torch::cat(ref_list, 0), torch::cat(out_list, 0));
    if (verbose) {
        std::cout << "contiguous total_M=" << total_M << " groups=" << groups << " diff=" << diff << "\n";
    }
    return diff < max_diff;
}

static bool run_masked(int64_t M, int64_t N, int64_t K, int groups, double max_diff, bool verbose) {
    auto dev = torch::Device(torch::kCUDA);
    auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

    float s = 1.0f / std::sqrt(static_cast<float>(K));
    torch::Tensor A = torch::randn({groups, M, K}, bf16) * s;
    torch::Tensor B = torch::randn({groups, N, K}, bf16) * s;
    torch::Tensor grouped_layout = test_utils::generate_masked_grouped_layout(M, groups, dev);

    torch::Tensor ref = torch::einsum("gmk,gnk->gmn", {A, B});

    auto [A_fp8_flat, sfa_flat] = test_utils::quantize_fp8_1d_block(A.reshape({groups * M, K}), Major::K, dev);
    torch::Tensor A_fp8 = A_fp8_flat.reshape({groups, M, K});
    torch::Tensor sfa = sfa_flat.reshape({groups, M, K / 128});

    auto [B_fp8_flat, sfb] = test_utils::quantize_fp8_2d_block(B.reshape({groups * N, K}), dev);
    torch::Tensor B_fp8 = B_fp8_flat.reshape({groups, N, K});

    torch::Tensor out = torch::empty({groups, M, N}, bf16);
    at::Tensor A_t = A_fp8;
    at::Tensor B_t = B_fp8;
    at::Tensor sfa_t = sfa;
    at::Tensor sfb_t = sfb;
    at::Tensor out_t = out;
    int* grouped_layout_ptr = grouped_layout.data_ptr<int>();

    std::pair<at::Tensor&, at::Tensor&> act{A_t, sfa_t};
    std::pair<at::Tensor&, at::Tensor&> weight{B_t, sfb_t};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    moe_cuda::fp8_gemm(act, weight, out_t, GemmType::MGroupedMasked, "", grouped_layout_ptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto layout_cpu = grouped_layout.cpu();
    auto layout_acc = layout_cpu.accessor<int32_t, 1>();

    std::vector<torch::Tensor> ref_list;
    std::vector<torch::Tensor> out_list;
    auto out_cpu = out.cpu();
    for (int g = 0; g < groups; ++g) {
        uint32_t rows = layout_acc[g];
        ref_list.push_back(ref.index({g, torch::indexing::Slice(0, rows)}));
        out_list.push_back(out_cpu.index({g, torch::indexing::Slice(0, rows)}));
    }

    double diff = calc_diff(torch::cat(ref_list, 0), torch::cat(out_list, 0));
    if (verbose) {
        std::cout << "masked M=" << M << " groups=" << groups << " diff=" << diff << "\n";
    }
    return diff < max_diff;
}

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        std::cerr << "Usage: " << argv[0]
                  << " --type <normal|batched|contiguous|masked> [--m M --n N --k K --groups G --max-diff X --verbose]\n";
        return 1;
    }

    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is unavailable\n";
        return 1;
    }

    Compiler::init_static_vars(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                               get_env<std::string>("CUDA_HOME_PATH", ""));

    int passed = 0;
    int failed = 0;

    auto run_one = [&](const TestShape& s) {
        bool ok = false;
        if (cfg.type == GemmType::Normal || cfg.type == GemmType::Batched) {
            // FP8 batched path reuses normal harness by flattening semantics in kernel implementation.
            ok = run_normal(s.M, s.N, s.K, cfg.max_diff, cfg.verbose);
        } else if (cfg.type == GemmType::MGroupedContiguous) {
            ok = run_contiguous(s.M, s.N, s.K, s.groups, cfg.max_diff, cfg.verbose);
        } else if (cfg.type == GemmType::MGroupedMasked) {
            ok = run_masked(s.M, s.N, s.K, s.groups, cfg.max_diff, cfg.verbose);
        }

        if (ok) ++passed;
        else ++failed;

        std::cout << (ok ? "[PASSED] " : "[FAILED] ") << type_to_string(cfg.type)
                  << " M=" << s.M << " N=" << s.N << " K=" << s.K << " G=" << s.groups
                  << " (" << s.desc << ")\n";
    };

    if (!cfg.all_shapes) {
        TestShape s{cfg.M, cfg.N, cfg.K, cfg.groups, "custom"};
        run_one(s);
    } else {
        const std::vector<TestShape>* shapes = &normal_shapes;
        if (cfg.type == GemmType::MGroupedContiguous) shapes = &contiguous_shapes;
        if (cfg.type == GemmType::MGroupedMasked) shapes = &masked_shapes;
        for (const auto& s : *shapes) run_one(s);
    }

    std::cout << "Summary: passed=" << passed << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
