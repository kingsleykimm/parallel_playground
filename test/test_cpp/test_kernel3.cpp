/**
 * @brief Test for kernel3: grouped FP8 GEMM with fused SwiGLU + FP8 requantization.
 *
 * The kernel fuses, per group:
 *   gate_out = A @ gate_weight.T           (FP8 GEMM)
 *   up_out   = A @ up_weight.T             (FP8 GEMM)
 *   result   = sigmoid(gate_out) * up_out  (gated activation)
 *   D        = fp8_quantize(result)        (FP8 output per col-block)
 *   scale_d  = per-(row, col-block) amax / 448  (output scales)
 *
 * Dequantize: D_fp32[m, cb*BN:(cb+1)*BN] = D_fp8[m, cb*BN:(cb+1)*BN] * scale_d[cb, m]
 *
 * N convention: each weight matrix (gate, up) has N rows per group.  The kernel
 * internally treats the combined gate+up space as "2*N", so the output columns
 * per row equal N/2.  Pass D with shape (total_M, N/2) and scale_d with shape
 * (N / (2*BN), total_M).
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "test_utils.h"
#include <kernels/internal_api.hpp>
#include <jit/compiler.hpp>
#include <moe_cuda/types.h>
#include <runtime/utils.h>

using test_utils::calc_diff;

// BN used by kernel3 (the only supported block-N size).
static constexpr int KERNEL3_BN = 128;
static constexpr int KERNEL4_BM = 128;

struct TestShape {
    int64_t M;        // expected tokens per group
    int64_t N;        // per-group weight rows; output has N/2 cols
    int64_t K;        // hidden dim (must be divisible by 128)
    int groups;
    GemmType type;
    int64_t expected_m_per_group;  // for masked: < M; for contiguous: == M
    const char* desc;
};

static const std::vector<TestShape> test_shapes = {
    // contiguous
    { 256, 2048, 2048, 1, GemmType::MGroupedContiguous, 256, "contig-1g"},
    { 256, 2048, 2048, 4, GemmType::MGroupedContiguous, 256, "contig-4g"},
    { 512, 1024, 1024, 2, GemmType::MGroupedContiguous, 512, "contig-2g"},
    // masked
    {4096, 1024, 1024, 1, GemmType::MGroupedMasked,     1024, "masked-1g"},
    {4096, 1024, 1024, 4, GemmType::MGroupedMasked,      512, "masked-4g"},
};

struct Config {
    bool all_shapes = true;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int groups = 1;
    GemmType type = GemmType::MGroupedContiguous;
    uint64_t seed = 1234;
    bool verbose = false;
    bool is_consumer_pp = false;
};

static bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      ((a == "--m")      && i + 1 < argc) { cfg.M = std::stoll(argv[++i]); cfg.all_shapes = false; }
        else if ((a == "--n")      && i + 1 < argc) { cfg.N = std::stoll(argv[++i]); cfg.all_shapes = false; }
        else if ((a == "--k")      && i + 1 < argc) { cfg.K = std::stoll(argv[++i]); cfg.all_shapes = false; }
        else if ((a == "--groups") && i + 1 < argc) { cfg.groups = std::stoi(argv[++i]); }
        else if ((a == "--seed")   && i + 1 < argc) { cfg.seed = std::stoull(argv[++i]); }
        else if  (a == "--verbose") { cfg.verbose = true; }
        else if (a == "--pingpong") { cfg.is_consumer_pp = true; }
        else if ((a == "--type")   && i + 1 < argc) {
            std::string t = argv[++i];
            if      (t == "contiguous") cfg.type = GemmType::MGroupedContiguous;
            else if (t == "masked")     cfg.type = GemmType::MGroupedMasked;
            else { std::cerr << "Unknown type: " << t << "\n"; return false; }
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

// Compute float32 SwiGLU reference for valid rows.
// (full silu = gate * sigmoid(gate) * up)
static torch::Tensor compute_swiglu_ref(
    const torch::Tensor& A_f32,    // (total_M, K)
    const torch::Tensor& gate_f32, // (groups, N, K)
    const torch::Tensor& up_f32,   // (groups, N, K)
    const std::vector<int32_t>& layout_cpu, // (total_M,) group index, -1=pad
    int64_t total_M, int groups, int64_t N_out,
    torch::Device dev)
{
    auto ref = torch::zeros({total_M, N_out},
                            torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    using S = torch::indexing::Slice;
    for (int g = 0; g < groups; ++g) {
        std::vector<int64_t> rows_g;
        for (int64_t r = 0; r < total_M; ++r)
            if (layout_cpu[r] == g) rows_g.push_back(r);
        if (rows_g.empty()) continue;

        auto idx      = torch::tensor(rows_g, torch::kLong).to(dev);
        auto A_g      = A_f32.index_select(0, idx);   // (m_g, K)
        auto gate_out = torch::mm(A_g, gate_f32[g].t()); // (m_g, N)
        auto up_out   = torch::mm(A_g, up_f32[g].t());   // (m_g, N)
        auto swiglu   = torch::sigmoid(gate_out) * gate_out * up_out;
        ref.index_put_({idx, S()}, swiglu);
    }
    return ref;
}

// Dequantize FP8 output using per-(col-block, row) scales.
//   D_fp8:   (total_M, N_out)       N_out = N/2, FP8
//   scale_d: (N_out / BN, total_M)  float32
// Returns float32 tensor of same shape as D_fp8.
static torch::Tensor dequantize_fp8(
    const torch::Tensor& D_fp8,  // (total_M, N_out)
    const torch::Tensor& scale_d) // (cblocks, total_M)
{
    int64_t total_M  = D_fp8.size(0);
    int64_t N_out    = D_fp8.size(1);
    int64_t cblocks  = scale_d.size(0);
    HOST_ASSERT(cblocks * KERNEL3_BN == N_out,
                "scale_d rows * BN must equal D columns");

    auto D_f32 = D_fp8.to(torch::kFloat32);              // (total_M, N_out)
    // scale_d: (cblocks, total_M) → transpose → (total_M, cblocks)
    // broadcast to (total_M, cblocks, BN) then reshape to (total_M, N_out)
    auto scale_t = scale_d.t().contiguous();              // (total_M, cblocks)
    auto scale_b = scale_t.unsqueeze(-1).expand({total_M, cblocks, KERNEL3_BN})
                          .reshape({total_M, N_out});     // (total_M, N_out)
    return (D_f32 * scale_b).to(torch::kBFloat16);
}

// ========================= Contiguous =========================
static bool run_contiguous(int64_t expected_m_per_group, int64_t N, int64_t K,
                            int groups, uint64_t seed, bool verbose, bool is_consumer_pp = false)
{
    auto dev      = torch::Device(torch::kCUDA);
    auto bf16     = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(dev);
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    int64_t N_out = N / 2;  // output columns after SwiGLU
    HOST_ASSERT(N_out % KERNEL3_BN == 0, "N/2 must be divisible by BN=128");
    int64_t cblocks = N_out / KERNEL3_BN;

    int total_rows = 0;
    auto [layout_gpu, actual_ms, aligned_ms] =
        test_utils::generate_contiguous_grouped_layout(
            total_rows, groups, static_cast<int>(expected_m_per_group),
            dev, /*padded=*/true, seed);
    int64_t total_M = total_rows;

    auto layout_cpu_t = layout_gpu.cpu();
    auto layout_acc   = layout_cpu_t.accessor<int32_t, 1>();
    std::vector<int32_t> layout_cpu(total_M);
    for (int64_t r = 0; r < total_M; ++r) layout_cpu[r] = layout_acc[r];

    float s = 1.0f / std::sqrt(static_cast<float>(K));
    auto A    = torch::randn({total_M, K}, bf16) * s;
    auto gate = torch::randn({(int64_t)groups, N_out, K}, bf16) * s;
    auto up   = torch::randn({(int64_t)groups, N_out, K}, bf16) * s;

    // Reference in float32
    auto ref = compute_swiglu_ref(
        A, gate, up,
        layout_cpu, total_M, groups, N_out, dev);

    // Quantize A: scale_a (K/128, total_M) MN-major
    auto [A_fp8, sfa] = test_utils::quantize_fp8_1d_block(A, Major::K, dev);
    sfa = sfa.reshape({total_M, K / 128}).transpose(-1, -2).contiguous();

    // Quantize gate/up with 2D block scales
    auto gate_flat = gate.reshape({(int64_t)groups * N_out, K});
    auto up_flat   = up.reshape({(int64_t)groups * N_out, K});
    auto [gate_fp8_flat, sf_gate] = test_utils::quantize_fp8_2d_block(gate_flat, dev);
    auto [up_fp8_flat,   sf_up]   = test_utils::quantize_fp8_2d_block(up_flat,   dev);
    auto gate_fp8 = gate_fp8_flat.reshape({(int64_t)groups, N_out, K});
    auto up_fp8   = up_fp8_flat.reshape({(int64_t)groups, N_out, K});

    // Outputs: D (FP8, total_M x N_out), scale_d (cblocks x total_M)
    auto D       = torch::zeros({total_M, N_out}, fp8_opts);
    auto scale_d = torch::zeros({cblocks, total_M}, f32_opts);

    at::Tensor A_t = A_fp8, gate_t = gate_fp8, up_t = up_fp8;
    at::Tensor sfa_t = sfa, sfg_t = sf_gate, sfu_t = sf_up;
    at::Tensor sd_t = scale_d, D_t = D;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    if (is_consumer_pp) {
        moe_cuda::kernels::fp8_grouped_gemm_swiglu_consumer_pp(
            A_t, gate_t, up_t, sfa_t, sfg_t, sfu_t, sd_t, D_t,
            GemmType::MGroupedContiguous, layout_gpu.data_ptr<int>(), stream);
    } else {
        moe_cuda::kernels::fp8_grouped_gemm_swiglu(
            A_t, gate_t, up_t, sfa_t, sfg_t, sfu_t, sd_t, D_t,
            GemmType::MGroupedContiguous, layout_gpu.data_ptr<int>(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Dequantize FP8 output and compare against reference
    auto D_dequant = dequantize_fp8(D, scale_d);  // (total_M, N_out) float32

    // Compare only valid (non-padding) rows
    std::vector<int64_t> valid_rows;
    valid_rows.reserve(total_M);
    for (int64_t r = 0; r < total_M; ++r)
        if (layout_cpu[r] >= 0) valid_rows.push_back(r);

    double diff = 0.0;
    if (!valid_rows.empty()) {
        auto idx   = torch::tensor(valid_rows, torch::kLong).to(dev);
        auto ref_v = ref.index_select(0, idx);
        auto out_v = D_dequant.index_select(0, idx);
        diff = calc_diff(ref_v, out_v);
        if (verbose) test_utils::check_tensor_close(ref_v, out_v, 0.1f, 0.1f);
        auto error = (D_dequant - ref);
        printf("mean error = %f, std error = %f\n", error.abs().mean().item<float>(), error.abs().std().item<float>());
    }

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << (diff < 0.005 ? "[PASSED] " : "[FAILED] ")
              << "contiguous M=" << expected_m_per_group
              << " N=" << N << " K=" << K << " G=" << groups
              << " total_M=" << total_M
              << " kernel_us=" << us
              << " diff=" << diff << "\n";
    return diff < 0.005;
}

// ========================= Masked =========================
static bool run_masked(int64_t max_M, int64_t expected_m_per_group, int64_t N,
                       int64_t K, int groups, uint64_t seed, bool verbose, bool is_consumer_pp = false)
{
    auto dev      = torch::Device(torch::kCUDA);
    auto bf16     = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(dev);
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    int64_t N_out   = N / 2;
    HOST_ASSERT(N_out % KERNEL3_BN == 0, "N/2 must be divisible by BN=128");
    int64_t cblocks = N_out / KERNEL3_BN;
    int64_t total_M = (int64_t)groups * max_M;

    auto layout_gpu = test_utils::generate_masked_grouped_layout(
        max_M, expected_m_per_group, groups, dev, seed);
    auto layout_cpu_t = layout_gpu.cpu();
    auto layout_acc   = layout_cpu_t.accessor<int32_t, 1>();

    // Build a flat contiguous-style layout so compute_swiglu_ref can be reused
    std::vector<int32_t> layout_flat(total_M, -1);
    for (int g = 0; g < groups; ++g) {
        int32_t actual_m = layout_acc[g];
        for (int32_t r = 0; r < actual_m; ++r)
            layout_flat[g * max_M + r] = g;
    }

    float s = 1.0f / std::sqrt(static_cast<float>(K));
    // A: (groups, max_M, K) — flatten to (total_M, K) for ref compute
    auto A    = torch::randn({(int64_t)groups, max_M, K}, bf16) * s;
    auto gate = torch::randn({(int64_t)groups, N_out, K}, bf16) * s;
    auto up   = torch::randn({(int64_t)groups, N_out, K}, bf16) * s;

    // Reference in float32
    auto A_flat = A.reshape({total_M, K});
    auto ref = compute_swiglu_ref(
        A_flat, gate, up,
        layout_flat, total_M, groups, N_out, dev);

    // Quantize A: reshape to (total_M, K) for 1D block quant, then MN-major
    auto [A_fp8_flat, sfa] = test_utils::quantize_fp8_1d_block(A_flat, Major::K, dev);
    sfa = sfa.reshape({total_M, K / 128}).transpose(-1, -2).contiguous();
    auto A_fp8 = A_fp8_flat.reshape({(int64_t)groups, max_M, K});

    auto gate_flat = gate.reshape({(int64_t)groups * N_out, K});
    auto up_flat   = up.reshape({(int64_t)groups * N_out, K});
    auto [gate_fp8_flat, sf_gate] = test_utils::quantize_fp8_2d_block(gate_flat, dev);
    auto [up_fp8_flat,   sf_up]   = test_utils::quantize_fp8_2d_block(up_flat,   dev);
    auto gate_fp8 = gate_fp8_flat.reshape({(int64_t)groups, N_out, K});
    auto up_fp8   = up_fp8_flat.reshape({(int64_t)groups, N_out, K});

    // Outputs
    auto D       = torch::zeros({total_M, N_out}, fp8_opts);
    auto scale_d = torch::zeros({cblocks, total_M}, f32_opts);

    at::Tensor A_t = A_fp8, gate_t = gate_fp8, up_t = up_fp8;
    at::Tensor sfa_t = sfa, sfg_t = sf_gate, sfu_t = sf_up;
    at::Tensor sd_t = scale_d, D_t = D;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    if (is_consumer_pp) {
        moe_cuda::kernels::fp8_grouped_gemm_swiglu_consumer_pp(
            A_t, gate_t, up_t, sfa_t, sfg_t, sfu_t, sd_t, D_t,
            GemmType::MGroupedMasked, layout_gpu.data_ptr<int>(), stream);
    } else {
        moe_cuda::kernels::fp8_grouped_gemm_swiglu(
            A_t, gate_t, up_t, sfa_t, sfg_t, sfu_t, sd_t, D_t,
            GemmType::MGroupedMasked, layout_gpu.data_ptr<int>(), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto D_dequant = dequantize_fp8(D, scale_d);

    // Compare only valid rows per group
    std::vector<torch::Tensor> ref_parts, out_parts;
    for (int g = 0; g < groups; ++g) {
        int32_t am = layout_acc[g];
        if (am == 0) continue;
        using S = torch::indexing::Slice;
        ref_parts.push_back(ref.index({S(g * max_M, g * max_M + am)}));
        out_parts.push_back(D_dequant.index({S(g * max_M, g * max_M + am)}));
    }
    double diff = 0.0;
    if (!ref_parts.empty()) {
        auto ref_cat = torch::cat(ref_parts, 0);
        auto out_cat = torch::cat(out_parts, 0);
        diff = calc_diff(ref_cat, out_cat);
        if (verbose) test_utils::check_tensor_close(ref_cat, out_cat, 0.1f, 0.1f);
        auto error = (out_cat - ref_cat);
        printf("mean error = %f, std error = %f\n", error.abs().mean().item<float>(), error.abs().std().item<float>());
    }

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << (diff < 0.005 ? "[PASSED] " : "[FAILED] ")
              << "masked  max_M=" << max_M
              << " N=" << N << " K=" << K << " G=" << groups
              << " kernel_us=" << us
              << " diff=" << diff << "\n";
    return diff < 0.005;
}

// ========================= main =========================
int main(int argc, char** argv)
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        std::cerr << "Usage: " << argv[0]
                  << " [--type contiguous|masked --m M --n N --k K"
                  << " --groups G --seed S --verbose]\n";
        return 1;
    }
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA unavailable\n";
        return 1;
    }
    Compiler::init_static_vars(get_env<std::string>("LIBRARY_ROOT_PATH", ""),
                               get_env<std::string>("CUDA_HOME_PATH", ""));

    int passed = 0, failed = 0;

    auto run_one = [&](const TestShape& s) {
        bool ok = (s.type == GemmType::MGroupedContiguous)
            ? run_contiguous(s.M, s.N, s.K, s.groups, cfg.seed, cfg.verbose, cfg.is_consumer_pp)
            : run_masked(s.M, s.expected_m_per_group, s.N, s.K,
                         s.groups, cfg.seed, cfg.verbose, cfg.is_consumer_pp);
        if (ok) ++passed; else ++failed;
    };

    if (!cfg.all_shapes) {
        if (cfg.M == 0) cfg.M = 256;
        if (cfg.N == 0) cfg.N = 256;
        if (cfg.K == 0) cfg.K = 128;
        printf("Running custom shape: M=%d, N=%d, K=%d\n", cfg.M, cfg.N, cfg.K);
        run_one({cfg.M, cfg.N, cfg.K, cfg.groups, cfg.type, cfg.M, "custom"});
    } else {
        for (const auto& s : test_shapes) run_one(s);
    }

    std::cout << "Summary: passed=" << passed << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
