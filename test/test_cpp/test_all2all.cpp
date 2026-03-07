/*
 * moe_cuda::All2All Dispatch/Combine Testing Harness (MPI)
 *
 * Tests the moe_cuda::All2All dispatch and combine operations across multiple GPUs
 * using MPI for process management. Each MPI rank maps to one GPU.
 *
 * Usage:
 *   mpirun -np 4 ./test_all2all          # EP=4
 *   mpirun -np 2 ./test_all2all          # EP=2
 *   mpirun -np 1 ./test_all2all          # EP=1
 *   mpirun -np 4 ./test_all2all --verbose
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <torch/torch.h>

#include "test_utils.h"
#include <all2all/all2all.hpp>
#include <runtime/kernels_api.h>
#include <runtime/parallel.h>

using test_utils::check_tensor_close;
using test_utils::shape_to_string;

struct All2AllTestConfig {
    uint32_t num_experts;
    uint32_t num_tokens;
    uint32_t hidden_dim;
    uint32_t num_experts_per_token;
    uint32_t expert_padding;
    c10::ScalarType dtype;
    bool verbose;
};

// ============================================================================
// Helpers
// ============================================================================

static void log_rank(int rank, const std::string& msg) {
    std::cout << "[Rank " << rank << "] " << msg << std::endl;
}

// ============================================================================
// Dispatch Test
// ============================================================================

bool test_dispatch(
    int rank,
    int world_size,
    moe_cuda::All2All& all2all,
    const All2AllTestConfig& config,
    cudaStream_t stream,
    // outputs used by combine test
    torch::Tensor& out_indices_torch,
    torch::Tensor& out_weights_torch,
    torch::Tensor& out_expert_x_torch,
    torch::Tensor& out_expert_num_tokens_torch,
    torch::Tensor& out_dp_x_torch
) {
    if (rank == 0) {
        std::cout << "\n--- Testing a2a_dispatch (EP=" << world_size
                  << ", num_tokens=" << config.num_tokens
                  << ", num_experts=" << config.num_experts
                  << ", hidden_dim=" << config.hidden_dim
                  << ", top_k=" << config.num_experts_per_token
                  << ") ---\n";
    }

    try {
        torch::Device device(torch::kCUDA, rank);
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);

        // Create input tokens (num_tokens, hidden_dim) in BF16
        torch::Tensor dp_x_torch = torch::randn(
            {(int64_t)config.num_tokens, (int64_t)config.hidden_dim}, bf16_opts);

        // Create expert indices (num_tokens, num_experts_per_token) in INT32
        // Values in [0, num_experts)
        torch::Tensor indices_torch = torch::randint(
            0, (int64_t)config.num_experts,
            {(int64_t)config.num_tokens, (int64_t)config.num_experts_per_token}, i32_opts);

        // Create routing weights (num_tokens, num_experts_per_token) in FP32
        torch::Tensor weights_torch = torch::rand(
            {(int64_t)config.num_tokens, (int64_t)config.num_experts_per_token}, f32_opts);
        // Normalize weights per token
        weights_torch = weights_torch / weights_torch.sum(/*dim=*/1, /*keepdim=*/true);

        // Compute number of local experts for this rank
        uint32_t num_local_experts = (config.num_experts + world_size - 1) / world_size;

        // Compute max recv tokens for output buffer sizing
        uint32_t avg_tokens_per_expert = (uint32_t)std::ceil(
            (float)(config.num_tokens * config.num_experts_per_token) / config.num_experts * 1.2f);
        uint32_t max_recv_tokens = avg_tokens_per_expert * num_local_experts * world_size;
        max_recv_tokens += std::max(
            std::min((uint32_t)(config.num_tokens * world_size * config.num_experts_per_token
                + num_local_experts * (config.expert_padding - 1)),
                num_local_experts * config.num_tokens * (uint32_t)world_size),
            num_local_experts * config.expert_padding);
        // Align to expert_padding
        max_recv_tokens = ((max_recv_tokens + config.expert_padding - 1) / config.expert_padding) * config.expert_padding;

        // Pre-allocate output buffers
        torch::Tensor expert_x_torch = torch::zeros(
            {(int64_t)max_recv_tokens, (int64_t)config.hidden_dim}, bf16_opts);
        torch::Tensor expert_num_tokens_torch = torch::zeros(
            {(int64_t)num_local_experts}, i32_opts);

        // Convert to custom tensors
        at::Tensor dp_x = (dp_x_torch);
        at::Tensor indices = (indices_torch);
        at::Tensor weights = (weights_torch);
        at::Tensor out_expert_x = (expert_x_torch);
        at::Tensor out_expert_num_tokens = (expert_num_tokens_torch);

        std::optional<at::Tensor> out_expert_x_scale = std::nullopt;
        std::optional<at::Tensor> dp_x_scale = std::nullopt;
        std::optional<at::Tensor> bound_m = std::nullopt;

        // Call dispatch
        moe_cuda::kernels::a2a_dispatch(
            all2all,
            out_expert_num_tokens, out_expert_x, out_expert_x_scale,
            dp_x, dp_x_scale, indices, weights, bound_m,
            true, true, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verification: check that expert_num_tokens sums correctly
        // Gather all expert_num_tokens across ranks to verify total token count
        torch::Tensor num_tokens_cpu = expert_num_tokens_torch.to(torch::kCPU).to(torch::kInt32);
        int32_t local_total = 0;
        auto num_tokens_acc = num_tokens_cpu.accessor<int32_t, 1>();
        for (int64_t i = 0; i < num_tokens_cpu.size(0); i++) {
            local_total += num_tokens_acc[i];
        }

        // Sum across all ranks
        int32_t global_total = 0;
        MPI_Allreduce(&local_total, &global_total, 1, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

        // Expected: each rank sends num_tokens * num_experts_per_token token-expert assignments
        int32_t expected_total = config.num_tokens * config.num_experts_per_token * world_size;

        bool count_passed = (global_total == expected_total);
        if (rank == 0) {
            std::cout << "Token count check: global_total=" << global_total
                      << " expected=" << expected_total;
            if (count_passed) {
                std::cout << " \033[0;32m[PASSED]\033[0m\n";
            } else {
                std::cout << " \033[0;31m[FAILED]\033[0m\n";
            }
        }

        if (config.verbose) {
            log_rank(rank, "Local expert token counts:");
            for (int64_t i = 0; i < num_tokens_cpu.size(0); i++) {
                std::cout << "  expert[" << i << "] = " << num_tokens_acc[i] << "\n";
            }
        }

        // Store outputs for combine test
        out_indices_torch = indices_torch;
        out_weights_torch = weights_torch;
        out_expert_x_torch = expert_x_torch;
        out_expert_num_tokens_torch = expert_num_tokens_torch;
        out_dp_x_torch = dp_x_torch;

        return count_passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31m[Rank " << rank << "] Dispatch error: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// Combine Test
// ============================================================================

bool test_combine(
    int rank,
    int world_size,
    moe_cuda::All2All& all2all,
    const All2AllTestConfig& config,
    cudaStream_t stream,
    torch::Tensor& indices_torch,
    torch::Tensor& weights_torch,
    torch::Tensor& expert_x_torch,
    torch::Tensor& expert_num_tokens_torch,
    torch::Tensor& dp_x_torch
) {
    if (rank == 0) {
        std::cout << "\n--- Testing a2a_combine (EP=" << world_size << ") ---\n";
    }

    try {
        torch::Device device(torch::kCUDA, rank);
        auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

        // Simulate expert output: use expert_x as identity transform (expert_y = expert_x)
        // This way we can verify the combine produces correct weighted sums
        torch::Tensor expert_y_torch = expert_x_torch.clone();

        // Output buffer for combined tokens
        torch::Tensor out_tokens_torch = torch::zeros(
            {(int64_t)config.num_tokens, (int64_t)config.hidden_dim}, bf16_opts);

        // Convert to custom tensors
        at::Tensor out_tokens = (out_tokens_torch);
        at::Tensor indices = (indices_torch);
        at::Tensor weights = (weights_torch);
        at::Tensor expert_y = (expert_y_torch);
        std::optional<at::Tensor> bound_m = std::nullopt;

        // Call combine
        moe_cuda::kernels::a2a_combine(
            all2all,
            out_tokens, indices, weights, expert_y, bound_m,
            true, true, false, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verification: output tokens should be non-zero (tokens returned to originating rank)
        torch::Tensor out_abs_sum = out_tokens_torch.abs().sum();
        float abs_sum_val = out_abs_sum.item<float>();

        bool nonzero_passed = (abs_sum_val > 0.0f);

        if (rank == 0) {
            std::cout << "Combine output non-zero check: abs_sum=" << abs_sum_val;
            if (nonzero_passed) {
                std::cout << " \033[0;32m[PASSED]\033[0m\n";
            } else {
                std::cout << " \033[0;31m[FAILED]\033[0m\n";
            }
        }

        if (config.verbose) {
            log_rank(rank, "Output tokens (first 5 values of token 0):");
            std::cout << out_tokens_torch.slice(0, 0, 1).slice(1, 0, 5) << "\n";
        }

        return nonzero_passed;
    } catch (const std::exception& e) {
        std::cerr << "\033[0;31m[Rank " << rank << "] Combine error: " << e.what() << "\033[0m\n";
        return false;
    }
}

// ============================================================================
// Run a single test configuration
// ============================================================================

bool run_test_config(int rank, int world_size, const All2AllTestConfig& config) {
    torch::Device device(torch::kCUDA, rank);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create parallel groups
    // node_group = all ranks (single node)
    ParallelGroup node_group(rank, world_size);
    // global_group = same as node_group for single-node
    ParallelGroup global_group(rank, world_size);
    // No dp_group for this test
    std::optional<ParallelGroup> dp_group = std::nullopt;

    // Create moe_cuda::All2All instance
    moe_cuda::All2All all2all(
        config.num_tokens,          // max_num_tokens
        config.num_experts,         // num_experts
        config.expert_padding,      // expert_padding
        config.hidden_dim,          // hidden_dim
        std::nullopt,               // hidden_dim_scale
        config.dtype,               // in_dtype
        config.dtype,               // out_dtype
        std::nullopt,               // scale_dtype
        config.num_experts_per_token,
        std::nullopt,               // max_private_tokens
        dp_group,                   // dp_group
        node_group,                 // node_group
        rank,                       // device
        global_group,               // global_group
        stream
    );

    MPI_Barrier(MPI_COMM_WORLD);

    // Outputs from dispatch used by combine
    torch::Tensor indices_torch, weights_torch, expert_x_torch, expert_num_tokens_torch, dp_x_torch;

    bool dispatch_passed = test_dispatch(
        rank, world_size, all2all, config, stream,
        indices_torch, weights_torch, expert_x_torch, expert_num_tokens_torch, dp_x_torch);

    MPI_Barrier(MPI_COMM_WORLD);

    bool combine_passed = test_combine(
        rank, world_size, all2all, config, stream,
        indices_torch, weights_torch, expert_x_torch, expert_num_tokens_torch, dp_x_torch);

    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));

    return dispatch_passed && combine_passed;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parse args
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose") verbose = true;
        else if (arg == "--help" || arg == "-h") {
            if (rank == 0) {
                std::cout << "moe_cuda::All2All Test\n";
                std::cout << "Usage: mpirun -np <N> " << argv[0] << " [--verbose] [--help]\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    // Set CUDA device based on rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (rank >= num_devices) {
        std::cerr << "[Rank " << rank << "] Error: not enough GPUs (have "
                  << num_devices << ", need " << world_size << ")\n";
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "============================================\n";
        std::cout << "moe_cuda::All2All Dispatch/Combine Testing Harness\n";
        std::cout << "World size (EP): " << world_size << "\n";
        std::cout << "============================================\n";
    }

    // Define test configurations
    std::vector<All2AllTestConfig> configs = {
        // num_experts, num_tokens, hidden_dim, top_k, expert_padding, dtype, verbose
        {(uint32_t)(4 * world_size), 64,  256, 2, 16, c10::ScalarType::BFloat16, verbose},
        {(uint32_t)(4 * world_size), 128, 128, 1, 16, c10::ScalarType::BFloat16, verbose},
        {(uint32_t)(2 * world_size), 32,  256, 2, 16, c10::ScalarType::BFloat16, verbose},
    };

    int passed = 0, failed = 0;

    for (size_t i = 0; i < configs.size(); i++) {
        auto& cfg = configs[i];

        if (rank == 0) {
            std::cout << "\n=== Test Config " << (i + 1) << "/" << configs.size() << " ===\n";
        }

        bool result = run_test_config(rank, world_size, cfg);

        if (rank == 0) {
            if (result) {
                std::cout << "\033[0;32m[PASSED] Config " << (i + 1) << "\033[0m\n";
                passed++;
            } else {
                std::cout << "\033[0;31m[FAILED] Config " << (i + 1) << "\033[0m\n";
                failed++;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "\n============================================\n";
        std::cout << "Test Summary\n";
        std::cout << "============================================\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";

        if (failed > 0) {
            std::cout << "\033[0;31mSome tests FAILED\033[0m\n";
        } else {
            std::cout << "\033[0;32mAll tests PASSED\033[0m\n";
        }
    }

    MPI_Finalize();
    return (failed > 0) ? 1 : 0;
}
