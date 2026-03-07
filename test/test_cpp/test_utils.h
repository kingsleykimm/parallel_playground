#pragma once

#include "ATen/TensorIndexing.h"
#include "ATen/core/ivalue_inl.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "moe_cuda/kernels/common/common.hpp"
#include <moe_cuda/error.hpp>
#include <moe_cuda/dtype.h>
#include <c10/core/Device.h>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#include <runtime/tensor.h>

namespace test_utils {

    static constexpr float float8_e4m3_amax = 448.0;

inline std::string shape_to_string(const std::array<size_t, 5> &shape) {
    std::ostringstream stream;
    stream << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        stream << shape[i];
        if (i + 1 < shape.size()) {
            stream << ", ";
        }
    }
    stream << ")";
    return stream.str();
}
inline std::string shape_to_string(const std::vector<int64_t> &shape) {
    std::ostringstream stream;
    stream << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        stream << shape[i];
        if (i + 1 < shape.size()) {
            stream << ", ";
        }
    }
    stream << ")";
    return stream.str();
}

inline std::vector<size_t> to_size_t_dims(const std::vector<int64_t> &shape) {
    std::vector<size_t> dims;
    dims.reserve(shape.size());
    for (const auto dim : shape) {
        dims.push_back(static_cast<size_t>(dim));
    }
    return dims;
}

inline void inspect_tensor(const torch::Tensor &tensor, int n) {
    std::cout << tensor.flatten().slice(0, 0, n) << std::endl;
}

inline void inspect_tensor(at::Tensor &tensor, int n) {
    std::cout << custom::to_string(tensor, n, 0, -1) << std::endl;
}

inline void inspect_tensor(const torch::Tensor &tensor, int s, int e) {
    std::cout << tensor.flatten().slice(0, s, e) << std::endl;
}


inline void inspect_tensor(at::Tensor &tensor, int s, int e) {
    std::cout << custom::to_string(tensor, e - s, s, e) << std::endl;
}



inline bool check_tensor_close(const torch::Tensor &tensor1, const torch::Tensor &tensor2, float atol, float rtol) {
    bool pass = torch::allclose(tensor1, tensor2, rtol, atol);
    if (!pass) {
        int argmax_ind = torch::abs(tensor1 - tensor2).argmax().item().toInt();
        std::cout << "\033[0;31mTensor mismatch\033[0m" << std::endl;
        printf("\033[0;31mTensor1 shape: %s, Strides: %s\033[0m\n", shape_to_string(tensor1.sizes().vec()).c_str(), shape_to_string(tensor1.strides().vec()).c_str());
        printf("\033[0;31mTensor2 shape: %s, Strides: %s\033[0m\n", shape_to_string(tensor2.sizes().vec()).c_str(), shape_to_string(tensor2.strides().vec()).c_str());
        std::cout << "\033[0;31mInd Diff: " << torch::abs(tensor1 - tensor2).argmax().item() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mDiff: " << torch::abs(tensor1 - tensor2).max().item() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mAvg Diff : " << torch::abs(tensor1 - tensor2).mean().item() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mTensor 1 : " << tensor1.flatten()[argmax_ind].item() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mTensor 2 : " << tensor2.flatten()[argmax_ind].item() << "\033[0m" << std::endl;
    }
    else {
        int argmax_ind = torch::abs(tensor1 - tensor2).argmax().item().toInt();
        std::cout << "\033[0;32mTensor passed\033[0m" << std::endl;
        printf("\033[0;32mTensor1 shape: %s, Strides: %s\033[0m\n", shape_to_string(tensor1.sizes().vec()).c_str(), shape_to_string(tensor1.strides().vec()).c_str());
        printf("\033[0;32mTensor2 shape: %s, Strides: %s\033[0m\n", shape_to_string(tensor2.sizes().vec()).c_str(), shape_to_string(tensor2.strides().vec()).c_str());
        std::cout << "\033[0;32mInd Diff: " << torch::abs(tensor1 - tensor2).argmax().item().toFloat() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mAvg Diff : " << torch::abs(tensor1 - tensor2).mean().item() << "\033[0m" << std::endl;
        std::cout << "\033[0;31mDiff: " << torch::abs(tensor1 - tensor2).max().item().toFloat() << "\033[0m" << std::endl;
        std::cout << "\033[0;32mTensor 1 : " << tensor1.flatten()[argmax_ind].item() << "\033[0m" << std::endl;
        std::cout << "\033[0;32mTensor 2 : " << tensor2.flatten()[argmax_ind].item() << "\033[0m" << std::endl;
    }
    return pass;
}

// overloaded method to check between two custom class tensors
template <typename ScalarType>
inline bool check_tensor_close(torch::Tensor& tensor1, at::Tensor& tensor2, float atol, float rtol) {
    (void)sizeof(ScalarType);
    return check_tensor_close(static_cast<const torch::Tensor&>(tensor1),
                              static_cast<const torch::Tensor&>(tensor2),
                              atol,
                              rtol);
}


// Quantize a 2D tensor to FP8 with 1D block scaling (per-row, 128-element blocks)
// Returns a tuple of (quantized_tensor, scale_factors)
// Scale factors have shape (num_rows, num_k_blocks) where num_k_blocks = K / 128
inline std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_1d_block(
    const torch::Tensor& t,
    Major sf_major,
    torch::Device device = torch::kCUDA
) {
    HOST_ASSERT(t.ndimension() == 2 || t.ndimension() == 3,
                "quantizing tensor must be 2 or 3 dimensions");
    // Flatten leading dims to 2D for quantization
    int num_groups = t.ndimension() == 2 ? 1 : t.size(0);
    int m = t.ndimension() == 2 ? t.size(0) : t.size(1);
    auto orig_sizes = t.sizes().vec();
    auto t_float = t.ndimension() == 2 ? t.unsqueeze(0).to(torch::kFloat32).contiguous().cpu() : t.to(torch::kFloat32).contiguous().cpu();
    const auto shapes = t_float.sizes().vec();
    const int64_t k_dim = t_float.size(-1);
    HOST_ASSERT(k_dim % 128 == 0, "K dimension must be divisible by 128 for FP8 block scaling");

    int num_k_blocks = k_dim / 128;
    torch::TensorOptions sf_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sf;
    if (sf_major == Major::K) {
        sf = torch::empty({num_groups, m, num_k_blocks}, sf_options);
    }
    else {
        sf = torch::empty({num_groups, m, num_k_blocks}, sf_options);
        // Manually change strides to transpose layout without changing dimensions
        sf = sf.as_strided({num_groups, m, num_k_blocks}, {num_k_blocks * m, 1, m});
    }
    torch::Tensor quantized = torch::empty({num_groups, m, k_dim}, sf_options);

    auto t_accessor = t_float.accessor<float, 3>();
    auto sf_accessor = sf.accessor<float, 3>();
    auto q_accessor = quantized.accessor<float, 3>();

    for (int g = 0; g < num_groups; g++) {
        for (int64_t row = 0; row < m; row++) {
            for (int chunk = 0; chunk < num_k_blocks; chunk++) {
                float max_abs = 0.0f;
                for (int i = 0; i < 128; i++) {
                    float val = std::abs(t_accessor[g][row][chunk * 128 + i]);
                    max_abs = std::max(max_abs, val);
                }
                float scale = max_abs > 0.0f ? float8_e4m3_amax / max_abs : 1.0f;
                float inv_scale = max_abs > 0.0f ? max_abs / float8_e4m3_amax : 1.0f;
                sf_accessor[g][row][chunk] = inv_scale;
    
                for (int i = 0; i < 128; i++) {
                    float val = t_accessor[g][row][chunk * 128 + i] * scale;
                    val = std::max(-float8_e4m3_amax, std::min(float8_e4m3_amax, val));
                    q_accessor[g][row][chunk * 128 + i] = val;
                }
            }
    }
    }

    torch::Tensor quantized_fp8 = quantized.to(torch::kFloat8_e4m3fn).to(device);
    torch::Tensor sf_device = sf.to(device);

    // Reshape back to 3D if input was 3D
    if (t.ndimension() == 2) {
        quantized_fp8 = quantized_fp8.squeeze(0);
        sf_device = sf_device.squeeze(0);
        // quantized_fp8 = quantized_fp8.reshape({orig_sizes[0], orig_sizes[1], orig_sizes[2]});
        // sf_device = sf_device.reshape({orig_sizes[0], orig_sizes[1], num_k_blocks}).to(device);
    }

    return std::make_tuple(quantized_fp8, sf_device);
}

// Quantize a 2D tensor to FP8 with 2D block scaling (128x128 blocks)
// Returns a tuple of (quantized_tensor, scale_factors)
// Scale factors have shape (num_n_blocks, num_k_blocks) in K-major layout
inline std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_2d_block(
    const torch::Tensor& t,
    torch::Device device = torch::kCUDA
) {
    HOST_ASSERT(t.ndimension() == 2, "quantizing tensor must be 2 dimensions");
    auto t_float = t.to(torch::kFloat32).contiguous().cpu();
    const auto shapes = t_float.sizes().vec();
    const int64_t n_dim = shapes[0];
    const int64_t k_dim = shapes[1];
    HOST_ASSERT(k_dim % 128 == 0, "K dimension must be divisible by 128 for FP8 block scaling");
    HOST_ASSERT(n_dim % 128 == 0, "N dimension must be divisible by 128 for 2D FP8 block scaling");

    int num_n_blocks = n_dim / 128;
    int num_k_blocks = k_dim / 128;

    torch::TensorOptions sf_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sf = torch::empty({num_n_blocks, num_k_blocks}, sf_options);
    torch::Tensor quantized = torch::empty({n_dim, k_dim}, sf_options);

    auto t_accessor = t_float.accessor<float, 2>();
    auto sf_accessor = sf.accessor<float, 2>();
    auto q_accessor = quantized.accessor<float, 2>();

    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            float max_abs = 0.0f;
            for (int ni = 0; ni < 128; ni++) {
                for (int ki = 0; ki < 128; ki++) {
                    int64_t n_idx = n_block * 128 + ni;
                    int64_t k_idx = k_block * 128 + ki;
                    float val = std::abs(t_accessor[n_idx][k_idx]);
                    max_abs = std::max(max_abs, val);
                }
            }
            float scale = max_abs > 0.0f ? float8_e4m3_amax / max_abs : 1.0f;
            float inv_scale = max_abs > 0.0f ? max_abs / float8_e4m3_amax : 1.0f;
            sf_accessor[n_block][k_block] = inv_scale;

            for (int ni = 0; ni < 128; ni++) {
                for (int ki = 0; ki < 128; ki++) {
                    int64_t n_idx = n_block * 128 + ni;
                    int64_t k_idx = k_block * 128 + ki;
                    float val = t_accessor[n_idx][k_idx] * scale;
                    val = std::max(-float8_e4m3_amax, std::min(float8_e4m3_amax, val));
                    q_accessor[n_idx][k_idx] = val;
                }
            }
        }
    }

    torch::Tensor quantized_fp8 = quantized.to(torch::kFloat8_e4m3fn).to(device);
    torch::Tensor sf_device = sf.to(device);

    return std::make_tuple(quantized_fp8, sf_device);
}


inline double calc_diff(const torch::Tensor& ref, const torch::Tensor& out) {
    auto ref_d = ref.to(torch::kDouble);
    auto out_d= out.to(torch::kDouble);

    double denominator = (ref_d * ref_d + out_d * out_d).sum().item().toDouble();

    double similarity = (2 * (ref_d * out_d).sum().item().toDouble()) / denominator;
    return 1 - similarity;
}

// Generate grouped layout for MGroupedContiguous
// Returns a tensor where each element indicates the group index for that row
// Some rows are marked -1 (no-op) to simulate real MoE inference where tokens
// aren't assigned to any expert. -1 rows are placed at periodic intervals within
// each group section, never on block-aligned boundaries (to avoid TMA issues).
inline std::tuple<torch::Tensor, std::vector<size_t>, std::vector<size_t>> generate_contiguous_grouped_layout(
    int& total_rows,
    int num_groups,
    int expected_m_per_group,
    torch::Device device = torch::kCUDA
) {
    // Calculate padding interval to simulate sparse token assignment
    // Similar to Python's random.uniform(0.7, 1.3) * expected_m_per_group
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.7f, 1.3f);

    std::vector<size_t> actual_ms;
    std::vector<size_t> aligned_ms;
    total_rows = 0;
    for (int i = 0; i < num_groups; i++) {
        int actual_m = (int) (dist(gen) * expected_m_per_group);
        int aligned_m = ti_align(actual_m, 128);
        actual_ms.push_back(actual_m);
        aligned_ms.push_back(aligned_m);
        total_rows += aligned_m;
    }
    

    torch::Tensor layout = torch::empty({total_rows}, torch::TensorOptions().dtype(torch::kInt32));
    auto accessor = layout.accessor<int32_t, 1>();

    int start = 0;
    for (int g = 0; g < num_groups; g++) {
        int actual_m = actual_ms[g];
        int aligned_m = aligned_ms[g];
        for (int idx = start; idx < start + actual_m; idx++) {
            layout[idx] = g;
        }
        for (int idx = start + actual_m; idx < start + aligned_m; idx++) {
            layout[idx] = -1;
        }
        start += aligned_m;
    }
    return std::make_tuple(layout.to(device), actual_ms, aligned_ms);
}

// Generate grouped layout for MGroupedMasked
// Returns a tensor of shape (num_groups,) containing the number of valid rows per group
inline torch::Tensor generate_masked_grouped_layout(
    int64_t max_rows_per_group,
    int num_groups,
    torch::Device device = torch::kCUDA
) {
    torch::Tensor layout = torch::empty({num_groups}, torch::TensorOptions().dtype(torch::kInt32));
    auto accessor = layout.accessor<int32_t, 1>();

    // For testing, use varying row counts per group
    for (int g = 0; g < num_groups; g++) {
        // Use a deterministic pattern: each group gets slightly different row count
        int64_t rows = max_rows_per_group - (g % 4) * 32;
        rows = std::max(rows, (int64_t)128);  // Minimum 128 rows
        accessor[g] = static_cast<int32_t>(rows);
    }
    return layout.to(device);
}

}
