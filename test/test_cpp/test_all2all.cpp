/*
 * TK-backed moe_cuda::All2All dispatch test harness.
 *
 * The TK path is currently single-node and 4-GPU specific, so this test
 * launches one child process per local rank and validates dispatch routing
 * against a deterministic CPU oracle.
 */

#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "c10/core/ScalarType.h"
#include "test_utils.h"
#include <all2all/all2all_tk.hpp>
#include <runtime/parallel.h>

namespace {

constexpr int kWorldSize = 4;
constexpr int kNumExperts = 64;

struct TestCase {
  const char *name;
  uint32_t num_tokens;
  uint32_t hidden_dim;
  uint32_t top_k;
  uint32_t expert_padding;
};

std::vector<TestCase> kCases = {{// {"topk1_h128", 128, 128, 1, 16},
                                 // {"topk2_h256", 64, 256, 2, 16},
                                 {"topk10_h2048", 8192, 2048, 10, 128},
                                 {"topk12_h2048", 4096, 2048, 12, 128}}};

struct RankResult {
  int passed;
  char message[512];
};

struct SharedResults {
  RankResult results[kWorldSize];
  std::atomic<int> barrier_count = 0;
};

struct OracleBundle {
  std::vector<torch::Tensor> input_fp8_cpu;
  std::vector<torch::Tensor> input_scale_cpu;
  std::vector<torch::Tensor> input_dequant_cpu;
  std::vector<torch::Tensor> indices_cpu;
  std::vector<torch::Tensor> weights_cpu;
  std::vector<torch::Tensor> expected_expert_x_cpu;
  std::vector<torch::Tensor> expected_expert_x_scale_cpu;
  std::vector<torch::Tensor> expected_counts_cpu;
  std::vector<torch::Tensor> expected_expert_x_dequant_cpu;
  std::vector<torch::Tensor> expected_source_rank_cpu;
  std::vector<torch::Tensor> expected_source_dispatch_offset_cpu;
  std::vector<torch::Tensor> expected_combine_send_offset_cpu;
  std::vector<torch::Tensor> expected_padded_index_cpu;
  std::vector<torch::Tensor> route_positions_cpu;
  std::vector<uint32_t> expected_num_recv_tokens;
  uint32_t max_private_tokens = 0;
  uint32_t max_recv_tokens = 0;
  torch::Tensor expected_num_routed_cpu;
};

struct SourceRouteRef {
  uint32_t token;
  uint32_t route;
  uint32_t source_local_offset;
};

uint32_t align_to(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

uint32_t compute_max_private_tokens(const TestCase &cfg) {
  const uint32_t num_local_experts =
      (kNumExperts + kWorldSize - 1) / kWorldSize;
  const uint32_t avg_tokens_per_expert = static_cast<uint32_t>(
      ((cfg.num_tokens * cfg.top_k + kNumExperts - 1) / kNumExperts) * 1.2f);
  return avg_tokens_per_expert * num_local_experts;
}

uint32_t compute_max_recv_tokens(const TestCase &cfg) {
  const uint32_t num_local_experts =
      (kNumExperts + kWorldSize - 1) / kWorldSize;
  const uint32_t num_dp_groups = kWorldSize;
  const uint32_t max_private_tokens = compute_max_private_tokens(cfg);
  const uint32_t num_tokens = cfg.num_tokens * num_dp_groups;

  uint32_t max_recv_tokens = max_private_tokens * num_dp_groups;
  max_recv_tokens += align_to(
      std::max(std::min(num_tokens * cfg.top_k +
                            num_local_experts * (cfg.expert_padding - 1),
                        num_local_experts * num_tokens),
               num_local_experts * cfg.expert_padding),
      cfg.expert_padding);
  return max_recv_tokens;
}

// Returns random float32 input seeded by rank (call torch::manual_seed before
// use).
torch::Tensor make_float_input_cpu(const TestCase &cfg, int /*rank*/) {
  return torch::randn(
      {static_cast<int64_t>(cfg.num_tokens),
       static_cast<int64_t>(cfg.hidden_dim)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

// Random top-k indices via scores: each expert gets an uneven, non-uniform
// share.
torch::Tensor make_indices_cpu(const TestCase &cfg, int /*rank*/) {
  auto scores = torch::randn(
      {static_cast<int64_t>(cfg.num_tokens), kNumExperts},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  scores = scores.abs() + 1.0f;
  auto topk = torch::topk(scores, static_cast<int64_t>(cfg.top_k), /*dim=*/-1,
                          /*largest=*/true, /*sorted=*/true);
  return std::get<1>(topk).to(torch::kInt32);
}

// Random weights normalized to sum to 1 per token.
torch::Tensor make_weights_cpu(const TestCase &cfg) {
  auto w = torch::rand(
      {static_cast<int64_t>(cfg.num_tokens), static_cast<int64_t>(cfg.top_k)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  return w / w.sum(/*dim=*/-1, /*keepdim=*/true);
}

torch::Tensor dequantize_fp8_blocks(const torch::Tensor &fp8,
                                    const torch::Tensor &scales) {
  auto fp32 = fp8.to(torch::kFloat32).contiguous();
  auto out = torch::empty_like(fp32);
  const int64_t hidden_dim = fp32.size(1);
  const int64_t num_blocks = hidden_dim / 128;
  for (int64_t block = 0; block < num_blocks; ++block) {
    auto values = fp32.slice(1, block * 128, (block + 1) * 128);
    auto block_scale = scales.slice(1, block, block + 1);
    out.slice(1, block * 128, (block + 1) * 128).copy_(values * block_scale);
  }
  return out;
}

OracleBundle build_oracle(const TestCase &cfg) {
  OracleBundle oracle;
  oracle.max_private_tokens = compute_max_private_tokens(cfg);
  oracle.max_recv_tokens = compute_max_recv_tokens(cfg);

  oracle.input_fp8_cpu.reserve(kWorldSize);
  oracle.input_scale_cpu.reserve(kWorldSize);
  oracle.input_dequant_cpu.reserve(kWorldSize);
  oracle.indices_cpu.reserve(kWorldSize);
  oracle.weights_cpu.reserve(kWorldSize);
  oracle.expected_expert_x_cpu.reserve(kWorldSize);
  oracle.expected_expert_x_scale_cpu.reserve(kWorldSize);
  oracle.expected_counts_cpu.reserve(kWorldSize);
  oracle.expected_expert_x_dequant_cpu.reserve(kWorldSize);
  oracle.expected_source_rank_cpu.reserve(kWorldSize);
  oracle.expected_source_dispatch_offset_cpu.reserve(kWorldSize);
  oracle.expected_combine_send_offset_cpu.reserve(kWorldSize);
  oracle.expected_padded_index_cpu.reserve(kWorldSize);
  oracle.route_positions_cpu.reserve(kWorldSize);
  oracle.expected_num_recv_tokens.reserve(kWorldSize);

  const uint32_t experts_per_rank = kNumExperts / kWorldSize;
  std::vector<std::vector<uint32_t>> routed_count_by_source(
      kWorldSize, std::vector<uint32_t>(kNumExperts, 0));
  std::vector<std::vector<std::vector<SourceRouteRef>>> routes_by_source(
      kWorldSize, std::vector<std::vector<SourceRouteRef>>(kNumExperts));

  for (int rank = 0; rank < kWorldSize; ++rank) {
    // Reproducible per-rank seed: different ranks get different data,
    // same seed every run so the oracle is consistent across processes.
    torch::manual_seed(static_cast<uint64_t>(rank) * 98761 + 42);
    auto input_cpu = make_float_input_cpu(cfg, rank);
    auto [fp8_cpu, scale_cpu] = test_utils::quantize_fp8_1d_block(
        input_cpu, Major::K, torch::Device(torch::kCPU));
    oracle.input_fp8_cpu.push_back(fp8_cpu.contiguous());
    oracle.input_scale_cpu.push_back(scale_cpu.contiguous());
    oracle.input_dequant_cpu.push_back(dequantize_fp8_blocks(
        oracle.input_fp8_cpu.back(), oracle.input_scale_cpu.back()));
    oracle.indices_cpu.push_back(make_indices_cpu(cfg, rank));
    oracle.weights_cpu.push_back(make_weights_cpu(cfg));
    oracle.route_positions_cpu.push_back(torch::full(
        {static_cast<int64_t>(cfg.num_tokens), static_cast<int64_t>(cfg.top_k)},
        -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)));

    auto indices = oracle.indices_cpu.back().accessor<int, 2>();
    std::vector<uint32_t> expert_offsets(kNumExperts, 0);
    std::vector<uint32_t> routed_seen(kNumExperts, 0);

    for (uint32_t token = 0; token < cfg.num_tokens; ++token) {
      for (uint32_t route = 0; route < cfg.top_k; ++route) {
        routed_count_by_source[rank][indices[token][route]] += 1;
      }
    }

    uint32_t running_offset = 0;
    for (int expert = 0; expert < kNumExperts; ++expert) {
      expert_offsets[expert] = running_offset;
      running_offset += routed_count_by_source[rank][expert];
    }

    for (uint32_t token = 0; token < cfg.num_tokens; ++token) {
      for (uint32_t route = 0; route < cfg.top_k; ++route) {
        const uint32_t expert = indices[token][route];
        routes_by_source[rank][expert].push_back(SourceRouteRef{
            token,
            route,
            expert_offsets[expert] + routed_seen[expert]++,
        });
      }
    }
  }

  {
    auto t = torch::zeros(
        {static_cast<int64_t>(kWorldSize * kNumExperts)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    auto acc = t.accessor<uint32_t, 1>();
    for (int dp_group = 0; dp_group < kWorldSize; ++dp_group)
      for (int expert = 0; expert < kNumExperts; ++expert)
        acc[dp_group * kNumExperts + expert] =
            routed_count_by_source[dp_group][expert];
    oracle.expected_num_routed_cpu = t.contiguous();
  }

  for (int target_rank = 0; target_rank < kWorldSize; ++target_rank) {
    const int32_t first_local_expert = target_rank * experts_per_rank;
    auto counts = torch::zeros(
        {static_cast<int64_t>(experts_per_rank)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    auto counts_acc = counts.accessor<uint32_t, 1>();

    for (int source_rank = 0; source_rank < kWorldSize; ++source_rank) {
      for (uint32_t local_expert = 0; local_expert < experts_per_rank;
           ++local_expert) {
        counts_acc[local_expert] +=
            routed_count_by_source[source_rank]
                                  [first_local_expert + local_expert];
      }
    }

    std::vector<int32_t> dst_group_offset(kWorldSize, 0);
    int32_t num_recv_tokens = 0;
    for (int peer_group = 0; peer_group < kWorldSize; ++peer_group) {
      for (uint32_t local_expert = 0; local_expert < experts_per_rank;
           ++local_expert) {
        num_recv_tokens +=
            routed_count_by_source[peer_group]
                                  [first_local_expert + local_expert];
      }
      for (int expert = 0; expert < first_local_expert; ++expert) {
        dst_group_offset[peer_group] +=
            routed_count_by_source[peer_group][expert];
      }
    }
    oracle.expected_num_recv_tokens.push_back(
        static_cast<uint32_t>(num_recv_tokens));

    std::vector<int32_t> padded_offsets(experts_per_rank, 0);
    for (uint32_t local_expert = 0; local_expert < experts_per_rank;
         ++local_expert) {
      padded_offsets[local_expert] =
          local_expert == 0
              ? 0
              : padded_offsets[local_expert - 1] +
                    align_to(counts_acc[local_expert - 1], cfg.expert_padding);
    }

    auto expert_x = torch::zeros({static_cast<int64_t>(oracle.max_recv_tokens),
                                  static_cast<int64_t>(cfg.hidden_dim)},
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat8_e4m3fn)
                                     .device(torch::kCPU));
    auto expert_x_scale = torch::zeros(
        {static_cast<int64_t>(oracle.max_recv_tokens),
         static_cast<int64_t>(cfg.hidden_dim / 128)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto source_rank = torch::empty(
        {static_cast<int64_t>(num_recv_tokens)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    auto source_dispatch_offset = torch::empty(
        {static_cast<int64_t>(num_recv_tokens)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    auto combine_send_offset = torch::empty(
        {static_cast<int64_t>(num_recv_tokens)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    auto padded_index = torch::empty(
        {static_cast<int64_t>(num_recv_tokens)},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU));
    std::vector<int32_t> seen_per_expert(experts_per_rank, 0);
    std::vector<int32_t> src_dispatch_count(kWorldSize, 0);
    std::vector<int32_t> src_combine_count(kWorldSize, 0);
    auto source_rank_acc = source_rank.accessor<uint32_t, 1>();
    auto source_dispatch_offset_acc =
        source_dispatch_offset.accessor<uint32_t, 1>();
    auto combine_send_offset_acc = combine_send_offset.accessor<uint32_t, 1>();
    auto padded_index_acc = padded_index.accessor<uint32_t, 1>();
    int32_t next_recv_slot = 0;

    auto route_group = [&](int peer_group, bool is_self_group) {
      auto positions =
          oracle.route_positions_cpu[peer_group].accessor<int32_t, 2>();
      for (uint32_t local_expert = 0; local_expert < experts_per_rank;
           ++local_expert) {
        const int32_t expert = first_local_expert + local_expert;
        for (const auto &entry : routes_by_source[peer_group][expert]) {
          const int32_t position =
              padded_offsets[local_expert] + seen_per_expert[local_expert]++;
          source_rank_acc[next_recv_slot] = static_cast<uint32_t>(peer_group);
          if (is_self_group) {
            source_dispatch_offset_acc[next_recv_slot] =
                static_cast<uint32_t>(entry.source_local_offset);
            combine_send_offset_acc[next_recv_slot] =
                static_cast<uint32_t>(entry.source_local_offset);
          } else {
            const int32_t index_on_rank = src_dispatch_count[peer_group]++;
            if (index_on_rank <
                static_cast<int32_t>(oracle.max_private_tokens)) {
              source_dispatch_offset_acc[next_recv_slot] =
                  static_cast<uint32_t>(oracle.max_private_tokens *
                                        peer_group) +
                  static_cast<uint32_t>(index_on_rank);
            } else {
              source_dispatch_offset_acc[next_recv_slot] =
                  static_cast<uint32_t>(dst_group_offset[peer_group] +
                                        index_on_rank) |
                  (1u << 31);
            }
            combine_send_offset_acc[next_recv_slot] = static_cast<uint32_t>(
                dst_group_offset[peer_group] + src_combine_count[peer_group]++);
          }
          padded_index_acc[next_recv_slot] = static_cast<uint32_t>(position);
          expert_x[position].copy_(
              oracle.input_fp8_cpu[peer_group][entry.token]);
          expert_x_scale[position].copy_(
              oracle.input_scale_cpu[peer_group][entry.token]);
          positions[entry.token][entry.route] = position;
          next_recv_slot += 1;
        }
      }
    };

    for (int local_group = 1; local_group < kWorldSize; ++local_group) {
      route_group((target_rank + local_group) % kWorldSize, false);
    }
    route_group(target_rank, true);

    if (next_recv_slot != num_recv_tokens) {
      throw std::runtime_error(
          "single-node oracle produced inconsistent recv count");
    }

    oracle.expected_counts_cpu.push_back(counts);
    oracle.expected_expert_x_cpu.push_back(expert_x.contiguous());
    oracle.expected_expert_x_scale_cpu.push_back(expert_x_scale.contiguous());
    oracle.expected_expert_x_dequant_cpu.push_back(
        dequantize_fp8_blocks(expert_x, expert_x_scale));
    oracle.expected_source_rank_cpu.push_back(source_rank.contiguous());
    oracle.expected_source_dispatch_offset_cpu.push_back(
        source_dispatch_offset.contiguous());
    oracle.expected_combine_send_offset_cpu.push_back(
        combine_send_offset.contiguous());
    oracle.expected_padded_index_cpu.push_back(padded_index.contiguous());
  }

  return oracle;
}

bool tensor_equal(const torch::Tensor &actual, const torch::Tensor &expected,
                  const std::string &what, std::ostringstream &error) {
  if (!torch::equal(actual, expected)) {
    auto a = actual.to(torch::kInt64);
    auto e = expected.to(torch::kInt64);
    auto diff = torch::abs(a - e);
    error << what << " mismatch: max diff=" << diff.max().item<int64_t>()
          << " at index "
          << torch::nonzero(diff == diff.max()).flatten()[0].item<int64_t>();
    return false;
  }

  return true;
}

bool tensor_close(const torch::Tensor &actual, const torch::Tensor &expected,
                  double atol, double rtol, const std::string &what,
                  std::ostringstream &error) {
  if (!torch::allclose(actual, expected, atol, rtol)) {
    error << what
          << " mismatch: max diff=" << torch::abs(actual - expected).max()
          << " at index "
          << torch::nonzero(torch::abs(actual - expected) ==
                            torch::abs(actual - expected).max())
                 .flatten()[0]
                 .item<int64_t>();
    printf("Reference \n");
    test_utils::inspect_tensor(expected, 5);
    printf("Actual \n");
    test_utils::inspect_tensor(actual, 5);
    return false;
  }
  return true;
}

// Hashes a token as raw fp8 bytes concatenated with its scale bytes.
// Two tokens are "equal" iff both their fp8 values and scales are identical.
static std::string hash_token(const torch::Tensor &fp8_row,
                              const torch::Tensor &scale_row) {
  auto fp8_c = fp8_row.contiguous();
  auto scl_c = scale_row.contiguous();
  std::string key;
  key.append(static_cast<const char *>(fp8_c.data_ptr()), fp8_c.nbytes());
  key.append(static_cast<const char *>(scl_c.data_ptr()), scl_c.nbytes());
  return key;
}

// Mirrors pplx-garden's set-membership check: verifies that every input token
// routed to this rank's local experts appears somewhere in the dispatch output,
// without caring about the order of tokens within each expert's slot.
bool check_dispatch_set_membership(
    const TestCase &cfg, int rank, const OracleBundle &oracle,
    const torch::Tensor &actual_expert_x_cpu, // [max_recv_tokens, H] fp8
    const torch::Tensor
        &actual_expert_x_scale_cpu,         // [max_recv_tokens, H/128] f32
    const torch::Tensor &actual_counts_cpu, // [experts_per_rank] u32
    std::ostringstream &error) {

  const uint32_t experts_per_rank = kNumExperts / kWorldSize;
  const int32_t first_local_expert =
      rank * static_cast<int32_t>(experts_per_rank);

  // Collect hashes of every token that actually arrived at this rank.
  std::unordered_set<std::string> tokens_on_rank;
  auto counts_acc = actual_counts_cpu.accessor<uint32_t, 1>();
  int64_t slot = 0;
  for (uint32_t le = 0; le < experts_per_rank; ++le) {
    uint32_t n = counts_acc[le];
    for (int64_t i = slot; i < slot + static_cast<int64_t>(n); ++i) {
      tokens_on_rank.insert(
          hash_token(actual_expert_x_cpu[i], actual_expert_x_scale_cpu[i]));
    }
    slot = static_cast<int64_t>(
        align_to(static_cast<uint32_t>(slot + n), cfg.expert_padding));
  }

  // For every source rank, check that each token routed to a local expert is
  // present.
  int num_missing = 0;
  for (int src = kWorldSize - 1; src >= 0; --src) {
    auto indices = oracle.indices_cpu[src].accessor<int, 2>();
    for (uint32_t token = 0; token < cfg.num_tokens; ++token) {
      for (uint32_t route = 0; route < cfg.top_k; ++route) {
        const uint32_t expert = indices[token][route];
        if (static_cast<int32_t>(expert) < first_local_expert ||
            static_cast<int32_t>(expert) >=
                first_local_expert + static_cast<int32_t>(experts_per_rank))
          continue;

        const std::string key = hash_token(oracle.input_fp8_cpu[src][token],
                                           oracle.input_scale_cpu[src][token]);

        if (tokens_on_rank.find(key) == tokens_on_rank.end()) {
          ++num_missing;
          // if (num_missing <= 5) {
          error << "\n  missing token " << token << " from rank " << src
                << " routed to expert " << expert;
          // }
        }
      }
    }
  }

  if (num_missing > 0) {
    error << "\n"
          << cfg.name << " dispatch set check: " << num_missing
          << " token(s) missing on rank " << rank;
    return false;
  }
  return true;
}

// bool check_combine_set_membership(
//     const TestCase& cfg,
//     int rank,
//     const OracleBundle& oracle,
//     const at::Tensor& combine_output,
//     const at::Tensor& combine_reference,
//     const at::Tensor& actual_counts_cpu,
//     std::ostringstream& error
// ) {
//     const uint32_t experts_per_rank = kNumExperts / kWorldSize;
//     const int32_t  first_local_expert = rank *
//     static_cast<int32_t>(experts_per_rank); std::unordered_set<std::string>
//     output_tokens_on_rank, reference_tokens_on_rank;

//     auto counts_acc = actual_counts_cpu.accessor<int32_t, 1>();

//     int32_t index = 0;
//     for (int expert = 0; expert < experts_per_rank; expert++) {
//         uint32_t n = counts_acc[expert];
//         for (int s = index; s < index + n; s++) {
//             output_tokens_on_rank.insert(hash_tensor(combine_output));
//         }
//         index = align_to(index + n, cfg.expert_padding);
//     }

// }

template <int TopK, int HiddenDim>
bool run_case_for_rank(const TestCase &cfg, int rank, bool verbose,
                       std::ostringstream &error) {
  static_assert(HiddenDim % 128 == 0);
  CUDA_CHECK(cudaSetDevice(rank));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  printf("Initializing All2All for rank %d + setup\n", rank);
  using All2AllT = moe_cuda::All2All<TopK, kNumExperts, HiddenDim>;

  const OracleBundle oracle = build_oracle(cfg);
  const auto device = torch::Device(torch::kCUDA, rank);
  const auto fp8_opts =
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
  const auto f32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto u32_opts =
      torch::TensorOptions().dtype(torch::kUInt32).device(device);
  const uint32_t experts_per_rank = kNumExperts / kWorldSize;

  ParallelConfig parallel_config{
      .tp_size = 1,
      .dp_size = 1,
      .ep_size = kWorldSize,
      .node_size = kWorldSize,
      .world_size = kWorldSize,
  };

  auto dp_x = oracle.input_fp8_cpu[rank].to(device);
  auto dp_x_scale = oracle.input_scale_cpu[rank].t().contiguous().to(device);
  auto indices = oracle.indices_cpu[rank].to(device);
  auto weights = oracle.weights_cpu[rank].to(device);

  auto out_expert_x =
      torch::zeros({static_cast<int64_t>(oracle.max_recv_tokens),
                    static_cast<int64_t>(cfg.hidden_dim)},
                   fp8_opts);
  auto out_expert_x_scale =
      torch::zeros({static_cast<int64_t>(cfg.hidden_dim / 128),
                    static_cast<int64_t>(oracle.max_recv_tokens)},
                   f32_opts);
  auto out_expert_num_tokens =
      torch::zeros({static_cast<int64_t>(experts_per_rank)}, u32_opts);

  auto final_out_tokens = torch::empty_like(dp_x).to(torch::kBFloat16);

  std::optional<at::Tensor> bound_m = std::nullopt;
  std::optional<at::Tensor> dp_x_scale_opt = dp_x_scale;
  std::optional<at::Tensor> out_expert_x_scale_opt = out_expert_x_scale;

  All2AllT all2all(cfg.num_tokens, kNumExperts, cfg.expert_padding,
                   cfg.hidden_dim, std::nullopt, c10::ScalarType::Float8_e4m3fn,
                   c10::ScalarType::BFloat16, c10::ScalarType::Float, TopK,
                   std::nullopt, rank, parallel_config, stream);

  all2all.dispatch(out_expert_num_tokens, out_expert_x, out_expert_x_scale_opt,
                   dp_x, dp_x_scale_opt, indices, weights, bound_m, true, true,
                   stream);
  auto routing_state = all2all.debug_routing_state(stream);
  auto expert_y =
      dequantize_fp8_blocks(out_expert_x, out_expert_x_scale.t().contiguous())
          .to(torch::kBFloat16);

  all2all.combine(final_out_tokens, indices, weights, expert_y, bound_m, true,
                  true, false, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  auto actual_counts =
      out_expert_num_tokens.to(torch::kCPU).to(torch::kUInt32).contiguous();
  auto actual_expert_x = out_expert_x.to(torch::kCPU).contiguous();
  auto actual_expert_x_scale =
      out_expert_x_scale.to(torch::kCPU).t().contiguous();
  auto actual_dequant =
      dequantize_fp8_blocks(actual_expert_x, actual_expert_x_scale);
  auto actual_source_rank =
      routing_state.source_rank.to(torch::kCPU).to(torch::kUInt32).contiguous();
  auto actual_source_dispatch_offset =
      routing_state.source_dispatch_offset.to(torch::kCPU)
          .to(torch::kUInt32)
          .contiguous();
  auto actual_combine_send_offset =
      routing_state.combine_send_offset.to(torch::kCPU)
          .to(torch::kUInt32)
          .contiguous();
  auto actual_padded_index = routing_state.padded_index.to(torch::kCPU)
                                 .to(torch::kUInt32)
                                 .contiguous();
  auto actual_num_routed = all2all.debug_num_routed();

  if (!tensor_equal(actual_num_routed, oracle.expected_num_routed_cpu,
                    std::string(cfg.name) + " num_routed", error)) {
    return false;
  }

  if (routing_state.num_recv_tokens != oracle.expected_num_recv_tokens[rank]) {
    error << cfg.name << " num_recv_tokens mismatch: actual="
          << routing_state.num_recv_tokens
          << " expected=" << oracle.expected_num_recv_tokens[rank];
    return false;
  }
  if (static_cast<uint32_t>(actual_counts.sum().item<int64_t>()) !=
      routing_state.num_recv_tokens) {
    error << cfg.name << " counts sum does not match num_recv_tokens";
    return false;
  }

  if (!tensor_equal(actual_counts, oracle.expected_counts_cpu[rank],
                    std::string(cfg.name) + " counts", error)) {
    return false;
  }
  if (!tensor_equal(actual_source_rank, oracle.expected_source_rank_cpu[rank],
                    std::string(cfg.name) + " source_rank", error)) {
    return false;
  }
  // source_dispatch_offset and padded_index are order-dependent within each
  // expert group (kernel uses non-deterministic atomicAdd ordering), so we
  // skip exact positional checks here and rely on set-membership below.
  (void)actual_source_dispatch_offset;
  (void)actual_combine_send_offset;
  (void)actual_padded_index;
  if (!check_dispatch_set_membership(cfg, rank, oracle, actual_expert_x,
                                     actual_expert_x_scale, actual_counts,
                                     error)) {
    return false;
  }

  auto route_positions =
      oracle.route_positions_cpu[rank].accessor<int32_t, 2>();
  auto input_dequant = oracle.input_dequant_cpu[rank].to(torch::kBFloat16);
  auto weights_cpu = oracle.weights_cpu[rank].accessor<float, 2>();
  auto indices_cpu = oracle.indices_cpu[rank].accessor<int, 2>();

  auto final_out_tokens_cpu = final_out_tokens.to(torch::kCPU);
  auto actual_dequant_cpu = actual_dequant.to(torch::kCPU);

  if (!tensor_close(input_dequant, final_out_tokens_cpu, 1e-4, 1e-4,
                    std::string(cfg.name) + " combine oracle", error)) {
    return false;
  }
  // if (!test_utils::check_tensor_close(input_dequant, final_out_tokens_cpu,
  // 1e-4, 1e-4)) {
  //     return false;
  // }

  if (verbose) {
    std::cout << "[Rank " << rank << "] " << cfg.name << " passed\n";
  }
  return true;
}

bool run_rank(int rank, bool verbose, std::ostringstream &error,
              SharedResults *shared) {
  if (rank < 0 || rank >= kWorldSize) {
    error << "invalid rank " << rank;
    return false;
  }

  int num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));

  CUDA_CHECK(cudaSetDevice(rank));
  for (int peer = 0; peer < kWorldSize; ++peer) {
    if (peer != rank)
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
  }
  if (num_devices < kWorldSize) {
    error << "need " << kWorldSize << " GPUs, found " << num_devices;
    return false;
  }

  // Barrier to synchronize all processes between test cases, preventing
  // broker.sync() races where faster ranks start constructing the next
  // All2AllT (which calls broker.sync()) before slower ranks finish their
  // oracle checks from the previous case.
  auto inter_process_barrier = [&](int barrier_idx) {
    shared->barrier_count.fetch_add(1, std::memory_order_acq_rel);
    while (shared->barrier_count.load(std::memory_order_acquire) <
           kWorldSize * (barrier_idx + 1)) {
      std::this_thread::yield();
    }
  };

  bool passed = true;
  int case_idx = 0;
  for (const auto &cfg : kCases) {
    bool case_passed;
    if (cfg.top_k == 1 && cfg.hidden_dim == 128) {
      case_passed = run_case_for_rank<1, 128>(cfg, rank, verbose, error);
    } else if (cfg.top_k == 2 && cfg.hidden_dim == 256) {
      case_passed = run_case_for_rank<2, 256>(cfg, rank, verbose, error);
    } else if (cfg.top_k == 10 && cfg.hidden_dim == 2048) {
      case_passed = run_case_for_rank<10, 2048>(cfg, rank, verbose, error);
    } else if (cfg.top_k == 12 && cfg.hidden_dim == 2048) {
      case_passed = run_case_for_rank<12, 2048>(cfg, rank, verbose, error);
    } else {
      error << "unsupported test config " << cfg.name;
      case_passed = false;
    }
    passed &= case_passed;

    // All processes must reach this point before any can start the next case.
    if (case_idx + 1 < static_cast<int>(kCases.size())) {
      inter_process_barrier(case_idx);
    }
    case_idx++;
  }
  return passed;
}

void write_result(RankResult &slot, bool passed, const std::string &message) {
  slot.passed = passed ? 1 : 0;
  std::snprintf(slot.message, sizeof(slot.message), "%s", message.c_str());
}

} // namespace

int main(int argc, char **argv) {
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--verbose]\n";
      return 0;
    }
  }

  auto *shared = static_cast<SharedResults *>(
      mmap(nullptr, sizeof(SharedResults), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  std::memset(shared, 0, sizeof(SharedResults));

  std::array<pid_t, kWorldSize> children{};
  for (int rank = 0; rank < kWorldSize; ++rank) {
    const pid_t pid = fork();
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    if (pid == 0) {
      std::ostringstream error;
      const bool passed = run_rank(rank, verbose, error, shared);
      write_result(shared->results[rank], passed,
                   passed ? "passed" : error.str());
      std::_Exit(passed ? 0 : 1);
    }
    children[rank] = pid;
  }

  int failed = 0;
  for (int rank = 0; rank < kWorldSize; ++rank) {
    int status = 0;
    if (waitpid(children[rank], &status, 0) < 0) {
      std::perror("waitpid");
      failed += 1;
      continue;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      failed += 1;
    }
  }

  std::cout << "============================================\n";
  std::cout << "TK moe_cuda::All2All Dispatch Test\n";
  std::cout << "World size: " << kWorldSize << "\n";
  std::cout << "============================================\n";
  for (int rank = 0; rank < kWorldSize; ++rank) {
    const bool passed = shared->results[rank].passed != 0;
    std::cout << "[Rank " << rank << "] " << (passed ? "PASSED" : "FAILED")
              << ": " << shared->results[rank].message << "\n";
  }

  munmap(shared, sizeof(SharedResults));
  return failed == 0 ? 0 : 1;
}
