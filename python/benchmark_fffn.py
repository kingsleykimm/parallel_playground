"""
@file: Benchmarking different types of swiglu FFN combinations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import moe_cuda
from .common import quantize_1d_128, quantize_2d_128
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


class TESwiglu(nn.Module):
    """
    Single-expert SwiGLU FFN using te.LayerNormMLP.
    Includes a LayerNorm before the FFN (standard pre-norm transformer block).
    For correctness comparison, either account for LayerNorm in your custom path,
    or set return_layernorm_output=True and compare only the FFN output.

    te.LayerNormMLP weight layout (activation='swiglu'):
      fc1_weight: [2 * ffn_hidden_size, hidden_size]  -- gate rows first, then up rows
      fc2_weight: [hidden_size, ffn_hidden_size]
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fp8_recipe: DelayedScaling | None = None,
    ):
        super().__init__()
        self.fp8_recipe = fp8_recipe
        self.layer = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            activation="swiglu",
            bias=False,
        )

    def load_weights(
        self,
        gate_weight: torch.Tensor,  # [N, K]
        up_weight: torch.Tensor,    # [N, K]
        down_weight: torch.Tensor,  # [K, N]
    ):
        with torch.no_grad():
            self.layer.fc1_weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
            self.layer.fc2_weight.copy_(down_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with te.fp8_autocast(enabled=self.fp8_recipe is not None, fp8_recipe=self.fp8_recipe):
            return self.layer(x)


def moe_cuda_swiglu_pp(
    input: torch.Tensor,
    input_scales: torch.Tensor,
    gate: torch.Tensor,
    gate_scales: torch.Tensor,
    up: torch.Tensor,
    up_scales: torch.Tensor,
    gemm_type: moe_cuda.GemmType,
    D: torch.Tensor,
) -> torch.Tensor:
    return moe_cuda.swiglu_pp(
        input, input_scales,
        gate, gate_scales,
        up, up_scales,
        gemm_type, D,
    )


@torch.no_grad()
def run(
    M: int,
    N: int,
    K: int,
    rank: int,
):
    device = f"cuda:{rank}"

    inputs = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    up_weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    down_weight = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
    )

    # --- TE BF16 baseline (no FP8, LayerNorm is identity on random inputs — just for shape check) ---
    te_model_bf16 = TESwiglu(K, N, fp8_recipe=None).to(device)
    te_model_bf16.load_weights(gate_weight, up_weight, down_weight)
    te_out_bf16 = te_model_bf16(inputs)

    # --- TE FP8 ---
    te_model_fp8 = TESwiglu(K, N, fp8_recipe=fp8_recipe).to(device)
    te_model_fp8.load_weights(gate_weight, up_weight, down_weight)
    te_out_fp8 = te_model_fp8(inputs)

    # --- Custom CUDA FP8 path ---
    input_q, input_scales = quantize_1d_128(inputs)
    gate_q, gate_scales = quantize_2d_128(gate_weight)
    up_q, up_scales = quantize_2d_128(up_weight)
    down_q, down_scales = quantize_2d_128(down_weight)

    # custom_out = moe_cuda_swiglu_pp(
    #     input_q, input_scales,
    #     gate_q, gate_scales,
    #     up_q, up_scales,
    #     moe_cuda.GemmType.FP8, down_q,
    # )

    # --- Correctness ---
    diff_fp8_vs_bf16 = (te_out_fp8.float() - te_out_bf16.float()).abs().mean().item()
    print(f"[TE] FP8 vs BF16 mean abs diff: {diff_fp8_vs_bf16:.6f}")

    # Uncomment once custom_out is wired up:
    # diff_custom_vs_bf16 = (custom_out.float() - te_out_bf16.float()).abs().mean().item()
    # print(f"[Custom] FP8 vs TE BF16 mean abs diff: {diff_custom_vs_bf16:.6f}")


if __name__ == "__main__":
    run(M=4096, N=8192, K=4096, rank=0)
