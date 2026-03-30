"""
@file: Benchmarking different types of swiglu FFN combinations
"""
import torch
import torch.nn as nn
import moe_cuda
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from transformer_engine_torch import DType as TE_DType
import transformer_engine_torch as tex

from common import enumerate_grouped_gemms, generate_m_grouped_contiguous, generate_m_grouped_masked, setup, benchmark_no_l2_clear


class TESwiglu(nn.Module):
    """
    Grouped SwiGLU (fc1 only) using te.GroupedLinear with blockwise FP8 scaling.

    GroupedLinear stores one weight parameter per expert: weight0, weight1, ...
    Each weight{i} has shape [intermediate_size * 2, hidden_size] (gate + up packed).

    Float8BlockScaling defaults:
      x_block_scaling_dim=1  →  1×128 rowwise for activations
      w_block_scaling_dim=2  →  128×128 blockwise for weights
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_groups: int,
    ):
        super().__init__()
        self.recipe = Float8BlockScaling(
            use_f32_scales=True,
            x_block_scaling_dim=1,
            w_block_scaling_dim=2,
        )
        self.quantizer_1d = Float8BlockQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,  # activations are never transposed in fprop
            block_scaling_dim=1,
        )
        self.fc1 = te.GroupedLinear(
            num_groups, hidden_size, intermediate_size * 2, bias=False, params_dtype=torch.bfloat16)

    def load_weights(
        self,
        gate_weight: torch.Tensor,  # [G, I, H]
        up_weight: torch.Tensor,    # [G, I, H]
    ):
        """Copy BF16 per-expert weights into weight0..weight{G-1}."""
        with torch.no_grad():
            for i in range(self.fc1.num_gemms):
                # weight{i}: [2*I, H] — gate rows first, then up rows
                getattr(self.fc1, f"weight{i}").copy_(
                    torch.cat([gate_weight[i], up_weight[i]], dim=0)
                )

    def forward(self, x: torch.Tensor, m_splits: list[int]) -> torch.Tensor:
        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            x = self.fc1(inp=x, m_splits=m_splits)
            return tex.swiglu(x, quantizer=self.quantizer_1d)


def unfused_swiglu_mlp(
    *,
    cfg : torch.Tensor,
    input : torch.Tensor,
    input_scales : torch.Tensor,
    weight : torch.Tensor,
    weight_scales : torch.Tensor,
    gemm_type : moe_cuda.GemmType,
    D : torch.Tensor,
    scale_d : torch.Tensor,
    grouped_layout : torch.Tensor,
) -> torch.Tensor:
    moe_cuda.fp8_grouped_gemm_nt(
        input, input_scales,
        weight, weight_scales,
        D, gemm_type, grouped_layout)
    swiglu_out = torch.empty(shape = (cfg.B * cfg.S, cfg.I), dtype = torch.float8_e4m3fn, device = input.device)
    swiglu_out_scales = torch.empty(shape = (cfg.B * cfg.S, cfg.I // 128), dtype = torch.float, device = input.device)
    moe_cuda.fused_silu_mul_quant(
        gemm_out = D, swiglu_out = swiglu_out, scale = swiglu_out_scales
    )
    dequantized = (swiglu_out.float() * swiglu_out_scales.repeat_interleave(repeats = 128, dim = -1)).to(torch.bfloat16)
    return dequantized

def non_interleaved_moe_cuda_swiglu(
    *,
    input : torch.Tensor,
    scale_a : torch.Tensor,
    gate : torch.Tensor,
    scale_gate : torch.Tensor,
    up : torch.Tensor,
    scale_up : torch.Tensor,
    D : torch.Tensor,
    scale_d : torch.Tensor,
    gemm_type : moe_cuda.GemmType,
    grouped_layout : torch.Tensor,
) -> torch.Tensor:
    return moe_cuda.fp8_grouped_gemm_swiglu_sub(
        input, scale_a, gate, scale_gate, up, scale_up, scale_d, D, gemm_type, grouped_layout
    )

def moe_cuda_swiglu_pp(
    *,
    input: torch.Tensor,
    input_scales: torch.Tensor,
    gate: torch.Tensor,
    gate_scales: torch.Tensor,
    up: torch.Tensor,
    up_scales: torch.Tensor,
    gemm_type: moe_cuda.GemmType,
    D: torch.Tensor,
    scale_d: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    return moe_cuda.fp8_grouped_gemm_swiglu_pp(
        input, input_scales,
        gate, gate_scales,
        up, up_scales,
        D, scale_d, gemm_type, grouped_layout
    )


def moe_cuda_swiglu_coop(
    *,
    input: torch.Tensor,
    input_scales: torch.Tensor,
    gate: torch.Tensor,
    gate_scales: torch.Tensor,
    up: torch.Tensor,
    up_scales: torch.Tensor,
    gemm_type: moe_cuda.GemmType,
    D: torch.Tensor,
    scale_d: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    return moe_cuda.fp8_grouped_gemm_swiglu(
        input, input_scales,
        gate, gate_scales,
        up, up_scales,
        D, scale_d, gemm_type, grouped_layout
    )


@torch.no_grad()
def _prepare_inputs(
    *,
    num_groups: int,
    expected_m_per_group: int,
    max_m: int,
    H: int,
    I: int,
    rank: int,
    gemm_type: moe_cuda.GemmType,
):
    device = f"cuda:{rank}"
    if gemm_type == moe_cuda.GemmType.MGroupedContiguous:
        A, up_weight, gate_weight, grouped_layout, D, scale_D, aligned_ms = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, I, H)
        m_splits = aligned_ms
    elif gemm_type == moe_cuda.GemmType.MGroupedMasked:
        A, up_weight, gate_weight, grouped_layout, D, scale_D = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, I, H)
        m_splits = [max_m] * num_groups
    else:
        assert False, "not supported gem type for now"

    quantizer_1d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=1,
    )
    quantizer_2d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )

    te_model = TESwiglu(hidden_size=H, intermediate_size=I,
                        num_groups=num_groups).to(device)
    te_model.load_weights(gate_weight, up_weight)

    inputs_q_tensor: Float8BlockwiseQTensor = quantizer_1d(A)
    gate_q_tensor: Float8BlockwiseQTensor = quantizer_2d(gate_weight)
    up_q_tensor: Float8BlockwiseQTensor = quantizer_2d(up_weight)

    inputs_q, inputs_scales = inputs_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), inputs_q_tensor._rowwise_scale_inv
    gate_q, gate_q_scales = gate_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), gate_q_tensor._rowwise_scale_inv
    up_q, up_q_scales = up_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), up_q_tensor._rowwise_scale_inv

    return dict(
        A=A, up_weight=up_weight, gate_weight=gate_weight,
        grouped_layout=grouped_layout, D=D, scale_D=scale_D,
        m_splits=m_splits, te_model=te_model,
        inputs_q=inputs_q, inputs_scales=inputs_scales,
        gate_q=gate_q, gate_q_scales=gate_q_scales,
        up_q=up_q, up_q_scales=up_q_scales,
    )


@torch.no_grad()
def run(
    *,
    num_groups: int,
    expected_m_per_group: int,
    max_m: int,
    H: int,  # hidden
    I: int,  # intermediate
    rank: int,
    gemm_type: moe_cuda.GemmType,
    do_profile: bool = False
):
    print(
        f"Starting run for config num_groups: {num_groups}, gemm_type: {gemm_type}, expected_m_per_group: {expected_m_per_group}, H: {H}, I: {I}")

    inputs = _prepare_inputs(
        num_groups=num_groups, expected_m_per_group=expected_m_per_group,
        max_m=max_m, H=H, I=I, rank=rank, gemm_type=gemm_type)

    inputs_q = inputs["inputs_q"]
    inputs_scales = inputs["inputs_scales"]
    gate_q = inputs["gate_q"]
    gate_q_scales = inputs["gate_q_scales"]
    up_q = inputs["up_q"]
    up_q_scales = inputs["up_q_scales"]
    D = inputs["D"]
    scale_D = inputs["scale_D"]
    grouped_layout = inputs["grouped_layout"]

    def coop_run(): return moe_cuda_swiglu_coop(
        input=inputs_q,
        input_scales=inputs_scales,
        gate=gate_q,
        gate_scales=gate_q_scales, up=up_q, up_scales=up_q_scales, gemm_type=gemm_type, D=D, scale_d=scale_D,
        grouped_layout=grouped_layout,
    )

    D_copy = torch.empty_like(D)
    scale_D_copy = torch.empty_like(scale_D)

    def pingpong_run(): return moe_cuda_swiglu_pp(
        input=inputs_q,
        input_scales=inputs_scales,
        gate=gate_q,
        gate_scales=gate_q_scales,
        up=up_q,
        up_scales=up_q_scales,
        gemm_type=gemm_type,
        D=D_copy, scale_d=scale_D_copy, grouped_layout=grouped_layout
    )

    if do_profile:
        custom_avg_ms = benchmark_no_l2_clear(
            coop_run, num_warmup_iters=1, num_iters=20)
        pingpong_avg_ms = benchmark_no_l2_clear(
            pingpong_run, num_warmup_iters=1, num_iters=20)
        print(
            f"Coop kernel average time: {custom_avg_ms:.3f} ms for config num_groups: {num_groups}, expected_m_per_group: {expected_m_per_group}, H: {H}, I: {I}")
        print(
            f"Pingpong kernel average time: {pingpong_avg_ms:.3f} ms for config num_groups: {num_groups}, expected_m_per_group: {expected_m_per_group}, H: {H}, I: {I}")


    # Profiling
if __name__ == "__main__":
    setup()

    for num_groups, expected_m_per_group, n, k, gemm_type in enumerate_grouped_gemms():
        run(num_groups=num_groups,
            max_m=8192,
            expected_m_per_group=expected_m_per_group,
            H=k,
            I=n,
            gemm_type=gemm_type, rank=0, do_profile=True)
