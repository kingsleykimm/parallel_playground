"""
@file: Benchmarking different types of swiglu FFN combinations
"""
import argparse
from typing import Any
import torch
import torch.nn as nn
import moe_cuda
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from transformer_engine_torch import DType as TE_DType
import transformer_engine_torch as tex

from common import enumerate_grouped_gemms, generate_m_grouped_contiguous, generate_m_grouped_masked, quantize_1d_128, setup, benchmark_no_l2_clear, calc_cosine_diff, check_diff


def reference_fused_silu_mul_quant(input : torch.Tensor):
    # input here is shape [M, 2 * N]
    # output here is shape [M, N]
    gate, up = input.chunk(2, dim = -1)
    gate = torch.nn.functional.silu(gate)
    output = gate * up
    return output

def test_silu_mul_quant(input : torch.Tensor, output : torch.Tensor, output_scale : torch.Tensor):
    moe_cuda.fused_silu_mul_quant(gemm_out = input, swiglu_out = output, scale = output_scale)

    ref_output = reference_fused_silu_mul_quant(input)
    print(output.shape, output_scale.shape)
    dequantized_kernel = output.to(torch.float) * output_scale.repeat_interleave(repeats=  128, dim = -1)
    dequantized_kernel = dequantized_kernel.to(torch.bfloat16)

    check_diff("dequantized_kernel", dequantized_kernel, ref_output, single = True)

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
        # Run without fp8_autocast so TE does not re-quantize A and weights
        # independently — that would produce a different FP8 path than the kernel,
        # causing systematic ~FP8-precision divergence that exceeds the 0.01 threshold.
        x = self.fc1(inp=x, m_splits=m_splits)   # BF16 GEMM
        gate, up = x.chunk(2, dim=-1)
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def unfused_swiglu_mlp(
    *,
    input : torch.Tensor,
    input_scales : torch.Tensor,
    weight : torch.Tensor,
    weight_scales : torch.Tensor,
    gemm_type : moe_cuda.GemmType,
    D : torch.Tensor,
    grouped_layout : torch.Tensor,
    swiglu_out : torch.Tensor,
    swiglu_out_scales : torch.Tensor,
) -> torch.Tensor:
    moe_cuda.fp8_grouped_gemm_nt(
        input, input_scales,
        weight, weight_scales,
        D, gemm_type, grouped_layout)
    moe_cuda.fused_silu_mul_quant(
        gemm_out = D, swiglu_out = swiglu_out, scale = swiglu_out_scales
    )

def non_interleaved_moe_cuda_swiglu_coop(
    *,
    input : torch.Tensor,
    input_scales : torch.Tensor,
    gate : torch.Tensor,
    gate_scales : torch.Tensor,
    up : torch.Tensor,
    up_scales : torch.Tensor,
    D : torch.Tensor,
    scale_d : torch.Tensor,
    gemm_type : moe_cuda.GemmType,
    grouped_layout : torch.Tensor,
) -> torch.Tensor:
    return moe_cuda.fp8_grouped_gemm_swiglu_sub(
        input, input_scales, gate, gate_scales, up, up_scales, D, scale_d, gemm_type, grouped_layout
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

function_maps = {
    "coop": moe_cuda_swiglu_coop,
    "pp": moe_cuda_swiglu_pp,
    "non_interleaved": non_interleaved_moe_cuda_swiglu_coop,
    "unfused": unfused_swiglu_mlp,
}


@torch.no_grad()
def manual_swiglu_ref(
    *,
    A: torch.Tensor,            # (total_M, H) bf16
    gate_weight: torch.Tensor,  # (G, I, H)   bf16
    up_weight: torch.Tensor,    # (G, I, H)   bf16
    grouped_layout: torch.Tensor,  # (total_M,) int32  [contiguous: group idx / -1]
                                   # (G,)       int32  [masked: actual_m per group]
    gemm_type: moe_cuda.GemmType,
    num_groups: int,
    max_m: int = 0,             # required for MGroupedMasked
) -> torch.Tensor:
    """
    Exact Python port of compute_swiglu_ref from test_kernel3.cpp.
    Computes SwiGLU in float32 using raw BF16 weights, so there is no
    FP8 weight-quantisation error masking kernel bugs.

    SwiGLU:  silu(gate_out) * up_out  =  gate_out * sigmoid(gate_out) * up_out
    """
    total_M = A.shape[0]
    I = gate_weight.shape[1]
    device = A.device

    A_f32        = A.float()
    gate_f32     = gate_weight.float()   # (G, I, H)
    up_f32       = up_weight.float()

    out = torch.zeros((total_M, I), dtype=torch.bfloat16, device=device)

    if gemm_type == moe_cuda.GemmType.MGroupedContiguous:
        # grouped_layout[row] == g for valid rows of group g, -1 for padding
        for g in range(num_groups):
            row_mask = (grouped_layout == g)
            if not row_mask.any():
                continue
            A_g      = A_f32[row_mask]                  # (m_g, H)
            gate_out = A_g @ gate_f32[g].t()            # (m_g, I)
            up_out   = A_g @ up_f32[g].t()              # (m_g, I)
            swiglu   = gate_out * torch.sigmoid(gate_out) * up_out
            out[row_mask] = swiglu.to(torch.bfloat16)

    else:  # MGroupedMasked
        # grouped_layout[g] == actual number of valid rows for group g
        for g in range(num_groups):
            actual_m = grouped_layout[g].item()
            if actual_m == 0:
                continue
            start    = g * max_m
            A_g      = A_f32[start : start + actual_m]  # (actual_m, H)
            gate_out = A_g @ gate_f32[g].t()            # (actual_m, I)
            up_out   = A_g @ up_f32[g].t()              # (actual_m, I)
            swiglu   = gate_out * torch.sigmoid(gate_out) * up_out
            out[start : start + actual_m] = swiglu.to(torch.bfloat16)

    return out.to(torch.bfloat16)

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
    function_name: str,
):
    device = f"cuda:{rank}"
    if gemm_type == moe_cuda.GemmType.MGroupedContiguous:
        A, up_weight, gate_weight, grouped_layout, D, scale_D, aligned_ms, actual_ms = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, I, H)
        m_splits = aligned_ms
    elif gemm_type == moe_cuda.GemmType.MGroupedMasked:
        A, up_weight, gate_weight, grouped_layout, D, scale_D = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, I, H)
        m_splits = [max_m] * num_groups
    else:
        assert False, "not supported gem type for now"
    if function_name == "unfused":
        D = D.to(torch.bfloat16)
    

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
        actual_ms=actual_ms if gemm_type == moe_cuda.GemmType.MGroupedContiguous else None,
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
    function_name: str,
    do_profile: bool = False,
    do_ncu_profile: bool = False,
    check_correctness : bool = False
):
    print(
        f"Starting run for config num_groups: {num_groups}, gemm_type: {gemm_type}, expected_m_per_group: {expected_m_per_group}, H: {H}, I: {I}")

    inputs = _prepare_inputs(
        num_groups=num_groups, expected_m_per_group=expected_m_per_group,
        max_m=max_m, H=H, I=I, rank=rank, gemm_type=gemm_type, function_name=function_name)
    A = inputs["A"]
    m_splits = inputs["m_splits"]
    inputs_q = inputs["inputs_q"]
    inputs_scales = inputs["inputs_scales"]
    gate_q = inputs["gate_q"]
    gate_q_scales = inputs["gate_q_scales"]
    up_q = inputs["up_q"]
    up_q_scales = inputs["up_q_scales"]
    D = inputs["D"]
    scale_D = inputs["scale_D"]
    grouped_layout = inputs["grouped_layout"]
    D_copy = torch.empty_like(D)
    scale_D_copy = torch.empty_like(scale_D)
    te_model = inputs["te_model"]
    gate_weight = inputs["gate_weight"]
    up_weight = inputs["up_weight"]
    actual_ms = inputs["actual_ms"]  # list[int] for contiguous, None for masked
    if function_name == "unfused":
        if gemm_type == moe_cuda.GemmType.MGroupedMasked:
            inputs_q_kernel = inputs_q.view(num_groups, max_m, H)
        else:
            inputs_q_kernel = inputs_q
        concat_weight = torch.cat([gate_q, up_q], dim = -2)
        concat_weight_scales = torch.cat([gate_q_scales, up_q_scales], dim = -2)
        D = torch.empty((inputs_q.shape[0], I * 2), dtype = torch.bfloat16, device = inputs_q.device)
        
        swiglu_out = torch.empty((inputs_q.shape[0], I), dtype = torch.float8_e4m3fn, device = inputs_q.device)
        swiglu_out_scales = torch.empty((inputs_q.shape[0], I // 128), dtype = torch.float, device = inputs_q.device)
        
        custom_run = lambda : unfused_swiglu_mlp(
            input=inputs_q_kernel,
            input_scales=inputs_scales,
            weight=concat_weight,
            weight_scales=concat_weight_scales,
            gemm_type=gemm_type,
            D=D, grouped_layout=grouped_layout, swiglu_out=swiglu_out, swiglu_out_scales=swiglu_out_scales
        )
    else:
        # Masked kernel expects A as 3D (G, max_m, K); contiguous expects 2D (total_M, K).
        # The C++ test reshapes A_fp8 explicitly before the call (test_kernel3.cpp:315).
        if gemm_type == moe_cuda.GemmType.MGroupedMasked:
            inputs_q_kernel = inputs_q.view(num_groups, max_m, H)
        else:
            inputs_q_kernel = inputs_q

        custom_run = lambda : function_maps[function_name](
            input=inputs_q_kernel,
            input_scales=inputs_scales,
            gate=gate_q,
            gate_scales=gate_q_scales, up=up_q, up_scales=up_q_scales, gemm_type=gemm_type, D=D, scale_d=scale_D,
            grouped_layout=grouped_layout,
        )

    if do_ncu_profile:
        # one warmup iter for jit launching
        custom_run()
        torch.cuda.synchronize()

        with torch.cuda.nvtx.range("ncu_profile"):
            custom_run()
        torch.cuda.synchronize()

    if do_profile:
        custom_avg_ms = benchmark_no_l2_clear(custom_run, num_warmup_iters=1, num_iters=20)
        print(f"{function_name} kernel average time: {custom_avg_ms:.3f} ms for config num_groups: {num_groups}, expected_m_per_group: {expected_m_per_group}, H: {H}, I: {I}")
    if check_correctness:
        # Manual float32 reference — mirrors compute_swiglu_ref from test_kernel3.cpp.
        # Uses the raw BF16 weights directly so no TE quantisation error is mixed in.
        reference_output = manual_swiglu_ref(
            A=A,
            gate_weight=gate_weight,
            up_weight=up_weight,
            grouped_layout=grouped_layout,
            gemm_type=gemm_type,
            num_groups=num_groups,
            max_m=max_m,
        )

        custom_run()
        if function_name != "unfused":
            kernel_output = D.to(torch.float) * scale_D.transpose(-1, -2).contiguous().repeat_interleave(repeats=128, dim=-1)
        else:
            kernel_output = swiglu_out.to(torch.float) * swiglu_out_scales.repeat_interleave(repeats=128, dim=-1)
        kernel_output = kernel_output.to(torch.bfloat16)

        

        # Build a boolean mask covering only the rows that contain real tokens.
        # Contiguous layout: each group's block is padded to a 128-row multiple;
        #   actual_ms[g] real rows are followed by (aligned_ms[g] - actual_ms[g])
        #   padding rows (grouped_layout == -1).
        # Masked layout: each group occupies max_m rows; only the first
        #   grouped_layout[g] rows per group are valid.
        if gemm_type == moe_cuda.GemmType.MGroupedContiguous:
            valid_mask = torch.zeros(sum(m_splits), dtype=torch.bool, device=A.device)
            offset = 0
            for actual_m, aligned_m in zip(actual_ms, m_splits):
                valid_mask[offset : offset + actual_m] = True
                offset += aligned_m
        else:  # MGroupedMasked — grouped_layout holds per-group actual counts
            valid_mask = torch.zeros(num_groups * max_m, dtype=torch.bool, device=A.device)
            for g in range(num_groups):
                valid_mask[g * max_m : g * max_m + grouped_layout[g].item()] = True
        if function_name == "unfused":
            if swiglu_out_scales[valid_mask].isnan().sum() > 0:
                row_nan = swiglu_out_scales[valid_mask].isnan().sum(dim=-1)
                print(torch.where(row_nan > 0))
                import code; code.interact(local=dict(globals(), **locals()))
                assert False, "swiglu_out_scales is nan"
        k_valid   = kernel_output[valid_mask]
        ref_valid = reference_output[valid_mask]
        print(kernel_output.shape, reference_output.shape)
        print(k_valid.shape, ref_valid.shape)
        err = (k_valid - ref_valid).abs()
        # if function_name == "unfused":
        #     swiglu_out_copy = swiglu_out.clone()
        #     swiglu_out_scales_copy = swiglu_out_scales.clone()
        #     test_silu_mul_quant(D[valid_mask], swiglu_out_copy[valid_mask], swiglu_out_scales_copy[valid_mask])
        print(f"  mean_err={err.mean():.5f}  max_err={err.max():.5f}  "
              f"cosine_diff={calc_cosine_diff(k_valid, ref_valid):.6f}")
        cosine_diff = calc_cosine_diff(k_valid, ref_valid)
        if cosine_diff >= 0.01 or torch.isnan(cosine_diff):
            check_diff("kernel_output", k_valid, ref_valid, single = True)

    # Profiling
if __name__ == "__main__":
    setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--function_name", type=str, default="coop")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ncu_profile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    args = parser.parse_args()

    for num_groups, expected_m_per_group, n, k, gemm_type in enumerate_grouped_gemms():
        run(num_groups=num_groups,
            max_m=8192,
            expected_m_per_group=expected_m_per_group,
            H=k,
            I=n,
            gemm_type=gemm_type, rank=0, function_name=args.function_name, do_profile=args.profile, do_ncu_profile=args.ncu_profile, check_correctness=args.correctness)
        if args.ncu_profile:
            break
