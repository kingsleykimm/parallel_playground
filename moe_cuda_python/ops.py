"""Thin wrappers around ``torch.ops.moe_cuda.*`` registered in ``csrc/torch_api.cpp``."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "fp8_nt_gemm",
    "fp8_grouped_gemm_nt",
    "fp8_grouped_gemm_swiglu",
    "fp8_grouped_gemm_swiglu_contiguous",
    "fp8_grouped_gemm_swiglu_consumer_pp",
    "cast",
    "fused_silu_mul_quant",
]


def fp8_nt_gemm(
    act: Tensor,
    act_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    output: Tensor,
) -> None:
    torch.ops.moe_cuda.fp8_nt_gemm(
        act, act_scale, weight, weight_scale, output
    )


def fp8_grouped_gemm_nt(
    act: Tensor,
    act_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    output: Tensor,
    gemm_type: int,
    grouped_layout: Tensor | None = None,
) -> None:
    torch.ops.moe_cuda.fp8_grouped_gemm_nt(
        act,
        act_scale,
        weight,
        weight_scale,
        output,
        gemm_type,
        grouped_layout,
    )


def fp8_grouped_gemm_swiglu(
    act: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    scale_a: Tensor,
    gate_scale: Tensor,
    up_scale: Tensor,
    scale_d: Tensor,
    output: Tensor,
) -> None:
    torch.ops.moe_cuda.fp8_grouped_gemm_swiglu(
        act,
        gate_weight,
        up_weight,
        scale_a,
        gate_scale,
        up_scale,
        scale_d,
        output,
    )


def fp8_grouped_gemm_swiglu_contiguous(
    a: Tensor,
    scale_a: Tensor,
    gate_weight: Tensor,
    gate_scale: Tensor,
    up_weight: Tensor,
    up_scale: Tensor,
    d: Tensor,
    scale_d: Tensor,
) -> None:
    torch.ops.moe_cuda.fp8_grouped_gemm_swiglu_contiguous(
        a,
        scale_a,
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        d,
        scale_d,
    )


def fp8_grouped_gemm_swiglu_consumer_pp(
    a: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    scale_a: Tensor,
    gate_scale: Tensor,
    up_scale: Tensor,
    scale_d: Tensor,
    output: Tensor,
    gemm_type: int,
    grouped_layout: Tensor | None = None,
) -> None:
    torch.ops.moe_cuda.fp8_grouped_gemm_swiglu_consumer_pp(
        a,
        gate_weight,
        up_weight,
        scale_a,
        gate_scale,
        up_scale,
        scale_d,
        output,
        gemm_type,
        grouped_layout,
    )


def cast(
    inp: Tensor,
    out: Tensor,
    scale: Tensor | None = None,
) -> None:
    torch.ops.moe_cuda.cast(inp, out, scale)


def fused_silu_mul_quant(
    gemm_out: Tensor,
    swiglu_out: Tensor,
    scale: Tensor,
) -> None:
    torch.ops.moe_cuda.fused_silu_mul_quant(gemm_out, swiglu_out, scale)
