from dataclasses import dataclass
import torch
from deepspeed.moe.layer import MoE
import moe_cuda
from argparse import ArgumentParser
from .common import quantize_1d_128, quantize_2d_128
import torch.nn as nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
import transformer_engine_torch as tex


class TESwigluMLP(nn.Module):
    """
    Single-expert SwiGLU FFN using te.Linear with blockwise FP8 scaling.

    Float8BlockScaling defaults match our custom kernel's scaling scheme:
      x_block_scaling_dim=1  →  1×128 rowwise for activations  (== quantize_1d_128)
      w_block_scaling_dim=2  →  128×128 blockwise for weights   (== quantize_2d_128)

    fc1 packs gate + up projections: weight shape [2*I, H]
    fc2 is the down projection:      weight shape [H, I]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.recipe = Float8BlockScaling(
            x_block_scaling_dim=1,
            w_block_scaling_dim=2,
        )
        self.fc1 = te.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.fc2 = te.Linear(intermediate_size, hidden_size, bias=False)

    def load_weights(
        self,
        gate_weight: torch.Tensor,  # [I, H]
        up_weight: torch.Tensor,    # [I, H]
        down_weight: torch.Tensor,  # [H, I]
    ):
        """Copy BF16 weights in. TE quantizes them on the first forward pass."""
        with torch.no_grad():
            self.fc1.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
            self.fc2.weight.copy_(down_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            x = self.fc1(x)
            x = tex.swiglu(x)
            return self.fc2(x)


@dataclass
class TestConfig:
    max_num_tokens: int
    check_correctness: bool = False
    do_profile: bool = False
    num_warmup_iters: int = 1
    num_iters: int = 5


def naive_moe_forward_fp8(
    a2a_handle: moe_cuda.All2All,
    num_experts: int,
    experts_per_token: int,
    gemm_type: moe_cuda.GemmType,
    hidden_dim: int,
    input_x: torch.Tensor,
    input_x_scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_weight_scale: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    stream: torch.cuda.Stream
):
    # this is an inplace function, so input will be rewritten
    moe_cuda.naive_moe_forward_dispatch(
        a2a_handle,
        num_experts,
        experts_per_token,
        hidden_dim,
        gemm_type,
        input_x,
        input_x_scale,
        gate_weight,
        gate_weight_scale,
        up_weight,
        up_weight_scale,
        down_weight,
        down_weight_scale,
        indices,
        weights,
        stream
    )


def naive_moe_forward_megatron(
    H: int,
    I: int,
    num_groups: int,
    num_experts: int,
    experts_per_token: int,
    gemm_type: moe_cuda.GemmType,
    input_x: torch.Tensor,
    input_x_scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_weight_scale: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    ep_size: int
):
    mlp = TESwigluMLP(H, I, num_groups)
    mlp.load_weights(gate_weight, up_weight, down_weight)
    layer = MoE(H, mlp, num_experts, ep_size, experts_per_token)

    layer.forward()

# structure of this file is based off of TK's benchmark.py files in their kernel directory


def run(
        B: int,
        S: int,
        I: int,
        H: int,
        num_experts: int,
        experts_per_token: int,
        local_rank: int,
        local_world_size: int,
        config: TestConfig):

    assert H % 128 == 0 and I % 128 == 0, "quantization shape checks"

    device = f"cuda:{local_rank}"
    num_tokens_per_dev = B * S // local_world_size
    num_experts_per_dev = B * S / local_world_size
    inputs = torch.randn(num_tokens_per_dev, H,
                         dtype=torch.bfloat16, device=device) / (H ** 0.5)
    dp_x, dp_x_scale = quantize_1d_128(inputs)
    up_weights_bf = torch.randn(
        num_experts_per_dev, I, H, dtype=torch.bfloat16, device=device) / (H ** 0.5)
    up_weight, up_weight_scale = quantize_2d_128(up_weights_bf)
    gate_weights_bf = torch.randn(
        num_experts_per_dev, I, H, dtype=torch.bfloat16, device=device) / (H ** 0.5)
    gate_weight, gate_weight_scale = quantize_2d_128(gate_weights_bf)
    down_weights_bf = torch.randn(
        num_experts_per_dev, H, I, dtype=torch.bfloat16, device=device) / (I ** 0.5)
    down_weight, down_weight_scale = quantize_2d_128(down_weights_bf)

    current_stream = torch.cuda.current_stream(device)

    a2a_handle = moe_cuda.All2All(
        config.max_num_tokens, num_experts, experts_per_token, 128,
        H, H / 128, torch.float8_e4m3fn, torch.bfloat16, torch.float32, None,
        1, local_world_size, local_world_size, local_rank, current_stream)

    # indices = torch.empty(max_recv_tokens, dtype = torch.int32, device = device)
    # weights = torch.empty(max_recv_tokens, dtype = torch.float32, device = device)

    if local_rank == 0:
        routing_weights = torch.rand(
            num_experts, dtype=torch.float32, device=device)
        # renorm routing_weights here
        chosen_experts = torch.multinomial(
            routing_weights.repeat(B * S, 1), experts_per_token)
        routing_weights = torch.gather(routing_weights.repeat(
            B * S, 1), -1, chosen_experts).softmax(dim=-1)

    else:
        routing_weights = torch.empty(
            num_experts, dtype=torch.float32, device=device)
        chosen_experts = torch.multinomial(
            routing_weights.repeat(B * S, 1), experts_per_token)

    torch.distributed.broadcast(routing_weights, 0)
    torch.distributed.broadcast(chosen_experts, 0)


if __name__ == "__main__":
    parser = ArgumentParser()
