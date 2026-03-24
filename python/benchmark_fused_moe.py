import moe_cuda
from common import *
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from transformer_engine_torch import DType as TE_DType



def fused_dispatch_grouped_gemm_swiglu(
    in_tokens : moe_cuda.TKParallelTensor,
    in_tokens_scales : moe_cuda.TKParallelTensor,
    expert_x_tokens : torch.Tensor,
    expert_x_tokens_scale : torch.Tensor,
    comm_comp_barrier : torch.Tensor,
    gate : torch.Tensor,
    up : torch.Tensor,
    C : torch.Tensor,
    scale_gate : torch.Tensor,
    scale_up : torch.Tensor,
    out_scales : torch.Tensor,
    indices : torch.Tensor,
    global_num_routed : moe_cuda.TKParallelTensor,
    expert_to_token_map : moe_cuda.TKParallelTensor,
    padded_expert_counts : torch.Tensor,
    src_token_idx : torch.Tensor,
    src_dev_idx : torch.Tensor,
    barrier : moe_cuda.TKParallelTensor,
    num_tokens : int,
    num_recv_tokens : int, # this needs to be moved into cpp side
    dp_rank : int,
    rank : int,
    dp_size : int,
    cur_dp_group : int,
    num_dp_groups : int,
    num_experts : int,
    experts_per_token : int,
    num_comm_sms : int,
    num_comp_sms : int,
):
    moe_cuda.fused_dispatch_grouped_gemm_swiglu(
        in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
        comm_comp_barrier, gate, up, C, scale_gate, scale_up, out_scales,
        indices, global_num_routed, expert_to_token_map, padded_expert_counts,
        src_token_idx, src_dev_idx, barrier, num_tokens, num_recv_tokens,
        dp_rank, rank, dp_size, cur_dp_group, num_dp_groups,
        num_experts, experts_per_token, num_comm_sms, num_comp_sms)


@dataclass
class TestConfig:
    B : int
    S : int
    I : int
    H : int
    num_experts : int
    experts_per_token : int
    max_recv_tokens : int # size the buffers get padded to, this can be calculated from total tokens in each dp group
    dp_size : int
    num_comm_sms : int
    num_comp_sms : int
    local_rank: int
    local_world_size: int

def run(
  cfg : TestConfig):

    assert cfg.H % 128 == 0 and cfg.I % 128 == 0, "quantization shape checks"
    assert cfg.max_recv_tokens % 128 == 0, "max_recv_tokens must be divisible by 128"

    # Metadata setup
    local_rank = cfg.local_rank
    local_world_size = cfg.local_world_size
    num_experts_per_dev = cfg.num_experts // local_world_size
    device = torch.device(f"cuda:{local_rank}")

    dp_rank = local_rank % cfg.dp_size
    dp_group = local_rank // cfg.dp_size
    num_dp_groups = local_world_size // cfg.dp_size

    quantizer_2d = Float8BlockQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,   # weights need both orientations (fprop + dgrad)
            block_scaling_dim=2,
        )

    # create the tensors
    in_tokens = moe_cuda.TKParallelTensor(
        shape = (cfg.B * cfg.S, cfg.H),
        dtype = torch.float8_e4m3fn,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )
    
    in_tokens_scales = moe_cuda.TKParallelTensor(
        shape = (cfg.B * cfg.S, cfg.H / 128),
        dtype = torch.float8_e4m3fn,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )

    expert_x_tokens = torch.empty(
        (cfg.max_recv_tokens, cfg.H), dtype = torch.float8_e4m3fn, device = device
    )
    expert_x_tokens_scale = torch.empty(
        (cfg.max_recv_tokens, cfg.H / 128), dtype = torch.float32, device = device
    )
    comm_comp_barrier = torch.zeros(cfg.B, cfg.S, cfg.H)

    gate_bf16 = torch.empty((
        num_experts_per_dev, cfg.I, cfg.H
    ), dtype = torch.bfloat16, device = device)

    gate_tensor : Float8BlockwiseQTensor = quantizer_2d.quantize(gate_bf16)

    gate = gate_tensor._rowwise_data.view(torch.float8_e4m3fn)
    scale_gate = gate_tensor._rowwise_scale_inv.view(torch.float32)

    up_bf16 = torch.empty((
        num_experts_per_dev, cfg.I, cfg.H
    ), dtype = torch.bfloat16, device = device)

    up_tensor : Float8BlockwiseQTensor = quantizer_2d.quantize(up_bf16)

    up = up_tensor._rowwise_data.view(torch.float8_e4m3fn)
    scale_up = up_tensor._rowwise_scale_inv.view(torch.float32)

    expert_y = torch.empty((
        cfg.B * cfg.S, cfg.I
    ), dtype = torch.float8_e4m3fn, device = device)

    expert_y_scales = torch.empty((
        cfg.B * cfg.S, cfg.I / 128
    ), dtype = torch.float32, device = device)

    global_num_routed = moe_cuda.TKParallelTensor(
        shape = (num_dp_groups, cfg.num_experts),
        dtype = torch.int32,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )

    expert_to_token_map = moe_cuda.TKParallelTensor(
        shape = (cfg.num_experts, cfg.experts_per_token * cfg.B * cfg.S),
        dtype = torch.int,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )

    padded_expert_counts = torch.empty((
        num_experts_per_dev
    ), dtype = torch.int32, device = device)

    src_token_idx = torch.empty((cfg.max_recv_tokens), dtype = torch.int32, device = device)
    src_dev_idx = torch.empty((cfg.max_recv_tokens), dtype = torch.int32, device = device)

    # small barrier shape to track num dp group exchanges 
    barrier = moe_cuda.TKParallelTensor(
        shape = (1),
        dtype = torch.int32,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )
    num_recv_tokens = torch.zeros((1), dtype = torch.int32, device = device)

    weights = torch.rand((cfg.B * cfg.S, cfg.num_experts), dtype = torch.bfloat16, device = device)
    indices = torch.multinomial(input = weights, num_samples = cfg.experts_per_token, replacement = False)
    
    fused_run = lambda : fused_dispatch_grouped_gemm_swiglu(
        in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
        comm_comp_barrier, gate, up, expert_y, scale_gate, scale_up, expert_y_scales,
        indices, global_num_routed, expert_to_token_map, padded_expert_counts,
        src_token_idx, src_dev_idx, barrier, cfg.B * cfg.S, num_recv_tokens,
        dp_rank, local_rank, cfg.dp_size, dp_group, num_dp_groups,
        cfg.num_experts, cfg.experts_per_token, cfg.num_comm_sms, cfg.num_comp_sms)
