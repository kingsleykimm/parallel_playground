from dataclasses import dataclass
import os
from pathlib import Path
import torch
from deepspeed.moe.layer import MoE
import moe_cuda
from argparse import ArgumentParser
from .common import quantize_1d_128, quantize_2d_128

@dataclass
class TestConfig:
    max_num_tokens : int
    check_correctness : bool = False
    do_profile : bool = False
    num_warmup_iters: int = 1
    num_iters: int = 5


def setup():
    # initialize distributed environment, env variables
    library_root_path = os.getenv("LIBRARY_ROOT_PATH")
    if library_root_path is None:
        library_root_path = str(Path(__file__)).resolve().parents[2]
    cuda_home_path = os.getenv("CUDA_HOME_PATH") or os.getenv("CUDA_PATH") or "/usr/loca/cuda/"
    moe_cuda.init(library_root_path, cuda_home_path)

def init_distributed_environment():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK, 0"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert world_size == local_world_size, "No multi-node configs"
    assert rank == local_rank, "no multi-node configs"
    
    torch.distributed.init_process_group(
        backend = "nccl", device_id = local_rank, rank = rank, world_size = local_world_size
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.random.manual_seed(local_rank)

    return local_rank, local_world_size

def naive_moe_forward_fp8(
    a2a_handle : moe_cuda.All2All,
    num_experts : int,
    experts_per_token : int,
    gemm_type : moe_cuda.GemmType,
    hidden_dim : int,
    input_x : torch.Tensor,
    input_x_scale : torch.Tensor,
    up_weight : torch.Tensor,
    up_weight_scale : torch.Tensor,
    gate_weight : torch.Tensor, 
    gate_weight_scale : torch.Tensor,
    down_weight : torch.Tensor, 
    down_weight_scale : torch.Tensor,
    indices : torch.Tensor,
    weights : torch.Tensor,
    stream : torch.cuda.Stream
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

):
    layer = MoE()

# structure of this file is based off of TK's benchmark.py files in their kernel directory
def run(
    B : int,
    S : int,
    I : int,
    H : int,
    num_experts : int,
    experts_per_token : int,
    local_rank: int,
    local_world_size: int,
    config : TestConfig):

    assert H % 128 == 0 and I % 128 == 0, "quantization shape checks"

    device = f"cuda:{local_rank}"
    num_tokens_per_dev = B * S // local_world_size
    num_experts_per_dev = B * S / local_world_size
    inputs = torch.randn(num_tokens_per_dev, H, dtype = torch.bfloat16, device=device) / (H ** 0.5)
    dp_x, dp_x_scale = quantize_1d_128(inputs)
    up_weights_bf = torch.randn(num_experts_per_dev, I, H, dtype = torch.bfloat16, device = device) / (H ** 0.5)
    up_weight, up_weight_scale = quantize_2d_128(up_weights_bf)
    gate_weights_bf = torch.randn(num_experts_per_dev, I, H, dtype = torch.bfloat16, device = device) / (H ** 0.5)
    gate_weight, gate_weight_scale = quantize_2d_128(gate_weights_bf)
    down_weights_bf = torch.randn(num_experts_per_dev, H, I, dtype = torch.bfloat16, device = device) / (I ** 0.5)
    down_weight, down_weight_scale = quantize_2d_128(down_weights_bf)

    current_stream = torch.cuda.current_stream(device)
    
    a2a_handle = moe_cuda.All2All(
        config.max_num_tokens, num_experts, experts_per_token, 128,
        H, H / 128, torch.float8_e4m3fn, torch.bfloat16, torch.float32, None,
        1, local_world_size, local_world_size, local_rank, current_stream)
    
    # indices = torch.empty(max_recv_tokens, dtype = torch.int32, device = device)
    # weights = torch.empty(max_recv_tokens, dtype = torch.float32, device = device)
   
    if local_rank == 0:
        routing_weights = torch.rand(num_experts, dtype = torch.float32, device = device)
        # renorm routing_weights here
        chosen_experts = torch.multinomial(routing_weights.repeat(B * S, 1), experts_per_token)
        routing_weights = torch.gather(routing_weights.repeat(B * S, 1), -1, chosen_experts).softmax(dim=-1)

    else:
        routing_weights = torch.empty(num_experts, dtype = torch.float32, device = device)
        chosen_experts = torch.multinomial(routing_weights.repeat(B * S, 1), experts_per_token)
    
    torch.distributed.broadcast(routing_weights, 0)
    torch.distributed.broadcast(chosen_experts, 0)

if __name__ == "__main__":
    parser = ArgumentParser()