import moe_cuda
from common import *
from dataclasses import dataclass
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from transformer_engine_torch import DType as TE_DType
import transformer_engine_torch as tex
import vllm.model_executor.layers.fused_moe.deep_gemm_moe as deep_gemm_moe
import vllm.model_executor.layers.fused_moe.config as fused_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


def make_moe_configs(
    *,
    num_experts: int,
    num_experts_per_token: int,
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
    num_experts_per_dev: int,
    fc1_scales: torch.Tensor,
    fc2_scales: torch.Tensor,
    device: torch.device,
    local_rank: int,
    local_world_size: int,
):
    fused_moe_parallel_config = fused_moe_config.FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=local_world_size,
        ep_size=local_world_size,
        dp_rank=local_rank,
        ep_rank=local_rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="naive",
        enable_eplb=False,
    )
    # make quant config first
    quant_config = fused_moe_config.fp8_w8a8_moe_quant_config(
        w1_scale=fc1_scales,
        w2_scale=fc2_scales,
        block_shape=[128, 128]
    )

    moe_config = fused_moe_config.FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=num_experts_per_token,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts_per_dev,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=fused_moe_config.RoutingMethodType.TopK,
        in_dtype=torch.float8_e4m3fn,
        max_num_tokens=max_num_tokens,
        moe_parallel_config=fused_moe_parallel_config
    )

    deep_gem_moe_experts = deep_gemm_moe.DeepGemmExperts(
        moe_config=moe_config, quant_config=quant_config
    )

    return deep_gem_moe_experts


class TESwigluMLP(nn.Module):
    """
    Single-expert SwiGLU FFN using te.Linear with blockwise FP8 scaling.

    Float8BlockScaling defaults match our custom kernel's scaling scheme:
      x_block_scaling_dim=1  →  1×128 rowwise for activations  (== quantize_1d_128)
      w_block_scaling_dim=2  →  128×128 blockwise for weights   (== quantize_2d_128)

    fc1 packs gate + up projections: weight shape [2*N, K]
    fc2 is the down projection:      weight shape [K, N]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_groups: int,
    ):
        super().__init__()
        self.recipe = Float8BlockScaling(
            x_block_scaling_dim=1,
            w_block_scaling_dim=2,
        )

        self.fc1 = te.GroupedLinear(
            num_groups, hidden_size, intermediate_size * 2, bias=False)
        self.fc2 = te.GroupedLinear(
            num_groups, intermediate_size, hidden_size, bias=False)

    def load_weights(
        self,
        gate_weight: torch.Tensor,  # [N, K]
        up_weight: torch.Tensor,    # [N, K]
        down_weight: torch.Tensor,  # [K, N]
    ):
        """Copy BF16 weights in. TE quantizes them on the first forward pass."""
        with torch.no_grad():
            for i in range(self.fc1.num_gemms):
                getattr(self.fc1, f"weight{i}").copy_(
                    torch.cat([gate_weight[i], up_weight[i]], dim=0))
            for i in range(self.fc2.num_gemms):
                getattr(self.fc2, f"weight{i}").copy_(down_weight[i])

    def forward(self, x: torch.Tensor, m_splits: torch.Tensor) -> torch.Tensor:
        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            x = self.fc1(x, m_splits=m_splits)
            x = tex.swiglu(x, quantizer=self.quantizer_1d)
            return self.fc2(x, m_splits=m_splits)


@torch.no_grad()
def vllm_forward_fp8(
    device: torch.device,
    num_experts: int,
    num_experts_per_token: int,
    num_tokens: int,  # num tokens per device
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
    indices: torch.Tensor,
    weights: torch.Tensor,
    inputs: torch.Tensor,
    inputs_scales: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_weights_scales: torch.Tensor,
    up_weights: torch.Tensor,
    up_weights_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_weights_scales: torch.Tensor,
    local_rank: int,
    local_world_size: int,
):
    local_num_experts = num_experts // local_world_size
    # for some reason, the B scales are flattened into 2d when using TE quantizer, so we reshape
    gate_weights_scales = gate_weights_scales.view(
        local_num_experts, intermediate_size // 128, hidden_size // 128)
    up_weights_scales = up_weights_scales.view(
        local_num_experts, intermediate_size // 128, hidden_size // 128)
    down_weights_scales = down_weights_scales.view(
        local_num_experts, hidden_size // 128, intermediate_size // 128)
    fc1 = torch.concat([gate_weights, up_weights], dim=1)
    fc1_scales = torch.concat([gate_weights_scales, up_weights_scales], dim=1)
    deep_gemm_moe_experts = make_moe_configs(
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_num_tokens=max_num_tokens,
        num_experts_per_dev=num_experts // local_world_size,
        fc1_scales=fc1_scales,
        fc2_scales=down_weights_scales,
        device=device,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )

    workspace1_shape, workspace2_shape, output_shape = deep_gemm_moe_experts.workspace_shapes(
        M=num_tokens,
        N=intermediate_size * 2,
        K=hidden_size,
        topk=num_experts_per_token,
        global_num_experts=num_experts,
        local_num_experts=num_experts // local_world_size,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU
    )

    output = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
    workspace1 = torch.empty(
        workspace1_shape, dtype=torch.bfloat16, device=device)
    workspace2 = torch.empty(
        workspace2_shape, dtype=torch.bfloat16, device=device)

    expert_map = torch.arange(local_num_experts, device=device,
                              dtype=torch.int32) + local_rank * local_num_experts
    deep_gemm_moe_experts.apply(
        output=output,
        hidden_states=inputs,
        w1=fc1,
        w2=down_weights,
        topk_weights=weights,
        topk_ids=indices,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=expert_map,
        a1q_scale=inputs_scales,
        a2_scale=None,  # dummy
        workspace13=workspace1,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    return output


@dataclass
class TestConfig:
    max_num_tokens: int
    check_correctness: bool = False
    do_profile: bool = False
    num_warmup_iters: int = 1
    num_iters: int = 5
    gemm_type: moe_cuda.GemmType = moe_cuda.GemmType.MGroupedContiguous


def naive_moe_forward_fp8(
    *,
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
    out_tokens: torch.Tensor,
    expert_x: torch.Tensor,  # workspace1
    expert_x_scales: torch.Tensor,
    inter_y: torch.Tensor,
    inter_y_scales: torch.Tensor,  # workspace 2 tensors
    expert_y: torch.Tensor,
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
        out_tokens,
        expert_x,
        expert_x_scales,
        inter_y,
        inter_y_scales,
        expert_y
    )


def run(
        B: int,
        S: int,
        I: int,
        H: int,
        num_experts: int,
        experts_per_token: int,
        local_rank: int,
        local_world_size: int,
        cfg: TestConfig):

    assert H % 128 == 0 and I % 128 == 0, "quantization shape checks"

    device = f"cuda:{local_rank}"
    num_tokens_per_dev = B * S // local_world_size
    num_experts_per_dev = num_experts // local_world_size
    inputs = torch.randn(num_tokens_per_dev, H,
                         dtype=torch.bfloat16, device=device) / (H ** 0.5)
    up_weights_bf = torch.randn(
        num_experts_per_dev, I, H, dtype=torch.bfloat16, device=device) / (H ** 0.5)
    gate_weights_bf = torch.randn(
        num_experts_per_dev, I, H, dtype=torch.bfloat16, device=device) / (H ** 0.5)
    down_weights_bf = torch.randn(
        num_experts_per_dev, H, I, dtype=torch.bfloat16, device=device) / (I ** 0.5)

    quantized_1d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3, rowwise=True, columnwise=False, block_scaling_dim=1)
    quantized_2d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=2)

    inputs_q_tensor: Float8BlockwiseQTensor = quantized_1d(inputs)
    inputs_q, inputs_scales = inputs_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), inputs_q_tensor._rowwise_scale_inv
    up_q_tensor: Float8BlockwiseQTensor = quantized_2d(up_weights_bf)
    up_q, up_q_scales = up_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), up_q_tensor._rowwise_scale_inv
    gate_q_tensor: Float8BlockwiseQTensor = quantized_2d(gate_weights_bf)
    gate_q, gate_q_scales = gate_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), gate_q_tensor._rowwise_scale_inv
    down_q_tensor: Float8BlockwiseQTensor = quantized_2d(down_weights_bf)
    down_q, down_q_scales = down_q_tensor._rowwise_data.view(
        torch.float8_e4m3fn), down_q_tensor._rowwise_scale_inv

    weights = torch.rand((num_experts), dtype=torch.float32,
                         device=device).repeat(num_tokens_per_dev, 1)
    indices = torch.multinomial(
        input=weights, num_samples=experts_per_token, replacement=False).to(torch.int32)

    weights = torch.gather(input=weights, dim=-1, index=indices)

    # torch.distributed.scatter(weights, src = 0)
    # torch.distributed.broadcast(indices, src = 0)
    # To get the current CUDA stream in PyTorch:
    current_stream = torch.cuda.current_stream(device)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    clean_print("Initializing All2All", print_once=True)
    a2a_handle = moe_cuda.All2All(
        max_num_tokens=cfg.max_num_tokens,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        expert_padding=128,
        hidden_dim=H,
        hidden_dim_scale=H // 128,
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        scale_dtype=torch.float32,
        max_private_tokens=None,
        dp_group_size=1,
        node_group_size=local_world_size,
        ep_group_size=local_world_size,
        device=local_rank)

    torch.cuda.synchronize()

    clean_print("Starting moe forward pass", print_once=True)

    out_tokens = torch.zeros((num_tokens_per_dev, H),
                             dtype=torch.bfloat16, device=device)

    expert_x = torch.zeros((a2a_handle.max_recv_tokens, H, ),
                           dtype=torch.float8_e4m3fn, device=device)
    expert_x_scales = torch.zeros(
        (H // 128, a2a_handle.max_recv_tokens), dtype=torch.float32, device=device)

    inter_y = torch.zeros((a2a_handle.max_recv_tokens, I),
                          dtype=torch.float8_e4m3fn, device=device)
    inter_y_scales = torch.zeros(
        (I // 128, a2a_handle.max_recv_tokens), dtype=torch.float32, device=device)

    expert_y = torch.zeros((a2a_handle.max_recv_tokens, H),
                           dtype=torch.bfloat16, device=device)

    def custom_moe_forward(): return naive_moe_forward_fp8(
        a2a_handle=a2a_handle,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        gemm_type=cfg.gemm_type,
        hidden_dim=H,
        input_x=inputs_q,
        input_x_scale=inputs_scales,
        up_weight=up_q,
        up_weight_scale=up_q_scales,
        gate_weight=gate_q,
        gate_weight_scale=gate_q_scales,
        down_weight=down_q,
        down_weight_scale=down_q_scales,
        indices=indices,
        weights=weights,
        out_tokens=out_tokens,
        expert_x=expert_x,
        expert_x_scales=expert_x_scales,
        inter_y=inter_y,
        inter_y_scales=inter_y_scales,
        expert_y=expert_y
    )

    torch.distributed.barrier()
    torch.cuda.synchronize()

    def vllm_moe_forward(): return vllm_forward_fp8(
        device=device,
        num_experts=num_experts,
        num_experts_per_token=experts_per_token,
        num_tokens=num_tokens_per_dev,
        hidden_size=H,
        intermediate_size=I,
        max_num_tokens=cfg.max_num_tokens,
        indices=indices,
        weights=weights,
        inputs=inputs_q,
        inputs_scales=inputs_scales,
        gate_weights=gate_q,
        gate_weights_scales=gate_q_scales,
        up_weights=up_q,
        up_weights_scales=up_q_scales,
        down_weights=down_q,
        down_weights_scales=down_q_scales,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )

    torch.distributed.barrier()
    torch.cuda.synchronize()

    # with nvtx.range("custom_moe_forward"):
    #     custom_moe_forward()

    custom_moe_forward()
    # reference_outputs = vllm_moe_forward()

    # check_diff("vllm_moe_forward", out_tokens, reference_outputs)
    avg_ms = benchmark_no_l2_clear(
        custom_moe_forward, num_warmup_iters=1, num_iters=20)
    clean_print("custom naive moe fp8 forward ms: ", avg_ms)
    destroy_distributed_environment()


if __name__ == "__main__":
    setup()
    local_rank, local_world_size = init_distributed_environment()
    config = TestConfig(
        max_num_tokens=2048,
        do_profile=False,
        check_correctness=False,
        gemm_type=moe_cuda.GemmType.MGroupedContiguous
    )
    run(B=1,
        S=1024,
        I=4096,
        H=2048,
        num_experts=256,
        experts_per_token=8,
        local_rank=local_rank,
        local_world_size=local_world_size,
        cfg=config)
