import moe_cuda

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine_torch import DType as TE_DType
from common import init_distributed_environment, clean_print, setup, calc_cosine_diff, quantize_2d_128, quantize_1d_128, enumerate_grouped_gemms, enumerate_moe_configs


@dataclass
class TestConfig:
    check_dispatch_correctness : bool
    check_correctness : bool
    B: int
    S: int
    I: int
    H: int
    num_experts: int
    experts_per_token: int
    dp_size: int
    num_comm_sms: int
    num_comp_sms: int
    local_rank: int
    local_world_size: int = 4


def fused_dispatch_grouped_gemm_swiglu(
    in_tokens: moe_cuda.TKParallelTensor,
    in_tokens_scales: moe_cuda.TKParallelTensor,
    expert_x_tokens: torch.Tensor,
    expert_x_tokens_scale: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    C: torch.Tensor,
    scale_gate: torch.Tensor,
    scale_up: torch.Tensor,
    out_scales: torch.Tensor,
    indices: torch.Tensor,
    global_num_routed: moe_cuda.TKParallelTensor,
    expert_to_token_map: moe_cuda.TKParallelTensor,
    expert_to_slot_map : moe_cuda.TKParallelTensor,
    padded_expert_counts: torch.Tensor,
    src_token_idx: torch.Tensor,
    src_dev_idx: torch.Tensor,
    src_slot_idx: torch.Tensor,
    barrier: moe_cuda.TKParallelTensor,
    num_tokens: int,
    num_recv_tokens: torch.Tensor,  # this needs to be moved into cpp side
    dp_rank: int,
    rank: int,
    dp_size: int,
    cur_dp_group: int,
    num_dp_groups: int,
    world_size: int,
    num_experts: int,
    experts_per_token: int,
    num_comm_sms: int,
    num_comp_sms: int,
):
    # runtime dtype checks
    assert in_tokens.data().dtype == torch.float8_e4m3fn, "in_tokens must be of type float8_e4m3fn"
    assert in_tokens_scales.data(
    ).dtype == torch.float32, "in_tokens_scales must be of type float32"
    assert expert_x_tokens.dtype == torch.float8_e4m3fn, "expert_x_tokens must be of type float8_e4m3fn"
    assert expert_x_tokens_scale.dtype == torch.float32, "expert_x_tokens_scale must be of type float32"
    assert gate.dtype == torch.float8_e4m3fn, "gate must be of type float8_e4m3fn"
    assert up.dtype == torch.float8_e4m3fn, "up must be of type float8_e4m3fn"
    assert C.dtype == torch.float8_e4m3fn, "C must be of type float8_e4m3fn"
    assert scale_gate.dtype == torch.float32, "scale_gate must be of type float32"
    assert scale_up.dtype == torch.float32, "scale_up must be of type float32"
    assert out_scales.dtype == torch.float32, "out_scales must be of type float32"
    assert indices.dtype == torch.int32, "indices must be of type int32"
    assert global_num_routed.data(
    ).dtype == torch.int32, "global_num_routed must be of type int32"
    assert expert_to_token_map.data(
    ).dtype == torch.int32, "expert_to_token_map must be of type int32"
    assert padded_expert_counts.dtype == torch.int32, "padded_expert_counts must be of type int32"
    assert src_token_idx.dtype == torch.int32, "src_token_idx must be of type int32"
    assert src_slot_idx.dtype == torch.int32, "src_slot_idx must be of type int32"
    moe_cuda.fused_dispatch_grouped_gemm_swiglu(
        in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
        gate, up, C, scale_gate, scale_up, out_scales,
        indices, global_num_routed, expert_to_token_map, expert_to_slot_map, padded_expert_counts,
        src_token_idx, src_dev_idx, src_slot_idx, barrier, num_tokens, num_recv_tokens,
        dp_rank, rank, dp_size, cur_dp_group, num_dp_groups, world_size,
        num_experts, experts_per_token, num_comm_sms, num_comp_sms)

def fused_grouped_gemm_combine(
    out_tokens : moe_cuda.TKParallelTensor,
    expert_y_tokens : torch.Tensor,
    expert_y_tokens_scale : torch.Tensor,
    down : torch.Tensor,
    scale_down : torch.Tensor,
    C : torch.Tensor,
    weights : torch.Tensor,
    padded_expert_counts : torch.Tensor,
    src_token_idx : torch.Tensor,
    src_dev_idx : torch.Tensor,
    src_slot_idx : torch.Tensor,
    num_experts : int,
    experts_per_token : int,
    num_recv_tokens : torch.Tensor,
    dp_rank : int,
    rank : int,
    dp_size : int,
    cur_dp_group : int,
    num_dp_groups : int,
    num_comm_sms : int,
    num_comp_sms : int,
):
    moe_cuda.fused_grouped_gemm_combine(
        out_tokens,
        expert_y_tokens,
        expert_y_tokens_scale,
        down,
        scale_down,
        C,
        weights,
        padded_expert_counts,
        src_token_idx,
        src_dev_idx,
        src_slot_idx,
        num_experts,
        experts_per_token,
        num_recv_tokens,
        dp_rank,
        rank,
        dp_size,
        cur_dp_group,
        num_dp_groups,
        num_comm_sms,
        num_comp_sms,
    )

def reference_compute_routing_info(
    indices: torch.Tensor,
    num_experts: int,
    num_experts_per_token: int,
    num_tokens: int,
    max_recv_tokens: int,
    dp_rank: int,
    rank: int,
    dp_size: int,
    num_dp_groups: int,
    world_size: int,
):

    device = torch.device(f"cuda:{rank}")
    # compute local num routed first

    num_routed = torch.zeros(
        (num_experts), dtype=torch.int32, device=torch.device("cpu"))

    host_indices = indices.cpu()
    expert_to_token_map = torch.zeros(
        (num_experts, num_tokens), dtype=torch.int32, device=torch.device("cpu"))
    for i in range(num_tokens):
        for j in range(num_experts_per_token):
            expert_idx = host_indices[i, j]
            expert_to_token_map[expert_idx, num_routed[expert_idx]] = i
            num_routed[expert_idx] += 1

    num_routed = num_routed.to(device)
    expert_to_token_map = expert_to_token_map.to(device)
    global_num_routed = [
        torch.zeros((num_experts), dtype=torch.int32, device=device) for _ in range(world_size)
    ]
    global_expert_to_token_map = [
        torch.zeros((num_experts, num_tokens), dtype=torch.int32, device=device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(global_num_routed, num_routed)
    torch.distributed.all_gather(
        global_expert_to_token_map, expert_to_token_map)
    # shape (local_world_size, num_experts)
    global_num_routed = torch.stack(global_num_routed, dim=0)
    # shape (local_world_size, num_experts, num_tokens)
    global_expert_to_token_map = torch.stack(global_expert_to_token_map, dim=0)

    global_num_routed_cpu = global_num_routed.cpu()
    global_expert_to_token_map_cpu = global_expert_to_token_map.cpu()
    num_experts_per_dev = num_experts // world_size
    first_local_expert = rank * num_experts_per_dev
    last_local_expert = min(first_local_expert +
                            num_experts_per_dev, num_experts)

    src_token_idx = torch.zeros(
        (max_recv_tokens), dtype=torch.int32, device=torch.device("cpu"))
    src_dev_idx = torch.zeros(
        (max_recv_tokens), dtype=torch.int32, device=torch.device("cpu"))

    padded_expert_counts = torch.zeros(
        (num_experts_per_dev), dtype=torch.int32, device=torch.device("cpu"))
    src_group_offset = torch.zeros(
        (num_experts_per_dev, num_dp_groups), dtype=torch.int32, device=torch.device("cpu"))
    for expert in range(first_local_expert, last_local_expert):
        src_rank_offset = 0
        for group in range(num_dp_groups):
            cur_rank = group * dp_size + dp_rank
            num_tokens_from_group = global_num_routed_cpu[cur_rank, expert]
            src_rank_offset += num_tokens_from_group
            src_group_offset[expert - first_local_expert,
                             group] = src_rank_offset
            padded_expert_counts[expert -
                                 first_local_expert] += global_num_routed_cpu[cur_rank, expert]
    padded_expert_counts = ((padded_expert_counts + 127) // 128) * 128

    num_recv_tokens = padded_expert_counts.sum()
    expert_offsets = padded_expert_counts.cumsum(dim=0)
    for token in range(num_recv_tokens):
        cur_expert = -1
        for expert in range(num_experts_per_dev):
            if token < expert_offsets[expert]:
                cur_expert = expert
                break
        assert cur_expert > -1 and cur_expert < num_experts_per_dev, "token not found in any expert"
        intra_expert_offset = token - \
            (expert_offsets[cur_expert - 1] if cur_expert > 0 else 0)
        cur_group = -1
        intra_rank_offset = 0
        for group in range(num_dp_groups):
            if intra_expert_offset < src_group_offset[cur_expert, group]:
                cur_group = group
                intra_rank_offset = intra_expert_offset - \
                    (src_group_offset[cur_expert, group - 1]
                     if group > 0 else 0)
                break
        if cur_group > -1 and cur_group < num_dp_groups:
            src_dev_idx[token] = cur_group * dp_size + dp_rank
            src_token_idx[token] = global_expert_to_token_map_cpu[src_dev_idx[token],
                                                                  cur_expert + first_local_expert, intra_rank_offset]
        else:
            src_dev_idx[token] = -1
            src_token_idx[token] = -1

    return src_token_idx, src_dev_idx, padded_expert_counts, num_recv_tokens, expert_to_token_map, num_routed.cpu()

# only use this for correctness testing


def slow_reference_dispatch_group_gemm(
    *,
    cfg: TestConfig,
    in_tokens: torch.Tensor,  # (B * S, H)
    gate_bf16: torch.Tensor,
    up_bf16: torch.Tensor,
    num_experts_per_dev: int,
    src_token_idx: torch.Tensor,
    src_dev_idx: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    num_recv_tokens: int,
    dp_group: torch.distributed.ProcessGroup
):

    all_tokens = torch.empty(
        (cfg.local_world_size, in_tokens.shape[0], in_tokens.shape[1]), dtype=in_tokens.dtype, device=in_tokens.device)
    inputs_gathered = torch.empty(
        (num_recv_tokens, cfg.H), dtype=all_tokens.dtype, device=all_tokens.device)
    output = torch.empty((num_recv_tokens, cfg.I),
                         dtype=in_tokens.dtype, device=in_tokens.device)
    torch.distributed.all_gather_into_tensor(
        output_tensor=all_tokens, input_tensor=in_tokens, group=dp_group)
    for local_token_idx in range(src_token_idx.shape[0]):
        src_token = src_token_idx[local_token_idx]
        src_dev = src_dev_idx[local_token_idx]
        if src_dev >= 0 and src_token >= 0:
            inputs_gathered[local_token_idx] = all_tokens[src_dev, src_token]

    num_cumsum_tokens = 0
    for expert in range(num_experts_per_dev):
        num_expert_tokens = padded_expert_counts[expert]
        cur_tokens = inputs_gathered[num_cumsum_tokens:
                                     num_cumsum_tokens + num_expert_tokens]
        gate_out = torch.mm(cur_tokens, gate_bf16[expert].T)
        up_out = torch.mm(cur_tokens, up_bf16[expert].T)
        output[num_cumsum_tokens: num_cumsum_tokens +
               num_expert_tokens] = F.silu(gate_out) * up_out
        num_cumsum_tokens += num_expert_tokens
    return inputs_gathered, output

# def torch_dispatch_group_gemm(
#     *,
#     cfg: TestConfig,
#     in_tokens: torch.Tensor,  # (B * S, H)
#     gate_bf16: torch.Tensor,
#     up_bf16: torch.Tensor,
#     num_experts_per_dev: int,
#     src_token_idx: torch.Tensor,
#     src_dev_idx: torch.Tensor,
#     padded_expert_counts: torch.Tensor,
#     num_recv_tokens: int,
#     dp_group: torch.distributed.ProcessGroup
# ):
#     all_tokens = torch.empty(
#         (cfg.local_world_size, in_tokens.shape[0], in_tokens.shape[1]), dtype=in_tokens.dtype, device=in_tokens.device)
#     output = torch.empty((num_recv_tokens, cfg.I),
#                          dtype=in_tokens.dtype, device=in_tokens.device)
#     torch.distributed.all_gather_into_tensor(
#         output_tensor=all_tokens, input_tensor=in_tokens, group=dp_group)

def mlp(*, input : torch.Tensor, up_weight : torch.Tensor, gate_weight : torch.Tensor, down_weight : torch.Tensor):
    grouped_up = torch.nn.functional.grouped_mm(mat_a = input, mat_b = up_weight, out_dtype = torch.bfloat16)
    grouped_gate = torch.nn.functional.grouped_mm(mat_a = input, mat_b = gate_weight, out_dtype = torch.bfloat16)
    act = torch.nn.functional.silu(grouped_gate) * grouped_up
    output = torch.nn.functional.grouped_mm(mat_a = act, mat_b = down_weight, out_dtype = torch.bfloat16)
    return output

def reference_moe(
    *,
    cfg : TestConfig,
    in_tokens : torch.Tensor,
    up : torch.Tensor,
    gate : torch.Tensor,
    down : torch.Tensor,
    indices : torch.Tensor,
    weights : torch.Tensor,
    num_experts : int,
    experts_per_token : int,
    world_size : int,
    mlp_compiled : torch.compiler.FuncType # should be a torch.compiled of the forward pass
) -> torch.Tensor:

    up_gathered = torch.empty( shape = (world_size, *up.shape), dtype = up.dtype, device = up.device)
    down_gathered = torch.empty( shape = (world_size, *down.shape), dtype = down.dtype, device = down.device)
    gate_gathered = torch.empty( shape = (world_size, *gate.shape), dtype = gate.dtype, device = gate.device)

    torch.distributed.all_gather_into_tensor(output_tensor = up_gathered, input_tensor = up)
    torch.distributed.all_gather_into_tensor(output_tensor = down_gathered, input_tensor = down)
    torch.distributed.all_gather_into_tensor(output_tensor = gate_gathered, input_tensor = gate)

    up_gathered = up_gathered.reshape(num_experts, *up.shape[1:])
    down_gathered = down_gathered.reshape(num_experts, *down.shape[1:])
    gate_gathered = gate_gathered.reshape(num_experts, *gate.shape[1:])

    # indices metadata
    sorted_in = torch.empty(shape = (num_experts, in_tokens.size(0), cfg.H), dtype = in_tokens.dtype, device = in_tokens.device)
    src_token_idx = torch.empty(shape = (num_experts, in_tokens.size(0)), dtype = torch.int32, device = in_tokens.device)
    src_slot_idx = torch.empty(shape = (num_experts, in_tokens.size(0)), dtype = torch.int32, device = in_tokens.device)
    per_expert_count = [0 for _ in range(num_experts)]
    for token in range(in_tokens.size(0)):
        for slot in range(experts_per_token):
            cur_expert = indices[token, slot]
            sorted_in[cur_expert][per_expert_count[cur_expert]] = in_tokens[token]
            src_token_idx[cur_expert][per_expert_count[cur_expert]] = token
            src_slot_idx[cur_expert][per_expert_count[cur_expert]] = slot
            per_expert_count[cur_expert] += 1
    
    max_expert_count = max(per_expert_count)
    sorted_in = sorted_in[:, :max_expert_count]
    mlp_out = mlp_compiled(input = sorted_in, up_weight = up_gathered, gate_weight = gate_gathered, down_weight = down_gathered)

    output = torch.zeros_like(in_tokens)
    for expert, expert_count in enumerate(per_expert_count):
        for row in range(expert_count):
            src_token = src_token_idx[expert][row]
            src_slot = src_slot_idx[expert][row]
            output[src_token] += mlp_out[expert][row] * weights[src_token, src_slot] 
    return output







    
    

def check_fused_dispatch_correctness(
        *,
        cfg: TestConfig,
        device: torch.device,
        num_experts_per_dev: int,
        my_dp_group: torch.distributed.ProcessGroup,
        indices: torch.Tensor,
        dp_rank: int,
        dp_group: int,
        num_dp_groups: int,
        in_tokens_bf16: torch.Tensor,
        gate_bf16: torch.Tensor,
        up_bf16: torch.Tensor,
        num_recv_tokens: torch.Tensor,
        padded_expert_counts: torch.Tensor,
        src_token_idx: torch.Tensor,
        src_dev_idx: torch.Tensor,
        expert_y: torch.Tensor,
        expert_y_scales: torch.Tensor,
):
    max_recv_tokens_val = src_token_idx.shape[0]

    reference_src_token_idx, reference_src_dev_idx, reference_padded_expert_counts, reference_num_recv_tokens, reference_expert_to_token_map, reference_num_routed = reference_compute_routing_info(
        indices, cfg.num_experts, cfg.experts_per_token, cfg.B * cfg.S, max_recv_tokens_val,
        dp_rank, cfg.local_rank, cfg.dp_size, num_dp_groups, cfg.local_world_size
    )

    assert num_recv_tokens.item() == reference_num_recv_tokens.item(
    ), "num_recv_tokens mismatch"
    torch.testing.assert_close(
        padded_expert_counts.cpu(), reference_padded_expert_counts.cpu())

    src_token_idx_cpu = src_token_idx.cpu()[:reference_num_recv_tokens.item()]
    src_dev_idx_cpu = src_dev_idx.cpu()[:reference_num_recv_tokens.item()]
    reference_src_token_idx = reference_src_token_idx.cpu()[
        :reference_num_recv_tokens.item()]
    reference_src_dev_idx = reference_src_dev_idx.cpu()[
        :reference_num_recv_tokens.item()]
    src_token_dev_idx = torch.stack(
        (src_token_idx_cpu, src_dev_idx_cpu), dim=1)
    reference_src_token_dev_idx = torch.stack(
        (reference_src_token_idx, reference_src_dev_idx), dim=1)

    k_key = src_token_dev_idx[:, 0] * 100000 + src_token_dev_idx[:, 1]
    r_key = reference_src_token_dev_idx[:, 0] * \
        100000 + reference_src_token_dev_idx[:, 1]
    torch.testing.assert_close(
        src_token_dev_idx[k_key.argsort()], reference_src_token_dev_idx[r_key.argsort()])

    reference_inputs_gathered, reference_swiglu = slow_reference_dispatch_group_gemm(
        cfg=cfg,
        in_tokens=in_tokens_bf16,
        gate_bf16=gate_bf16,
        up_bf16=up_bf16,
        num_experts_per_dev=num_experts_per_dev,
        src_token_idx=reference_src_token_idx.to(device),
        src_dev_idx=reference_src_dev_idx.to(device),
        padded_expert_counts=padded_expert_counts.to(device),
        num_recv_tokens=reference_num_recv_tokens.item(),
        dp_group=my_dp_group
    )

    r_mask = r_key >= 0
    k_mask = k_key >= 0

    torch.testing.assert_close(
        src_token_dev_idx[k_mask][k_key[k_mask].argsort()],
        reference_src_token_dev_idx[r_mask][r_key[r_mask].argsort()]
    )

    expert_offsets = padded_expert_counts.cpu().cumsum(dim=0)
    expert_keys = torch.empty((num_recv_tokens.item()),
                              dtype=torch.int32, device=torch.device("cpu"))
    start = 0
    for e in range(num_experts_per_dev):
        end = expert_offsets[e].item()
        expert_keys[start:end] = e
        start = end
    k_key += expert_keys * 10000000000
    r_key += expert_keys * 10000000000

    expert_y_scales_trimmed = expert_y_scales[:, :num_recv_tokens.item()]
    expert_y_trimmed = expert_y[:num_recv_tokens.item()]
    clean_print(expert_y_scales_trimmed.shape, print_once=True)
    kernel_output = (expert_y_trimmed.to(torch.float32) * expert_y_scales_trimmed.transpose(-1, -
                     2).contiguous().repeat_interleave(repeats=128, dim=-1)).to(torch.bfloat16)

    reference_swiglu = reference_swiglu.to(torch.bfloat16)
    kernel_output = kernel_output[k_mask].gather(
        dim=0, index=k_key[k_mask].argsort().unsqueeze(1).expand(-1, cfg.I).to(device))
    reference_swiglu = reference_swiglu[r_mask].gather(
        dim=0, index=r_key[r_mask].argsort().unsqueeze(1).expand(-1, cfg.I).to(device))

    diff = calc_cosine_diff(kernel_output, reference_swiglu)
    clean_print(f"cosine diff: {diff}")
    assert diff < 0.01, "cosine diff between kernel and reference is too high"


def fused_dispatch_run(
        cfg: TestConfig,
        max_recv_tokens: int):

    assert cfg.H >= 512 and cfg.I >= 512, "H and I must be at least 512, since TE FP8 Quantizer rounds up to the nearest multiple of 4 for scale factors"
    assert cfg.H % 128 == 0 and cfg.I % 128 == 0, "quantization shape checks"
    assert max_recv_tokens % 128 == 0, "max_recv_tokens must be divisible by 128"
    clean_print(f"Starting test case with B = {cfg.B}, S = {cfg.S}, I = {cfg.I}, H = {cfg.H}, num_experts = {cfg.num_experts}, experts_per_token = {cfg.experts_per_token}, dp_size = {cfg.dp_size}, num_comm_sms = {cfg.num_comm_sms}, num_comp_sms = {cfg.num_comp_sms}, local_rank = {cfg.local_rank}, local_world_size = {cfg.local_world_size}", print_once=True)
    # Metadata setup
    local_rank = cfg.local_rank
    local_world_size = cfg.local_world_size
    num_experts_per_dev = cfg.num_experts // local_world_size
    device = torch.device(f"cuda:{local_rank}")

    dp_rank = local_rank % cfg.dp_size
    dp_group = local_rank // cfg.dp_size
    num_dp_groups = local_world_size // cfg.dp_size

    dp_groups = []
    for i in range(cfg.dp_size):
        ranks = [r for r in range(local_world_size) if r % cfg.dp_size == i]
        group = torch.distributed.new_group(ranks)
        dp_groups.append(group)
    my_dp_group = dp_groups[local_rank % cfg.dp_size]

    quantizer_1d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=1,
    )

    quantizer_2d = Float8BlockQuantizer(
        fp8_dtype=TE_DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,   # weights need both orientations (fprop + dgrad)
        block_scaling_dim=2,
    )
    in_tokens_bf16 = torch.randn((cfg.B * cfg.S, cfg.H),
                                 dtype=torch.bfloat16, device=device)

    # in_tokens_quantized: Float8BlockwiseQTensor = quantizer_1d(
    #     in_tokens_bf16)
    in_tokens_fp8, in_tokens_scales_fp32 = quantize_1d_128(in_tokens_bf16)

    # in_tokens_fp8 = in_tokens_quantized._rowwise_data.view(torch.float8_e4m3fn)
    # in_tokens_scales_fp32 = in_tokens_quantized._rowwise_scale_inv.view(
    #     torch.float32)

    # create the tensors
    in_tokens = moe_cuda.TKParallelTensor(
        shape=(cfg.B * cfg.S, cfg.H),
        dtype=torch.float8_e4m3fn,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    in_tokens.data().copy_(in_tokens_fp8)

    in_tokens_scales = moe_cuda.TKParallelTensor(
        shape=[cfg.H // 128, cfg.B * cfg.S],
        dtype=torch.float32,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    in_tokens_scales.data().copy_(in_tokens_scales_fp32)

    expert_x_tokens = torch.empty(
        (max_recv_tokens, cfg.H), dtype=torch.float8_e4m3fn, device=device
    )
    expert_x_tokens_scale = torch.empty(
        (cfg.H // 128, max_recv_tokens), dtype=torch.float32, device=device
    )

    gate_bf16 = torch.randn((
        num_experts_per_dev, cfg.I, cfg.H
    ), dtype=torch.bfloat16, device=device) / (cfg.H ** 0.5)

    # gate_tensor: Float8BlockwiseQTensor = quantizer_2d.quantize(gate_bf16)
    # clean_print(gate_bf16.shape, print_once=True)
    # clean_print(gate_tensor._rowwise_scale_inv.shape, print_once=True)

    # gate = gate_tensor._rowwise_data.view(torch.float8_e4m3fn).view(
    #     num_experts_per_dev * cfg.I, cfg.H)
    # scale_gate = gate_tensor._rowwise_scale_inv.view(torch.float32).view(
    #     num_experts_per_dev * (cfg.I // 128), cfg.H // 128)
    gate, scale_gate = quantize_2d_128(gate_bf16)
    # gate = gate.reshape(num_experts_per_dev * cfg.I, cfg.H)
    # scale_gate = scale_gate.reshape(num_experts_per_dev * (cfg.I // 128), cfg.H // 128)
    up_bf16 = torch.randn((
        num_experts_per_dev, cfg.I, cfg.H
    ), dtype=torch.bfloat16, device=device) / (cfg.H ** 0.5)

    # up_tensor: Float8BlockwiseQTensor = quantizer_2d.quantize(up_bf16)

    # up = up_tensor._rowwise_data.view(torch.float8_e4m3fn).view(
    #     num_experts_per_dev * cfg.I, cfg.H)
    # scale_up = up_tensor._rowwise_scale_inv.view(torch.float32).view(
    #     num_experts_per_dev * (cfg.I // 128), cfg.H // 128)
    up, scale_up = quantize_2d_128(up_bf16)
    # up = up.reshape(num_experts_per_dev * cfg.I, cfg.H)
    # scale_up = scale_up.reshape(num_experts_per_dev * (cfg.I // 128), cfg.H // 128)
    expert_y = torch.empty((
        max_recv_tokens, cfg.I
    ), dtype=torch.float8_e4m3fn, device=device)

    expert_y_scales = torch.empty((
        cfg.I // 128, max_recv_tokens
    ), dtype=torch.float32, device=device)

    global_num_routed = moe_cuda.TKParallelTensor(
        shape=(num_dp_groups, cfg.num_experts),
        dtype=torch.int32,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )

    expert_to_token_map = moe_cuda.TKParallelTensor(
        # for any given expert, a max of all the tokens can be routed to it
        shape=(cfg.num_experts, cfg.B * cfg.S),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )

    expert_to_slot_map = moe_cuda.TKParallelTensor(
        shape=(cfg.num_experts, cfg.B * cfg.S),
        dtype=torch.int32,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    padded_expert_counts = torch.empty((
        num_experts_per_dev
    ), dtype=torch.int32, device=device)

    src_token_idx = torch.empty(
        (max_recv_tokens), dtype=torch.int32, device=device)
    src_dev_idx = torch.empty((max_recv_tokens),
                              dtype=torch.int32, device=device)
    src_slot_idx = torch.empty((max_recv_tokens),
                               dtype=torch.int32, device=device)
    # small barrier shape to track num dp group exchanges
    barrier = moe_cuda.TKParallelTensor(
        shape=(1,),
        dtype=torch.int32,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    num_recv_tokens = torch.zeros((1,), dtype=torch.int32, device=device)

    weights = torch.rand((cfg.B * cfg.S, cfg.num_experts),
                         dtype=torch.bfloat16, device=device)
    indices = torch.multinomial(
        input=weights, num_samples=cfg.experts_per_token, replacement=False).to(torch.int32).to(device)
    weights = torch.gather(input = weights, dim = 1, index = indices).softmax(dim = -1)

    fused_dispatch_run  = lambda : fused_dispatch_grouped_gemm_swiglu(
        in_tokens, in_tokens_scales, expert_x_tokens, expert_x_tokens_scale,
        gate, up, expert_y, scale_gate, scale_up, expert_y_scales,
        indices, global_num_routed, expert_to_token_map, expert_to_slot_map, padded_expert_counts,
        src_token_idx, src_dev_idx, src_slot_idx, barrier, cfg.B * cfg.S, num_recv_tokens,
        dp_rank, local_rank, cfg.dp_size, dp_group, num_dp_groups,
        local_world_size,
        cfg.num_experts, cfg.experts_per_token, cfg.num_comm_sms, cfg.num_comp_sms)
    clean_print("Initialized tensors", print_once=True)
    fused_dispatch_run()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    clean_print("Fused run completed", print_once=True)

    if cfg.check_dispatch_correctness:
        check_fused_dispatch_correctness(
            cfg=cfg,
            device=device,
            num_experts_per_dev=num_experts_per_dev,
            my_dp_group=my_dp_group,
            indices=indices,
            dp_rank=dp_rank,
            dp_group=dp_group,
            num_dp_groups=num_dp_groups,
            in_tokens_bf16=in_tokens_bf16,
            gate_bf16=gate_bf16,
            up_bf16=up_bf16,
            num_recv_tokens=num_recv_tokens,
            padded_expert_counts=padded_expert_counts,
            src_token_idx=src_token_idx,
            src_dev_idx=src_dev_idx,
            expert_y=expert_y,
            expert_y_scales=expert_y_scales,
        )


    # combine step
    out_tokens = moe_cuda.TKParallelTensor(
        shape = (cfg.B * cfg.S, cfg.H),
        dtype = torch.bfloat16,
        local_rank = local_rank,
        local_world_size = local_world_size,
        multicast = False
    )

    down_bf16 = torch.rand((cfg.num_experts, cfg.H, cfg.I), dtype=torch.bfloat16, device=device) / (cfg.I ** 0.5)

    down, scale_down = quantize_2d_128(down_bf16)

    C = torch.rand((max_recv_tokens, cfg.H), dtype=torch.bfloat16, device=device) / (cfg.H ** 0.5)
    clean_print("beginning fused combine run", print_once=True)
    fused_combine_run = lambda : fused_grouped_gemm_combine(
        out_tokens,
        expert_y,
        expert_y_scales,
        down,
        scale_down,
        C,
        weights,
        padded_expert_counts,
        src_token_idx,
        src_dev_idx,
        src_slot_idx,
        cfg.num_experts,
        cfg.experts_per_token,
        num_recv_tokens,
        dp_rank,
        local_rank,
        cfg.dp_size,
        dp_group,
        num_dp_groups,
        cfg.num_comm_sms,
        cfg.num_comp_sms,
    )

    fused_combine_run()

    torch.distributed.barrier()
    torch.cuda.synchronize()
    clean_print("Fused combine run completed", print_once=True)

    mlp_compiled = torch.compile(mlp, fullgraph=True)
    full_reference_moe = lambda : reference_moe(
        cfg = cfg, in_tokens = in_tokens_bf16, up = up_bf16, gate = gate_bf16, down = down_bf16,
        indices = indices, weights = weights, num_experts = num_experts, experts_per_token = experts_per_token,
        world_size = local_world_size, mlp_compiled = mlp_compiled)

    if cfg.check_correctness:
        output = full_reference_moe()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        cosine_diff = calc_cosine_diff(output, out_tokens.data())
        clean_print(f"Moe cosine diff: {cosine_diff}")

        
if __name__ == "__main__":
    setup()
    local_rank, world_size = init_distributed_environment()

    for b, s, h, i, num_experts, experts_per_token, num_comm_sms in enumerate_moe_configs():
        cfg = TestConfig(
            B=b,
            S=s,
            I=i,
            H=h,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            dp_size=1,
            num_comm_sms=num_comm_sms,
            num_comp_sms=132 - num_comm_sms,
            local_rank=local_rank,
            local_world_size=world_size,
            check_dispatch_correctness = True,
            check_correctness = False
        )
    # max recv tokens is calculated as the maximum possible number of tokens that could be received by any dp group
    fused_dispatch_run(cfg, cfg.B * cfg.S *
                       cfg.experts_per_token * (world_size // cfg.dp_size))
