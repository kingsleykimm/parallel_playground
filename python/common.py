import os
import torch
from time import perf_counter
import random
from typing import Callable, Generator
from enum import Enum
import moe_cuda
from pathlib import Path


class Major(Enum):
    K = 0
    MN = 1


def calc_cosine_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def quantize_2d_128(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float or input.dtype == torch.bfloat16
    assert input.size(-1) % 128 == 0 and input.size(-2) % 128 == 0
    num_groups = input.size(0)
    num_k_blocks = input.size(-1) / 128
    num_n_blocks = input.size(-2) / 128

    quantized = torch.empty_like(
        input, dtype=torch.float8_e4m3fn, device=input.device)
    scales = torch.empty(num_groups, num_n_blocks, num_k_blocks,
                         dtype=torch.float32, device=input.device)

    for n_block in range(num_n_blocks):
        for k_block in range(num_k_blocks):
            slice2d = input[n_block * 128: (n_block + 1)
                            * 128, k_block * 128: (k_block + 1) * 128]
            cur_scale = slice2d.abs().amax() / 448.0
            scales[n_block, k_block] = cur_scale
            quantized[n_block * 128: (n_block + 1) * 128, k_block * 128: (
                k_block + 1) * 128] = (slice2d / cur_scale).to(torch.float8_e4m3fn)
    return quantized, scales


def clean_print(*args, **kwargs):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if kwargs.pop("print_once", False):
        if local_rank == 0:
            print(*args, **kwargs)
        torch.distributed.barrier()
    else:
        for i in range(local_world_size):
            if i == local_rank:
                print(f"[Rank {i}]", *args, **kwargs)
            torch.distributed.barrier()


def align(val: int, alignment: int) -> int:
    return ((val + alignment - 1) // alignment) * alignment


def get_mk_alignment_for_contiguous_layout():
    return 128


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor, single: bool = False):
    if single:
        print("===============================================================================")
        print(f"<{name}>")
        print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        print(f"Mean:      {A.abs().mean().item():.10f}")
        print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        print(f"Max:       {A.abs().max().item():.10f}")
        print(f"Ref max:   {A_ref.abs().max().item():.10f}")
    else:
        clean_print(
            "===============================================================================", print_once=True)
        clean_print(f"<{name}>", print_once=True)
        clean_print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        clean_print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        clean_print(f"Mean:      {A.abs().mean().item():.10f}")
        clean_print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        clean_print(f"Max:       {A.abs().max().item():.10f}")
        clean_print(f"Ref max:   {A_ref.abs().max().item():.10f}")


def benchmark_no_l2_clear(
    func: Callable,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    single: bool = False,  # for the sake of consistency with benchmark_l2_clear
    use_events: bool = True  # only valid if using default stream
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    if use_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / num_iters

    else:
        start_time = perf_counter()
        for _ in range(num_iters):
            func()
        torch.cuda.synchronize()
        end_time = perf_counter()
        avg_ms = (end_time - start_time) * 1000 / num_iters

    return avg_ms


def profile(
    func: Callable,
    num_iters: int = 5,
    suffix: str = ""
) -> None:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True
    ) as profiler:
        for _ in range(num_iters):
            func()

    # Export to Chrome trace format
    trace_filename = f"rank_{int(os.environ.get('LOCAL_RANK', 0))}{'_' if suffix else ''}{suffix}_trace.json"
    profiler.export_chrome_trace(trace_filename)
    # clean_print(f"Profiler trace exported to {trace_filename}")


def quantize_1d_128(input: torch.Tensor):
    assert input.dtype == torch.float or input.dtype == torch.bfloat16
    flattened = input.reshape(-1, input.size(-1))
    num_blocks = input.size(-1) // 128

    quantized = torch.empty_like(
        flattened, dtype=torch.float8_e4m3fn, device=flattened.device)
    scales = torch.empty((num_blocks, flattened.size(0)),
                         dtype=torch.float32, device=flattened.device)
    for i in range(flattened.size(0)):

        for block in range(num_blocks):
            slice = flattened[i, block * 128: (block + 1) * 128]
            cur_scale = slice.abs().amax() / 448.0
            scales[block, i] = cur_scale
            quantized[i, block * 128: (block + 1) * 128] = (slice /
                                                            cur_scale).to(torch.float8_e4m3fn)
    return quantized, scales


def quantize_2d_128(input: torch.Tensor):
    assert input.dtype == torch.float or input.dtype == torch.bfloat16
    assert input.size(-1) % 128 == 0 and input.size(-2) % 128 == 0
    num_groups = input.size(0)
    num_k_blocks = input.size(-1) // 128
    num_n_blocks = input.size(-2) // 128

    quantized = torch.empty_like(
        input, dtype=torch.float8_e4m3fn, device=input.device)
    scales = torch.empty((num_groups, num_n_blocks, num_k_blocks),
                         dtype=torch.float32, device=input.device)

    for g in range(num_groups):
        for n_block in range(num_n_blocks):
            for k_block in range(num_k_blocks):
                slice2d = input[g, n_block *
                                128: (n_block + 1) * 128, k_block * 128: (k_block + 1) * 128]
                cur_scale = slice2d.abs().amax() / 448.0
                scales[g, n_block, k_block] = cur_scale
                quantized[g, n_block * 128: (n_block + 1) * 128, k_block * 128: (
                    k_block + 1) * 128] = (slice2d / cur_scale).to(torch.float8_e4m3fn)
    return quantized, scales


def setup():
    # initialize distributed environment, env variables
    library_root_path = os.getenv("LIBRARY_ROOT_PATH")
    if library_root_path is None:
        library_root_path = str(Path(__file__)).resolve().parents[2]
    cuda_home_path = os.getenv("CUDA_HOME_PATH") or os.getenv(
        "CUDA_PATH") or "/usr/loca/cuda/"
    moe_cuda.init(library_root_path, cuda_home_path)


def init_distributed_environment():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert world_size == local_world_size, "No multi-node configs"
    assert rank == local_rank, "no multi-node configs"

    torch.distributed.init_process_group(
        backend="nccl", device_id=local_rank, rank=rank, world_size=local_world_size
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(local_rank)
    torch.random.manual_seed(local_rank)

    return local_rank, local_world_size


def destroy_distributed_environment():
    torch.distributed.destroy_process_group()


# from DeepGEMM's python testing suite, modified for this testing

def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3))
                 for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout())
                  for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) / (k ** 0.5)
    up = torch.randn((num_groups, n, k), device='cuda',
                     dtype=torch.bfloat16) / (k ** 0.5)
    gate = torch.randn((num_groups, n, k), device='cuda',
                       dtype=torch.bfloat16) / (k ** 0.5)
    grouped_layout = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)
    scale_d = torch.empty((n // 128, m), device='cuda', dtype=torch.float32)
    # ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        grouped_layout[start: actual_end] = i
        grouped_layout[actual_end: aligned_end] = -1
        a[actual_end: aligned_end] = 0
        # ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
        start = aligned_end

    # if use_bf16:
    #     b = b if major_b.is_k_major() else b.mT.contiguous().mT
    #     return m, a, b, grouped_layout, d, ref_d

    # assert major_a.is_k_major()
    # a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    # b = grouped_cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)
    return a, up, gate, grouped_layout, d, scale_d, aligned_ms


def generate_m_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, n: int, k: int, use_bf16: bool = False,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((num_groups * max_m, k), device='cuda',
                    dtype=torch.bfloat16) / (k ** 0.5)
    up = torch.randn((num_groups, n, k), device='cuda',
                     dtype=torch.bfloat16) / (k ** 0.5)
    gate = torch.randn((num_groups, n, k), device='cuda',
                       dtype=torch.bfloat16) / (k ** 0.5)
    d = torch.empty((num_groups * max_m, n), device='cuda',
                    dtype=torch.float8_e4m3fn)
    scale_d = torch.empty((n // 128, num_groups * max_m),
                          device='cuda', dtype=torch.float32)
    # ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    # if use_bf16:
    #     return a, b, masked_m, d

    # quant_config = QuantConfig() if quant_config is None else quant_config
    # a = grouped_cast_fp8_fp4_with_major(a, MajorTypeAB.KMajor, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    # b = grouped_cast_fp8_fp4_with_major(b, MajorTypeAB.KMajor, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)

    return a, up, gate, masked_m, d, scale_d


def enumerate_grouped_gemms() -> Generator:
    m_group_list = [(4, 8192), (8, 4096), [32, 1024]]
    n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]

    for gemm_type in [moe_cuda.GemmType.MGroupedContiguous, moe_cuda.GemmType.MGroupedMasked]:
        for num_groups, expected_m_per_group in m_group_list:
            for n, k in n_k_list:
                yield num_groups, expected_m_per_group, n, k, gemm_type


# use these later

# def enumerate_m_grouped_contiguous(dtype: torch.dtype) -> Generator:
#     m_group_list = [(4, 8192), (8, 4096)]
#     n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
#     for kernel_type in get_kernel_types(dtype):
#             for num_groups, expected_m_per_group in m_group_list:
#                 for n, k in n_k_list:
#                     for major_a, major_b in get_major_ab(False, get_arch_major() != 9 or dtype != torch.float8_e4m3fn):
#                             yield kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout


# def enumerate_m_grouped_masked(dtype: torch.dtype) -> Generator:
#     max_m = 4096
#     m_group_list = [(6, 1024), (32, 192), (32, 50)]
#     n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
#     for num_groups, m in m_group_list:
#         for n, k in n_k_list:
#             yield , num_groups, max_m, m, n, k, use_psum_layout
