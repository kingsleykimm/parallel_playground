# parallel_playground (moe_cuda)

Worklog of optimizing FP8/BF16 MoE Pipelines on H100 with ThunderKittens. WIP + Experimental

## Build

```bash
./run_cmake.sh
```

## Python Development Build

```bash
./develop.sh
```

This mirrors the `h100_gdn_cuda` flow: it builds a `moe_cuda` Python extension through `setup.py` + CMake, then symlinks the built `.so` into the repo root.

Or manually:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Required dependencies

- NCCL is mandatory and is discovered via `find_package(NCCL)` first, with `NCCL_ROOT` fallback search.
- If `Torch_DIR` points to a pip/venv torch install, CMake also checks sibling `site-packages/nvidia/nccl` automatically.

## JIT Environment

JIT include root is environment-driven. There is no hardcoded library subfolder in include path resolution.

`Compiler::init_static_vars(library_root, cuda_home)` resolves include path as:
1. `LIBRARY_INCLUDE_PATH` (if set)
2. `${library_root}/include` (fallback)

Recommended `.env` variables:

```bash
export LIBRARY_ROOT_PATH="/abs/path/to/library/root"
export LIBRARY_INCLUDE_PATH="/abs/path/to/include/root"
export CUDA_HOME_PATH="/usr/local/cuda"
export JIT_CACHE_DIR="/tmp/.moe_cuda"
export JIT_USE_NVRTC=1
```

# Current Issues:
- [ ] we need to add better sychronization / overlap between the host side and the dispatch_send kernel
    - [ ] once the num_routed is copied in, we can make the kernel issue nvlink scatters for the num_routed tensors
    - [ ] once the tokens are copied into the send buffers in dispatch-send, the worker's process_routing_info can be set up, preparing for dispatch_recv, we dont need to use the dispatch-send_done flag, this is mainly for send buffer RDMA copies
- [ ] zero initialize sync_counter, right now we get away with it since it's auto zero initialized
- [ ] edit the dispatch kernels to take in an elemsize
- [] fuse topk into dispatch_send

# TODO WORKLOG:
- [ ] determine whether we dispatch NUM_EXPERTS_PER_TOKEN and NUM_EXPERTS at compile time, or if this is templated all the way up to the a2a state (I think it should probably jsut be templated all the way up, since it's initialized once)
- [ ] link the a2a_dispatch_recv
- [ ] fix the barrier issue, investigate if this is just clangd complaining or a serious compile time issue
- [ ] after a2a dispatch is tested and confirms it works, we need to add in mn-layout friendly dispatch-recv/send, or do the transpose when permuting the tokens before grouped-gemm

- plan for a2a structure:

always initialize the ParallelTensors, recv buffer, send buffer, num routed tensor and barrier tensor (metadata) on the state initialization
then persist these down into the worker - putting them there allows the state + context + worker to all access them
