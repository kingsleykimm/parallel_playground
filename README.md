# parallel_playground (moe_cuda)

It says moe_cuda, but I'm just experimenting with a bunch of different parallelism strategies, like EP, TP + SP, Ring attention, etc.

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

# Roadmap:
- [ ] implement a fused dispatch + swiglu grouped gemm
    - [ ] inside here, enforce transposed sfa majors
    - [ ] keep linking this together, and also remember to zero out the comm_comp_barriers after each dispatch
- [] fuse gate matmul + grouped topk kernels into one kernel
- [ ] look at fusing SP + TP kernels to match megatron forward pass


# TO-DO today:
- [ ] ncu swiglu grouped gemm across different shapes and tactics
- [ ] work on testing and benchmarking swiglu + full a2a, find optimization spots
- [ ] i suspect a dispatch gemm is what's next
- [ ] compare performance of fused swiglu with liger kernel