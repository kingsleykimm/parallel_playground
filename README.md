# parallel_playground (moe_cuda)

Worklog of optimizing FP8/BF16 MoE Pipelines on H100

## Build

```bash
./run_cmake.sh
```

Or manually:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Required dependencies

- MPI is mandatory at configure/build time.
- NCCL is mandatory and is discovered via `find_package(NCCL)` first, with `NCCL_ROOT` fallback search.
- `MOE_CUDA_USE_MPI=1` is always enabled for this project.
- If `Torch_DIR` points to a pip/venv torch install, CMake also checks sibling `site-packages/nvidia/nccl` automatically.
- On this cluster/CMake combo, set `-DMPI_SKIP_COMPILER_WRAPPER=TRUE` if `FindMPI` mis-parses wrapper include flags.

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

