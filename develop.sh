#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

# --- Submodules (only when inside a git repo) ---
if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  git submodule update --init --recursive \
    third-party/fmt \
    third-party/ThunderKittens
fi

# --- Verify required third-party trees exist ---
check_dir() {
  local label="$1" path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} not found at ${path}" >&2
    echo "  Run: git submodule update --init --recursive third-party/$(basename "$(dirname "${path}")")" >&2
    exit 1
  fi
}

mkdir -p moe_cuda/include
check_dir "ThunderKittens headers"     "${ROOT_DIR}/third-party/ThunderKittens/include"

# --- CMake configure + build ---
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

CMAKE_EXTRA_ARGS=()
if [[ -n "${Torch_DIR:-}" ]]; then
  CMAKE_EXTRA_ARGS+=("-DTorch_DIR=${Torch_DIR}")
fi

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMPI_SKIP_COMPILER_WRAPPER=TRUE \
  -U MPI_* \
  "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"


# Symlink compile_commands.json to root so clangd picks it up automatically
ln -sfn "${BUILD_DIR}/compile_commands.json" "${ROOT_DIR}/compile_commands.json"
echo "Linked compile_commands.json -> ${BUILD_DIR}/compile_commands.json"

echo "Build complete. Artifacts: ${BUILD_DIR}/"
