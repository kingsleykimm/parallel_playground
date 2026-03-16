#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

BUILD_DIR="${1:-build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

cmake -S . -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKITTENS_HOPPER=ON \
  -DFETCH_LIBTORCH=ON \
  -DCMAKE_CUDA_FLAGS="-lineinfo"

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "Build complete."
echo "Artifacts are in: ${BUILD_DIR}/"
