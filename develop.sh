#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set +u
  source .env
  set -u
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

# Wipe old artifacts to avoid stale extension/binary mismatches.
# if possible, do this manually, since if you want to return to using the csrc, refetching and installing libtorch takes a while, upwards of 10 minutes
# rm -rf build build_* dist 

rm -rf ./*.egg-info
rm -f ./*.so

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Error: python not found. Set PYTHON=/path/to/python or activate an environment." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" setup.py build

so_file="$("${PYTHON_BIN}" - <<'PY'
import glob
matches = sorted(glob.glob("build/**/moe_cuda*.so", recursive=True))
print(matches[0] if matches else "")
PY
)"
if [[ -z "${so_file}" ]]; then
  echo "Error: no built moe_cuda*.so found under build/" >&2
  exit 1
fi

ln -sfn "${so_file}" "./$(basename "${so_file}")"
echo "Linked module: $(basename "${so_file}") -> ${so_file}"
