#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"

export ENABLE_CUDA="${ENABLE_CUDA:-1}"
export LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_dir}/build-cuda}"
export BUILD_JOBS="${BUILD_JOBS:-8}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<EOF
nvcc was not found in PATH, so a CUDA llama.cpp build cannot be created.

Install or expose the CUDA toolkit in WSL, then retry:
  BUILD_JOBS=8 scripts/wsl/build_llama_cpp_cuda.sh
EOF
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<EOF
Warning: nvcc is available, but nvidia-smi failed in this shell.
The build may still succeed, but GPU runtime smoke tests need WSL GPU access.
EOF
  fi
else
  echo "Warning: nvidia-smi was not found; CUDA runtime access is not verified." >&2
fi

exec "${repo_root}/scripts/wsl/build_llama_cpp.sh" "$@"
