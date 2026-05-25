#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
llama_cpp_repo="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}"
clone_llama_cpp="${CLONE_LLAMA_CPP:-0}"
build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_dir}/build}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
build_jobs="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
enable_cuda="${ENABLE_CUDA:-auto}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

need_cmd git
need_cmd cmake

if [[ ! -d "${llama_cpp_dir}/.git" ]]; then
  if [[ "${clone_llama_cpp}" != "1" ]]; then
    cat >&2 <<EOF
llama.cpp was not found at:
  ${llama_cpp_dir}

Set LLAMA_CPP_DIR to an existing clone, or run with:
  CLONE_LLAMA_CPP=1 LLAMA_CPP_DIR="${llama_cpp_dir}" scripts/wsl/build_llama_cpp.sh

This script does not install system packages and does not clone unless requested.
EOF
    exit 2
  fi
  mkdir -p "$(dirname "${llama_cpp_dir}")"
  git clone --depth=1 "${llama_cpp_repo}" "${llama_cpp_dir}"
fi

cmake_args=(-B "${build_dir}" -S "${llama_cpp_dir}" "-DCMAKE_BUILD_TYPE=${build_type}")

if [[ "${enable_cuda}" == "auto" ]]; then
  if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
    enable_cuda="1"
  else
    enable_cuda="0"
  fi
fi

if [[ "${enable_cuda}" == "1" ]]; then
  cmake_args+=("-DGGML_CUDA=ON")
  if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  fi
else
  cmake_args+=("-DGGML_CUDA=OFF")
fi

echo "Configuring llama.cpp:"
printf '  %q' cmake "${cmake_args[@]}"
echo
cmake "${cmake_args[@]}"

echo "Building llama-server, llama-cli, and llama-mtmd-cli"
cmake --build "${build_dir}" --config "${build_type}" -j "${build_jobs}" \
  --target llama-server llama-cli llama-mtmd-cli

echo "Built binaries under ${build_dir}/bin"
