#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"

export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build-cuda/bin/llama-server}"
export CTX_SIZE="${CTX_SIZE:-512}"
export N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
export LLAMA_THREADS="${LLAMA_THREADS:-2}"
export LLAMA_THREADS_BATCH="${LLAMA_THREADS_BATCH:-${LLAMA_THREADS}}"
export LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-128}"
export LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-32}"
export LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"

exec "${repo_root}/scripts/wsl/run_minicpmv46_llama.sh" "$@"
