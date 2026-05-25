#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
server_bin="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build/bin/llama-server}"

host="${VLM_SERVER_HOST:-127.0.0.1}"
port="${VLM_SERVER_PORT:-8080}"
model_ref="${MODEL_REF:-ggml-org/gemma-4-E2B-it-GGUF:Q8_0}"
ctx_size="${CTX_SIZE:-1024}"
n_gpu_layers="${N_GPU_LAYERS:-0}"
model_alias="${MODEL_ALIAS:-gemma4-e2b-it-q8}"
model_path="${MODEL_PATH:-${GEMMA4_GGUF_MODEL:-}}"
mmproj_path="${MMPROJ_PATH:-${GEMMA4_MMPROJ:-}}"
llama_threads="${LLAMA_THREADS:-4}"
llama_threads_batch="${LLAMA_THREADS_BATCH:-${llama_threads}}"
llama_parallel="${LLAMA_PARALLEL:-1}"
llama_batch_size="${LLAMA_BATCH_SIZE:-256}"
llama_ubatch_size="${LLAMA_UBATCH_SIZE:-64}"
cache_type_k="${CACHE_TYPE_K:-q8_0}"
cache_type_v="${CACHE_TYPE_V:-q8_0}"

if [[ ! -x "${server_bin}" ]]; then
  echo "llama-server not found or not executable: ${server_bin}" >&2
  echo "Build first with scripts/wsl/build_llama_cpp.sh or set LLAMA_SERVER_BIN." >&2
  exit 2
fi

server_lib_dir="$(cd "$(dirname "${server_bin}")" && pwd)"
export LD_LIBRARY_PATH="${server_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

model_args=()
if [[ -n "${model_path}" || -n "${mmproj_path}" ]]; then
  if [[ -z "${model_path}" || -z "${mmproj_path}" ]]; then
    echo "Set both MODEL_PATH and MMPROJ_PATH for local Gemma GGUF runtime." >&2
    exit 2
  fi
  [[ -f "${model_path}" ]] || { echo "MODEL_PATH not found: ${model_path}" >&2; exit 2; }
  [[ -f "${mmproj_path}" ]] || { echo "MMPROJ_PATH not found: ${mmproj_path}" >&2; exit 2; }
  model_args=(-m "${model_path}" --mmproj "${mmproj_path}")
else
  model_args=(-hf "${model_ref}")
fi

extra_args=()
if [[ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${LLAMA_SERVER_EXTRA_ARGS}"
fi

exec "${server_bin}" \
  "${model_args[@]}" \
  --alias "${model_alias}" \
  --host "${host}" \
  --port "${port}" \
  -c "${ctx_size}" \
  --threads "${llama_threads}" \
  --threads-batch "${llama_threads_batch}" \
  --parallel "${llama_parallel}" \
  --batch-size "${llama_batch_size}" \
  --ubatch-size "${llama_ubatch_size}" \
  --cache-type-k "${cache_type_k}" \
  --cache-type-v "${cache_type_v}" \
  --n-gpu-layers "${n_gpu_layers}" \
  "${extra_args[@]}" \
  "$@"
