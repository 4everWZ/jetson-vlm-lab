#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
server_bin="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build/bin/llama-server}"

host="${VLM_SERVER_HOST:-127.0.0.1}"
port="${VLM_SERVER_PORT:-8080}"
ctx_size="${CTX_SIZE:-512}"
n_gpu_layers="${N_GPU_LAYERS:-0}"
model_alias="${MODEL_ALIAS:-minicpmv46-q4}"
model_ref="${MODEL_REF:-}"
default_model_path="${repo_root}/models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf"
default_mmproj_path="${repo_root}/models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf"
model_path="${MODEL_PATH:-${MINICPMV_GGUF_MODEL:-${default_model_path}}}"
mmproj_path="${MMPROJ_PATH:-${MINICPMV_MMPROJ:-${default_mmproj_path}}}"
llama_threads="${LLAMA_THREADS:-2}"
llama_threads_batch="${LLAMA_THREADS_BATCH:-${llama_threads}}"
llama_parallel="${LLAMA_PARALLEL:-1}"
llama_batch_size="${LLAMA_BATCH_SIZE:-128}"
llama_ubatch_size="${LLAMA_UBATCH_SIZE:-32}"

if [[ ! -x "${server_bin}" ]]; then
  echo "llama-server not found or not executable: ${server_bin}" >&2
  echo "Build first with scripts/wsl/build_llama_cpp.sh or set LLAMA_SERVER_BIN." >&2
  exit 2
fi

server_lib_dir="$(cd "$(dirname "${server_bin}")" && pwd)"
export LD_LIBRARY_PATH="${server_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

model_args=()
if [[ -n "${model_ref}" ]]; then
  model_args=(-hf "${model_ref}")
else
  if [[ -z "${model_path}" || -z "${mmproj_path}" ]]; then
    cat >&2 <<EOF
MiniCPM-V 4.6 requires either:
  scripts/wsl/prepare_minicpmv46_q4.sh

or explicitly supplied local GGUF files:
  MODEL_PATH=/path/to/MiniCPM-V-4_6-Q4_K_M.gguf \\
  MMPROJ_PATH=/path/to/mmproj-model-f16.gguf \\
  scripts/wsl/run_minicpmv46_llama.sh

The default files are downloaded from openbmb/MiniCPM-V-4.6-gguf.
EOF
    exit 2
  fi
  [[ -f "${model_path}" ]] || { echo "MODEL_PATH not found: ${model_path}" >&2; exit 2; }
  [[ -f "${mmproj_path}" ]] || { echo "MMPROJ_PATH not found: ${mmproj_path}" >&2; exit 2; }
  model_args=(-m "${model_path}" --mmproj "${mmproj_path}")
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
  --n-gpu-layers "${n_gpu_layers}" \
  "${extra_args[@]}" \
  "$@"
