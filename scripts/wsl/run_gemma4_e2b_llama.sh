#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
server_bin="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build/bin/llama-server}"

host="${VLM_SERVER_HOST:-127.0.0.1}"
port="${VLM_SERVER_PORT:-8080}"
model_ref="${MODEL_REF:-ggml-org/gemma-4-E2B-it-GGUF}"
ctx_size="${CTX_SIZE:-4096}"
n_gpu_layers="${N_GPU_LAYERS:-99}"
model_alias="${MODEL_ALIAS:-gemma4-e2b-it-q4}"

if [[ ! -x "${server_bin}" ]]; then
  echo "llama-server not found or not executable: ${server_bin}" >&2
  echo "Build first with scripts/wsl/build_llama_cpp.sh or set LLAMA_SERVER_BIN." >&2
  exit 2
fi

extra_args=()
if [[ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${LLAMA_SERVER_EXTRA_ARGS}"
fi

exec "${server_bin}" \
  -hf "${model_ref}" \
  --alias "${model_alias}" \
  --host "${host}" \
  --port "${port}" \
  -c "${ctx_size}" \
  --n-gpu-layers "${n_gpu_layers}" \
  "${extra_args[@]}" \
  "$@"
