#!/usr/bin/env bash
set -Eeuo pipefail

image="${LLAMA_CPP_DOCKER_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
model_dir="${MODEL_DIR:-/mnt/nvme/models}"
hf_home_on_host="${HF_HOME:-${model_dir}/hf-cache}"
host="${VLM_SERVER_HOST:-0.0.0.0}"
port="${VLM_SERVER_PORT:-8080}"
model_ref="${MODEL_REF:-ggml-org/gemma-4-E2B-it-GGUF}"
ctx_size="${CTX_SIZE:-2048}"
n_gpu_layers="${N_GPU_LAYERS:-99}"
model_alias="${MODEL_ALIAS:-gemma4-e2b-it-q4}"
docker_gpu_args="${DOCKER_GPU_ARGS:---runtime nvidia}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on Jetson for this runtime path." >&2
  exit 2
fi

mkdir -p "${model_dir}" "${hf_home_on_host}"
read -r -a gpu_args <<< "${docker_gpu_args}"

exec docker run --rm -it \
  "${gpu_args[@]}" \
  -p "${port}:8080" \
  -v "${model_dir}:/models" \
  -v "${hf_home_on_host}:/hf-cache" \
  -e HF_HOME=/hf-cache \
  "${image}" \
  -hf "${model_ref}" \
  --alias "${model_alias}" \
  --host "${host}" \
  --port 8080 \
  -c "${ctx_size}" \
  --n-gpu-layers "${n_gpu_layers}" \
  "$@"
