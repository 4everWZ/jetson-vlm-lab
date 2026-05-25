#!/usr/bin/env bash
set -Eeuo pipefail

image="${LLAMA_CPP_DOCKER_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
model_dir="${MODEL_DIR:-/mnt/nvme/models}"
hf_home_on_host="${HF_HOME:-${model_dir}/hf-cache}"
host="${VLM_SERVER_HOST:-0.0.0.0}"
port="${VLM_SERVER_PORT:-8080}"
model_ref="${MODEL_REF:-ggml-org/gemma-4-E2B-it-GGUF:Q8_0}"
ctx_size="${CTX_SIZE:-2048}"
n_gpu_layers="${N_GPU_LAYERS:-99}"
model_alias="${MODEL_ALIAS:-gemma4-e2b-it-q8}"
model_path="${MODEL_PATH:-}"
mmproj_path="${MMPROJ_PATH:-}"
docker_gpu_args="${DOCKER_GPU_ARGS:---runtime nvidia}"
dry_run="${JETSON_DRY_RUN:-0}"

mkdir -p "${model_dir}" "${hf_home_on_host}"
read -r -a gpu_args <<< "${docker_gpu_args}"

model_args=()
if [[ -n "${model_path}" || -n "${mmproj_path}" ]]; then
  if [[ -z "${model_path}" || -z "${mmproj_path}" ]]; then
    echo "Set both MODEL_PATH and MMPROJ_PATH for local Gemma GGUF runtime." >&2
    exit 2
  fi
  [[ -f "${model_path}" ]] || { echo "MODEL_PATH not found on host: ${model_path}" >&2; exit 2; }
  [[ -f "${mmproj_path}" ]] || { echo "MMPROJ_PATH not found on host: ${mmproj_path}" >&2; exit 2; }
  if [[ "${model_path}" != "${model_dir}/"* || "${mmproj_path}" != "${model_dir}/"* ]]; then
    echo "MODEL_PATH and MMPROJ_PATH must be under MODEL_DIR so Docker can mount them." >&2
    echo "MODEL_DIR=${model_dir}" >&2
    exit 2
  fi
  model_args=(-m "/models/${model_path#${model_dir}/}" --mmproj "/models/${mmproj_path#${model_dir}/}")
else
  model_args=(-hf "${model_ref}")
fi

docker_cmd=(docker run --rm -it \
  "${gpu_args[@]}" \
  -p "${port}:8080" \
  -v "${model_dir}:/models" \
  -v "${hf_home_on_host}:/hf-cache" \
  -e HF_HOME=/hf-cache \
  "${image}" \
  "${model_args[@]}" \
  --alias "${model_alias}" \
  --host "${host}" \
  --port 8080 \
  -c "${ctx_size}" \
  --n-gpu-layers "${n_gpu_layers}" \
  "$@")

if [[ "${dry_run}" == "1" ]]; then
  printf '%q ' "${docker_cmd[@]}"
  printf '\n'
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on Jetson for this runtime path." >&2
  exit 2
fi

exec "${docker_cmd[@]}"
