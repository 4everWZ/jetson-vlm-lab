#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve_llama_cpp_image.sh
source "${script_dir}/resolve_llama_cpp_image.sh"

image="$(resolve_llama_cpp_image)"
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
llama_server_cmd="${LLAMA_SERVER_CMD:-}"

mkdir -p "${model_dir}" "${hf_home_on_host}"
read -r -a gpu_args <<< "${docker_gpu_args}"

server_cmd=()
if [[ -n "${llama_server_cmd}" ]]; then
  read -r -a server_cmd <<< "${llama_server_cmd}"
else
  server_cmd=(/bin/bash -lc 'if command -v llama-server >/dev/null 2>&1; then server="$(command -v llama-server)"; elif [[ -x /usr/local/bin/llama-server ]]; then server=/usr/local/bin/llama-server; elif [[ -x /opt/llama.cpp/build/bin/llama-server ]]; then server=/opt/llama.cpp/build/bin/llama-server; elif command -v server >/dev/null 2>&1; then server="$(command -v server)"; else echo "llama-server not found in container; set LLAMA_SERVER_CMD" >&2; exit 127; fi; exec "${server}" "$@"' --)
fi

model_args=()
if [[ -n "${model_path}" || -n "${mmproj_path}" ]]; then
  if [[ -z "${model_path}" || -z "${mmproj_path}" ]]; then
    echo "Set both MODEL_PATH and MMPROJ_PATH for local Gemma GGUF runtime." >&2
    exit 2
  fi
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
  "${server_cmd[@]}" \
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

if [[ -n "${model_path}" || -n "${mmproj_path}" ]]; then
  [[ -f "${model_path}" ]] || { echo "MODEL_PATH not found on host: ${model_path}" >&2; exit 2; }
  [[ -f "${mmproj_path}" ]] || { echo "MMPROJ_PATH not found on host: ${mmproj_path}" >&2; exit 2; }
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on Jetson for this runtime path." >&2
  exit 2
fi

exec "${docker_cmd[@]}"
