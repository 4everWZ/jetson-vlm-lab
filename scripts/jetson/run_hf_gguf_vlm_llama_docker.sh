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
model_ref="${MODEL_REF:-ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0}"
ctx_size="${CTX_SIZE:-512}"
n_gpu_layers="${N_GPU_LAYERS:-99}"
model_alias="${MODEL_ALIAS:-${model_ref%%:*}}"
docker_gpu_args="${DOCKER_GPU_ARGS:---runtime nvidia}"
docker_tty="${DOCKER_TTY:-1}"
dry_run="${JETSON_DRY_RUN:-0}"
llama_server_cmd="${LLAMA_SERVER_CMD:-}"

mkdir -p "${model_dir}" "${hf_home_on_host}"
read -r -a gpu_args <<< "${docker_gpu_args}"
tty_args=()
if [[ "${docker_tty}" == "1" ]]; then
  tty_args=(-it)
elif [[ "${docker_tty}" != "0" ]]; then
  echo "DOCKER_TTY must be 0 or 1." >&2
  exit 2
fi

server_cmd=()
if [[ -n "${llama_server_cmd}" ]]; then
  read -r -a server_cmd <<< "${llama_server_cmd}"
else
  server_cmd=(/bin/bash -lc 'if command -v llama-server >/dev/null 2>&1; then server="$(command -v llama-server)"; elif [[ -x /usr/local/bin/llama-server ]]; then server=/usr/local/bin/llama-server; elif [[ -x /opt/llama.cpp/build/bin/llama-server ]]; then server=/opt/llama.cpp/build/bin/llama-server; elif command -v server >/dev/null 2>&1; then server="$(command -v server)"; else echo "llama-server not found in container; set LLAMA_SERVER_CMD" >&2; exit 127; fi; exec "${server}" "$@"' --)
fi

docker_cmd=(docker run --rm \
  "${tty_args[@]}" \
  "${gpu_args[@]}" \
  -p "${port}:8080" \
  -v "${model_dir}:/models" \
  -v "${hf_home_on_host}:/hf-cache" \
  -e HF_HOME=/hf-cache \
  "${image}" \
  "${server_cmd[@]}" \
  -hf "${model_ref}" \
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
