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
repo_id="${MODEL_REPO:-${model_ref%%:*}}"
model_subdir="${MODEL_SUBDIR:-${repo_id}}"
model_file="${MODEL_FILE:-SmolVLM2-256M-Video-Instruct-Q8_0.gguf}"
mmproj_file="${MMPROJ_FILE:-mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf}"
ctx_size="${CTX_SIZE:-512}"
n_gpu_layers="${N_GPU_LAYERS:-99}"
model_alias="${MODEL_ALIAS:-${model_ref%%:*}}"
docker_gpu_args="${DOCKER_GPU_ARGS:---runtime nvidia}"
docker_tty="${DOCKER_TTY:-1}"
dry_run="${JETSON_DRY_RUN:-0}"
llama_server_cmd="${LLAMA_SERVER_CMD:-}"

host_model_path="${MODEL_PATH_ON_HOST:-${model_dir}/${model_subdir}/${model_file}}"
host_mmproj_path="${MMPROJ_PATH_ON_HOST:-${model_dir}/${model_subdir}/${mmproj_file}}"
container_model_path="${MODEL_PATH:-/models/${model_subdir}/${model_file}}"
container_mmproj_path="${MMPROJ_PATH:-/models/${model_subdir}/${mmproj_file}}"

mkdir -p "${model_dir}" "${hf_home_on_host}" "$(dirname "${host_model_path}")" "$(dirname "${host_mmproj_path}")"
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
  -m "${container_model_path}" \
  --mmproj "${container_mmproj_path}" \
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

download_hf_file() {
  local filename="$1"
  local destination="$2"
  if [[ -f "${destination}" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download ${repo_id}/${filename}; pre-place the file or install curl." >&2
    exit 2
  fi
  local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
  local partial="${destination}.partial"
  echo "Downloading ${url} -> ${destination}" >&2
  curl --fail --location --retry 3 --continue-at - --output "${partial}" "${url}"
  mv "${partial}" "${destination}"
}

download_hf_file "${model_file}" "${host_model_path}"
download_hf_file "${mmproj_file}" "${host_mmproj_path}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on Jetson for this runtime path." >&2
  exit 2
fi

exec "${docker_cmd[@]}"
