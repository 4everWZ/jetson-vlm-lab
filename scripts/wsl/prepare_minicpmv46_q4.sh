#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_cmd="${PYTHON_CMD:-conda run -n transformers python}"

model_dir="${MODEL_DIR:-${repo_root}/models/MiniCPM-V-4.6-gguf}"
hf_repo="${MODEL_REF:-openbmb/MiniCPM-V-4.6-gguf}"
model_file="${MODEL_FILE:-MiniCPM-V-4_6-Q4_K_M.gguf}"
mmproj_file="${MMPROJ_FILE:-mmproj-model-f16.gguf}"
min_model_bytes="${MIN_MODEL_BYTES:-400000000}"
min_mmproj_bytes="${MIN_MMPROJ_BYTES:-900000000}"

file_size() {
  stat -c '%s' "$1"
}

remove_if_too_small() {
  local path="$1"
  local min_bytes="$2"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  local size
  size="$(file_size "${path}")"
  if (( size < min_bytes )); then
    echo "Existing file is too small (${size} bytes < ${min_bytes}); replacing: ${path}" >&2
    rm -f "${path}"
  fi
}

mkdir -p "${model_dir}"

remove_if_too_small "${model_dir}/${model_file}" "${min_model_bytes}"
remove_if_too_small "${model_dir}/${mmproj_file}" "${min_mmproj_bytes}"

echo "Downloading pre-built MiniCPM-V 4.6 Q4_K_M GGUF files to ${model_dir}"
${python_cmd} -c '
import sys
from huggingface_hub import hf_hub_download

repo_id, local_dir, model_file, mmproj_file = sys.argv[1:5]
for filename in (model_file, mmproj_file):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    print(path, flush=True)
' "${hf_repo}" "${model_dir}" "${model_file}" "${mmproj_file}"

cat <<EOF
MiniCPM-V 4.6 Q4_K_M local artifacts:
  MODEL_PATH=${model_dir}/${model_file}
  MMPROJ_PATH=${model_dir}/${mmproj_file}

Suggested WSL CUDA launch:
  MODEL_PATH=${model_dir}/${model_file} MMPROJ_PATH=${model_dir}/${mmproj_file} MODEL_ALIAS=minicpmv46-q4 scripts/wsl/run_minicpmv46_llama_cuda.sh
EOF
