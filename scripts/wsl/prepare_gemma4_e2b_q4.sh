#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
quantize_bin="${LLAMA_QUANTIZE_BIN:-${llama_cpp_dir}/build/bin/llama-quantize}"
python_cmd="${PYTHON_CMD:-conda run -n transformers python}"

model_dir="${MODEL_DIR:-${repo_root}/models/gemma-4-E2B-it-GGUF}"
hf_repo="${MODEL_REF:-ggml-org/gemma-4-E2B-it-GGUF}"
source_model_file="${SOURCE_MODEL_FILE:-gemma-4-E2B-it-bf16.gguf}"
source_mmproj_file="${SOURCE_MMPROJ_FILE:-mmproj-gemma-4-E2B-it-bf16.gguf}"
quant_type="${QUANT_TYPE:-Q4_K_M}"
quantize_threads="${QUANTIZE_THREADS:-1}"
min_output_bytes="${MIN_OUTPUT_BYTES:-1048576000}"
output_model_file="${OUTPUT_MODEL_FILE:-gemma-4-E2B-it-${quant_type}.gguf}"
output_mmproj_file="${OUTPUT_MMPROJ_FILE:-${source_mmproj_file}}"
allow_high_memory_quantize="${ALLOW_HIGH_MEMORY_QUANTIZE:-0}"

if [[ "${allow_high_memory_quantize}" != "1" ]]; then
  cat >&2 <<EOF
Gemma BF16 -> ${quant_type} quantization is disabled by default on low-memory WSL.

This path was observed to be killed while quantizing the large embedding tensor.
Use the low-memory prepared artifact path instead:
  scripts/wsl/prepare_gemma4_e2b_q8.sh

If you are on a larger machine and intentionally want to run local quantization:
  ALLOW_HIGH_MEMORY_QUANTIZE=1 scripts/wsl/prepare_gemma4_e2b_q4.sh
EOF
  exit 2
fi

if [[ ! -x "${quantize_bin}" ]]; then
  echo "llama-quantize not found: ${quantize_bin}" >&2
  echo "Build it first with scripts/wsl/build_llama_cpp.sh" >&2
  exit 2
fi

quantize_lib_dir="$(cd "$(dirname "${quantize_bin}")" && pwd)"
export LD_LIBRARY_PATH="${quantize_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

file_size() {
  stat -c '%s' "$1"
}

needs_quantize() {
  local output_path="$1"
  if [[ ! -f "${output_path}" ]]; then
    return 0
  fi
  local size
  size="$(file_size "${output_path}")"
  if (( size < min_output_bytes )); then
    echo "Existing quantized output is too small (${size} bytes < ${min_output_bytes}); replacing: ${output_path}" >&2
    rm -f "${output_path}"
    return 0
  fi
  return 1
}

mkdir -p "${model_dir}"

echo "Downloading Gemma source GGUF files to ${model_dir}"
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
' "${hf_repo}" "${model_dir}" "${source_model_file}" "${source_mmproj_file}"

source_model_path="${model_dir}/${source_model_file}"
output_model_path="${model_dir}/${output_model_file}"

if needs_quantize "${output_model_path}"; then
  echo "Quantizing ${source_model_path} -> ${output_model_path} (${quant_type}, threads=${quantize_threads})"
  "${quantize_bin}" "${source_model_path}" "${output_model_path}" "${quant_type}" "${quantize_threads}"
else
  echo "Quantized model already exists: ${output_model_path} ($(file_size "${output_model_path}") bytes)"
fi

if [[ "${output_mmproj_file}" != "${source_mmproj_file}" ]]; then
  cp -n "${model_dir}/${source_mmproj_file}" "${model_dir}/${output_mmproj_file}"
fi

cat <<EOF
Gemma 4 E2B-it local artifacts:
  MODEL_PATH=${output_model_path}
  MMPROJ_PATH=${model_dir}/${output_mmproj_file}
EOF
