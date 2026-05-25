#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
convert_script="${LLAMA_CONVERT_SCRIPT:-${llama_cpp_dir}/convert_hf_to_gguf.py}"
quantize_bin="${LLAMA_QUANTIZE_BIN:-${llama_cpp_dir}/build/bin/llama-quantize}"
python_cmd="${PYTHON_CMD:-conda run -n transformers python}"

hf_repo="${MODEL_REF:-openbmb/MiniCPM-V-4_6}"
model_dir="${MODEL_DIR:-${repo_root}/models/MiniCPM-V-4_6}"
f16_model="${F16_MODEL:-${model_dir}/ggml-model-f16.gguf}"
f16_mmproj="${F16_MMPROJ:-${model_dir}/mmproj-model-f16.gguf}"
quant_type="${QUANT_TYPE:-Q4_K_M}"
quantize_threads="${QUANTIZE_THREADS:-1}"
min_output_bytes="${MIN_OUTPUT_BYTES:-1048576000}"
q_model="${Q_MODEL:-${model_dir}/ggml-model-${quant_type}.gguf}"

if [[ ! -f "${convert_script}" ]]; then
  echo "convert_hf_to_gguf.py not found: ${convert_script}" >&2
  echo "Build or clone llama.cpp first with scripts/wsl/build_llama_cpp.sh" >&2
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

echo "Downloading MiniCPM-V 4.6 HF checkpoint to ${model_dir}"
${python_cmd} -c '
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:3]
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    ],
)
print(path, flush=True)
' "${hf_repo}" "${model_dir}"

if [[ ! -f "${f16_model}" ]]; then
  echo "Converting MiniCPM-V 4.6 language model to ${f16_model}"
  ${python_cmd} "${convert_script}" "${model_dir}" --outfile "${f16_model}"
else
  echo "Language-model GGUF already exists: ${f16_model}"
fi

if [[ ! -f "${f16_mmproj}" ]]; then
  echo "Converting MiniCPM-V 4.6 mmproj to ${f16_mmproj}"
  ${python_cmd} "${convert_script}" "${model_dir}" --mmproj --outfile "${f16_mmproj}"
else
  echo "MMPROJ GGUF already exists: ${f16_mmproj}"
fi

if needs_quantize "${q_model}"; then
  echo "Quantizing ${f16_model} -> ${q_model} (${quant_type}, threads=${quantize_threads})"
  "${quantize_bin}" "${f16_model}" "${q_model}" "${quant_type}" "${quantize_threads}"
else
  echo "Quantized model already exists: ${q_model} ($(file_size "${q_model}") bytes)"
fi

cat <<EOF
MiniCPM-V 4.6 local artifacts:
  MODEL_PATH=${q_model}
  MMPROJ_PATH=${f16_mmproj}
EOF
