#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-${repo_root}/tmp/llama.cpp}"
convert_script="${LLAMA_CONVERT_SCRIPT:-${llama_cpp_dir}/convert_hf_to_gguf.py}"
conversion_dir="${LLAMA_CONVERSION_DIR:-${llama_cpp_dir}/conversion}"
python_cmd="${PYTHON_CMD:-conda run -n transformers python}"
hf_repo="${MODEL_REF:-openbmb/MiniCPM-V-4_6}"

echo "Inspecting MiniCPM-V 4.6 metadata without downloading model weights"
echo "  HF repo: ${hf_repo}"
echo "  llama.cpp convert script: ${convert_script}"
echo "  llama.cpp conversion modules: ${conversion_dir}"

${python_cmd} -c '
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
info = HfApi().model_info(repo_id=repo_id, files_metadata=True)

total_size = 0
known_size_count = 0
print("HF files:", flush=True)
for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
    name = sibling.rfilename
    size = getattr(sibling, "size", None)
    if size is None:
        print(f"  {name}\tunknown-size", flush=True)
        continue
    total_size += size
    known_size_count += 1
    print(f"  {name}\t{size} bytes", flush=True)

if known_size_count:
    gib = total_size / (1024 ** 3)
    print(f"Known total file size: {total_size} bytes ({gib:.2f} GiB)", flush=True)
else:
    print("Known total file size: unavailable from HF metadata", flush=True)
' "${hf_repo}"

if [[ -f "${convert_script}" ]]; then
  echo
  echo "llama.cpp conversion-script signals:"
  if [[ -d "${conversion_dir}" ]] && grep -R -q "MiniCPMV4_6ForConditionalGeneration" "${conversion_dir}"; then
    echo "  conversion modules register MiniCPM-V 4.6"
  else
    echo "  MiniCPM-V 4.6 registration was not found in conversion modules"
  fi
  if [[ -d "${conversion_dir}" ]] && grep -R -q "MINICPMV4_6" "${conversion_dir}"; then
    echo "  conversion modules include MiniCPM-V 4.6 projector metadata"
  else
    echo "  MiniCPM-V 4.6 projector metadata was not found in conversion modules"
  fi
  if ${python_cmd} "${convert_script}" --help 2>/dev/null | grep -q -- "--mmproj"; then
    echo "  --mmproj option is present"
  else
    echo "  --mmproj option was not found in --help output"
  fi
else
  echo
  echo "llama.cpp convert script not found. Build or clone llama.cpp first:"
  echo "  CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh"
fi

cat <<'EOF'

No model files were downloaded by this inspection.
If the metadata and llama.cpp revision look acceptable, run full preparation only on a machine with enough memory/storage:
  ALLOW_MINICPM_FULL_PREPARE=1 scripts/wsl/prepare_minicpmv46_q4.sh
EOF
