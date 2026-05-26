#!/usr/bin/env bash
set -Eeuo pipefail

python_cmd="${PYTHON_CMD:-conda run -n transformers python}"
hf_repo="${MODEL_REF:-openbmb/MiniCPM-V-4.6-gguf}"

echo "Inspecting MiniCPM-V 4.6 pre-built GGUF metadata without downloading model weights"
echo "  HF repo: ${hf_repo}"

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

cat <<'EOF'

No model files were downloaded by this inspection.
To download the official Q4_K_M model and mmproj files:
  scripts/wsl/prepare_minicpmv46_q4.sh
EOF
