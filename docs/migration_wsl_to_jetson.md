# WSL To Jetson Migration

This workflow keeps WSL as the development machine and Jetson Orin / Orin Nano as the runtime target. The Jetson should receive a small runtime package, not reference repositories, build artifacts from WSL, or large development caches.

## Copy To Jetson

Copy:

- `src/edge_vlm/`
- `configs/`
- `scripts/`
- `docs/`
- `tests/` if you want a local sanity check

Do not copy:

- `tmp/references/`
- `.git/` unless you intentionally clone the repo on Jetson
- WSL llama.cpp build directories
- Conda environments
- Python caches
- benchmark output from unrelated runs
- model files stored outside the chosen Jetson model cache

Example sync from WSL:

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '.vscode/' \
  --exclude 'tmp/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  ./ jetson:/home/jetson/edge-vlm-lab/
```

## Model Storage

Prefer NVMe or known external storage:

```bash
sudo mkdir -p /mnt/nvme/models
sudo chown "$USER:$USER" /mnt/nvme/models
```

For Gemma 4 E2B-it, the Jetson script can use `MODEL_REF=ggml-org/gemma-4-E2B-it-GGUF` and an HF cache under the model directory.

For MiniCPM-V 4.6, pre-place converted files:

```text
/mnt/nvme/models/MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf
/mnt/nvme/models/MiniCPM-V-4_6/mmproj-model-f16.gguf
```

The conversion itself is better done on WSL or another larger Linux machine. Jetson should not be the primary conversion/build machine unless there is no alternative.

## First-Run Checklist

1. Confirm Jetson has Docker and NVIDIA container runtime configured.
2. Confirm available storage under `/mnt/nvme/models` or set `MODEL_DIR`.
3. Confirm model files or HF cache are present.
4. Start with `CTX_SIZE=2048` and Q4 quantization.
5. Start `tegrastats` logging before benchmark runs.
6. Run a text-only case before image cases.
7. Record server command, model ref, quantization, context size, and power mode with benchmark output.

## Gemma 4 E2B-it On Jetson

```bash
MODEL_DIR=/mnt/nvme/models \
CTX_SIZE=2048 \
N_GPU_LAYERS=99 \
scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

In another terminal:

```bash
scripts/common/check_server.sh
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q4.yaml \
  --output outputs/benchmarks/gemma4-e2b-jetson.jsonl
```

## MiniCPM-V 4.6 On Jetson

```bash
MODEL_DIR=/mnt/nvme/models \
CTX_SIZE=2048 \
N_GPU_LAYERS=99 \
scripts/jetson/run_minicpmv46_llama_docker.sh
```

In another terminal:

```bash
scripts/common/check_server.sh
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/minicpmv46_q4.yaml \
  --output outputs/benchmarks/minicpmv46-jetson.jsonl
```

## Common Failure Modes

- `docker: unknown runtime nvidia`: NVIDIA container runtime is not configured. Fix Jetson Docker runtime before benchmarking.
- `Model GGUF not found`: MiniCPM-V local GGUF files are missing or paths do not match `MODEL_DIR`.
- Server starts but image cases fail: mmproj may be missing, incompatible, or not loaded. Check `/v1/models` capabilities and server logs.
- Out-of-memory or process killed: lower `CTX_SIZE`, reduce parallelism, use Q4, close other processes, or move model cache to faster storage.
- Very slow first request: separate cold-start and steady-state measurements. Do not mix model download, load, first image preprocessing, and decode speed into one benchmark claim.
- Python import errors on Jetson: use the system Python or a lightweight venv; avoid Conda-heavy workflows unless explicitly needed.
