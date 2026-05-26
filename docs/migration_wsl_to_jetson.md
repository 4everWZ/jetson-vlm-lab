# WSL To Jetson Migration

This workflow keeps WSL as the development machine and Jetson Orin / Orin Nano as the runtime target. The Jetson should receive a small runtime package, not reference repositories, build artifacts from WSL, or large development caches.

## Copy To Jetson

Copy:

- `src/edge_vlm/`
- `configs/`
- `scripts/`
- `docs/`
- `data/sample_images/` and `data/sample_stream/` for small payload sanity checks
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

Before moving to the device, you can validate Docker command construction from any shell without requiring Docker or Jetson hardware:

```bash
JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

The Jetson launchers use dusty-nv `llama_cpp` containers by default. On Jetson, install jetson-containers or otherwise provide `autotag`; the scripts use `autotag llama_cpp` to select a JetPack/L4T-compatible image. If `autotag` is unavailable, the fallback image is `dustynv/llama_cpp:r36.4.0`. Override the image with `LLAMA_CPP_DOCKER_IMAGE=...`, and override the server binary path inside the container with `LLAMA_SERVER_CMD=...` if a particular image places `llama-server` somewhere unusual.

## Model Storage

Prefer NVMe or known external storage:

```bash
sudo mkdir -p /mnt/nvme/models
sudo chown "$USER:$USER" /mnt/nvme/models
```

For Gemma 4 E2B-it, use the Q8_0 GGUF model and mmproj files as the first migration artifact. You can either let the Jetson Docker script fetch `MODEL_REF=ggml-org/gemma-4-E2B-it-GGUF:Q8_0` into the HF cache under the model directory, or pre-place these files under `MODEL_DIR`:

```text
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf
/mnt/nvme/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf
```

If Jetson memory is tighter, use the pre-built Gemma Q4 files instead:

```text
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf
```

For MiniCPM-V 4.6, pre-place the official pre-built GGUF files downloaded by `scripts/wsl/prepare_minicpmv46_q4.sh`:

```text
/mnt/nvme/models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf
/mnt/nvme/models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf
```

Do not use Jetson as the primary conversion or quantization machine. The current WSL workflow uses downloaded GGUF artifacts for Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4.

## First-Run Checklist

1. Confirm Jetson has Docker and NVIDIA container runtime configured.
2. Confirm `autotag llama_cpp` resolves a dusty-nv `llama_cpp` image, or set `LLAMA_CPP_DOCKER_IMAGE` explicitly.
3. Confirm available storage under `/mnt/nvme/models` or set `MODEL_DIR`.
4. Confirm model files or HF cache are present.
5. Start Gemma with `CTX_SIZE=2048` and the Q8_0 config; lower context first if memory is tight.
6. Start `tegrastats` logging before benchmark runs.
7. Run a text-only case before image cases.
8. Record server command, container image, model ref, quantization, context size, and power mode with benchmark output.

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
  --config configs/models/gemma4_e2b_q8.yaml \
  --output outputs/benchmarks/gemma4-e2b-q8-jetson.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-jetson.md
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
  --output outputs/benchmarks/minicpmv46-jetson.jsonl \
  --summary-output outputs/benchmarks/minicpmv46-jetson.md
```

## Common Failure Modes

- `docker: unknown runtime nvidia`: NVIDIA container runtime is not configured. Fix Jetson Docker runtime before benchmarking.
- `autotag: command not found`: install jetson-containers on Jetson or set `LLAMA_CPP_DOCKER_IMAGE` to a compatible `dustynv/llama_cpp` tag.
- `llama-server not found in container`: set `LLAMA_SERVER_CMD` to the server binary path inside that dusty-nv image, or switch to a tag that includes the installed llama.cpp server binary.
- `Model GGUF not found`: model files are missing or paths do not match `MODEL_DIR`.
- Server starts but image cases fail: mmproj may be missing, incompatible, or not loaded. Check `/v1/models` capabilities and server logs.
- Out-of-memory or process killed: lower `CTX_SIZE`, reduce parallelism, close other processes, or use externally prepared lower-bit quantization. Do not run BF16-to-Q4 conversion on a memory-constrained Jetson.
- Very slow first request: separate cold-start and steady-state measurements. Do not mix model download, load, first image preprocessing, and decode speed into one benchmark claim.
- Python import errors on Jetson: use the system Python or a lightweight venv; avoid Conda-heavy workflows unless explicitly needed.
