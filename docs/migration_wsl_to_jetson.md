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

The Jetson launchers default to the self-built official llama.cpp image used by
the observed multimodal smoke runs. The dusty-nv `llama_cpp` image is not the
default because it has not provided the multimodal `llama-server` path this repo
needs. Override the image with `LLAMA_CPP_DOCKER_IMAGE=...`, set
`LLAMA_CPP_USE_AUTOTAG=1` only when you intentionally want `autotag llama_cpp`
selection, and override the server binary path inside the container with
`LLAMA_SERVER_CMD=...` if a particular image places `llama-server` somewhere
unusual. The Jetson sweep plan also probes `llama-server --help` inside the
selected image and records whether the runtime exposes `--mmproj`; image-capable
variants are skipped with `runtime_missing_mmproj_support` when the runtime
cannot satisfy this repo's multimodal path.
For the Qwen3-VL 2B Instruct pair, `scripts/jetson/select_qwen3_instruct_variant.sh`
wraps the current runtime probe plus preflight gate into one Q4-first / Q8
fallback decision and writes the result as JSON.

Remote Jetson connection settings belong in the ignored `.env.jetson` file.
`scripts/jetson/remote_exec.sh` sources it automatically. Do not put SSH hosts,
passwords, tokens, or private paths into tracked documentation.

The observed Jetson smoke runs used:

```bash
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87
```

## Model Storage

Prefer NVMe or known external storage:

```bash
sudo mkdir -p /mnt/nvme/models
sudo chown "$USER:$USER" /mnt/nvme/models
```

For Gemma 4 E2B-it, use the pre-built Q4 files for the observed Jetson smoke path:

```text
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf
```

The Q8_0 GGUF model and mmproj files remain the verified WSL CUDA baseline and can be staged for a future Jetson Q8 check, but Jetson Q8 is not yet observed:

```text
/mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf
/mnt/nvme/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf
```

For MiniCPM-V 4.6, pre-place the official pre-built GGUF files downloaded by `scripts/wsl/prepare_minicpmv46_q4.sh`:

```text
/mnt/nvme/models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf
/mnt/nvme/models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf
```

Do not use Jetson as the primary conversion or quantization machine. The current WSL workflow uses downloaded GGUF artifacts for Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4.

## First-Run Checklist

1. Confirm Jetson has Docker and NVIDIA container runtime configured.
2. Confirm the default self-built official llama.cpp image is present, or set `LLAMA_CPP_DOCKER_IMAGE` explicitly.
3. Confirm available storage under `/mnt/nvme/models` or set `MODEL_DIR`.
4. Confirm model files or HF cache are present.
5. Start the observed Q4 smoke paths first: MiniCPM-V 4.6 Q4 with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, batch 128, ubatch 32; or Gemma Q4 with `CTX_SIZE=512`, `N_GPU_LAYERS=12`, batch 512, ubatch 512.
6. Start `tegrastats` logging before benchmark runs.
7. Run a text-only case before image cases.
8. Record server command, container image, model ref, quantization, context size, and power mode with benchmark output.
9. For formal runs, prefer `scripts/jetson/run_formal_benchmark.sh` so benchmark JSONL, summary, manifest, profile files, and `tegrastats` share one run id.

## Gemma 4 E2B-it On Jetson

```bash
MODEL_DIR=$PWD/models \
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
CTX_SIZE=512 \
N_GPU_LAYERS=12 \
scripts/jetson/run_gemma4_e2b_llama_docker.sh \
  -fit off \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-warmup
```

In another terminal:

```bash
scripts/common/check_server.sh
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q4.yaml \
  --output outputs/benchmarks/gemma4-e2b-q4-jetson.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q4-jetson.md \
  --max-tokens 64 \
  --temperature 0
```

For the observed Gemma Q4 smoke, keep mmproj on GPU. The comparison run with `--no-mmproj-offload` also completed, but image throughput was lower.

## MiniCPM-V 4.6 On Jetson

```bash
MODEL_DIR=$PWD/models \
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
CTX_SIZE=512 \
N_GPU_LAYERS=32 \
scripts/jetson/run_minicpmv46_llama_docker.sh \
  --parallel 1 \
  --batch-size 128 \
  --ubatch-size 32 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-warmup
```

In another terminal:

```bash
scripts/common/check_server.sh
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/minicpmv46_q4.yaml \
  --output outputs/benchmarks/minicpmv46-q4-jetson.jsonl \
  --summary-output outputs/benchmarks/minicpmv46-q4-jetson.md \
  --max-tokens 64 \
  --temperature 0
```

## Common Failure Modes

- `docker: unknown runtime nvidia`: NVIDIA container runtime is not configured. Fix Jetson Docker runtime before benchmarking.
- `dustynv/llama_cpp` was selected unexpectedly: unset `LLAMA_CPP_USE_AUTOTAG`,
  or set `LLAMA_CPP_DOCKER_IMAGE` to the self-built official llama.cpp image.
- Sweep skipped with `runtime_missing_mmproj_support`: the selected image did
  not expose `--mmproj` in `llama-server --help`, so use the self-built
  official llama.cpp image or another image with confirmed multimodal support.
- Selector returned no usable variant: inspect the JSON from
  `scripts/jetson/select_qwen3_instruct_variant.sh` to see whether the block was
  strict `lfb`, missing artifacts, or runtime multimodal support.
- `llama-server not found in container`: set `LLAMA_SERVER_CMD` to the server
  binary path inside that image, or switch to a tag that includes the installed
  llama.cpp server binary.
- `Model GGUF not found`: model files are missing or paths do not match `MODEL_DIR`.
- Server starts but image cases fail: mmproj may be missing, incompatible, or not loaded. Check `/v1/models` capabilities and server logs.
- Out-of-memory or process killed: lower `CTX_SIZE`, reduce parallelism, close other processes, or use externally prepared lower-bit quantization. Do not run BF16-to-Q4 conversion on a memory-constrained Jetson.
- Very slow first request: separate cold-start and steady-state measurements. Do not mix model download, load, first image preprocessing, and decode speed into one benchmark claim.
- Python import errors on Jetson: use the system Python or a lightweight venv; avoid Conda-heavy workflows unless explicitly needed.
