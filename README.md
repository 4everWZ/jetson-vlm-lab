# Jetson VLM Lab

WSL-first edge VLM workflow for validating GGUF vision-language models before moving the smallest runtime package to Jetson Orin / Orin Nano.

The default path is deliberately practical: download pre-built GGUF artifacts, start `llama-server`, run the shared benchmark, then copy only source/configs/scripts/docs and model files to Jetson storage. Local model quantization is not part of the normal WSL flow for this machine.

## What This Repo Provides

- llama.cpp `llama-server` as the first supported backend for GGUF models.
- Shared Python client and benchmark harness for WSL and Jetson.
- Separate WSL and Jetson scripts.
- Runtime model configs outside source code.
- Reference notes from the OrangePi MiniCPM-V 4.6 project without porting Ascend-specific code.
- A documented migration path from WSL validation to Jetson runtime checks.

## Verified Status

| Area | Status |
|---|---|
| Python environment | Uses local Conda env: `conda run -n transformers python`. |
| llama.cpp CPU build | Present under `tmp/llama.cpp/build/bin`; useful as fallback when GPU access is blocked. |
| llama.cpp CUDA build | Verified locally under `tmp/llama.cpp/build-cuda` with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, and `BUILD_JOBS=8`. |
| WSL GPU visibility | `nvcc` is available. `nvidia-smi` works with full access and shows an RTX 3060 Laptop GPU; sandboxed commands may not see NVML. |
| Gemma 4 E2B-it Q8 | Official pre-quantized Q8_0 model and mmproj files are present under ignored `models/` storage. |
| Gemma 4 E2B-it Q4 | Uses pre-built `Q4_K_M` GGUF from `mradermacher/gemma-4-E2B-it-GGUF`. WSL CUDA text, sample-image benchmark, and one-frame fake-stream checks passed with `VLM_SERVER_PORT=18083`. |
| Gemma Q8 WSL CUDA smoke | Text and sample-image benchmark passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, one server slot, and `VLM_SERVER_PORT=18081`. The wrapper-default real run wrote `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl` and `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`. |
| MiniCPM-V 4.6 | Official pre-built `Q4_K_M` model and F16 mmproj files from `openbmb/MiniCPM-V-4.6-gguf` are downloaded under ignored `models/` storage. WSL CUDA text, sample-image benchmark, and one-frame fake-stream checks passed with `VLM_SERVER_PORT=18082`. |
| Jetson runtime | Scripted and documented with dusty-nv `llama_cpp` containers through Jetson Docker launchers, but not yet observed on Jetson hardware in this repository. |

Do not treat dry runs or server startup as performance results. Performance claims need real benchmark JSONL from a running model/server. The current observed runtime support covers Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 on WSL CUDA only; it does not validate Jetson runtime or broad performance.

## Repository Layout

```text
configs/models/                  model runtime configs
configs/benchmark/               shared benchmark prompt cases
docs/                            design, migration, benchmark, matrix, and reference notes
scripts/wsl/                     WSL build, prepare, and run scripts
scripts/jetson/                  Jetson Docker launch and monitor scripts
scripts/common/                  shared helper scripts
src/edge_vlm/                    OpenAI-compatible client and benchmark code
tests/                           lightweight contract tests
tmp/references/                  ignored reference clones
models/                          ignored local model artifacts
outputs/                         ignored benchmark logs
```

## Prerequisites

- WSL on the Windows host.
- Conda environment named `transformers` for Python commands.
- `git` and `cmake` for llama.cpp builds.
- CUDA toolkit in WSL for CUDA builds; this workspace has `nvcc` 12.0 available.
- Optional full-access shell for GPU runtime checks when sandboxed commands cannot access NVML.

Do not install dependencies into Conda `base`, system Python, or global Python for this project.

## Quick Start On WSL

Run the contract tests first:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

Download pre-built model artifacts. Use Gemma Q8 for the already verified WSL CUDA baseline, Gemma Q4 when memory/storage pressure matters, and MiniCPM Q4 for the verified smaller WSL CUDA VLM path:

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
scripts/wsl/prepare_gemma4_e2b_q4.sh
scripts/wsl/prepare_minicpmv46_q4.sh
```

Build llama.cpp if the local build directories are missing:

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
scripts/wsl/build_llama_cpp_cuda.sh
```

Start the verified Gemma Q8 CUDA baseline:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
VLM_SERVER_PORT=18081 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

In another terminal, run the real benchmark against that server:

```bash
VLM_SERVER_PORT=18081 scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=18081 conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl.md \
  --max-tokens 64 \
  --temperature 0
```

Use dry-run only to validate payload construction and JSONL logging without a server:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-dryrun.md \
  --dry-run
```

## llama.cpp Builds

CPU fallback build, useful when GPU runtime access is blocked:

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
```

CUDA build for this WSL machine:

```bash
scripts/wsl/build_llama_cpp_cuda.sh
```

The CUDA wrapper defaults to:

- `LLAMA_CPP_BUILD_DIR=$PWD/tmp/llama.cpp/build-cuda`
- `BUILD_JOBS=8`
- `CMAKE_CUDA_ARCHITECTURES=86`

Keep `BUILD_JOBS=8` for this 12 GiB RAM + 6 GiB swap WSL setup unless you intentionally retune the machine. More build parallelism is not required for the current workflow.

## Model Artifacts

Q8 baseline:

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

Downloads:

- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf`
- `models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf`

Q4 lower-memory option:

```bash
scripts/wsl/prepare_gemma4_e2b_q4.sh
```

Downloads:

- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf`
- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf`

These scripts download existing GGUF artifacts. They do not run `llama-quantize`.

## Run Gemma 4 E2B-it

CPU fallback, useful when GPU access is blocked:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
scripts/wsl/run_gemma4_e2b_llama.sh
```

WSL CUDA path:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

Q4 uses the same launcher with different artifacts and alias:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

The CUDA launcher defaults to `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, two threads, one server slot, and no warmup. The 512 batch/ubatch setting is required for the observed Gemma Q8 image path on this llama.cpp build; the lower 32 ubatch text-only setting triggered a llama.cpp image assertion. On a memory-rich run, raise the offload explicitly:

```bash
N_GPU_LAYERS=48 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
N_GPU_LAYERS=99 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

Use `nvidia-smi` in another terminal to watch VRAM. Using 8-10 GiB of WSL host memory for build/runtime is acceptable here; avoid settings that push the process into OOM. `BUILD_JOBS` mostly spends host RAM and CPU, so the default is intentionally aggressive for this 12 GiB RAM + 6 GiB swap WSL setup. `N_GPU_LAYERS` spends GPU VRAM, so tune it against the 6 GiB RTX 3060 Laptop budget.

When running Q4, use `configs/models/gemma4_e2b_q4.yaml` for benchmark records.

## MiniCPM-V 4.6

Inspect the official pre-built GGUF repo without downloading weights:

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

Download the official pre-built Q4_K_M model and F16 mmproj files:

```bash
scripts/wsl/prepare_minicpmv46_q4.sh
```

Downloads:

- `models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf`
- `models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf`

Run the WSL CUDA path after the files exist:

```bash
VLM_SERVER_PORT=18082 \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

Then benchmark it:

```bash
VLM_SERVER_PORT=18082 scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=18082 conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/minicpmv46_q4.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/minicpmv46-q4-wsl-cuda.jsonl \
  --summary-output outputs/benchmarks/minicpmv46-q4-wsl-cuda.md \
  --max-tokens 64 \
  --temperature 0
```

The WSL CUDA smoke has passed for text, committed sample images, and one fake-stream frame. Jetson runtime remains unverified.

## Fake Stream

Folder-based image stream dry run:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "Describe this frame." \
  --output outputs/fake_stream/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

Small non-private sample images are included under `data/sample_images/` and `data/sample_stream/` so dry runs and payload checks work after clone. Do not commit large or private media.

## Jetson Migration

Copy source, configs, scripts, docs, and optional tests. Do not copy WSL build directories, Conda environments, reference repos, or unrelated benchmark outputs.

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '.vscode/' \
  --exclude 'tmp/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  ./ jetson:/home/jetson/edge-vlm-lab/
```

Use NVMe or external storage for models:

```bash
sudo mkdir -p /mnt/nvme/models
sudo chown "$USER:$USER" /mnt/nvme/models
```

Run Gemma with Docker on Jetson:

```bash
MODEL_DIR=/mnt/nvme/models \
CTX_SIZE=2048 \
N_GPU_LAYERS=99 \
scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

The Jetson scripts default to a dusty-nv `llama_cpp` image. On a Jetson with `autotag` from jetson-containers installed, they use `autotag llama_cpp` to select a JetPack/L4T-compatible image. Without `autotag`, dry-run and fallback commands use `dustynv/llama_cpp:r36.4.0`. Override with `LLAMA_CPP_DOCKER_IMAGE=...` if your JetPack requires a different tag.

Use `JETSON_DRY_RUN=1` to print the Docker command without requiring Docker or Jetson hardware:

```bash
JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

See [docs/migration_wsl_to_jetson.md](docs/migration_wsl_to_jetson.md) for the full checklist.

## Documentation

- [Runtime matrix](docs/runtime_matrix.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [WSL to Jetson migration](docs/migration_wsl_to_jetson.md)
- [Implementation plan](docs/implementation_plan.md)
- [APEX workflow matrix](docs/matrix_edge_vlm_workflow.md)
- [OrangePi MiniCPM-V 4.6 notes](docs/reference_notes/orangepi_minicpmv46_notes.md)
