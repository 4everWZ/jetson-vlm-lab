# Jetson VLM Lab

WSL-first edge VLM experimentation workflow for Jetson Orin and Orin Nano class devices.

The repository keeps development, reference inspection, client code, benchmark harnesses, and migration notes on WSL first. Jetson receives the smallest practical runtime package: source, configs, scripts, benchmark code, and model artifacts placed on NVMe or external storage.

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
| llama.cpp CPU build | Present under `tmp/llama.cpp/build/bin`; Gemma Q8 text-only smoke previously passed with low-memory CPU settings. |
| llama.cpp CUDA build | Verified locally under `tmp/llama.cpp/build-cuda` with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, and `BUILD_JOBS=8`. |
| WSL GPU visibility | `nvcc` is available. `nvidia-smi` works with full access and shows an RTX 3060 Laptop GPU; sandboxed commands may not see NVML. |
| Gemma 4 E2B-it | Official pre-quantized Q8_0 model and mmproj files are present under ignored `models/` storage. |
| Gemma Q8 WSL CUDA smoke | Text-only smoke passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, one server slot, and `VLM_SERVER_PORT=18081`. The benchmark harness wrote `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-smoke.jsonl`; image cases failed because local sample images are absent. |
| Gemma BF16 to Q4 | Not a WSL baseline here; local quantization was killed by memory pressure. Use pre-quantized GGUF or a larger conversion host. |
| MiniCPM-V 4.6 | Metadata-only inspection passed; local conversion and runtime are still unverified. |
| Jetson runtime | Scripted and documented, but not yet observed in this repository. |

Do not treat dry runs or server startup as performance results. Performance claims need real benchmark JSONL from a running model/server. The current CUDA smoke validates Gemma Q8 text inference only; it does not validate image prompts, MiniCPM-V 4.6, or Jetson runtime.

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

Run tests first:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

Prepare Gemma Q8_0 artifacts if they are not already present:

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

Run a dry benchmark to validate payloads and JSONL logging:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

## llama.cpp Builds

CPU fallback build:

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

The CUDA launcher defaults to `CTX_SIZE=512`, `N_GPU_LAYERS=32`, two threads, one server slot, and no warmup. This is a moderate WSL GPU smoke setting. On a memory-rich run, raise the offload explicitly:

```bash
N_GPU_LAYERS=48 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
N_GPU_LAYERS=99 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

Use `nvidia-smi` in another terminal to watch VRAM. Using 8-10 GiB of WSL host memory for build/runtime is acceptable here; avoid settings that push the process into OOM. `BUILD_JOBS` mostly spends host RAM and CPU, so the default is intentionally aggressive for this 12 GiB RAM + 6 GiB swap WSL setup. `N_GPU_LAYERS` spends GPU VRAM, so tune it against the 6 GiB RTX 3060 Laptop budget.

In another terminal:

```bash
scripts/common/check_server.sh
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl
```

## MiniCPM-V 4.6

Start with metadata inspection. This does not download weights:

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

Full preparation is guarded because it downloads the HF checkpoint, creates F16 GGUF, and quantizes to Q4_K_M:

```bash
ALLOW_MINICPM_FULL_PREPARE=1 scripts/wsl/prepare_minicpmv46_q4.sh
```

Run only after converted model and mmproj files exist:

```bash
MODEL_PATH=$PWD/models/MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/MiniCPM-V-4_6/mmproj-model-f16.gguf \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

MiniCPM-V 4.6 remains unverified until conversion completes on the selected llama.cpp revision and a real request succeeds.

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

Image cases require small local sample images under `data/`. Do not commit large or private media.

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

See [docs/migration_wsl_to_jetson.md](docs/migration_wsl_to_jetson.md) for the full checklist.

## Documentation

- [Runtime matrix](docs/runtime_matrix.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [WSL to Jetson migration](docs/migration_wsl_to_jetson.md)
- [Implementation plan](docs/implementation_plan.md)
- [APEX workflow matrix](docs/matrix_edge_vlm_workflow.md)
- [OrangePi MiniCPM-V 4.6 notes](docs/reference_notes/orangepi_minicpmv46_notes.md)
