# Jetson VLM Lab

WSL-first edge VLM experimentation workflow for Jetson Orin and Orin Nano class devices.

The repository is organized so development, reference inspection, client code, benchmark harnesses, and migration docs happen on WSL first. Jetson receives only the small runtime package, configs, scripts, benchmark code, and model artifacts placed on NVMe or external storage.

## Current Status

- llama.cpp is the first supported backend through `llama-server` and GGUF models.
- Python project commands use the local Conda environment: `conda run -n transformers python`.
- A local CPU-only llama.cpp build exists under `tmp/llama.cpp/build/bin` in this workspace.
- WSL sees `nvidia-smi`, but `nvcc` is not installed, so the observed local llama.cpp build is CPU-only.
- Gemma BF16 GGUF source files were downloaded, but BF16 to Q4_K_M quantization was killed by WSL memory pressure. The incomplete Q4 output was removed.
- The low-memory Gemma path is official pre-quantized Q8_0 GGUF, prepared by `scripts/wsl/prepare_gemma4_e2b_q8.sh`. In this workspace, the Q8_0 model and Q8_0 mmproj files are present under ignored `models/` storage.
- MiniCPM-V 4.6 preparation is still a local conversion path and must be verified on the exact llama.cpp revision before claiming runtime support.
- Jetson inference is not yet observed in this repository.

Full filesystem access does not reduce model quantization peak memory. For constrained WSL, prefer already-quantized GGUF files, smaller context, CPU-only smoke tests, and one server slot.

## Layout

```text
configs/models/                  model runtime configs
configs/benchmark/               shared benchmark prompt cases
docs/                            design, migration, benchmark, and reference notes
scripts/wsl/                     WSL build, prepare, and run scripts
scripts/jetson/                  Jetson Docker launch and monitor scripts
scripts/common/                  shared helper scripts
src/edge_vlm/                    OpenAI-compatible client and harness code
tests/                           lightweight contract tests
tmp/references/                  ignored reference clones
models/                          ignored local model artifacts
outputs/                         ignored benchmark logs
```

## WSL Setup

Use the local `transformers` Conda environment. Do not install project dependencies into Conda `base` or system Python.

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

Build or reuse llama.cpp:

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
```

The build script does not install system packages. It enables CUDA only when `nvcc` is available, or when explicitly requested with a working CUDA toolkit.

## Prepare Gemma 4 E2B-it

Low-memory WSL path:

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

This downloads the official Q8_0 model and mmproj GGUF files into ignored `models/` storage. It does not run BF16 to Q4 quantization.

Start a conservative WSL server after the files exist:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
CTX_SIZE=1024 \
N_GPU_LAYERS=0 \
scripts/wsl/run_gemma4_e2b_llama.sh
```

In another terminal:

```bash
scripts/common/check_server.sh
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl
```

## MiniCPM-V 4.6

MiniCPM-V 4.6 is tracked as a llama.cpp local conversion path:

```bash
scripts/wsl/prepare_minicpmv46_q4.sh
```

This downloads the official HF checkpoint subset, converts language and mmproj GGUF files, and quantizes the language GGUF to Q4_K_M. Treat this as unverified until the command completes on the current machine and a real llama-server request succeeds.

Start it only after local files exist:

```bash
MODEL_PATH=$PWD/models/MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/MiniCPM-V-4_6/mmproj-model-f16.gguf \
CTX_SIZE=1024 \
N_GPU_LAYERS=0 \
scripts/wsl/run_minicpmv46_llama.sh
```

## Benchmarks

Dry-run payload and logging validation:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

Fake stream dry run:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "Describe this frame." \
  --output outputs/fake_stream/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

Image cases require local sample images under `data/`; do not commit large or private media.

## Jetson Path

Copy source, configs, scripts, docs, and tests to Jetson. Do not copy WSL build directories, Conda environments, reference repos, or unrelated benchmark outputs.

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

For local model files on Jetson, place them under `MODEL_DIR` and pass both `MODEL_PATH` and `MMPROJ_PATH`.

## References

- OrangePi MiniCPM-V 4.6 reference notes: `docs/reference_notes/orangepi_minicpmv46_notes.md`
- Runtime matrix: `docs/runtime_matrix.md`
- Benchmark protocol: `docs/benchmark_protocol.md`
- WSL to Jetson migration: `docs/migration_wsl_to_jetson.md`
- APEX implementation matrix: `docs/matrix_edge_vlm_workflow.md`
