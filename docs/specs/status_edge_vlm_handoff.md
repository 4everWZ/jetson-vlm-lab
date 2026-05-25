# Edge VLM Handoff Status

Last updated: 2026-05-26T04:29:43+09:30
Current code baseline: `df63290 fix: use image-safe gemma cuda batch defaults`

## Current Objective

Prepare and hand off the WSL-first, Jetson-Orin-targeted edge VLM experimentation workflow for MiniCPM-V 4.6 and Gemma 4 E2B-it. WSL remains the development and validation machine. Jetson receives only the minimal runtime package, configs, scripts, benchmark harness, docs, and model artifacts staged on NVMe or external storage.

## Accepted Scope

### In Scope

- llama.cpp / `llama-server` as the first backend.
- GGUF plus `mmproj` model layout for Gemma 4 E2B-it and MiniCPM-V 4.6.
- WSL scripts for build, model preparation, server launch, and metadata inspection.
- Jetson Docker-oriented launch scripts and `tegrastats` monitoring.
- Shared Python OpenAI-compatible client, benchmark harness, image payload builder, and fake stream runner.
- Runtime configs under `configs/models/`.
- Benchmark cases under `configs/benchmark/prompt_cases.jsonl`.
- Small non-private sample assets under `data/sample_images/` and `data/sample_stream/`.
- Documentation for reference notes, runtime matrix, benchmark protocol, implementation plan, migration, tradeoffs, and APEX matrix.
- Local planning files (`task_plan.md`, `findings.md`, `progress.md`) are ignored and are not part of the committed source tree.

### Explicitly Out Of Scope

- Porting OrangePi Ascend/CANN/ACL/aclnn/custom AscendC kernels to Jetson.
- Custom CUDA kernels, TensorRT/TensorRT-LLM, or TensorRT vision tower extraction in the first version.
- Using Jetson as the primary development or conversion machine.
- Installing dependencies into Conda `base`, system Python, or a global Python environment.
- Claiming image, MiniCPM-V 4.6, Jetson, or broad performance support without observed runs.
- Re-running local BF16-to-Q4 Gemma quantization on this WSL host as a baseline.

## Current State

### Implemented

- Project structure is in place:
  - `src/edge_vlm/` for client, config loading, image payloads, benchmark, and fake stream.
  - `scripts/wsl/`, `scripts/jetson/`, and `scripts/common/` for separated runtime workflows.
  - `configs/models/` and `configs/benchmark/` for runtime and benchmark data.
  - `docs/` for design, specs, matrix, migration, runtime strategy, benchmark protocol, tradeoffs, and reference notes.
- WSL CPU fallback path exists under `scripts/wsl/run_gemma4_e2b_llama.sh`.
- WSL CUDA path exists under:
  - `scripts/wsl/build_llama_cpp_cuda.sh`
  - `scripts/wsl/run_gemma4_e2b_llama_cuda.sh`
  - `scripts/wsl/run_minicpmv46_llama_cuda.sh`
- CUDA build defaults are tuned for this WSL host:
  - `BUILD_JOBS=8`
  - `CMAKE_CUDA_ARCHITECTURES=86`
  - build directory `tmp/llama.cpp/build-cuda`
- Gemma Q8 runtime config is present at `configs/models/gemma4_e2b_q8.yaml`.
- Gemma Q8 WSL CUDA launcher defaults are image-safe for the observed local llama.cpp build:
  - `CTX_SIZE=512`
  - `N_GPU_LAYERS=32`
  - `LLAMA_BATCH_SIZE=512`
  - `LLAMA_UBATCH_SIZE=512`
- MiniCPM-V 4.6 guarded preparation and metadata inspection scripts are present:
  - `scripts/wsl/inspect_minicpmv46_hf.sh`
  - `scripts/wsl/prepare_minicpmv46_q4.sh`
- Jetson Docker launch scripts are present:
  - `scripts/jetson/run_gemma4_e2b_llama_docker.sh`
  - `scripts/jetson/run_minicpmv46_llama_docker.sh`
  - `scripts/jetson/monitor_tegrastats.sh`
- Jetson launchers support `JETSON_DRY_RUN=1` for command-construction checks without Docker, model files, or Jetson hardware.
- Shared benchmark cases now reference committed sample images:
  - `data/sample_images/image_caption_single.png`
  - `data/sample_images/image_safety_scene_single.png`
  - `data/sample_stream/frame_001.png`
- Benchmark runs can optionally write a Markdown summary with `--summary-output`.
- README and Chinese README are project entry points, not progress logs.

### Partially Implemented

- Gemma 4 E2B-it:
  - Official pre-quantized Q8_0 model and mmproj files exist locally under ignored `models/` storage in this workspace.
  - Text-only CPU fallback smoke has passed.
  - Text-only WSL CUDA smoke has passed.
  - WSL CUDA text, committed sample-image benchmark, and a real fake-stream frame have passed with the current wrapper defaults.
  - Jetson runtime remains unobserved.
- MiniCPM-V 4.6:
  - Metadata-only inspection has passed without downloading weights.
  - Local llama.cpp conversion signals were observed and documented.
  - Full conversion, quantization, and runtime are still unverified.
- Benchmark harness:
  - JSONL output and dry-run paths are implemented and tested.
  - Markdown summary output is implemented and tested.
  - Real Gemma Q8 WSL CUDA text and sample-image cases have been observed.
- Jetson path:
  - Scripts, dry-run command construction, and migration docs exist.
  - Runtime on Jetson has not been observed.

### Deferred / Not Implemented

- MiniCPM-V 4.6 real GGUF conversion and `llama-server` runtime.
- Jetson Docker execution and benchmark collection.
- Ollama, NanoLLM, vLLM, TensorRT/TensorRT-LLM, and custom kernels.

## Validation Snapshot

### Verified

- `git status --short` was clean before this handoff refresh.
- Local Python commands use the `transformers` Conda environment.
- Shell syntax passed:
  - `bash -n scripts/common/*.sh scripts/wsl/*.sh scripts/jetson/*.sh`
- Python syntax passed:
  - `env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m py_compile src/edge_vlm/*.py`
- Unit tests passed:
  - `env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m unittest discover -s tests -v`
  - 18 tests passed.
- Diff whitespace check passed:
  - `git diff --check`
- Benchmark dry-run with committed sample images and Markdown summary passed:
  - `env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m edge_vlm.benchmark --config configs/models/gemma4_e2b_q8.yaml --cases configs/benchmark/prompt_cases.jsonl --output outputs/benchmarks/final-handoff-dryrun.jsonl --summary-output outputs/benchmarks/final-handoff-dryrun.md --dry-run`
  - 6 records written; all dry-run cases succeeded.
- Fake stream dry-run with committed sample frame passed:
  - `env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m edge_vlm.fake_stream --config configs/models/gemma4_e2b_q8.yaml --image-dir data/sample_stream --output outputs/fake_stream/final-handoff-dryrun.jsonl --prompt "Describe this frame." --interval-s 0 --max-frames 1 --dry-run`
  - 1 frame record written.
- Jetson command-construction dry-runs passed:
  - `JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models VLM_SERVER_PORT=19090 bash scripts/jetson/run_gemma4_e2b_llama_docker.sh`
  - `JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-minicpm-models VLM_SERVER_PORT=19091 bash scripts/jetson/run_minicpmv46_llama_docker.sh`
- WSL CUDA llama.cpp build passed:
  - `GGML_CUDA:BOOL=ON`
  - `CMAKE_CUDA_ARCHITECTURES=86`
  - `llama-server` links `libggml-cuda`, `cudart`, and `cublas`.
- Full-access `llama-server --version` passed from `tmp/llama.cpp/build-cuda/bin/llama-server`.
- Gemma Q8 WSL CUDA text smoke passed:
  - `CTX_SIZE=512`
  - `N_GPU_LAYERS=32`
  - `LLAMA_BATCH_SIZE=512`
  - `LLAMA_UBATCH_SIZE=512`
  - one server slot
  - `VLM_SERVER_PORT=18081`
  - `/v1/models` reported `multimodal` capability.
  - `nvidia-smi` observed `llama-server` on the RTX 3060 Laptop GPU during validation.
- Gemma Q8 WSL CUDA wrapper-default benchmark passed and wrote:
  - `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.md`
  - 6 records written: 3 text successes, 2 image successes, and 1 fake-stream marker success.
- Gemma Q8 WSL CUDA real fake-stream run passed and wrote:
  - `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`
  - 1 committed sample frame succeeded.
- Earlier Gemma image request attempt with `LLAMA_UBATCH_SIZE=32` failed in llama.cpp:
  - `GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed`
  - The current Gemma CUDA wrapper default uses `LLAMA_BATCH_SIZE=512` and `LLAMA_UBATCH_SIZE=512`.
- CUDA smoke server was shut down after validation:
  - no listener remained on `:18081`.
  - final `nvidia-smi` was back near idle, with no `llama-server` process.

### Not Yet Verified

- MiniCPM-V 4.6 conversion, quantization, and runtime.
- Jetson Docker runtime and benchmark collection.
- Jetson `tegrastats` capture paired with benchmark output.
- Any formal performance comparison or throughput claim beyond the observed smoke context.

### Relevant Commands

WSL tests:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
bash -n scripts/common/*.sh scripts/wsl/*.sh scripts/jetson/*.sh
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m py_compile src/edge_vlm/*.py
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m unittest discover -s tests -v
git diff --check
```

WSL Gemma Q8 CUDA server:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
VLM_SERVER_PORT=18081 \
N_GPU_LAYERS=32 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

WSL Gemma Q8 benchmark:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
VLM_SERVER_PORT=18081 scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=18081 conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.md \
  --max-tokens 64 \
  --temperature 0
```

WSL Gemma Q8 fake stream:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src VLM_SERVER_PORT=18081 conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --output outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl \
  --prompt "Describe this frame." \
  --interval-s 0 \
  --max-frames 1
```

## Active Blockers Or Open Questions

- MiniCPM-V 4.6 runtime is blocked on deliberate model preparation: HF download, GGUF conversion, mmproj creation, and quantization must run on WSL or another machine with enough memory and disk.
- Jetson validation is blocked on access to the target device, model placement under Jetson storage, and Docker/NVIDIA runtime readiness.
- WSL sandboxed commands may not access NVML/GPU. Use full-access local shell for GPU runtime observations.
- The project has no dependency file yet. That is intentional because the current Python client uses standard library HTTP and tests run in the user-provided `transformers` Conda environment.

## Recommended Next Steps

1. Decide where MiniCPM-V 4.6 conversion should happen. Use `scripts/wsl/inspect_minicpmv46_hf.sh` first; only use `ALLOW_MINICPM_FULL_PREPARE=1` on a machine with enough memory and disk.
2. Sync the minimal runtime package to Jetson, excluding `tmp/`, `outputs/`, Conda environments, and local WSL build artifacts.
3. Start Jetson with Gemma Q8 text-only benchmark first, while recording `tegrastats`.
4. Only after Gemma text works on Jetson, try image cases and then MiniCPM-V 4.6.

## Key References

- Overview:
  - `README.md`
  - `README.zh-CN.md`
- Relevant leaf docs:
  - `docs/specs/00_edge_vlm_workflow.md`
  - `docs/specs/dev_client_and_benchmark.md`
  - `docs/specs/integration_wsl_to_jetson.md`
- Runtime strategy:
  - `docs/runtime_matrix.md`
- Benchmark protocol:
  - `docs/benchmark_protocol.md`
- Migration:
  - `docs/migration_wsl_to_jetson.md`
- Matrix:
  - `docs/matrix_edge_vlm_workflow.md`
- Tradeoff IDs:
  - `TRD-001`: Standard-library client instead of OpenAI SDK.
  - `TRD-002`: MiniCPM-V 4.6 defaults to local GGUF plus mmproj.
  - `TRD-003`: Docker-first Jetson scripts.
  - `TRD-004`: Gemma low-memory baseline uses pre-quantized Q8_0.
  - `TRD-005`: MiniCPM full preparation requires explicit opt-in.
- Reference notes:
  - `docs/reference_notes/orangepi_minicpmv46_notes.md`
- Design docs:
  - `docs/design/edge_vlm_architecture.md`

## Handoff Notes

- Preserve the WSL-first workflow. Do not turn Jetson into the development, cloning, or model-conversion machine unless the user explicitly accepts that tradeoff.
- Preserve script separation between WSL and Jetson.
- Keep runtime configs separate from source code.
- Keep `BUILD_JOBS=8` as the current WSL CUDA build default. The user explicitly stopped further tuning upward.
- Do not conflate WSL host RAM with GPU VRAM. `BUILD_JOBS` spends CPU/RAM; `N_GPU_LAYERS` spends VRAM.
- Image support for Gemma Q8 on WSL is backed by a real image prompt run; do not generalize that to MiniCPM-V 4.6, Jetson, or broad performance.
- Do not claim MiniCPM-V 4.6 support until conversion and a real request are observed on the selected llama.cpp revision.
- Do not commit local models, benchmark outputs, reference clones, or planning files.
- If adding dependencies later, introduce repo dependency files and target the project environment. Do not install into Conda `base` or global Python.

## Optional Recent Accepted Milestones

- 2026-05-26 — Planning files were created and ignored by git.
- 2026-05-26 — WSL CUDA wrappers were added and defaults settled at `BUILD_JOBS=8`, `CMAKE_CUDA_ARCHITECTURES=86`, and runtime `N_GPU_LAYERS=32`.
- 2026-05-26 — README and README.zh-CN were reorganized into project entry-point docs.
- 2026-05-26 — WSL CUDA llama.cpp build completed successfully.
- 2026-05-26 — Gemma Q8 WSL CUDA text smoke and benchmark text cases passed.
- 2026-05-26 — Sample image assets, Markdown benchmark summary output, and Jetson launcher dry-run checks were added and verified.
- 2026-05-26 — Gemma Q8 WSL CUDA sample-image benchmark and one-frame fake-stream real run passed with wrapper defaults.
