# Edge VLM Handoff Status

Last updated: 2026-05-27T14:44:22+09:30
Current code baseline: current Git HEAD plus the Jetson Q4 smoke-result documentation update and Gemma Jetson explicit-model dry-run fix ready for commit.

## Current Objective

Prepare and hand off the WSL-first, Jetson-Orin-targeted edge VLM experimentation workflow for MiniCPM-V 4.6 and Gemma 4 E2B-it. WSL remains the development and validation machine. Jetson receives only the minimal runtime package, configs, scripts, benchmark harness, docs, and model artifacts staged on NVMe or external storage.

## Accepted Scope

### In Scope

- llama.cpp / `llama-server` as the first backend.
- GGUF plus `mmproj` model layout for Gemma 4 E2B-it and MiniCPM-V 4.6.
- WSL scripts for build, model artifact download, server launch, and metadata inspection.
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
- Using Jetson as the primary development, cloning, conversion, or quantization machine.
- Installing dependencies into Conda `base`, system Python, or a global Python environment.
- Claiming Jetson runtime support beyond observed Q4 smoke paths, or broad performance without formal Jetson runs.
- Running local Gemma BF16-to-Q4, Gemma Q8-to-Q4, or MiniCPM HF-to-GGUF conversion/quantization on this WSL host as a baseline.

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
- Gemma Q4 runtime config is present at `configs/models/gemma4_e2b_q4.yaml`.
- MiniCPM-V 4.6 Q4 runtime config is present at `configs/models/minicpmv46_q4.yaml`.
- Gemma Q8 preparation downloads official pre-quantized artifacts from `ggml-org/gemma-4-E2B-it-GGUF`.
- Gemma Q4 preparation downloads pre-built artifacts from `mradermacher/gemma-4-E2B-it-GGUF`:
  - model file: `gemma-4-E2B-it.Q4_K_M.gguf`
  - mmproj file: `gemma-4-E2B-it.mmproj-Q8_0.gguf`
- MiniCPM-V 4.6 preparation downloads official pre-built artifacts from `openbmb/MiniCPM-V-4.6-gguf`:
  - model file: `MiniCPM-V-4_6-Q4_K_M.gguf`
  - mmproj file: `mmproj-model-f16.gguf`
- MiniCPM-V 4.6 WSL and Jetson launchers default to the same `MiniCPM-V-4.6-gguf/` file layout.
- MiniCPM-V 4.6 inspection checks the official pre-built GGUF repo metadata without downloading model files.
- Benchmark/fake-stream timestamps now compute `end_time` from captured wall-clock start plus monotonic latency, so WSL wall-clock rollback cannot produce `end_time < start_time` for real request records.
- Jetson Docker launch scripts are present:
  - `scripts/jetson/run_gemma4_e2b_llama_docker.sh`
  - `scripts/jetson/run_minicpmv46_llama_docker.sh`
  - `scripts/jetson/monitor_tegrastats.sh`
- Jetson launchers use dusty-nv `llama_cpp` containers by default:
  - `LLAMA_CPP_DOCKER_IMAGE` pins an explicit image.
  - `autotag llama_cpp` is used on Jetson when available.
  - `dustynv/llama_cpp:r36.4.0` is the non-Jetson/dry-run fallback.
  - `LLAMA_SERVER_CMD` can override the server binary path inside a specific image.
- Jetson launchers support `JETSON_DRY_RUN=1` for command-construction checks without Docker, model files, or Jetson hardware.
- Gemma Jetson launcher keeps explicit `MODEL_PATH` and `MMPROJ_PATH` under `MODEL_DIR`, but no longer requires those files to exist during `JETSON_DRY_RUN=1`; real runs still require both files.
- Copied Jetson smoke outputs under ignored `outputs/` document successful MiniCPM-V 4.6 Q4 and Gemma Q4 runs with the pinned Jetson llama.cpp image:
  - `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`
- Shared benchmark cases reference committed sample images:
  - `data/sample_images/image_caption_single.png`
  - `data/sample_images/image_safety_scene_single.png`
  - `data/sample_stream/frame_001.png`
- Benchmark runs can optionally write a Markdown summary with `--summary-output`.
- README and Chinese README are project entry points, not progress logs.

### Verified On WSL

- Gemma 4 E2B-it Q8:
  - Official pre-quantized Q8_0 model and mmproj files exist locally under ignored `models/` storage in this workspace.
  - Text-only CPU fallback smoke passed.
  - WSL CUDA text, committed sample-image benchmark, and one-frame fake-stream run passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, one server slot, and `VLM_SERVER_PORT=18081`.
- Gemma 4 E2B-it Q4:
  - Pre-built Q4_K_M model and mmproj artifacts exist locally under ignored `models/` storage.
  - WSL CUDA text, committed sample-image benchmark, and one-frame fake-stream run passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, one server slot, and `VLM_SERVER_PORT=18083`.
- MiniCPM-V 4.6 Q4:
  - Official pre-built Q4_K_M model and F16 mmproj artifacts exist locally under ignored `models/` storage.
  - WSL CUDA text, committed sample-image benchmark, and one-frame fake-stream run passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=128`, `LLAMA_UBATCH_SIZE=32`, one server slot, and `VLM_SERVER_PORT=18082`.
- Benchmark harness:
  - JSONL output and dry-run paths are implemented and tested.
  - Markdown summary output is implemented and tested.
  - Real WSL CUDA text and sample-image cases have been observed for Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4.
- Local conversion residue was removed from ignored storage:
  - old `models/MiniCPM-V-4_6/` local-conversion directory
  - MiniCPM metadata-test GGUF outputs
  - Gemma BF16 and Q8-to-Q4 local-conversion/quantization outputs

### Verified On Jetson

- MiniCPM-V 4.6 Q4:
  - Copied benchmark output: `outputs/benchmarks/minicpmv46-q4-jetson.jsonl`
  - Copied summary output: `outputs/benchmarks/minicpmv46-q4-jetson.md`
  - Benchmark result: 6/6 cases passed.
  - Text average: 41.49 tok/s across three 64-token text cases.
  - Image average: 34.10 tok/s across two 64-token sample-image cases.
  - Copied fake-stream output: `outputs/fake_stream/minicpmv46-q4-jetson.jsonl`
  - Fake-stream result: one frame passed in 3.27 s.
  - Observed launch shape: `CTX_SIZE=512`, `N_GPU_LAYERS=32`, batch 128, ubatch 32, `q8_0` KV cache, no warmup.
- Gemma 4 E2B-it Q4:
  - Copied benchmark output: `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl`
  - Copied summary output: `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.md`
  - Benchmark result: 6/6 cases passed.
  - Text average: 6.84 tok/s across three 64-token text cases.
  - Image average: 5.91 tok/s across two 64-token sample-image cases.
  - Copied fake-stream output: `outputs/fake_stream/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl`
  - Fake-stream result: one frame passed in 18.79 s.
  - Observed launch shape: `CTX_SIZE=512`, `N_GPU_LAYERS=12`, batch 512, ubatch 512, `q8_0` KV cache, no warmup, mmproj kept on GPU.
- Gemma Q4 no-mmproj-offload comparison:
  - Copied benchmark output: `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu.jsonl`
  - Benchmark result: 6/6 cases passed.
  - Text average: 7.09 tok/s; image average: 2.87 tok/s.
  - Use the mmproj-on-GPU smoke path unless retesting.
- Earlier Gemma Q4 `outputs/benchmarks/gemma4-e2b-q4-jetson.jsonl` is not a success case:
  - 1 fake-stream marker success.
  - 5 connection-refused benchmark failures.

### Deferred / Not Implemented

- Jetson Q8 execution.
- Jetson `tegrastats` capture paired with benchmark output.
- Formal Jetson performance reporting with repeated trials, power mode, and thermal context.
- Ollama, NanoLLM, vLLM, TensorRT/TensorRT-LLM, and custom kernels.
- Camera access or live video stream capture.
- Local MiniCPM HF checkpoint conversion or local quantization as a supported baseline.

## Validation Snapshot

### Verified Before This Active Phase

- WSL CUDA llama.cpp build passed:
  - `GGML_CUDA:BOOL=ON`
  - `CMAKE_CUDA_ARCHITECTURES=86`
  - `llama-server` links `libggml-cuda`, `cudart`, and `cublas`.
- Full-access `llama-server --version` passed from `tmp/llama.cpp/build-cuda/bin/llama-server`.
- Gemma Q8 WSL CUDA wrapper-default benchmark wrote:
  - `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.md`
- Gemma Q8 WSL CUDA real fake-stream run wrote:
  - `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`
- MiniCPM-V 4.6 official pre-built WSL CUDA benchmark wrote:
  - `outputs/benchmarks/minicpmv46-q4-wsl-cuda-official-prebuilt-timestamp-fixed.jsonl`
  - `outputs/benchmarks/minicpmv46-q4-wsl-cuda-official-prebuilt-timestamp-fixed.md`
- MiniCPM-V 4.6 official pre-built WSL CUDA fake-stream run wrote:
  - `outputs/fake_stream/minicpmv46-q4-wsl-cuda-official-prebuilt-timestamp-fixed.jsonl`
- Gemma Q4 WSL CUDA benchmark wrote:
  - `outputs/benchmarks/gemma4-e2b-q4-wsl-cuda-official-prebuilt.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q4-wsl-cuda-official-prebuilt.md`
- Gemma Q4 WSL CUDA fake-stream run wrote:
  - `outputs/fake_stream/gemma4-e2b-q4-wsl-cuda-official-prebuilt.jsonl`
- Earlier Gemma image request attempt with `LLAMA_UBATCH_SIZE=32` failed in llama.cpp:
  - `GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed`
  - The current Gemma CUDA wrapper default uses `LLAMA_BATCH_SIZE=512` and `LLAMA_UBATCH_SIZE=512`.

### Jetson Smoke Evidence Copied Into Ignored Outputs

- MiniCPM-V 4.6 Q4 Jetson benchmark wrote:
  - `outputs/benchmarks/minicpmv46-q4-jetson.jsonl`
  - `outputs/benchmarks/minicpmv46-q4-jetson.md`
- MiniCPM-V 4.6 Q4 Jetson fake-stream run wrote:
  - `outputs/fake_stream/minicpmv46-q4-jetson.jsonl`
- Gemma Q4 Jetson benchmark with mmproj on GPU wrote:
  - `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.md`
- Gemma Q4 Jetson fake-stream with mmproj on GPU wrote:
  - `outputs/fake_stream/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl`
- Gemma Q4 Jetson no-mmproj-offload comparison wrote:
  - `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu.md`
- Failed Gemma Q4 connection attempt wrote:
  - `outputs/benchmarks/gemma4-e2b-q4-jetson.jsonl`
  - `outputs/benchmarks/gemma4-e2b-q4-jetson.md`

### Current Local Artifact Snapshot

Ignored `models/` storage currently contains only the expected downloaded GGUF artifacts and HF local cache directories for this workflow:

```text
models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf
models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf
models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf
models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf
models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf
models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf
```

### Fresh Verification Passed For Recent Phases

The following checks passed after the MiniCPM/Gemma pre-built-artifact updates:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
bash -n scripts/common/*.sh scripts/wsl/*.sh scripts/jetson/*.sh
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m py_compile src/edge_vlm/*.py
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m unittest discover -s tests -v
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models VLM_SERVER_PORT=19090 bash scripts/jetson/run_gemma4_e2b_llama_docker.sh
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-minicpm-models VLM_SERVER_PORT=19091 bash scripts/jetson/run_minicpmv46_llama_docker.sh
git diff --check
pgrep -af llama-server
```

Unit tests: 21 tests passed. `pgrep -af llama-server` returned no running server.

The following checks passed after the Jetson dusty-nv launcher correction:

```bash
bash -n scripts/common/*.sh scripts/wsl/*.sh scripts/jetson/*.sh
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m py_compile src/edge_vlm/*.py tests/test_edge_vlm.py
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m unittest discover -s tests -v
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models VLM_SERVER_PORT=19090 bash scripts/jetson/run_gemma4_e2b_llama_docker.sh
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-minicpm-models VLM_SERVER_PORT=19091 bash scripts/jetson/run_minicpmv46_llama_docker.sh
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models LLAMA_CPP_DOCKER_IMAGE=dustynv/llama_cpp:b5283-r36.4-cu128-24.04 LLAMA_SERVER_CMD=/usr/local/bin/llama-server bash scripts/jetson/run_gemma4_e2b_llama_docker.sh
git diff --check
```

Unit tests: 22 tests passed. The two default Jetson dry-runs emitted `dustynv/llama_cpp:r36.4.0` fallback commands, and the explicit override dry-run emitted `dustynv/llama_cpp:b5283-r36.4-cu128-24.04` with `/usr/local/bin/llama-server`.

The following checks passed after documenting copied Jetson Q4 smoke outputs and fixing Gemma explicit-model dry-run:

```bash
bash -n scripts/common/*.sh scripts/wsl/*.sh scripts/jetson/*.sh
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m py_compile src/edge_vlm/*.py tests/test_edge_vlm.py
env PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/edge-vlm-pycache conda run -n transformers python -m unittest discover -s tests -v
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models VLM_SERVER_PORT=19090 bash scripts/jetson/run_gemma4_e2b_llama_docker.sh
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-minicpm-models VLM_SERVER_PORT=19091 bash scripts/jetson/run_minicpmv46_llama_docker.sh
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-models LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 MODEL_PATH=/tmp/edge-vlm-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf MMPROJ_PATH=/tmp/edge-vlm-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf MODEL_ALIAS=gemma4-e2b-it-q4 CTX_SIZE=512 N_GPU_LAYERS=12 bash scripts/jetson/run_gemma4_e2b_llama_docker.sh -fit off --parallel 1 --batch-size 512 --ubatch-size 512 --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup
JETSON_DRY_RUN=1 MODEL_DIR=/tmp/edge-vlm-minicpm-models LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 CTX_SIZE=512 N_GPU_LAYERS=32 bash scripts/jetson/run_minicpmv46_llama_docker.sh --parallel 1 --batch-size 128 --ubatch-size 32 --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup
git diff --check
```

Unit tests: 23 tests passed. The pinned Gemma Q4 dry-run emitted the explicit model/mmproj mount paths and Jetson image even without local model files, matching the documented dry-run contract.

## Relevant Commands

WSL Gemma Q8 CUDA server:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
VLM_SERVER_PORT=18081 \
N_GPU_LAYERS=32 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

WSL Gemma Q4 CUDA server:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
VLM_SERVER_PORT=18083 \
N_GPU_LAYERS=32 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

WSL MiniCPM-V 4.6 Q4 CUDA server:

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
VLM_SERVER_PORT=18082 \
N_GPU_LAYERS=32 \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

Benchmark:

```bash
VLM_SERVER_PORT=<port> scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=<port> conda run -n transformers python -m edge_vlm.benchmark \
  --config <model-config.yaml> \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output <output.jsonl> \
  --summary-output <summary.md> \
  --max-tokens 64 \
  --temperature 0
```

Fake stream:

```bash
PYTHONPATH=src VLM_SERVER_PORT=<port> conda run -n transformers python -m edge_vlm.fake_stream \
  --config <model-config.yaml> \
  --image-dir data/sample_stream \
  --output <output.jsonl> \
  --prompt "Describe this frame." \
  --interval-s 0 \
  --max-frames 1
```

## Active Blockers Or Open Questions

- Jetson Q4 smoke is no longer blocked: copied outputs show MiniCPM-V 4.6 Q4 and Gemma Q4 ran successfully.
- Jetson Q8, formal `tegrastats`-paired performance reporting, camera input, long-run behavior, power mode, and thermal context remain open.
- WSL sandboxed commands may not access NVML/GPU. Use full-access local shell for GPU runtime observations.
- The project has no dependency file yet. That is intentional because the current Python client uses standard library HTTP and tests run in the user-provided `transformers` Conda environment.

## Recommended Next Steps

1. Treat MiniCPM-V 4.6 Q4 as the fastest observed Jetson smoke path for first demos.
2. Keep Gemma Q4 mmproj on GPU for the observed Jetson smoke path unless retesting.
3. Repeat the Jetson Q4 runs with `tegrastats`, power mode, and thermal context captured beside the benchmark JSONL.
4. Add Jetson Q8 only after the Q4 path has a clean formal benchmark record.
5. Add camera/live stream only after the fake-stream path is stable under repeated Jetson runs.

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
  - `TRD-002`: MiniCPM-V 4.6 uses official pre-built GGUF plus mmproj.
  - `TRD-003`: Docker-first Jetson scripts.
  - `TRD-004`: Gemma uses pre-quantized GGUF artifacts.
  - `TRD-005`: Local MiniCPM conversion is not a baseline path.
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
- Image support for Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 on WSL is backed by real image prompt runs. Jetson image support is backed only for MiniCPM-V 4.6 Q4 and Gemma Q4 smoke logs; do not generalize that to Jetson Q8, camera input, or broad performance.
- Do not restart local conversion or quantization on this WSL host unless the user explicitly re-authorizes it.
- Keep Jetson launchers on the dusty-nv/jetson-containers path unless a real Jetson run proves a different container is required.
- The currently observed Jetson runs used `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`.
- Do not commit local models, benchmark outputs, reference clones, or planning files.
- If adding dependencies later, introduce repo dependency files and target the project environment. Do not install into Conda `base` or global Python.

## Optional Recent Accepted Milestones

- 2026-05-26 - Planning files were created and ignored by git.
- 2026-05-26 - WSL CUDA wrappers were added and defaults settled at `BUILD_JOBS=8`, `CMAKE_CUDA_ARCHITECTURES=86`, and runtime `N_GPU_LAYERS=32`.
- 2026-05-26 - README and README.zh-CN were reorganized into project entry-point docs.
- 2026-05-26 - WSL CUDA llama.cpp build completed successfully.
- 2026-05-26 - Gemma Q8 WSL CUDA text smoke and benchmark text cases passed.
- 2026-05-26 - Sample image assets, Markdown benchmark summary output, and Jetson launcher dry-run checks were added and verified.
- 2026-05-26 - Gemma Q8 WSL CUDA sample-image benchmark and one-frame fake-stream real run passed with wrapper defaults.
- 2026-05-26 - Gemma Q4 preparation switched to pre-built GGUF artifacts and WSL CUDA text/image/fake-stream checks passed.
- 2026-05-26 - MiniCPM-V 4.6 preparation switched to official pre-built GGUF artifacts and WSL CUDA text/image/fake-stream checks passed.
- 2026-05-27 - MiniCPM-V 4.6 Q4 and Gemma Q4 Jetson smoke outputs were copied under ignored `outputs/` and summarized in README/status docs.
