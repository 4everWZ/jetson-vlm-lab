# Jetson VLM Lab

WSL-first edge VLM workflow for validating GGUF vision-language models before moving the smallest runtime package to Jetson Orin / Orin Nano.

The default path is deliberately practical: download pre-built GGUF artifacts, start `llama-server`, run the shared benchmark, then copy only source/configs/scripts/docs and model files to Jetson storage. Local model quantization is not part of the normal WSL flow for this machine.

## Goal

- Iterate on LLM/VLM candidates at or below the 2B class. Prefer pre-built Q4 GGUF artifacts; fall back to Q8 when the Q4 artifact is missing or the strict Jetson gate keeps the Q4 lane unusable.
- Use the self-built official llama.cpp image as the default Jetson multimodal runtime. The dusty-nv `llama_cpp` image is not the default VLM path because it has not provided the validated multimodal `llama-server` route this repo needs.
- Prioritize lower-level infrastructure work: launchers, artifact download/resume, profile capture, runtime image reproducibility, and multimodal load paths. Parameter changes should be scoped to a profile-backed bottleneck.
- Keep separate tasks on separate branches or worktrees, then merge successful verified work into `main`.

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
| Gemma 4 E2B-it Q4 | Uses pre-built `Q4_K_M` GGUF from `mradermacher/gemma-4-E2B-it-GGUF`. WSL CUDA and Jetson text, sample-image benchmark, and one-frame fake-stream checks passed. |
| Gemma Q8 WSL CUDA smoke | Text and sample-image benchmark passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, one server slot, and `VLM_SERVER_PORT=18081`. The wrapper-default real run wrote `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl` and `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`. |
| MiniCPM-V 4.6 | Official pre-built `Q4_K_M` model and F16 mmproj files from `openbmb/MiniCPM-V-4.6-gguf` are downloaded under ignored `models/` storage. WSL CUDA and Jetson text, sample-image benchmark, and one-frame fake-stream checks passed. |
| Qwen3-VL 2B Instruct fallback lane | Strict `--min-lfb-blocks 150` still blocks both lanes on Jetson, but the shared relaxed `100-LFB` compare `qwen3-instruct-q4q8-lfb100-20260609T091700Z` ran both rows successfully. Q4 stayed faster at 34.865 text tok/s, 31.958 image tok/s, and 1.827 s fake-stream latency; Q8 reached 31.346 text tok/s, 29.393 image tok/s, and 2.144 s fake-stream latency. Q8 remains a fallback lane, not the preferred default, but its prepare delta was much larger (`lfb +88`, `MemAvailable +963.6 MB`). |
| Jetson runtime | MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 have observed Jetson smoke outputs through the Docker launchers with the self-built official llama.cpp image `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`. The dusty-nv `llama_cpp` image is not the default because it has not provided the multimodal server path this repo needs. |

Do not treat dry runs or server startup as performance results. Performance claims need real benchmark JSONL from a running model/server. The current observed runtime support covers Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 on WSL CUDA, plus Jetson smoke coverage for MiniCPM-V 4.6 Q4 and Gemma Q4. It does not validate Jetson Q8, camera input, long-run behavior, power/thermal behavior, or broad performance.

## Observed Jetson Smoke

The ignored `outputs/` directory contains user-copied Jetson logs from 2026-05-27. These are short 64-token smoke checks against committed text/image cases, not a formal performance suite.

| Model | Jetson Command Shape | Benchmark Output | Result |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `CTX_SIZE=512`, `N_GPU_LAYERS=32`, batch `128`, ubatch `32`, KV cache `q8_0`, no warmup | `outputs/benchmarks/minicpmv46-q4-jetson.jsonl` | 6/6 benchmark cases passed. Text avg 41.49 tok/s; image avg 34.10 tok/s. One-frame fake stream passed in 3.27 s. |
| Gemma 4 E2B-it Q4 | `CTX_SIZE=512`, `N_GPU_LAYERS=12`, batch `512`, ubatch `512`, KV cache `q8_0`, no warmup, mmproj kept on GPU | `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl` | 6/6 benchmark cases passed. Text avg 6.84 tok/s; image avg 5.91 tok/s. One-frame fake stream passed in 18.79 s. |
| Gemma 4 E2B-it Q4, comparison | `CTX_SIZE=512`, `N_GPU_LAYERS=12`, batch `256`, ubatch `256`, `--no-mmproj-offload`, no warmup | `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu.jsonl` | 6/6 benchmark cases passed, but image avg dropped to 2.87 tok/s. Keep mmproj on GPU for this smoke path unless retesting. |

The earlier `outputs/benchmarks/gemma4-e2b-q4-jetson.jsonl` run is a failed startup/connection attempt: 1 marker success and 5 connection-refused benchmark failures. Do not cite it as a runtime success.

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
Jetson GGUF launchers validate host-side model and mmproj files by checking the
GGUF magic bytes before starting `llama-server`. If a cached artifact fails this
check, remove or restore that file and rerun the launcher so the HF path can
download it again.
For a no-start preflight manifest on Jetson, run:

```bash
scripts/jetson/check_gguf_artifacts.sh check \
  --artifact model /mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
  --artifact mmproj /mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
  --output outputs/artifacts/gemma-q4.gguf-artifacts.json
```

The manifest records `missing`, `invalid_magic`, or `ok` per artifact and exits
non-zero when any artifact is not ready.
Remote suite wrappers also enable advisory
`JETSON_REMOTE_GGUF_PREFLIGHT=1` by default, writing
`outputs/optimization_sweeps/<run-prefix>/<run-prefix>.gguf-artifacts.json`
before server startup. Set `JETSON_REMOTE_GGUF_PREFLIGHT_FAIL=1` to abort on
missing or invalid cached artifacts.

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

The WSL CUDA and Jetson smoke checks have passed for text, committed sample images, and one fake-stream frame. The Jetson result is limited to the Q4 files and parameters listed in [Observed Jetson Smoke](#observed-jetson-smoke).

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

Run the observed MiniCPM-V 4.6 Q4 smoke path on Jetson:

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

Run the observed Gemma Q4 smoke path on Jetson:

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

The Jetson scripts default to the self-built official llama.cpp image used by
the observed multimodal smoke runs:
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`. The dusty-nv
`llama_cpp` image is not used by default because direct Jetson evidence on
June 9, 2026 still rejected the actual multimodal flag this repo uses:
`dustynv/llama_cpp:b5283-r36.4-cu128-24.04` let the probe find
`/usr/local/bin/llama-server`, but an actual Qwen3-VL 2B Instruct Q4 smoke
exited before ready with `error: invalid argument: --mmproj`. Override with
`LLAMA_CPP_DOCKER_IMAGE=...` for a specific image, or set
`LLAMA_CPP_USE_AUTOTAG=1` only when you intentionally want `autotag llama_cpp`
selection. The Jetson sweep plan now probes `llama-server --help` inside the
selected image and records whether the runtime exposes `--mmproj`; image-capable
variants are skipped with `runtime_missing_mmproj_support` when that probe says
the runtime cannot serve this repo's multimodal path.
The direct VLM Docker launchers run the same probe before a real startup, before
artifact checks or downloads. `JETSON_DRY_RUN=1` still only prints the Docker
command. Set `LLAMA_CPP_RUNTIME_PROBE_OUTPUT=...` when you need the launcher to
write the probe JSON to a fixed path for review.
Probe a candidate image directly before running a sweep when the runtime itself
is under review:

```bash
PYTHONPATH=src python -m edge_vlm.llama_cpp_runtime probe-image \
  --image ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
  --output outputs/jetson_inspect/llama_cpp_runtime_probe.json
```

The JSON includes Docker image labels, the resolved `llama-server` path, exact
`--mmproj` support, and `multimodal_ready`. A generic `mmproj` help marker
without the exact `--mmproj` flag does not count as multimodal-ready evidence.

For the Qwen3-VL 2B Instruct pair, use the selector entrypoint when you want an
actual Q4-first / Q8 fallback decision instead of manually choosing one smoke
variant:

```bash
scripts/jetson/select_qwen3_instruct_variant.sh \
  --min-lfb-blocks 150 \
  --fallback-min-lfb-blocks 100 \
  --output outputs/jetson_inspect/qwen3-instruct-selector.json
```

The selector writes a JSON decision with runtime probe, preflight sample, per-
candidate GGUF artifact manifest, block reasons, and the chosen variant id.
Cached artifacts that exist but fail the GGUF magic check are blocked as
`invalid_model_artifact` or `invalid_mmproj_artifact`, so a corrupt Q4 file can
still fall through to a valid Q8 fallback. Its standalone default keeps the same
strict gate for both lanes unless you are intentionally doing scoped fallback
triage.
Pass `--memory-diagnostics-output <path>` when the selector is blocked by LFB
and you need lower-level memory evidence. The selector records that sidecar path
and a summary in its JSON; the full sidecar captures `/proc/meminfo` including
CMA, `/proc/buddyinfo`, swap/zram, compaction vmstat counters, memory pressure,
top RSS processes, debugfs availability, and any parseable nvmap/dma-buf totals
without changing the LFB gate. It also records read-only boot-time memory
configuration evidence: the `/proc/cmdline` `cma=` token when present and a
summary of device-tree `reserved-memory` nodes. Device-tree string properties
are preserved as strings, while binary properties such as `reg` or `size` are
preserved as hex bytes. When the parent `reserved-memory` node exposes
unambiguous `#address-cells` / `#size-cells` metadata, the sidecar also decodes
the `linux,cma` reservation size into bytes; otherwise it leaves the binary
properties as hex-only evidence.

The remote lightweight suite now uses this selector automatically instead of
pinning `qwen3-vl-2b-instruct-q4-smoke` in the candidate list. By default it
keeps the suite-wide `JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS=150` primary gate and
applies `JETSON_LIGHTWEIGHT_QWEN3_FALLBACK_MIN_LFB_BLOCKS=100` only to the
Qwen3 Instruct fallback lane. Override `JETSON_LIGHTWEIGHT_QWEN3_SELECTOR=0`
if you need to go back to a fully manual candidate list for a scoped rerun.
The remote sweep wrapper now also forwards that selector decision into the sweep
manifest, so `edge_vlm.optimization compare` can show the auto-selected lane in
its `Selection` column instead of leaving the Qwen3 row as an unlabeled static
variant id. The comparison table also shows a `Required lfb` column so the
effective preflight gate is visible next to the observed `Preflight lfb`.
If one remote drop-cache/compact-memory pass still leaves the Jetson below the
requested LFB gate, set `JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS=<N>` together
with `JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`. This repeats preparation
before selector inspection and variant preflight; it does not relax the gate.
The remote Qwen selector also writes
`<selector-output>.memory-diagnostics.json` by default; set
`JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS=0` only when you need to skip
that read-only sidecar.
If the sidecar reports debugfs paths as unreadable and you need the actual
nvmap/dma-buf/CMA debugfs previews, set
`JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO=1` with
`JETSON_REMOTE_SUDO_PASSWORD` in `.env.jetson`. The wrapper then writes a second
read-only sidecar named `<selector-output>.memory-diagnostics.sudo.json` after
the selector run. Its compact summary records debugfs status plus parseable
nvmap/dma-buf totals when those debugfs files expose stable `total` lines; it
is also written back to the selector JSON as `sudo_memory_diagnostics_summary`
so downstream `selection_contexts` carry the readable debugfs evidence. It does
not relax the LFB gate or change model selection.
The selector and sudo enrichment also add conservative
`memory_blocker_assessment` / `sudo_memory_blocker_assessment` objects. These
summarize observed-vs-required LFB blocks, CMA headroom, debugfs-tracked
allocator bytes, boot-time CMA configuration evidence such as `cmdline_cma` and
`reserved_memory` summaries, and evidence signals such as `lfb_below_required`;
they are evidence summaries only, not new gates or root-cause claims.
The forwarded `selection_contexts` now preserve selector candidate summaries,
including `block_reasons` and the GGUF `artifact_manifest`, so a Q8 row chosen
as fallback keeps the Q4 blocker evidence in the sweep plan and compare
eligibility JSON. They also preserve the regular and sudo memory diagnostics
paths and compact summaries. Launcher environment blocks are intentionally not
copied into that context.
When diagnostics summaries are present, compare adds a `Selector memory` column
and exports a flat `selection_memory_diagnostics` object so LFB/CMA and readable
debugfs totals are visible without opening nested sidecars. When boot-time
memory evidence is present, that column also shows `cmdline_cma=<token|none>`
and a compact `reserved_memory=<count>[:linux,cma]` summary. If `linux,cma`
size was decoded from device-tree cells, it also shows `linux_cma=<size>`.
The JSON export also keeps `selection_memory_assessment`, and the Markdown
column appends the assessment status and largest LFB deficit when present. If
the selector chooses no variant, compare still attaches that context to rows
whose variant id appears in the selector's primary, fallback, or candidate list.
When you also pass `--ranking-min-lfb-blocks <strict-gate>`, the report adds a
`Ranking precheck` column so relaxed fallback rows remain visible without being
mistaken for promotable strict-gate evidence.
Add `--ranking-require-startup-precheck` when ranking or export decisions must
also reject first-download rows and older rows without cached-startup evidence.
When startup time itself is part of the decision, also pass
`--startup-require-cached-artifacts`. That adds a `Startup precheck` column and
only passes rows whose profile phase timings explicitly recorded
`artifact_check_or_download = cached`, so first-download rows stay visible
without being mistaken for cached-startup baselines.
The remote suite wrappers now also run `edge_vlm.sweep_quality_review` with
`configs/benchmark/quality_review_policy.json` after the sweep finishes. That
writes per-run `quality_review_json` and `quality_review_markdown` sidecars
back into the manifest paths, so `edge_vlm.optimization compare` can add a
`Quality review` column without rerunning the policy review by hand.
When compare is used for promotion-oriented review, add
`--promotion-require-startup-precheck` and
`--promotion-require-quality-review` so `Promotion precheck` also requires
cached-startup evidence plus the structured `Quality review` sidecar instead
of only checking the mechanical run conditions.
If you want the remote promotion-oriented wrappers to stop the suite on that
gate, enable `JETSON_CURRENT_DEFAULTS_FAIL_ON_PROMOTION_PRECHECK=1` or
`JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK=1` or
`JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK=1`. They stay off by default
so diagnostic runs can still emit a comparison report without being treated as
a hard failure. For text-only rows, `Promotion precheck` skips the fake-stream
requirement because their configs declare `capabilities.image=false`.
Those same wrappers now also pass `--startup-require-cached-artifacts` by
default, so their compare reports always show `Startup precheck` for cached
versus first-download rows. Set
`JETSON_CURRENT_DEFAULTS_FAIL_ON_STARTUP_PRECHECK=1` or
`JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK=1` or
`JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK=1` when a suite should return
non-zero if any row fails that cached-startup gate; the default remains `0` so
diagnostic runs can still keep first-download evidence in the report.
They also pass `--ranking-require-startup-precheck` by default, so `Ranking precheck`
will only pass rows that already passed `Startup precheck`. Set
`JETSON_CURRENT_DEFAULTS_FAIL_ON_RANKING_PRECHECK=1` or
`JETSON_LIGHTWEIGHT_FAIL_ON_RANKING_PRECHECK=1` or
`JETSON_TENCENT_TEXT_FAIL_ON_RANKING_PRECHECK=1` when a suite should return
non-zero if any row fails that ranking gate; the default remains `0`.
When downstream ranking/export automation needs a machine-readable decision
artifact instead of only Markdown, also pass `--eligibility-output
outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json`.
That JSON sidecar carries the same `Startup precheck`, `Ranking precheck`, and
`Promotion precheck` states per row, plus top-level eligible row lists for each
gate. The remote suite wrappers now emit that
`comparison.eligibility.json` sidecar by default next to `comparison.md`.
To turn that compare-sidecar into a filtered downstream artifact, run
`python -m edge_vlm.optimization select-eligible --input
.../comparison.eligibility.json --gate ranking --output
.../ranking.selection.json` or switch the gate to `promotion` and write
`promotion.selection.json`. The remote suite wrappers now do both by default,
so ranking/promotion automation can consume filtered JSON directly instead of
re-parsing Markdown tables.
For scoped <=2B exports, the same command also accepts
`--require-leq2b-candidate` and `--candidate-lane <vlm|text>`. The lightweight
suite now uses those filters to emit `ranking.leq2b-vlm.selection.json` and
`promotion.leq2b-vlm.selection.json`, and the Tencent text suite now emits
`ranking.leq2b-text.selection.json` and
`promotion.leq2b-text.selection.json`, next to the unfiltered
`ranking.selection.json` and `promotion.selection.json`.
Use `python -m edge_vlm.optimization bundle-selections` to merge those lane
artifacts into one `leq2b.candidate_bundle.json`; on Jetson,
`scripts/jetson/build_remote_leq2b_candidate_bundle.sh` accepts
`JETSON_LEQ2B_VLM_SELECTION_DIR` and `JETSON_LEQ2B_TEXT_SELECTION_DIR` and
writes the same unified bundle remotely. It also exports `leq2b.routes.json`
from that bundle by default; set `JETSON_LEQ2B_BUILD_ROUTES=0` to build only
the bundle, or override `JETSON_LEQ2B_ROUTE_GATE` /
`JETSON_LEQ2B_ROUTES_OUTPUT` for a scoped route artifact.
Use `python -m edge_vlm.optimization export-routes --input
.../leq2b.candidate_bundle.json --gate promotion --output
.../leq2b.routes.json` when a downstream router needs lane-grouped primary and
backup candidates. The route export applies the `q4_first_q8_fallback` policy
inside each `comparison_group`: if a Q4 and Q8 row for the same group both pass
the chosen gate, Q4 becomes the route primary for that group and Q8 is emitted as
fallback metadata. Fallback metadata carries the row's `selection` and
`selection_context` when present, so selector-chosen Q8 routes keep the Q4
blocker evidence. It preserves cross-group bundle order and does not redefine
the ranking/promotion gate semantics.
When the selector chooses the Q8 fallback under a relaxed fallback gate, the
wrapper also forwards `--variant-min-lfb-blocks
qwen3-vl-2b-instruct-q8-smoke=...`, and the sweep plan records that under
`variant_min_lfb_blocks` so the selected fallback lane is not re-blocked by the
stricter suite-wide `--min-lfb-blocks`.

Remote Jetson connection settings live in the ignored `.env.jetson` file. The
remote helpers source it automatically; do not copy SSH hosts, passwords,
tokens, or private paths into tracked docs.
For Tailscale-style `100.x` targets, `scripts/jetson/check_remote_access.sh`
also prints `tailnet_probe=...` to distinguish local tailnet CLI/status issues
from SSH authentication failures.
On WSL hosts where Windows has the Tailscale route, set
`JETSON_SSH_BIN=/mnt/c/Windows/System32/OpenSSH/ssh.exe` and
`JETSON_SSH_PASSWORD_HELPER=none` to use Windows OpenSSH keys/agent; the access
precheck can also use host-side `tailscale.exe` and PowerShell TCP probes.

Use `JETSON_DRY_RUN=1` to print the Docker command without requiring Docker or Jetson hardware:

```bash
JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

For formal Jetson runs, start one of the model servers above first, then run the benchmark wrapper so JSONL, Markdown, manifest, profile files, and optional `tegrastats` output share one run id:

```bash
EDGE_VLM_FORMAL_RUN_ID=minicpmv46-q4-jetson-formal-001 \
EDGE_VLM_CONFIG=configs/models/minicpmv46_q4.yaml \
EDGE_VLM_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.jsonl \
EDGE_VLM_SUMMARY_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.md \
EDGE_VLM_METADATA_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.manifest.json \
EDGE_VLM_TRIAL_COUNT=3 \
scripts/jetson/run_formal_benchmark.sh
```

Use `EDGE_VLM_FORMAL_DRY_RUN=1 EDGE_VLM_SKIP_TEGRASTATS=1` to validate the wrapper without a running server or Jetson hardware.

See [docs/migration_wsl_to_jetson.md](docs/migration_wsl_to_jetson.md) for the full checklist.

## Documentation

- [Runtime matrix](docs/runtime_matrix.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [Next phase roadmap](docs/specs/next_phase_benchmark_and_models.md)
- [WSL to Jetson migration](docs/migration_wsl_to_jetson.md)
- [Implementation plan](docs/implementation_plan.md)
- [APEX workflow matrix](docs/matrix_edge_vlm_workflow.md)
- [OrangePi MiniCPM-V 4.6 notes](docs/reference_notes/orangepi_minicpmv46_notes.md)
