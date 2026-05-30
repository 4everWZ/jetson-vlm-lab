# Benchmark Protocol

The benchmark harness is designed to run the same prompt cases on WSL and Jetson. It records observations; it does not claim accurate performance unless the server and model actually ran.

## Cases

`configs/benchmark/prompt_cases.jsonl` contains:

- `text_cn_short`
- `text_en_reasoning_short`
- `text_code_short`
- `image_caption_single`
- `image_safety_scene_single`
- `fake_stream_folder_sample`

Small non-private sample images are included under `data/sample_images/` so dry runs and payload checks work after clone. Do not commit large or private media.

`fake_stream_folder_sample` is a marker case in the shared prompt list. The benchmark runner records it as a reminder to use the fake-stream harness; folder iteration itself is handled by `python -m edge_vlm.fake_stream`.

## Raw Output

The benchmark writes JSONL records with:

- `model`
- `backend`
- `quantization`
- `model_ref`
- `device`
- `run_id`
- `trial_index`
- `case_index`
- `prompt_case_id`
- `input_type`
- `image_path`
- `start_time`
- `end_time`
- `latency_s`
- `tokens`
- `tokens_per_sec`
- `success`
- `error`
- `output_excerpt`

Token counts are recorded only when the backend response exposes usage fields. If token counts are missing, `tokens` and `tokens_per_sec` remain null.

## Manifest Sidecar

Use `--metadata-output` for formal runs. The manifest records:

- run id, start/end timestamps, and device label
- config/cases/output/summary/metadata paths
- model name, family, backend, model ref, and quantization
- benchmark arguments, including dry-run mode, token limit, temperature, stream mode, and trial count
- success/failure counts for the current run
- selected runtime environment variables such as model paths, context size, GPU layers, batch/ubatch, container image, and server port
- Jetson profile pointers for `tegrastats`, `nvpmodel`, and `jetson_clocks` when captured by the formal wrapper

Example:

```bash
PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/minicpmv46_q4.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/minicpmv46-q4-formal.jsonl \
  --summary-output outputs/benchmarks/minicpmv46-q4-formal.md \
  --metadata-output outputs/benchmarks/minicpmv46-q4-formal.manifest.json \
  --run-id minicpmv46-q4-formal-001 \
  --trial-count 3 \
  --max-tokens 64 \
  --temperature 0
```

## WSL Dry Run

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl-dryrun.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl-dryrun.md \
  --dry-run
```

Dry run validates payload construction and logging only. It does not contact a server and must not be used as a performance result.

## WSL Real Run

Start a server first, then check health. Prefer the CUDA wrapper on this WSL host after `tmp/llama.cpp/build-cuda` exists; use the CPU fallback when GPU access is unavailable.

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
scripts/common/check_server.sh
```

In a second terminal:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl.md
```

Observed WSL CUDA smoke for Gemma Q8 used `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, one server slot, and `VLM_SERVER_PORT=18081`. The benchmark harness recorded three successful text cases, two successful sample-image cases, and the fake-stream marker case. A real fake-stream run against `data/sample_stream/` also succeeded with one frame.

The lower text-only setting `LLAMA_UBATCH_SIZE=32` triggered this llama.cpp assertion on the first image request:

```text
GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed
```

Keep the Gemma CUDA wrapper's 512 batch/ubatch defaults for image smoke runs unless you are deliberately retesting that boundary.

For Gemma Q4, use the downloaded pre-built Q4 artifacts and the Q4 config:

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
VLM_SERVER_PORT=18083 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

For MiniCPM-V 4.6, inspect the official pre-built GGUF repo metadata without downloading model files:

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

Then download the official pre-built Q4_K_M model and F16 mmproj files:

```bash
scripts/wsl/prepare_minicpmv46_q4.sh
```

Start MiniCPM-V 4.6 from the default downloaded paths:

```bash
VLM_SERVER_PORT=18082 \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

Observed WSL CUDA smoke for MiniCPM-V 4.6 Q4 used `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=128`, `LLAMA_UBATCH_SIZE=32`, one server slot, and `VLM_SERVER_PORT=18082`. The benchmark harness recorded three successful text cases, two successful sample-image cases, and the fake-stream marker case. A real fake-stream run against `data/sample_stream/` also succeeded with one frame.

## Fake Stream Run

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "For this frame, describe the most important object or activity in one sentence." \
  --output outputs/fake_stream/gemma4-e2b-q8-wsl.jsonl
```

Add `--dry-run` to validate folder iteration and JSONL logging without contacting a server.

## Jetson Run

For smoke runs, start `tegrastats` in one terminal:

```bash
scripts/jetson/monitor_tegrastats.sh
```

Start the model server in another terminal, then run the same benchmark command with `EDGE_VLM_DEVICE=jetson-orin`.

```bash
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-jetson.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-jetson.md
```

For formal Jetson runs, start the model server first and then use the wrapper so JSONL, Markdown summary, manifest, device profile, and optional `tegrastats` log use one run id:

```bash
EDGE_VLM_FORMAL_RUN_ID=minicpmv46-q4-jetson-formal-001 \
EDGE_VLM_CONFIG=configs/models/minicpmv46_q4.yaml \
EDGE_VLM_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.jsonl \
EDGE_VLM_SUMMARY_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.md \
EDGE_VLM_METADATA_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.manifest.json \
EDGE_VLM_TRIAL_COUNT=3 \
EDGE_VLM_MAX_TOKENS=64 \
EDGE_VLM_TEMPERATURE=0 \
scripts/jetson/run_formal_benchmark.sh
```

Use `EDGE_VLM_FORMAL_DRY_RUN=1 EDGE_VLM_SKIP_TEGRASTATS=1` to validate the wrapper without a running server or Jetson hardware.

## Jetson Optimization Sweep

Use the sweep wrapper when comparing server parameter variants. It starts each
variant, runs the formal benchmark, optionally runs one fake-stream frame, and
builds an optimization report that excludes sanity-failed output from ranking.
The report includes fake-stream latency and fake-stream guard failures when the
fake-stream sidecar exists.

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --run-prefix minicpm-opt-001 \
  --model minicpmv46-q4 \
  --trial-count 3 \
  --max-tokens 64 \
  --temperature 0
```

Validate planned commands first with:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --dry-run \
  --plan-output outputs/optimization_sweeps/plan.json \
  --run-prefix opt-plan \
  --model minicpmv46-q4
```

Variants are defined in `configs/benchmark/jetson_optimization_variants.jsonl`;
the promotion rules are documented in `docs/specs/jetson_optimization_loop.md`.
Each variant also writes `preflight/*.preflight.json` under the sweep output
root so Jetson `lfb` and memory state are visible before server startup.
For promotion sweeps, add `--min-lfb-blocks 150` or a stricter threshold learned
from prior runs so memory-fragmented starts are skipped and labeled before
Docker launches.
When comparing variants back-to-back, add `--pre-variant-command` to run the
same cleanup before each preflight, for example:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --run-prefix minicpm-promo-001 \
  --model minicpmv46-q4 \
  --trial-count 5 \
  --max-tokens 64 \
  --temperature 0 \
  --min-lfb-blocks 150 \
  --pre-variant-command "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
```

A non-zero preparation command skips that variant before preflight and records
the failure in the sweep manifest.

## Remote Jetson Execution

For repeatable remote runs, keep SSH settings in ignored `.env.jetson`:

```bash
cp docs/examples/jetson_remote.env.example .env.jetson
```

Then check command construction without connecting:

```bash
JETSON_REMOTE_DRY_RUN=1 scripts/jetson/remote_exec.sh \
  git status --short --branch
```

Run a command on the Jetson worktree:

```bash
scripts/jetson/remote_exec.sh \
  bash scripts/jetson/run_optimization_sweep.sh \
    --dry-run \
    --run-prefix remote-plan \
    --variant minicpm-q4-baseline-b128-u32-kvq8
```

The helper supports SSH keys by default. If `JETSON_SSH_PASSWORD` or
`JETSON_SSH_PASSWORD_FILE` is set in `.env.jetson`, it uses `sshpass` when
available and otherwise falls back to `SSH_ASKPASS` with `setsid`. Set
`JETSON_SSH_PASSWORD_HELPER=sshpass` or `JETSON_SSH_PASSWORD_HELPER=askpass` to
force one mode. Dry-run output never prints the password. Do not commit
`.env.jetson`.

For optimization sweeps, prefer the remote sweep wrapper so the Jetson worktree
is updated and the pinned llama.cpp image is applied consistently:

```bash
scripts/jetson/run_remote_optimization_sweep.sh \
  --run-prefix minicpm-promo-iso-001 \
  --variant minicpm-q4-baseline-b128-u32-kvq8 \
  --variant minicpm-q4-b512-u128-kvq8 \
  --trial-count 5 \
  --max-tokens 64 \
  --temperature 0 \
  --min-lfb-blocks 150 \
  --pre-variant-command "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" \
  --wait-timeout-s 180
```

Set `JETSON_REMOTE_SYNC=0` to skip the initial `git pull --ff-only`, or set
`JETSON_REMOTE_LLAMA_CPP_IMAGE` to test another pinned llama.cpp image.
The generated sweep plan records inherited launcher environment, including the
pinned llama.cpp image, so the plan itself contains the server image evidence.

## Reporting Rules

- Report dry-run logs as payload/logging validation only.
- Report WSL and Jetson logs separately.
- Include model ref, quantization, context size, `N_GPU_LAYERS`, Jetson power mode, storage location, run id, trial count, manifest path, `tegrastats` status, and whether image cases succeeded.
- Do not compare Jetson and WSL as equivalent hardware. Use WSL for development correctness and Jetson for edge runtime observations.
