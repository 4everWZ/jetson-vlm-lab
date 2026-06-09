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

`configs/benchmark/text_prompt_cases.jsonl` is the text/router-only subset. It
contains no image or fake-stream cases, and is used by text-only model variants
such as Tencent Hy-MT1.5, Hy-MT2, and Youtu-LLM. Do not mix text/router rows
into VLM rankings.

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
- `quality_terms_any`
- `input_timing`

Token counts are recorded only when the backend response exposes usage fields. If token counts are missing, `tokens` and `tokens_per_sec` remain null.

`input_timing` records client-side input preparation and request overhead when
available. Current keys include:

- `image_bytes`
- `mime_detect_s`
- `image_read_s`
- `base64_encode_s`
- `data_url_build_s`
- `payload_build_s`
- `json_serialize_s`
- `request_body_bytes`
- `http_request_s`
- `response_parse_s` for non-streaming responses when parsing completes

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
variant, runs the formal benchmark, optionally runs the default three-frame
fake-stream check for image-capable configs, and builds an optimization report that excludes
sanity-failed output from ranking. The report includes fake-stream latency and
fake-stream guard failures when the fake-stream sidecar exists.

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
  --pre-variant-command "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'"
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

Before starting a remote sweep, run the sanitized connectivity probe:

```bash
JETSON_REMOTE_PROBE_DRY_RUN=1 scripts/jetson/remote_probe.sh
scripts/jetson/remote_probe.sh
```

The probe prints `remote_probe=ok` only when SSH reaches the configured Jetson
worktree and the remote command returns the expected marker. On failure, it
classifies the first SSH boundary as `ssh_connect_timeout`,
`ssh_network_unreachable`, `ssh_auth_failed`, `ssh_failed`, or
`remote_command_failed` without printing `.env.jetson` secrets. A successful
probe is a connectivity preflight, not benchmark evidence and not a model
runtime success.

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
JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
scripts/jetson/run_remote_optimization_sweep.sh \
  --run-prefix minicpm-promo-iso-001 \
  --variant minicpm-q4-baseline-b128-u32-kvq8 \
  --variant minicpm-q4-b512-u128-kvq8 \
  --trial-count 5 \
  --max-tokens 64 \
  --temperature 0 \
  --min-lfb-blocks 150 \
  --wait-timeout-s 180
```

The wrapper syncs the remote Jetson worktree to `main` by default with
`git fetch origin main` and `git checkout --detach FETCH_HEAD` before the sweep.
Detached sync avoids Git worktree conflicts when `main` is already checked out
in another Jetson directory. Set `JETSON_REMOTE_BRANCH` to run an unmerged
branch intentionally, set `JETSON_REMOTE_SYNC=0` to skip the initial branch
sync, or set `JETSON_REMOTE_LLAMA_CPP_IMAGE` to test another pinned llama.cpp
image.
The generated sweep plan records inherited launcher environment, including the
pinned llama.cpp image. It also records safe Docker image metadata under
`plan.variants[].server_runtime` when `docker image inspect` is available:
image tag, image id, repo digests, creation time, OCI source revision, base
image, and `org.opencontainers.image.version` as the llama.cpp ref. The sweep
does not copy container environment variables into the manifest.
Set `JETSON_REMOTE_PREPARE_MAX_CLOCKS=1` for promotion/comparison sweeps. The
wrapper runs `sudo jetson_clocks` and captures `sudo jetson_clocks --show` under
ignored `outputs/jetson_inspect/` before launching the sweep. It reads the sudo
password from `JETSON_REMOTE_SUDO_PASSWORD` or `JETSON_SSH_PASSWORD` in the
ignored `.env.jetson` file and passes it over stdin; do not put passwords in
tracked files or command-line arguments.
Set `JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1` for back-to-back promotion
comparisons where memory fragmentation can bias later variants. The wrapper
uses the sudo password from stdin to feed a per-run 0600 FIFO, then appends a
recorded pre-variant command shaped like
`sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' < /tmp/...`.
The manifest records the command and FIFO path, not the password. Do not
combine this env flag with a manual `--pre-variant-command`; use the lower-level
local sweep command only when a custom preparation command is required.

Build a comparison table from one or more sweep manifests with:

```bash
PYTHONPATH=src python -m edge_vlm.optimization compare \
  --manifest outputs/optimization_sweeps/minicpm-promo-iso-001/minicpm-promo-iso-001.manifest.json \
  --baseline-variant minicpm-q4-baseline-b128-u32-kvq8 \
  --output outputs/optimization_sweeps/minicpm-promo-iso-001/comparison.md
```

The comparison report reads each sweep manifest, matching benchmark JSONL,
fake-stream sidecar, benchmark metadata, and `tegrastats` log. It adds
runtime image/id/llama.cpp ref, preflight `lfb`, trial count, startup time,
guard status, success counts, throughput, latency, max temperature, average
`VDD_IN` power, average GR3D utilization, average EMC utilization, minimum
profiled `lfb`, conservative bottleneck labels, and per-model delta columns
against the selected baseline variant. Copy only the defensible summary rows
into tracked benchmark docs; keep raw generated reports under ignored
`outputs/`.

For route-specific output review, run the benchmark JSONL through the quality
review policy before promoting a candidate beyond smoke/repeat evidence:

```bash
PYTHONPATH=src python -m edge_vlm.quality_review \
  --input outputs/optimization_sweeps/<run-prefix>/benchmarks/<run-id>.jsonl \
  --policy configs/benchmark/quality_review_policy.json \
  --output outputs/optimization_sweeps/<run-prefix>/<run-id>.quality.json \
  --markdown-output outputs/optimization_sweeps/<run-prefix>/<run-id>.quality.md
```

This check is deliberately stricter than the optimization sanity guard for
route-sensitive prompts such as WSL reasoning, code, safety, and translation.
It is a structured review aid, not a replacement for human review of raw
excerpts.

Successful sweep variants also write derived profile artifacts under
`outputs/optimization_sweeps/<run-prefix>/profiles/` and launcher lifecycle
events under `outputs/optimization_sweeps/<run-prefix>/lifecycle/`:

- `<run-id>.profile.jsonl` contains parsed `tegrastats` samples. When the raw
  log line has a UTC timestamp prefix, each profile record also includes
  `captured_at` and `elapsed_s` so `tegrastats` evidence can be aligned with
  benchmark, fake-stream, and lifecycle phases. `run_formal_benchmark.sh`
  prefixes `tegrastats` lines this way for formal runs.
- `<run-id>.summary.json` contains aggregate memory, power, thermal,
  GR3D/EMC/CPU, bottleneck labels, profile file pointers, and available phase
  timings. For timestamped logs it also records `first_captured_at`,
  `last_captured_at`, and `captured_duration_s`. It includes
  `input_timing_summary` aggregated from benchmark and fake-stream JSONL, with
  payload overhead, estimated end-to-end latency, request wait, request body
  bytes, image bytes, and per-source record counts.
- `<run-id>.lifecycle.jsonl` contains optional launcher phase records emitted
  through `EDGE_VLM_LAUNCH_PHASE_LOG`.

The currently instrumented phase timings are `artifact_check_or_download` when
the launcher emits lifecycle JSONL, `server_startup`, `formal_text`,
`formal_image`, `fake_stream`, and `shutdown`. Warmup is now classified instead
of left as generic `not_recorded`: variants with `--no-warmup` record
`disabled_by_variant`; variants without that flag record
`included_in_server_startup` until server logs or runtime hooks can separate
the internal llama.cpp warmup duration. Gemma `-hf` runtime downloads that
happen inside `llama-server` are labeled as not separated rather than guessed
as launcher time.

The `input_payload` bottleneck label is emitted only when average payload
preparation is non-trivial and accounts for a material share of estimated
end-to-end latency. `runtime_overhead` is emitted only when request wait
dominates while GR3D, EMC, and CPU utilization are not saturated. Treat both as
triage labels that choose the next investigation lane, not as proof of model
decode speed.

Benchmark and fake-stream JSONL records include `input_timing` when the client
path can measure it. Current fields include image byte count, MIME detection
time, image read time, base64 encoding time, data URL construction time,
payload build time, JSON serialization time, request body size, HTTP elapsed
time, and response parse time when applicable. Treat these as client-side
pipeline timings; they are separate from server decode throughput.

Fake-stream JSONL records also include `stream_timing`:

- `interval_s`: configured source-frame interval.
- `effective_interval_s`: source-frame interval used for this scheduled frame,
  which can differ from `interval_s` in adaptive experiments.
- `scheduled_offset_s`: nominal frame offset from stream start.
- `pre_frame_sleep_s`: time slept before starting the frame to match the fixed cadence.
- `schedule_delay_s`: how late the frame started relative to its nominal offset.
- `backpressure_s`: current accumulated delay from the fixed-cadence source schedule.
- `skipped_frames_before`: number of source frames skipped immediately before this processed record.
- `frame_elapsed_s`: local elapsed time spent processing the frame record.

The fake-stream runner now schedules frame starts against the nominal stream
clock instead of sleeping a fixed interval after each frame. This makes
backpressure visible when model/request latency exceeds the target frame
interval.

For routing and stream-control experiments, `edge_vlm.fake_stream` also accepts
`--skip-late-frames`. When enabled, non-final source frames whose current
backpressure is at or above `--skip-threshold-s` are dropped before model
inference, and the next processed record carries the accumulated
`skipped_frames_before` count. The default threshold is the configured
`--interval-s`, and the default behavior remains no skipping.

`edge_vlm.fake_stream` also accepts `--adaptive-interval` for controlled
stream-control experiments. When enabled, the next frame's effective interval
is derived from the previous processed frame's local elapsed time, bounded by
the configured base `--interval-s`, optional `--adaptive-interval-scale`, and
optional `--adaptive-interval-max-s`. This is an explicit experiment knob; the
default benchmark cadence remains fixed.

The sweep planner can pass these fake-stream controls through with
`--fake-stream-interval-s`, `--fake-stream-skip-late-frames`,
`--fake-stream-skip-threshold-s`, `--fake-stream-adaptive-interval`,
`--fake-stream-adaptive-interval-scale`, and
`--fake-stream-adaptive-interval-max-s`.

Text-only configs set `capabilities.image=false`. The sweep planner still runs
their formal benchmark, but it does not attach a fake-stream command even when
fake-stream is enabled globally. Put text-only variants on
`configs/benchmark/text_prompt_cases.jsonl` with `EDGE_VLM_CASES` in the variant
environment.

To refresh the current MiniCPM/Gemma default reference in one step, use:

```bash
scripts/jetson/run_remote_current_defaults_suite.sh
```

To run the fixed-policy lightweight model ladder with the current MiniCPM and
Gemma baselines plus the guard-passing lightweight candidates, use:

```bash
scripts/jetson/run_remote_lightweight_model_suite.sh
```

The wrapper runs the selected defaults for both target models with
`JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, `--trial-count 5`,
`--fake-stream-max-frames 3`, `--min-lfb-blocks 150`, and
`--wait-timeout-s 600`, then runs the mechanical comparison report against both
baseline variants. Override `JETSON_LIGHTWEIGHT_RUN_PREFIX` to make the output
path stable, or override `JETSON_LIGHTWEIGHT_TRIAL_COUNT`,
`JETSON_LIGHTWEIGHT_MAX_TOKENS`, `JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS`,
`JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S`, `JETSON_LIGHTWEIGHT_BASELINE_VARIANTS`,
`JETSON_LIGHTWEIGHT_CANDIDATE_VARIANTS`, or
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS` for scoped validation runs.
The default candidate list excludes HunyuanOCR and official Youtu-VL Q8:
HunyuanOCR loaded after its launcher-resume smoke but failed the guard with
repetitive punctuation output, and official Youtu-VL Q8 failed before server
ready with CUDA OOM on the BF16 mmproj buffer. Pass
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=hunyuanocr-q8-smoke` or
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=youtu-vl-4b-q8-smoke` only for scoped
quality/runtime triage reruns.
The default Qwen candidates include both Qwen3-VL 2B Thinking Q4 and
Qwen3-VL 2B Instruct Q4. The Instruct row is a newly configured Q4-first
candidate with no Jetson evidence yet; keep its first run at smoke scope until
it has guard, fake-stream, and quality-review evidence.

If a host-side HF GGUF download is interrupted after the bytes have completed
but before the launcher renames the `.partial` file, the generic HF GGUF
launchers recover a subsequent HTTP 416 resume response only when the existing
partial starts with GGUF magic bytes. This path was added after the first
HunyuanOCR run timed out during mmproj download; it is an artifact recovery
mechanism, not a model-quality signal.

For current Tencent text-only GGUF candidates, use the generic text launcher and
text cases through the Tencent text suite wrapper:

```bash
scripts/jetson/run_remote_tencent_text_suite.sh
```

The wrapper defaults to all eleven configured Tencent text GGUF rows, runs with
locked clocks, drops caches before each variant, sets
`--fake-stream-max-frames 0`, and writes a comparison report. Low-bit rows are
runtime-compatibility canaries inside that dedicated text suite; the first
Hy-MT1.5 1.25bit Jetson smoke failed before server ready on the pinned llama.cpp
image with `invalid ggml type 42`. The HY-MT1.5 Q4/Q6/Q8 and Youtu-LLM Q8 rows
are executable defaults but still need Jetson evidence before route use. Override
`JETSON_TENCENT_TEXT_RUN_PREFIX`,
`JETSON_TENCENT_TEXT_TRIAL_COUNT`, `JETSON_TENCENT_TEXT_MAX_TOKENS`,
`JETSON_TENCENT_TEXT_MIN_LFB_BLOCKS`, `JETSON_TENCENT_TEXT_WAIT_TIMEOUT_S`,
`JETSON_TENCENT_TEXT_VARIANTS`, or `JETSON_TENCENT_TEXT_EXTRA_VARIANTS` for
scoped validation. Those variants use
`scripts/jetson/run_hf_gguf_llama_docker.sh` and
`configs/benchmark/text_prompt_cases.jsonl`; fake-stream is also skipped by
their text-only configs.

Observed one-trial full-suite evidence is recorded in
`docs/benchmarks/jetson_lightweight_models_20260531.md` from
`tencent-text-smoke64-20260531T120054Z`: Hy-MT2 Q4 passed at 33.541 tok/s and
1.688 s average text latency, Hy-MT2 Q6 passed at 26.287 tok/s and 2.145 s
average text latency, and Q8 full-suite row is invalidated by the pre-fix
launcher cleanup/stale-port issue. Cached Q6 run
`tencent-hy-mt2-q6-smoke64-cached-20260531a` passed at 26.298 tok/s and
2.144 s average text latency with `terminate_group` shutdown. The fixed-harness
cached Q8 rerun `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` passed at
30.756 tok/s and 1.841 s average text latency with `terminate_group` shutdown.
The 5-trial text repeat `tencent-text-repeat5-20260531a` then passed Hy-MT2 Q4,
Q6, and Q8 at 20/20 records each: Q4 reached 34.528 tok/s and 1.643 s average
text latency, Q6 reached 26.778 tok/s and 2.108 s, and Q8 reached 31.395 tok/s
and 1.806 s. Low-bit rows still fail before server ready with the same
`invalid ggml type 42` and tensor-offset errors.
The fixed sweep harness now starts launcher processes in their own process
group, terminates the group, rejects a variant with
`server_port_still_open_before_start` when `/v1/models` is already served on
the target port, and records whether the server port closes after shutdown.
Keep the invalidated full-suite Q8 row separate from these cached rows.

## llama.cpp Runtime Image Builds

When testing a newer llama.cpp runtime, keep the build path artifact based:

```bash
LLAMA_CPP_REF=d749821db3bd587932d1ed57d43626cd552c9909 \
scripts/build_llama_cpp_image.sh
```

The artifact builder compiles llama.cpp inside the CUDA builder container and
writes only `artifacts/llama.cpp-install/`. The image builder then copies that
install tree into the runtime image and tags the image as
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-<ref7>` by default.
It passes only reproducibility labels (`BUILD_DATE`, `VCS_REF`, and
`LLAMA_CPP_REF`) to Docker. The repository `.dockerignore` narrows the build
context to the Dockerfile and `artifacts/llama.cpp-install/**`; local `.env`,
SSH settings, model weights, caches, generated outputs, and benchmark artifacts
are not sent to the Docker daemon and are not copied into the image.

Use the resulting tag through the remote sweep wrapper:

```bash
JETSON_REMOTE_LLAMA_CPP_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-d749821 \
scripts/jetson/run_remote_current_defaults_suite.sh
```

## Reporting Rules

- Report dry-run logs as payload/logging validation only.
- Report WSL and Jetson logs separately.
- Include model ref, quantization, context size, `N_GPU_LAYERS`, Jetson power mode, storage location, run id, trial count, manifest path, `tegrastats` status, and whether image cases succeeded.
- Do not compare Jetson and WSL as equivalent hardware. Use WSL for development correctness and Jetson for edge runtime observations.
