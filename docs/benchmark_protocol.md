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
root so `meminfo_kb`, `/proc/buddyinfo`, and structured `tegrastats` `lfb`
state are visible before server startup.
When `--pre-variant-command` is used, the sweep now also writes
`preflight/*.preflight-before-prepare.json` first, keeps
`preflight/*.preflight.json` as the post-prepare gate sample, and records a
machine-readable delta in the sweep manifest so the effect of
cache-drop/`compact_memory` can be compared directly instead of inferred from a
single snapshot.
For promotion sweeps, add `--min-lfb-blocks 150` or a stricter threshold learned
from prior runs so memory-fragmented starts are skipped and labeled before
Docker launches.
The sweep plan now persists that global threshold under `plan.min_lfb_blocks`,
so later comparison/report passes can still recover the intended strict gate
even when an older result row does not yet carry
`preflight_required_lfb_blocks`.
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
image, and `org.opencontainers.image.version` as the llama.cpp ref. It also
probes `llama-server --help` inside the selected image, recording whether the
server binary was found, which path resolved, and whether the help output
exposes `--mmproj`. The sweep does not copy container environment variables
into the manifest.
For image-capable variants, this repo treats `--mmproj` as the mechanical proxy
for the required llama.cpp multimodal path. When the runtime probe says the
container lacks that support, the sweep skips the row before server startup with
`preflight_reason=runtime_missing_mmproj_support` instead of paying model
download and startup cost first.
The direct VLM Docker launchers use the same lower-level check before real
startup and before artifact checks or downloads. `JETSON_DRY_RUN=1` remains a
command preview only; set `LLAMA_CPP_RUNTIME_PROBE_OUTPUT=...` to choose the
launcher probe JSON path for a reproducible artifact.
This exact-flag check matters in practice: on June 9, 2026, the Jetson probe
found `/usr/local/bin/llama-server` inside
`dustynv/llama_cpp:b5283-r36.4-cu128-24.04`, but the real Qwen3-VL 2B Instruct
Q4 smoke still exited with `error: invalid argument: --mmproj`, so a generic
`mmproj` substring is not enough to count as multimodal support.
When evaluating a runtime image directly, run the same lower-level check before
the sweep:

```bash
PYTHONPATH=src python -m edge_vlm.llama_cpp_runtime probe-image \
  --image ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
  --output outputs/jetson_inspect/llama_cpp_runtime_probe.json
```

The probe JSON records Docker metadata, resolved `llama-server` path, exact
`--mmproj` support, and `multimodal_ready`. Treat `multimodal_ready=false` as a
runtime-infra block for VLM rows until the image is replaced or rebuilt.
For the Qwen3-VL 2B Instruct Q4/Q8 pair, use
`scripts/jetson/select_qwen3_instruct_variant.sh` when you need an explicit
selection decision rather than a manual variant choice. The selector emits JSON
with the shared runtime probe, current preflight sample, per-candidate block
reasons, and `selected_variant_id`. Keep the conservative gate the same for both
lanes by default; use `--fallback-min-lfb-blocks` only for scoped diagnostic
fallback triage. `edge_vlm.jetson_sweep` also accepts
`--selection-context-json <path>` so a wrapper can forward that selector JSON
into the sweep manifest as normalized `selection_contexts`. When a selector run
chooses a fallback lane under a relaxed gate, forward the same decision as
`--variant-min-lfb-blocks <variant-id>=<min-blocks>`; the sweep plan records
those per-variant overrides under `variant_min_lfb_blocks`, and execution uses
them instead of re-applying the stricter global `--min-lfb-blocks` to the
selected fallback row.
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
The remote wrapper now also forwards a structured `prepare_context` into the
sweep plan so later compare/promote steps can recover how the run was prepared
without inferring it from shell snippets alone. When those env flags are used,
`plan.prepare_context` records booleans such as `max_clocks_enabled` and
`drop_caches_before_variant`, plus supporting fields like
`max_clocks_capture` and `pre_variant_command_source`.

Build a comparison table from one or more sweep manifests with:

```bash
PYTHONPATH=src python -m edge_vlm.optimization compare \
  --manifest outputs/optimization_sweeps/minicpm-promo-iso-001/minicpm-promo-iso-001.manifest.json \
  --baseline-variant minicpm-q4-baseline-b128-u32-kvq8 \
  --ranking-min-lfb-blocks 150 \
  --promotion-precheck-stage formal-repeat \
  --output outputs/optimization_sweeps/minicpm-promo-iso-001/comparison.md \
  --eligibility-output outputs/optimization_sweeps/minicpm-promo-iso-001/comparison.eligibility.json
```

The comparison report reads each sweep manifest, matching benchmark JSONL,
fake-stream sidecar, benchmark metadata, and `tegrastats` log. It adds
runtime image/id/llama.cpp ref, preflight `lfb`, trial count, startup time,
guard status, success counts, throughput, latency, max temperature, average
`VDD_IN` power, average GR3D utilization, average EMC utilization, minimum
profiled `lfb`, conservative bottleneck labels, delta columns against the
selected baseline variant, and any recorded prepare-phase
preflight deltas such as `lfb` or `MemAvailable` changes after cache-drop plus
`compact_memory`. When variant metadata defines a shared
`comparison_group`, the delta columns use that group instead of raw model id,
so Q4/Q8 fallback lanes can be compared directly. When the sweep manifest also
includes `selection_contexts`, the report adds a `Selection` column so
auto-selected lanes remain visible in promotion evidence instead of reading like
anonymous static variant ids. The sweep manifest also carries any
`variant_min_lfb_blocks` overrides used to keep a selected fallback lane
aligned with its selector gate, and the comparison table surfaces that gate in
its `Required lfb` column next to the observed `Preflight lfb`. Compare first
uses a row's explicit `preflight_required_lfb_blocks`; if that is missing, it
backs off to `variant_min_lfb_blocks` and then `plan.min_lfb_blocks` from the
sweep manifest before labeling the row as missing required-LFB evidence.
When `plan.prepare_context` is present, the comparison table also adds a
`Prepare ctx` column. Today it summarizes the mechanically important prepare
signals as `max_clocks` and `drop_caches`, so promotion review can tell whether
a row came from a locked-clocks run, a cache-drop plus `compact_memory` run, or
both without reopening the raw manifest.
When profile phase timings include `artifact_check_or_download`, compare also
adds `Artifact phase` and `Artifact s` columns so first-download rows can be
separated from cached-start rows without opening lifecycle or profile-summary
sidecars by hand.
Host-side GGUF launchers also validate model and mmproj artifacts with a GGUF
magic-byte check before `llama-server` startup. A corrupt cached artifact must
be removed or restored and rerun; direct local-artifact launchers record
`invalid_model` or `invalid_mmproj` in the artifact phase before exiting.
When startup time itself is under review, add
`--startup-require-cached-artifacts`. Compare then adds a `Startup precheck`
column and only marks rows as startup-comparable when
`artifact_check_or_download` explicitly recorded `cached`; first-download rows
and older manifests without that phase remain visible but fail the startup
precheck.
For ranking decisions, keep only the defensible strict rows: when compare also receives
`--ranking-min-lfb-blocks`, it adds a `Ranking precheck` column so rows that
ran under a more relaxed gate stay in the report but are mechanically marked as
non-ranking evidence. Copy only the defensible summary rows into tracked
benchmark docs; keep raw generated reports under ignored `outputs/`.
Add `--ranking-require-startup-precheck` when ranking or export decisions must
also reject first-download rows and rows without explicit cached-startup
evidence.
For promotion-oriented review, compare also accepts
`--promotion-precheck-stage formal-repeat` and
`--promotion-precheck-stage promotion-reference`. This adds a
`Promotion precheck` column that checks the mechanical gate only: locked clocks,
cache drop, strict required-LFB floor, `max_tokens >= 64`, `temperature = 0`,
full benchmark success, fake-stream success for image-capable rows, and the
stage-specific trial floor. Text-only rows whose configs declare
`capabilities.image=false` do not need fake-stream records for this gate. Add
`--promotion-require-startup-precheck` when the promotion gate should also
require a passing `Startup precheck`, which means the same row already proved
cached-startup evidence under `--startup-require-cached-artifacts`. Add
`--promotion-require-quality-review` when the promotion gate should also require
a passing structured `Quality review` sidecar. Use
`formal-repeat` for the 5-trial lightweight ladder and
`promotion-reference` for 10-trial baseline/reference refreshes. The raw excerpt review remains manual and is not replaced by this column.
When ranking/export automation needs the same gate state without scraping the
Markdown table, add `--eligibility-output
outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json`. That
JSON sidecar records the same per-row `Startup precheck`, `Ranking precheck`,
and `Promotion precheck` results, plus top-level eligible row lists for each
gate.
To export a filtered downstream artifact from that sidecar, run:

```bash
PYTHONPATH=src python -m edge_vlm.optimization select-eligible \
  --input outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json \
  --gate ranking \
  --output outputs/optimization_sweeps/<run-prefix>/ranking.selection.json
```

Switch `--gate` to `promotion` and write `promotion.selection.json` when the
consumer should only see promotion-pass rows. This command reads the compare
eligibility JSON directly instead of scraping Markdown.
For scoped <=2B candidate exports, add `--require-leq2b-candidate` and
`--candidate-lane <vlm|text>`. For example, the lightweight suite now emits
`ranking.leq2b-vlm.selection.json` and `promotion.leq2b-vlm.selection.json`
with:

```bash
PYTHONPATH=src python -m edge_vlm.optimization select-eligible \
  --input outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json \
  --gate ranking \
  --require-leq2b-candidate \
  --candidate-lane vlm \
  --output outputs/optimization_sweeps/<run-prefix>/ranking.leq2b-vlm.selection.json
```

Switch `--candidate-lane` to `text` and write
`ranking.leq2b-text.selection.json` or `promotion.leq2b-text.selection.json`
when the consumer wants only the `<=2B` text/router candidate rows.
When a consumer needs both lanes in one artifact, merge the scoped selection
files with `bundle-selections`:

```bash
PYTHONPATH=src python -m edge_vlm.optimization bundle-selections \
  --input outputs/optimization_sweeps/<vlm-prefix>/ranking.leq2b-vlm.selection.json \
  --input outputs/optimization_sweeps/<vlm-prefix>/promotion.leq2b-vlm.selection.json \
  --input outputs/optimization_sweeps/<text-prefix>/ranking.leq2b-text.selection.json \
  --input outputs/optimization_sweeps/<text-prefix>/promotion.leq2b-text.selection.json \
  --output outputs/optimization_sweeps/<bundle-prefix>/leq2b.candidate_bundle.json
```

For a remote Jetson bundle, set `JETSON_LEQ2B_VLM_SELECTION_DIR` and
`JETSON_LEQ2B_TEXT_SELECTION_DIR`, then run
`scripts/jetson/build_remote_leq2b_candidate_bundle.sh`. The helper now exports
`leq2b.routes.json` with the `promotion` gate by default after the bundle step.
Set `JETSON_LEQ2B_BUILD_ROUTES=0` to skip that second remote command, or set
`JETSON_LEQ2B_ROUTE_GATE` and `JETSON_LEQ2B_ROUTES_OUTPUT` to produce a scoped
route artifact.

Export a router-facing view from that bundle with `export-routes`:

```bash
PYTHONPATH=src python -m edge_vlm.optimization export-routes \
  --input outputs/optimization_sweeps/<bundle-prefix>/leq2b.candidate_bundle.json \
  --gate promotion \
  --output outputs/optimization_sweeps/<bundle-prefix>/leq2b.routes.json
```

The route artifact groups candidates by lane and applies the
`q4_first_q8_fallback` export policy inside each `comparison_group`. If Q4 and
Q8 rows for the same group both pass the selected gate, the Q4 row is promoted
to that group's route primary and the Q8 row is recorded in `fallbacks` /
`fallback_groups`. Cross-group order still follows the bundle's selected order,
and empty lanes stay explicit with `primary: null`. This does not change the
Ranking precheck / Promotion precheck semantics.

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

For finished sweep manifests, use the sweep-level helper instead of invoking
the review command one JSONL at a time:

```bash
PYTHONPATH=src python -m edge_vlm.sweep_quality_review \
  --manifest outputs/optimization_sweeps/<run-prefix>/<run-prefix>.manifest.json \
  --policy configs/benchmark/quality_review_policy.json \
  --allow-failures
```

This writes `<run-id>.quality.json` and `<run-id>.quality.md` sidecars next to
the manifest, records those paths under each variant's
`paths.quality_review_json` and `paths.quality_review_markdown`, and lets
`edge_vlm.optimization compare` add a `Quality review` column when those
sidecars are present. The remote current-defaults, lightweight, and Tencent
text suite wrappers now run `edge_vlm.sweep_quality_review` automatically
before compare so the structured review state stays attached to the same sweep
artifact set. The current-defaults and lightweight wrappers also pass
`--promotion-require-quality-review` into compare, so their
`Promotion precheck` rows only pass when the structured sidecar passes as well.

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
Host-side GGUF artifact checks include a GGUF magic-byte validation step before
server startup, so corrupt model or mmproj files fail as artifact problems
instead of surfacing later as runtime startup failures.

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

That wrapper now also runs `edge_vlm.sweep_quality_review` on the finished
manifest and calls compare with `--promotion-require-startup-precheck` plus
`--promotion-require-quality-review`, so the reference `Promotion precheck`
includes both the cached-startup gate and the structured `Quality review`
gate. It also passes `--startup-require-cached-artifacts` by default, so the
same report always shows `Startup precheck` for cached versus first-download
rows.
It also writes `comparison.eligibility.json` next to `comparison.md`, using
compare's `--eligibility-output` to persist the same gate state in a
machine-readable form.
It also runs `edge_vlm.optimization select-eligible` for both `ranking` and
`promotion`, writing `ranking.selection.json` and `promotion.selection.json`
next to the compare outputs so downstream automation can consume filtered rows
without re-implementing gate parsing.
Set `JETSON_CURRENT_DEFAULTS_FAIL_ON_PROMOTION_PRECHECK=1` when the wrapper
should return non-zero if that promotion gate fails; the default remains `0`
so comparison evidence can still be collected during diagnostic runs. Set
`JETSON_CURRENT_DEFAULTS_FAIL_ON_STARTUP_PRECHECK=1` when the wrapper should
also return non-zero if any row fails the cached-startup gate; that default
also remains `0`.
The wrapper also passes `--ranking-require-startup-precheck` by default, so
`Ranking precheck` only passes rows that already passed `Startup precheck`.
Set `JETSON_CURRENT_DEFAULTS_FAIL_ON_RANKING_PRECHECK=1` when the wrapper
should also return non-zero if any row fails that ranking gate; the default
remains `0`.

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
baseline variants. It also runs `edge_vlm.sweep_quality_review` with
`configs/benchmark/quality_review_policy.json` against the sweep manifest so
the recorded `quality_review_json` sidecars can feed the compare report's
`Quality review` column. The wrapper also passes
`--promotion-require-startup-precheck` and
`--promotion-require-quality-review`, so the reported `Promotion precheck`
requires cached-startup evidence and that structured sidecar to pass. It also passes
`--startup-require-cached-artifacts`, so the same report always includes
`Startup precheck` for cached versus first-download rows. Set
The wrapper also writes `comparison.eligibility.json` next to `comparison.md`
through compare's `--eligibility-output`, so downstream ranking/export steps
can consume gate results without parsing Markdown.
It also runs `edge_vlm.optimization select-eligible` for `ranking` and
`promotion`, writing `ranking.selection.json` and `promotion.selection.json`
next to the compare outputs. It also emits the scoped <=2B VLM exports
`ranking.leq2b-vlm.selection.json` and
`promotion.leq2b-vlm.selection.json` by calling `select-eligible` with
`--require-leq2b-candidate --candidate-lane vlm`.
`JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK=1` when the wrapper should
return non-zero on a failed promotion gate; the default remains `0` so
lightweight diagnostic ladders can still emit comparison evidence. Set
`JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK=1` when the wrapper should also
return non-zero on a failed cached-startup gate; that default remains `0`.
The wrapper also passes `--ranking-require-startup-precheck` by default, so
`Ranking precheck` only passes rows that already passed `Startup precheck`.
Set `JETSON_LIGHTWEIGHT_FAIL_ON_RANKING_PRECHECK=1` when the wrapper should
also return non-zero on a failed ranking gate; that default remains `0`.
Override `JETSON_LIGHTWEIGHT_RUN_PREFIX` to make the output
path stable, or override `JETSON_LIGHTWEIGHT_TRIAL_COUNT`,
`JETSON_LIGHTWEIGHT_MAX_TOKENS`, `JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS`,
`JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S`, `JETSON_LIGHTWEIGHT_BASELINE_VARIANTS`,
`JETSON_LIGHTWEIGHT_CANDIDATE_VARIANTS`, or
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS` for scoped validation runs.
The suite also enables `JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR=1` by default, so
the Qwen3-VL 2B Instruct lane is resolved remotely through
`scripts/jetson/select_qwen3_instruct_variant.sh` after sync, max-clocks, and
the current runtime probe. The suite keeps the primary Qwen3 gate at the global
`JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS` value and uses
`JETSON_LIGHTWEIGHT_QWEN3_FALLBACK_MIN_LFB_BLOCKS=100` only for the Q8 fallback
lane. Set `JETSON_LIGHTWEIGHT_QWEN3_SELECTOR=0` if you need a fully manual
candidate list for a scoped rerun.
The default candidate list excludes HunyuanOCR and official Youtu-VL Q8:
HunyuanOCR loaded after its launcher-resume smoke but failed the guard with
repetitive punctuation output, and official Youtu-VL Q8 failed before server
ready with CUDA OOM on the BF16 mmproj buffer. Pass
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=hunyuanocr-q8-smoke` or
`JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=youtu-vl-4b-q8-smoke` only for scoped
quality/runtime triage reruns.
The default Qwen candidates include Qwen3-VL 2B Thinking Q4 plus an automatic
Qwen3-VL 2B Instruct Q4-first / Q8 fallback lane. The Instruct Q4 row is a
newly configured candidate with one diagnostic locked-clocks smoke on June 9,
2026:
`qwen3-instruct-q4-smoke-lfb32-20260609T075208Z` passed 6/6 formal records and
1/1 fake-stream record at relaxed `--min-lfb-blocks 32`, reaching 34.349 text
tok/s, 26.957 image tok/s, and 1.827 s fake-stream latency. The strict
`--min-lfb-blocks 150` rerun `qwen3-instruct-q4-smoke-compact-20260609T074600Z`
still skipped in preflight at `lfb 46x4MB` even after cache-drop plus
`compact_memory`, so keep this row at smoke/triage scope until it can satisfy
the conservative gate and pass quality review. The suite-level selector now
uses that strict 150-LFB gate for the primary lane and a 100-LFB fallback gate
for Q8, so the default lightweight ladder can automatically keep a Qwen3
Instruct row when Q4 is blocked only by fragmentation pressure.
The scoped Q8 fallback row `qwen3-vl-2b-instruct-q8-smoke` is now executable
through the same launcher path. Strict-gate smoke
`qwen3-instruct-q8-smoke-20260609T082222Z` also skipped in preflight at
`lfb 106x4MB`, so do not silently swap it into the default lightweight suite.
Relaxed-gate smoke `qwen3-instruct-q8-smoke-lfb100-20260609T082321Z` did pass
6/6 formal records and 1/1 fake-stream record with guard pass at
`--min-lfb-blocks 100`, reaching 31.076 text tok/s, 25.997 image tok/s, and
2.142 s fake-stream latency, but it included a 323.05 s first-download artifact
phase. Manual `JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=qwen3-vl-2b-instruct-q8-smoke`
reruns are still useful for scoped fallback triage when you need the Q8 row
even with `JETSON_LIGHTWEIGHT_QWEN3_SELECTOR=0`.
A shared relaxed-gate compare `qwen3-instruct-q4q8-lfb100-20260609T091700Z`
then ran both Instruct lanes under max clocks, per-variant cache-drop plus
`compact_memory`, three formal trials, and one fake-stream frame. Q4 passed
18/18 formal records plus 1/1 fake-stream record at 34.865 text tok/s, 31.958
image tok/s, and 1.827 s fake-stream latency with post-prepare preflight
`lfb 121x4MB`; its prepare delta was effectively flat at `lfb -1` and
`MemAvailable +1.816 MB`. Q8 also passed 18/18 formal plus 1/1 fake-stream at
31.346 text tok/s, 29.393 image tok/s, and 2.144 s fake-stream latency with
post-prepare preflight `lfb 125x4MB`, but its prepare delta was much larger at
`lfb +88` and `MemAvailable +963.598 MB`. Under the same relaxed gate, Q4
remained the better default lane while Q8 stayed a fallback path with much
stronger prepare sensitivity.

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
`--fake-stream-max-frames 0`, runs `edge_vlm.sweep_quality_review`, and writes
a comparison report with `--promotion-precheck-stage formal-repeat` plus
`--promotion-require-startup-precheck` and
`--promotion-require-quality-review`. It also passes
`--startup-require-cached-artifacts`, so the text compare report still shows
whether startup came from cached artifacts or a first download, and
`Promotion precheck` now consumes that cached-startup gate as well. It also
passes `--ranking-require-startup-precheck`, so `Ranking precheck` likewise
rejects first-download rows. Because these rows are text-only,
`Promotion precheck` skips the fake-stream requirement. Set
The wrapper also writes `comparison.eligibility.json` next to `comparison.md`
with compare's `--eligibility-output`, so the text/router ranking flow can
reuse the same machine-readable gate state.
It also runs `edge_vlm.optimization select-eligible` for `ranking` and
`promotion`, writing `ranking.selection.json` and `promotion.selection.json`
next to the compare outputs. It also emits scoped `<=2B` text artifacts,
`ranking.leq2b-text.selection.json` and
`promotion.leq2b-text.selection.json`, by calling `select-eligible` with
`--require-leq2b-candidate --candidate-lane text`.
Use `scripts/jetson/build_remote_leq2b_candidate_bundle.sh` after the
lightweight and Tencent text suites when both scoped lane outputs should be
published as one `leq2b.candidate_bundle.json`.
`JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK=1` when the wrapper should
return non-zero if any row fails that promotion gate; the default remains `0`
so diagnostic text ladders can still emit comparison evidence. Set
`JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK=1` when the wrapper should also
return non-zero if any row fails the cached-startup gate; that default remains
`0`. Set `JETSON_TENCENT_TEXT_FAIL_ON_RANKING_PRECHECK=1` when the wrapper
should also return non-zero on a failed ranking gate; that default remains
`0`. Low-bit rows are
runtime-compatibility canaries inside that dedicated text suite; the first
Hy-MT1.5 1.25bit Jetson smoke failed before server ready on the pinned llama.cpp
image with `invalid ggml type 42`. HY-MT1.5 Q4/Q6/Q8 rows remain executable
defaults without Jetson evidence. Prepared repeat
`tencent-text-repeat5-prepctx-20260609T131412Z` now provides strict
`--min-lfb-blocks 150` evidence for Hy-MT2 Q4/Q6/Q8 and Youtu-LLM 2B Q8 with
`prepare_context={"max_clocks_enabled": true, "drop_caches_before_variant":
true}` captured in the sweep plan; compare renders that as
`Prepare ctx = max_clocks, drop_caches`, and each scoped row reports
`Required lfb = 150`. In that run the Hy-MT2 rows all passed formal records and
`Ranking precheck`, but `Promotion precheck` failed on the quality gate only:
Q4 reported `quality_review_failed 15/20 text_en_reasoning_short`, while Q6
and Q8 both reported
`quality_review_failed 10/20 text_en_reasoning_short,text_code_short`.
Youtu-LLM 2B Q8 passed `20/20` records, `Quality review = yes (20/20)`, and
`Promotion precheck = yes`; its reported `354.568 s` startup includes the
first artifact download and should not be treated as cached startup evidence.
Cached rerun `youtu-llm-q8-cached-20260609T133339Z` then reused the local
artifact under the same strict gate: lifecycle timing recorded
`artifact_check_or_download = cached` in `0.002 s`, compare kept
`Prepare ctx = max_clocks, drop_caches` plus `Required lfb = 150`, and the row
still passed `Quality review = yes (20/20)` with `Promotion precheck = yes`.
The cached row now also surfaces `Artifact phase = cached` and
`Artifact s = 0.002` in compare, alongside `5.015 s` startup, `24.621` text
tok/s, and `1.861 s` average text latency.
Override
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
