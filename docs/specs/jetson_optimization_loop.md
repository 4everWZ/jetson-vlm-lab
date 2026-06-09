# Jetson Optimization Loop

## Objective

Find the fastest usable MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 settings on the
Jetson without accepting obviously degraded output. The loop optimizes server
parameters, records formal benchmark evidence, and ranks only runs that pass a
lightweight output sanity guard.

## Guardrail

`python -m edge_vlm.optimization report` ranks benchmark JSONL files, and can
also read matching fake-stream JSONL files, only after checking:

- every non-placeholder benchmark record succeeded
- output excerpts are not empty
- output excerpts meet a minimum length
- output excerpts are not dominated by obvious repetition
- output excerpts include at least one case-specific `quality_terms_any` canary
  term when the prompt case defines one
- fake-stream frame outputs pass the same lightweight output checks when
  provided

This is a sanity guard, not a full quality evaluation. The canary terms catch
obvious off-topic or collapsed answers before speed ranking, but a promotion
still needs review against the raw excerpts when a speedup comes from precision
or cache changes. A high-speed candidate that fails the guard is kept in the
report but excluded from ranked candidates.

## Variant Source

Variants live in `configs/benchmark/jetson_optimization_variants.jsonl`.

Current sweep knobs are deliberately narrow:

- `CTX_SIZE`
- `N_GPU_LAYERS`
- `LLAMA_BATCH_SIZE`
- `LLAMA_UBATCH_SIZE`
- llama.cpp server args observed in the repo or verified in the pinned
  Jetson container help: `--parallel`,
  `--batch-size`, `--ubatch-size`, `--cache-type-k`, `--cache-type-v`,
  `--flash-attn`, `--mlock`, `--mmap`/`--no-mmap`, `--cont-batching`,
  `--no-warmup`, and Gemma `-fit off`
- Docker launch env `DOCKER_GPU_ARGS` only for runtime capability flags needed
  by a measured server flag, such as raising `memlock` for `--mlock`

Do not add speculative llama.cpp flags until the container help or a dry-run
command confirms the flag exists in the pinned Jetson image.

## Dry Run

Use dry-run mode before launching Docker:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --dry-run \
  --plan-output outputs/optimization_sweeps/plan.json \
  --run-prefix opt-plan \
  --model minicpmv46-q4
```

The plan records server commands, benchmark output paths, preflight output
paths, environment overrides, and fake-stream commands for image-capable
configs. Text-only configs set `capabilities.image=false`; the planner keeps
their formal benchmark but omits the fake-stream sidecar. The sweep sets
`DOCKER_TTY=0` so Docker can run under background automation instead of
requiring an interactive terminal.
Launcher-affecting inherited environment variables such as
`LLAMA_CPP_DOCKER_IMAGE`, `LLAMA_CPP_DOCKER_IMAGE_FALLBACK`,
`LLAMA_SERVER_CMD`, and `DOCKER_GPU_ARGS` are copied into each variant
`server_env` so dry-run plans remain reproducible.

## Real Sweep

Run a small sweep first:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --run-prefix minicpm-opt-001 \
  --model minicpmv46-q4 \
  --trial-count 3 \
  --max-tokens 64 \
  --temperature 0
```

Outputs stay under ignored `outputs/optimization_sweeps/<run-prefix>/`:

- `benchmarks/*.jsonl`
- `benchmarks/*.md`
- `benchmarks/*.manifest.json`
- `fake_stream/*.jsonl`
- `lifecycle/*.lifecycle.jsonl`
- `profiles/*.profile.jsonl`
- `profiles/*.summary.json`
- `preflight/*.preflight-before-prepare.json` when a pre-variant prepare command is used
- `preflight/*.preflight.json`
- `server_logs/*.server.log`
- `optimization_report.md`
- `<run-prefix>.manifest.json`

Each variant captures a preflight JSON file before server startup. On Jetson,
this includes `/proc/meminfo` and a short `tegrastats` sample with parsed
`lfb` when available. Use this to distinguish memory-state-sensitive startup
failures from parameter-incompatible failures.
If the sweep uses a pre-variant prepare command, it also captures a
before-prepare sidecar sample and stores a delta in the manifest so the effect
of cache-drop or `compact_memory` can be measured explicitly.

Each planned variant also records `server_runtime` when the launcher environment
contains a pinned `LLAMA_CPP_DOCKER_IMAGE` and Docker can inspect it. This
captures the image tag, image id, repo digests, base image, source revision,
llama.cpp ref from OCI labels, plus a runtime probe that resolves the
`llama-server` path and checks whether `llama-server --help` exposes `--mmproj`.
It intentionally excludes container environment variables.
For image-capable variants, `--mmproj` support is the mechanical proxy for this
repo's llama.cpp multimodal load path. If the runtime probe reports that the
selected image lacks it, the sweep skips the row early with
`runtime_missing_mmproj_support` instead of attempting server startup.

Text-only model variants should set `EDGE_VLM_CASES` to
`configs/benchmark/text_prompt_cases.jsonl` in their variant environment. Do
not mix those rows into VLM promotion comparisons.

For promotion or final comparison sweeps, lock Jetson clocks before the run:

```bash
JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
scripts/jetson/run_remote_optimization_sweep.sh \
  --run-prefix minicpm-promo-001 \
  --variant minicpm-q4-baseline-b128-u32-kvq8 \
  --trial-count 5 \
  --max-tokens 64 \
  --temperature 0 \
  --min-lfb-blocks 150
```

The max-clocks repeat in `docs/benchmarks/jetson_isolated_repeats_20260531.md`
showed this is not just bookkeeping: MiniCPM and especially Gemma improved
substantially after clocks were locked. Treat runs without confirmed
`jetson_clocks` state as exploratory unless the comparison is explicitly about
dynamic-clock behavior. The remote wrapper captures `sudo jetson_clocks --show`
under ignored `outputs/jetson_inspect/` when
`JETSON_REMOTE_PREPARE_MAX_CLOCKS=1` is set.

The sweep manifest also records server startup timing for variants that reach
Docker startup:

- `server_started_at`
- `server_ready_at`
- `server_wait_seconds`
- `server_startup_seconds`

Skipped variants keep these fields as `null`. A variant that starts Docker but
does not become ready records `server_wait_seconds` and keeps
`server_startup_seconds=null`, so load-path experiments can separate startup
behavior from steady-state benchmark throughput.

For variants that complete the formal benchmark, the sweep also writes derived
profile artifacts under `profiles/`:

- `<run-id>.profile.jsonl`: one parsed `tegrastats` sample per line. When
  `tegrastats` lines carry the UTC prefix emitted by
  `run_formal_benchmark.sh`, each record also includes `captured_at` and
  `elapsed_s` for phase-window alignment.
- `<run-id>.summary.json`: aggregate RAM/`lfb`, GR3D, EMC, CPU, temperature,
  power, conservative bottleneck labels, available phase timings,
  `input_timing_summary`, profile file pointers, and timestamp bounds
  (`first_captured_at`, `last_captured_at`, `captured_duration_s`) when
  available.

The first profile-summary phase timings are `artifact_check_or_download` when
the launcher emits lifecycle JSONL, `server_startup`, `formal_text`,
`formal_image`, `fake_stream`, and `shutdown`. Warmup is classified from the
variant command: `--no-warmup` records `disabled_by_variant`, while warmup-on
variants record `included_in_server_startup` until server logs or runtime hooks
can separate the internal llama.cpp warmup duration. Launchers that delegate
artifact download to `llama-server` inside the runtime container emit a
not-separated lifecycle record.

Profile summaries also aggregate benchmark and fake-stream `input_timing`
records. `input_payload` is emitted only when payload preparation is both
non-trivial and a material share of estimated end-to-end latency.
`runtime_overhead` is emitted only when request wait dominates while GR3D, EMC,
and CPU utilization are not saturated; it is a triage label for runtime/API
investigation, not proof that decode kernels are slow.

After a promotion or comparison sweep, generate the mechanical comparison table
before writing tracked benchmark notes:

```bash
PYTHONPATH=src python -m edge_vlm.optimization compare \
  --manifest outputs/optimization_sweeps/minicpm-promo-001/minicpm-promo-001.manifest.json \
  --baseline-variant minicpm-q4-baseline-b128-u32-kvq8 \
  --promotion-precheck-stage promotion-reference \
  --output outputs/optimization_sweeps/minicpm-promo-001/comparison.md
```

The comparison report joins the sweep manifest with each benchmark JSONL,
fake-stream JSONL, benchmark manifest, and `tegrastats` log. It reports
runtime image/id/ref, preflight `lfb`, trial count, startup seconds, sanity
guard, success counts, formal throughput/latency, fake-stream latency, max
temperature, average `VDD_IN` power, average GR3D utilization, average EMC
utilization, minimum profiled `lfb`, conservative bottleneck labels, any
recorded prepare-phase preflight deltas, and deltas versus the selected
baseline variant. When variant metadata provides a shared `comparison_group`,
the delta columns use that group instead of raw model id so Q4/Q8 fallback
lanes can share one baseline. When the sweep manifest carries normalized
`selection_contexts`, the comparison table also adds a `Selection` column so an
auto-selected fallback row stays tied to its selector decision. The sweep plan
also carries `variant_min_lfb_blocks` so a fallback-selected row can keep its
relaxed per-variant gate during execution instead of being re-blocked by the
suite-wide `--min-lfb-blocks`. The comparison table exposes that effective gate
in a `Required lfb` column so reviewers can see immediately whether a row ran
under a relaxed fallback threshold. Treat this as
the source table for tracked benchmark docs; do not hand-copy raw metrics from
multiple JSON files when the comparison command can derive them.
When compare also receives `--promotion-precheck-stage formal-repeat` or
`--promotion-precheck-stage promotion-reference`, it adds a `Promotion
precheck` column for the mechanical gate only: locked clocks, cache drop,
strict required-LFB floor, `max_tokens >= 64`, `temperature = 0`, full
benchmark success, fake-stream success, and the stage-specific trial floor.
Add `--promotion-require-quality-review` when the gate should also require a
passing structured `Quality review` sidecar. Use `formal-repeat` for 5-trial
lightweight ranking passes and `promotion-reference` for 10-trial
baseline/reference refreshes. The raw excerpt review remains manual.

When a sweep has route-sensitive outputs, run `edge_vlm.sweep_quality_review`
with `configs/benchmark/quality_review_policy.json` against the manifest before
or as part of compare. That helper writes per-run `quality_review_json` and
`quality_review_markdown` sidecars, updates the sweep manifest paths, and lets
the comparison table add a `Quality review` column without reopening each
benchmark JSONL separately. When compare also receives
`--promotion-require-quality-review`, that same sidecar becomes part of the
`Promotion precheck` gate instead of remaining report-only evidence.

For the recurring current-defaults baseline refresh, prefer the wrapper:

```bash
JETSON_CURRENT_DEFAULTS_RUN_PREFIX=current-defaults-clocks10-YYYYMMDDa \
scripts/jetson/run_remote_current_defaults_suite.sh
```

Set `JETSON_CURRENT_DEFAULTS_FAIL_ON_PROMOTION_PRECHECK=1` when that wrapper
should return non-zero after compare marks any row as failing the promotion
gate. The default remains diagnostic-friendly and leaves this off.

It fixes the two current default variants, enables remote max clocks and
per-variant cache dropping, runs 10 formal trials plus the three-frame
fake-stream sidecar, and generates `comparison.md` from the sweep manifest.
Use this before container, llama.cpp, or model-family A/B runs so the reference
baseline comes from the same automation path as candidates.

Use `--min-lfb-blocks <N>` for promotion or repeatability sweeps. When set, the
sweep skips a variant before server startup if parsed `lfb` free blocks are
below the threshold and records `preflight_passed=false` plus a
`preflight_reason` in the sweep manifest. The default is unset, so exploratory
runs still execute and gather evidence.

For promotion comparisons where memory fragmentation can bias later variants,
enable the remote wrapper's page-cache preparation:

```bash
JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
scripts/jetson/run_remote_optimization_sweep.sh \
  --run-prefix minicpm-promo-001 \
  --variant minicpm-q4-baseline-b128-u32-kvq8 \
  --trial-count 5 \
  --max-tokens 64 \
  --temperature 0 \
  --min-lfb-blocks 150
```

The wrapper uses the sudo password from stdin to feed a per-run 0600 FIFO, then
appends a pre-variant command shaped like
`sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' < /tmp/...`
without putting the password in command-line arguments, the dry-run plan, or
the sweep manifest. The command itself and the FIFO path are recorded in the
dry-run plan and sweep manifest. If it returns a non-zero exit code, that
variant is skipped before preflight or Docker startup and the manifest records
`pre_variant_command_passed=false` plus
`preflight_reason=pre_variant_command_failed returncode <N>`.

## Promotion Rule

A candidate can become the new baseline only when:

1. the formal benchmark completes successfully
2. the fake-stream check completes successfully; use the default multi-frame
   `data/sample_stream` set for streaming-sensitive candidates
3. the optimization report marks the candidate guard as `yes`
4. when compare is run with `--promotion-precheck-stage promotion-reference`,
   the optimization report marks the candidate `Promotion precheck` as `yes`
5. raw excerpt review has been completed by a human reviewer
6. its throughput or latency improves over the prior baseline
7. the fake-stream latency is not worse enough to invalidate the use case
8. the exact server parameters and Jetson memory notes are documented in a
   tracked benchmark result file
9. promotion comparisons were run under confirmed `sudo jetson_clocks`, unless
   the candidate is explicitly scoped to dynamic-clock operation

If a faster run fails the sanity guard, keep it as a failed optimization
candidate. Do not promote it.
