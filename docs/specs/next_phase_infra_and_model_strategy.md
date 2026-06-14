# Next Phase Infra And Model Strategy

Status: active strategy spec, 2026-05-31.

This spec narrows the next phase after the MiniCPM-V 4.6 Q4 and Gemma 4
E2B-it Q4 llama.cpp sweeps. The current evidence says broad server-flag tuning
has low remaining upside. The next work should prove bottlenecks, expand
lighter model choices, compare artifacts cleanly, and only then pay the cost of
runtime or lower-level infra changes.

## Decisions

- Stop broad llama.cpp parameter sweeps for the current MiniCPM and Gemma
  defaults. New llama.cpp flags can still be tested when container help or a
  dry-run confirms support and the flag maps to a specific bottleneck.
- Keep MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 as reference baselines, not as
  the main source of further speedup work. Gemma stays in the larger reference
  lane for this phase, not the sub-2B expansion lane.
- Make locked clocks, fresh memory state, and three-frame fake-stream evidence
  mandatory for promotion or comparison claims.
- Treat model artifact, quantization, mmproj placement, runtime, and input
  pipeline as separate comparison axes. A faster row is not a model win if it
  also changed runtime, artifact provenance, prompt handling, or preprocessing.
- Defer TensorRT, TensorRT-LLM, NanoLLM/JPS, Ollama, vLLM, and custom kernels
  until profiling identifies a bottleneck that a different runtime can plausibly
  address.

## Evidence Snapshot

The current default references are the max-clocks current-defaults suites:

| Model | Current default | Reference evidence |
|---|---|---|
| MiniCPM-V 4.6 Q4 | `batch=128`, `ubatch=32`, `N_GPU_LAYERS=32`, q8_0 KV cache, `--no-warmup` | 10-trial max-clocks repeats keep this ahead of `b512/u128`; no current flag candidate is ahead. |
| Gemma 4 E2B-it Q4 | `batch=512`, `ubatch=512`, `N_GPU_LAYERS=12`, q8_0 KV cache, `--no-warmup`, `-fit off` | 10-trial max-clocks DirectIO confirmation regressed formal and fake-stream latency; no current flag candidate is ahead. |

The new canonical llama.cpp image, `d749821db3bd`, is valid for continued work
and newer multimodal/model support, but it is not a performance promotion over
the previous canonical image.

Lightweight model evidence is promising but not promotable yet:

| Candidate | Current status | Next gate |
|---|---|---|
| SmolVLM2 256M Q8 | 5-trial fixed-policy repeat `lightweight-repeat5-20260531T134554Z` passed guard at 199.847 text tok/s, 164.551 image tok/s, 0.462 s fake-stream latency, and minimum profiled `lfb` 220. `edge_vlm.quality_review` with `configs/benchmark/quality_review_policy.json` passed 20/30 excerpt records and failed `text_cn_short` plus `text_en_reasoning_short`; raw excerpts still show weak semantics, including a WSL prompt answered as if WSL meant a generic web-services layer. | Role `latency_floor`; use for cheap routing/harness stress only, not text reasoning routes, unless prompt/runtime changes produce fresh quality evidence. |
| Qwen3-VL 2B Thinking Q4 | 5-trial fixed-policy repeat `lightweight-repeat5-20260531T134554Z` passed guard at 34.761 text tok/s, 34.290 image tok/s, 1.942 s fake-stream latency, average GR3D 96.457%, and minimum profiled `lfb` 154. `edge_vlm.quality_review` passed 20/30 excerpt records and failed `text_cn_short` plus `text_code_short`, so it is slower than MiniCPM but still far ahead of Gemma for the surviving route. | Qwen remains image/fake-stream balanced_candidate; do human full-output review and route-specific tests against MiniCPM before any broader text/code route use. |
| Qwen3-VL 2B Instruct Q4 | One locked-clocks diagnostic smoke on June 9, 2026, `qwen3-instruct-q4-smoke-lfb32-20260609T075208Z`, passed guard with 6/6 formal and 1/1 fake-stream records at relaxed `--min-lfb-blocks 32`, reaching 34.349 text tok/s, 26.957 image tok/s, and 1.827 s fake-stream latency. The strict-gate reruns `qwen3-instruct-q4-smoke-compact-20260609T074600Z` and `qwen3-instruct-q4-buddyinfo150-20260609T081121Z` still skipped at preflight with `lfb 46x4MB` and `117x4MB` after cache-drop plus `compact_memory`, so this is diagnostic startup evidence rather than ranking evidence. A shared relaxed-gate compare `qwen3-instruct-q4q8-lfb100-20260609T091700Z` later passed 18/18 formal and 1/1 fake-stream with 34.865 text tok/s, 31.958 image tok/s, 1.827 s fake-stream latency, and post-prepare preflight `lfb 121x4MB`, but its prepare delta was almost flat (`lfb -1`, `MemAvailable +1.816 MB`). | Keep it out of ranking tables until a locked-clocks smoke passes guard at `--min-lfb-blocks 150`, then run quality review and compare it against the Thinking row. |
| Qwen3-VL 2B Instruct Q8 fallback | Added as the Q8 fallback lane from `Qwen/Qwen3-VL-2B-Instruct-GGUF`, using `Qwen3VL-2B-Instruct-Q8_0.gguf` with `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`. Strict-gate smoke `qwen3-instruct-q8-smoke-20260609T082222Z` still skipped at preflight with `lfb 106x4MB`, but relaxed-gate smoke `qwen3-instruct-q8-smoke-lfb100-20260609T082321Z` passed guard with 6/6 formal and 1/1 fake-stream records, reaching 31.076 text tok/s, 25.997 image tok/s, 2.142 s fake-stream latency, and a 323.05 s first-download artifact phase. The shared relaxed-gate compare `qwen3-instruct-q4q8-lfb100-20260609T091700Z` also passed 18/18 formal and 1/1 fake-stream at 31.346 text tok/s, 29.393 image tok/s, and 2.144 s fake-stream latency with post-prepare preflight `lfb 125x4MB`; its prepare delta was much larger than the Q4 row (`lfb +88`, `MemAvailable +963.598 MB`). | Treat this as scoped fallback evidence only. Do not replace the default Q4 lightweight row or mix it into ranking tables unless the Q8 row can also satisfy the conservative 150-LFB gate, or the gate policy is explicitly revised. The mechanical comparison report now supports a shared `comparison_group`, so Q4-vs-Q8 percent deltas can be computed when both lanes declare the same fallback group. |

The repo now also has a selector entrypoint,
`scripts/jetson/select_qwen3_instruct_variant.sh`, which emits a structured
Q4-first / Q8 fallback decision based on the current runtime probe, artifact
presence, and preflight gate. Its default policy keeps the same conservative
gate for both lanes; a separate fallback gate is opt-in for scoped triage only.
The remote lightweight suite now consumes that selector as part of its default
execution path, keeping the suite-wide 150-LFB primary gate while applying a
100-LFB fallback gate only to the auto-selected Qwen3 Instruct Q8 lane. The
remote sweep wrapper now also forwards that selector JSON into the sweep
manifest, so comparison reports can label the selected row in a `Selection`
column instead of treating it as an unlabeled static variant id. It also
forwards the fallback lane's relaxed gate as `--variant-min-lfb-blocks`, and
the sweep plan persists that under `variant_min_lfb_blocks` so the selected Q8
row is not re-blocked by the stricter suite-wide threshold during execution.
The same wrapper now persists a structured `prepare_context` block in the sweep
plan. That block records whether `max_clocks` and `drop_caches` preparation was
mechanically enabled, instead of leaving later review to reconstruct intent
from shell wrappers or ad hoc filenames.
The comparison report now also shows that effective threshold in a `Required
lfb` column, which makes relaxed fallback evidence mechanically distinguishable
from strict-gate rows. When compare also receives
`--ranking-min-lfb-blocks 150`, it adds a `Ranking precheck` column so
promotion/ranking flows can keep fallback rows visible while still marking them
as non-ranking evidence when their effective gate is looser than the requested
strict floor. The sweep plan now also persists the suite-wide strict gate as
`plan.min_lfb_blocks`, and compare backfills missing
`preflight_required_lfb_blocks` from `variant_min_lfb_blocks` and then
`plan.min_lfb_blocks` so older result rows are not mislabeled as lacking
required-LFB evidence when the manifest still carries the gate contract. When
`prepare_context` is present, compare also emits a `Prepare ctx` summary column
with labels such as `max_clocks` and `drop_caches`, so ranking evidence can
distinguish strict prepared rows from unprepared or partially prepared runs
without hand-opening the manifest JSON. When profile phase timings include
`artifact_check_or_download`, compare also emits `Artifact phase` and
`Artifact s` so first-download rows can be separated from cached-start rows at
the report layer instead of by hand-opening lifecycle JSONL. When startup time
itself is under review, compare also accepts
`--startup-require-cached-artifacts` and emits a `Startup precheck` column that
only passes rows whose profile phase timings explicitly recorded
`artifact_check_or_download = cached`; first-download rows and older manifests
without that phase remain visible but fail the startup precheck. Compare can also add
`--ranking-require-startup-precheck`, so `Ranking precheck` can reject the same
first-download rows instead of letting them remain ranking-eligible just because
their required LFB gate was strict enough. Compare can now also add a
`Promotion precheck` column with `--promotion-precheck-stage formal-repeat` or
`--promotion-precheck-stage promotion-reference`. That precheck is
intentionally mechanical only: it checks locked clocks, cache drop, strict required-LFB floor, `max_tokens >= 64`, `temperature = 0`, full benchmark success, fake-stream success for image-capable rows, and the stage-specific trial floor. Text-only rows whose configs declare `capabilities.image=false` do not need fake-stream records for this gate. The raw excerpt review remains manual, and route-sensitive promotion still requires `edge_vlm.quality_review` plus human review before any role change.
When compare also receives `--promotion-require-startup-precheck`, the same
`Promotion precheck` gate also requires a passing `Startup precheck`, which in
practice means the row already proved cached-startup evidence under
`--startup-require-cached-artifacts`.
When compare also receives `--promotion-require-quality-review`, the same
`Promotion precheck` gate requires the structured `Quality review` sidecar to
pass as well. This still does not replace human raw excerpt review; it only
pulls the policy-level quality evidence into the same promotion row.
The suite wrappers now also run `edge_vlm.sweep_quality_review` with
`configs/benchmark/quality_review_policy.json` against each finished sweep
manifest. That writes `quality_review_json` and `quality_review_markdown`
sidecars per variant, and compare surfaces them in a `Quality review` column.
The current-defaults and lightweight wrappers also pass
`--promotion-require-quality-review`, so those promotion-oriented reports now
consume the same sidecars as part of their `Promotion precheck`.
When downstream ranking/export steps need the same evidence without scraping
Markdown, compare also accepts `--eligibility-output
outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json`. That
machine-readable sidecar records the per-row `Startup precheck`,
`Ranking precheck`, and `Promotion precheck` states plus top-level eligible row
lists, and the remote current-defaults, lightweight, and Tencent text suite
wrappers now emit `comparison.eligibility.json` by default next to
`comparison.md`.
The next downstream step now also exists in code instead of prose only:
`python -m edge_vlm.optimization select-eligible --input
.../comparison.eligibility.json --gate ranking --output
.../ranking.selection.json` (or `--gate promotion` for
`promotion.selection.json`). The suite wrappers run both exports by default, so
later ranking/promotion automation can consume filtered JSON artifacts rather
than duplicating Markdown parsing or gate logic.
The same export step now also accepts `--require-leq2b-candidate` plus
`--candidate-lane <vlm|text>`, so the lightweight suite can emit a scoped
`<=2B` VLM artifact (`ranking.leq2b-vlm.selection.json` and
`promotion.leq2b-vlm.selection.json`) and the Tencent text suite can emit a
scoped `<=2B` text artifact (`ranking.leq2b-text.selection.json` and
`promotion.leq2b-text.selection.json`) without mixing those rows with the
current-default baselines or the VLM lane.
`edge_vlm.optimization bundle-selections` now merges those scoped lane artifacts
into one `leq2b.candidate_bundle.json`; the remote helper
`scripts/jetson/build_remote_leq2b_candidate_bundle.sh` builds the same bundle
from `JETSON_LEQ2B_VLM_SELECTION_DIR` and `JETSON_LEQ2B_TEXT_SELECTION_DIR` on
the Jetson worktree.
| HunyuanOCR 1B Q8 | Jetson smoke `hunyuanocr-q8-smoke64-20260531c` loaded through `ggml-org/HunyuanOCR-GGUF` after the launcher-resume fix and completed formal/fake-stream records, but the guard failed because all output excerpts were repetitive exclamation-mark strings. This is not an official Tencent-owned GGUF artifact. | Do not run formal repeats for the current Q8 artifact/runtime path. Revisit only with bounded quality triage of artifact, prompt/template handling, or runtime lane. |
| Tencent Youtu-VL-4B official Q8/BF16-mmproj | Downloaded but failed before server ready with CUDA OOM allocating the mmproj buffer. | Defer until lower-bit official artifact or different runtime path exists. |
| Youtu-VL-4B third-party Q4 | 5-trial fixed-policy repeat `lightweight-repeat5-20260531T134554Z` passed guard and `edge_vlm.quality_review` passed 30/30 excerpt records, but it only reached 7.502 text tok/s, 7.249 image tok/s, 9.186 s fake latency, average GR3D 19.546%, and minimum profiled `lfb` 1 with `runtime_overhead`. It is also not official Tencent support. | Role `failed_artifact` for default ranking; revisit only for a scoped artifact/runtime A/B. |
| Tencent Hy-MT1.5/Hy-MT2/Youtu-LLM small GGUF text variants | Added to the text/router lane only: Hy-MT1.5 1.25bit/2bit/Q4_K_M/Q6_K/Q8_0, Hy-MT2 1.25Bit/2Bit/Q4_K_M/Q6_K/Q8_0, and Youtu-LLM 2B Q8_0 are all included in the dedicated Tencent text suite by default. Low-bit rows are runtime-compatibility canaries; the 1.25bit rows fail on pinned llama.cpp with `invalid ggml type 42`, and the 2bit rows fail with tensor offset mismatches. Earlier Hy-MT2 Q4 cached smoke `tencent-hy-mt2-q4-smoke64-cached-20260531b` passed the text guard at 19.759 tok/s and 3.070 s average text latency. Full-suite run `tencent-text-smoke64-20260531T120054Z` produced valid Hy-MT2 Q4/Q6 text rows: Q4 passed at 33.541 tok/s and 1.688 s average text latency, Q6 passed at 26.287 tok/s and 2.145 s average text latency. Cached Q6 run `tencent-hy-mt2-q6-smoke64-cached-20260531a` passed at 26.298 tok/s and 2.144 s average text latency. Q8 full-suite row is invalidated by the pre-fix launcher cleanup/stale-port issue; fixed-harness cached rerun `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` passed at 30.756 tok/s and 1.841 s average text latency. Five-trial repeat `tencent-text-repeat5-20260531a` passed Q4/Q6/Q8 at 20/20 records each: Q4 34.528 tok/s and 1.643 s latency, Q6 26.778 tok/s and 2.108 s, Q8 31.395 tok/s and 1.806 s. Prepared repeat `tencent-text-repeat5-prepctx-20260609T131412Z` then refreshed Hy-MT2 Q4/Q6/Q8 and Youtu-LLM 2B Q8 under `prepare_context = max_clocks, drop_caches` and `Required lfb = 150`: all four rows passed 20/20 formal records and `Ranking precheck`, Hy-MT2 Q4 failed `Promotion precheck` on `quality_review_failed 15/20 text_en_reasoning_short`, Hy-MT2 Q6/Q8 failed on `quality_review_failed 10/20 text_en_reasoning_short,text_code_short`, and Youtu-LLM 2B Q8 passed `Quality review = yes (20/20)` with `Promotion precheck = yes`. Cached rerun `youtu-llm-q8-cached-20260609T133339Z` then confirmed the same strict gate without download pollution: lifecycle recorded `artifact_check_or_download = cached (0.002 s)`, startup dropped to `5.015 s`, throughput held at `24.621` text tok/s, and the row remained promotion-pass. HY-MT1.5 Q4/Q6/Q8 rows still have no Jetson evidence. All valid rows used `terminate_group` shutdown with closed-port evidence. | Run `edge_vlm.quality_review` with `configs/benchmark/quality_review_policy.json`, then do human output review and route-specific translation/router checks before use; keep text rows separate from VLM image/fake-stream rows. Low-bit failures remain runtime-compatibility evidence unless a newer llama.cpp runtime can load them. |

Structured quality review on `lightweight-repeat5-20260531T134554Z` produced
30/30 policy passes for MiniCPM-V 4.6 Q4 and third-party Youtu Q4, and 20/30
passes for Gemma, SmolVLM2, and Qwen. SmolVLM2 failed text-resource and WSL
reasoning checks, which reinforces `latency_floor`. Qwen failed the Chinese
resource-term and code-function checks but kept image and fake-stream cases
passing. Qwen remains image/fake-stream balanced_candidate, not a general
text/code route, until a human output review or prompt/policy change proves a
broader role.

Tencent Hub note from the 2026-05-31 refresh: the current official small GGUF
Tencent rows such as `tencent/Hy-MT1.5-1.8B-1.25bit-GGUF`,
`tencent/Hy-MT1.5-1.8B-2bit-GGUF`, `tencent/HY-MT1.5-1.8B-GGUF`,
`tencent/Hy-MT2-1.8B-GGUF`,
`tencent/Hy-MT2-1.8B-1.25Bit-GGUF`,
`tencent/Hy-MT2-1.8B-2Bit-GGUF`, and `tencent/Youtu-LLM-2B-GGUF` are
text/translation or text-generation models, not VLM candidates. They are
configured for text/router benchmarks only. Low-bit rows stay in the dedicated
text-suite default as runtime-compatibility canaries, with failures recorded as
runtime support evidence rather than VLM ranking evidence. The Youtu-LLM F16
GGUF sibling is intentionally deferred from the default suite because Q8 covers
the model with lower memory pressure.
Latest Hy-MT1.5 1.8B Safetensors quant rows
`tencent/Hy-MT1.5-1.8B-1.25bit` and `tencent/Hy-MT1.5-1.8B-2bit`
are deferred because this bench lane is GGUF/llama.cpp only until a bounded
Transformers or conversion path is selected.
Newer Tencent small VLM-like releases such as `tencent/Penguin-VL-2B`,
`tencent/HY-Embodied-0.5`, `tencent/HY-Embodied-0.5-X`, and
`tencent/Youtu-Parsing` are Transformers/Safetensors/custom-code paths, not
low-friction GGUF. `tencent/HunyuanOCR` is a 1B image-text model; the directly
actionable low-friction route is currently `ggml-org/HunyuanOCR-GGUF`, which is
a Tencent-base GGUF artifact rather than an official Tencent-owned GGUF repo.
The custom-code Tencent rows stay in the deferred runtime lane unless a GGUF or
bounded Transformers runtime is selected.

## Non-Goals

- Do not optimize Gemma as if it were a sub-2B target.
- Do not promote a model, runtime, or artifact from a one-trial smoke.
- Do not claim official Tencent Youtu support from the third-party Q4 artifact.
- Do not run local conversion or quantization on the memory-constrained host as
  a default path.
- Do not put secrets, passwords, tokens, private keys, `.env` content, full
  container environments, or SSH credentials into tracked docs, manifests, or
  Docker image layers.
- Do not change the Python package into a weight-loading runtime. The repo
  remains a thin OpenAI-compatible client and harness; `llama-server` or an
  explicitly selected backend owns inference and preprocessing internals.

## Workstreams

### 1. Profiling Harness

Purpose: prove where latency is spent before changing runtime, kernels, or
pipeline design.

Required phase timing:

- `artifact_check_or_download`
- `server_startup`
- `warmup`, when enabled
- `formal_text`
- `formal_image`
- `fake_stream`
- `shutdown`

Required per-run profile data:

- `tegrastats` parsed into structured samples for `RAM`, `SWAP`, `lfb`,
  `CPU`, `GR3D_FREQ`, `EMC_FREQ`, temperatures, `VDD_IN`, and other available
  power rails.
- Formal-wrapper `tegrastats` logs carry a UTC timestamp prefix; derived
  profile JSONL records expose `captured_at` and `elapsed_s`, and summaries
  expose capture bounds so utilization samples can be aligned with lifecycle,
  benchmark, and fake-stream windows.
- `jetson_clocks --show`, `nvpmodel -q`, `uname -a`, Docker version, runtime
  image tag/id/digest, and llama.cpp ref.
- Host memory snapshot before startup and before each variant, including
  `/proc/meminfo`, `/proc/buddyinfo`, and `tegrastats` `lfb` so fragmentation
  can be separated from simple free-memory pressure.
- Startup timing separated from benchmark request timing.
- First-run download time separated from cached startup time for host-side Hub
  GGUF launchers through `EDGE_VLM_LAUNCH_PHASE_LOG`. Runtime-internal `-hf`
  downloads are labeled not separated until that path is replaced or parsed from
  runtime logs.
- Warmup policy recorded from variant args. `--no-warmup` rows are marked
  disabled; warmup-on rows are marked as included in server startup until
  runtime logs or hooks can split the internal llama.cpp warmup duration.
- Benchmark and fake-stream `input_timing` records are summarized into profile
  summaries so payload preparation and request-wait evidence can be compared
  with `tegrastats` utilization in the same run artifact.

Profiling output should be machine-readable under ignored output paths, for
example:

- `outputs/profiles/<run_prefix>/profile.jsonl` for parsed and time-aligned
  samples.
- `outputs/profiles/<run_prefix>/summary.json` for capture bounds and derived
  bottleneck labels.
- Existing sweep manifests should link to those profile files rather than copy
  sensitive environment state.

Acceptance:

- A profile can support one of these bottleneck labels with evidence:
  `gpu_compute`, `emc_memory_bandwidth`, `cpu_prepost`, `power_or_thermal`,
  `startup_or_download`, `input_payload`, `runtime_overhead`, or
  `not_identified`.
- `input_payload` requires both non-trivial average payload overhead and a
  material payload share of estimated end-to-end latency. `runtime_overhead`
  requires high request-wait share while GR3D, EMC, and CPU utilization are not
  saturated. Treat both as conservative triage labels, not as decode-speed
  proof.
- A runtime or kernel proposal must cite a profile row showing the bottleneck it
  is meant to address.
- Formal comparisons use `TEGR_STATS_INTERVAL_MS=200` for profiling runs unless
  the interval itself perturbs results; standard current-default refreshes may
  keep the existing lower-frequency logging.

### 2. 2B Model Expansion And Formal Repeat

Purpose: find faster usable VLMs before spending time on lower-level runtime
integration.

Candidate policy:

- Primary lane: VLMs at or below the 2B class with GGUF plus mmproj or an
  equivalent confirmed llama.cpp multimodal load path.
- Quantization policy: prefer pre-built Q4 GGUF artifacts for <=2B LLM/VLM
  candidates; fall back to Q8 when Q4 is unavailable or unusable. A fallback
  row is still not a ranking row unless it passes the same conservative Jetson
  gate used for the surrounding comparison. Do not use local conversion or
  quantization on the memory-constrained host as a default path.
- Secondary lane: slightly larger models only when they fill a specific
  comparison role, such as official Tencent/Youtu evidence or quality anchor.
- Text-only small models are allowed only for a separate text/router study; they
  must use `configs/benchmark/text_prompt_cases.jsonl` and must not be mixed
  into VLM rankings.

Repeat ladder:

| Stage | Trials | Frames | Use |
|---|---:|---:|---|
| Payload/dry-run | 0 | 0 | Validate launcher, config, local file paths, and command shape. |
| Smoke | 1 | 3 | Establish that the model loads, answers, passes the guard, and records manifests. |
| Formal repeat | 5 | 3 | Rank candidate families and catch obvious thermal/memory variance. |
| Promotion/reference | 10 | 3 | Replace a baseline, canonical artifact, or runtime image. |

Required run settings for ranking:

- `sudo jetson_clocks` confirmed.
- Cache dropped before each comparison variant.
- `--min-lfb-blocks 150` or stricter.
- `--max-tokens 64` or higher.
- `--temperature 0`.
- Guard pass, full success counts, fake-stream success, and raw excerpt review.

### 3. Artifact And Quantization Matrix

Purpose: compare model bytes, mmproj bytes, and placement without hiding
artifact differences inside a model label.

Each artifact row must record:

- Model repo/ref, filename, quantization, source provenance, file size, and
  checksum when practical.
- mmproj repo/ref, filename, quantization, source provenance, file size, and
  checksum when practical.
- CPU/GPU mmproj placement and GPU-layer policy.
- Runtime image tag/id/digest and llama.cpp ref.
- Whether the row is official, third-party, local conversion, or unknown.

Allowed A/B comparisons:

- Same model, same runtime, CPU mmproj versus GPU mmproj.
- Same official model, different quantization, same mmproj.
- Same model quantization, different mmproj quantization.
- Same artifacts, different runtime.

Acceptance:

- A controlled artifact comparison changes one axis at a time.
- A third-party artifact can be reported as useful evidence, but cannot become
  official-model evidence.
- A first-run row that includes downloads must be paired with a cached-startup
  row before startup conclusions are written.

### 4. Runtime Comparison Gate

Purpose: test different runtimes only after there is a concrete bottleneck and
an equivalent-enough artifact path.

Baseline runtime remains llama.cpp `llama-server` with local GGUF and mmproj
files. Other runtime lanes are gated as follows:

| Runtime lane | Entry criteria | Equivalence requirement |
|---|---|---|
| NVIDIA JPS / NanoLLM | Profiling shows input stream, routing, or deployment overhead that JPS can reduce; model support and storage budget are explicit. | Same prompt cases or documented API adapter, image path preserved, runtime/container version recorded. |
| TensorRT / TensorRT-LLM | Profiling shows GPU compute or memory-bound decode bottleneck and a bounded conversion path exists for a selected model. | Same model family and prompt/image semantics, engine build metadata, quantization and calibration recorded. |
| Ollama | A model bundle exists that preserves VLM behavior and exposes comparable API timing. | Same artifact provenance or explicitly non-equivalent row. |
| Transformers/PyTorch | Only for a model with no GGUF path and a bounded Jetson memory plan. | Separate runtime lane; not compared as a llama.cpp artifact win. |
| Custom kernels | Only after profiles and runtime A/B show a model-specific bottleneck still worth targeting. | Design doc required before implementation. |

Acceptance:

- Runtime rows record version/container, model format, preprocessing path, API
  adapter, startup time, formal throughput/latency, fake-stream latency, power,
  temperature, `lfb`, and guard result.
- A runtime is not "faster" unless the artifact and prompt/image path are
  equivalent, or the row is explicitly labeled non-equivalent.

### 5. Pipeline, Routing, And Input

Purpose: reduce end-to-end latency without pretending it is model decode speed.

Instrument the client/input path separately:

- Image file read. Implemented in `input_timing.image_read_s`.
- MIME detection. Implemented in `input_timing.mime_detect_s`.
- Base64 encoding and data URL construction. Implemented in
  `input_timing.base64_encode_s` and `input_timing.data_url_build_s`.
- JSON serialization. Implemented in `input_timing.json_serialize_s`.
- HTTP request/response elapsed time. Implemented in `input_timing.http_request_s`.
- Server-reported token usage and request latency when available.
- Profile summaries aggregate `input_timing` from benchmark and fake-stream
  JSONL into `input_timing_summary`, including payload overhead, estimated
  end-to-end latency, request wait, request body bytes, image bytes, and source
  record counts.
- Fake-stream scheduling delay and backpressure. Implemented in
  `stream_timing.schedule_delay_s`, `stream_timing.backpressure_s`, and
  `stream_timing.pre_frame_sleep_s` using fixed-cadence frame scheduling.
- Explicit late-frame skipping for stream-control experiments. Implemented as
  `edge_vlm.fake_stream --skip-late-frames`; skipped source frames are not sent
  to the model, and the next processed record reports
  `stream_timing.skipped_frames_before`.
- Adaptive fake-stream interval experiments. Implemented as
  `edge_vlm.fake_stream --adaptive-interval`; processed records report both
  the base `stream_timing.interval_s` and the per-frame
  `stream_timing.effective_interval_s`. Sweeps can pass the base interval,
  skip, and adaptive controls through with the corresponding
  `--fake-stream-*` options.

Routing policy candidates:

- Use a fast low-end VLM for simple frame descriptions or low-risk triage.
- Fall back to MiniCPM or another reference model when guard terms fail, output
  is too short/repetitive, or the route is quality-sensitive.
- Use explicit frame skipping when fake-stream latency exceeds the target frame
  interval.
- Use adaptive fake-stream intervals only as an explicitly labeled
  stream-control experiment; do not mix adaptive cadence rows into fixed-cadence
  ranking tables without labeling the cadence difference.
- Keep text-only small models out of image routes unless a text-only stage is
  explicitly added after image understanding.
- Run `edge_vlm.quality_review` with
  `configs/benchmark/quality_review_policy.json` on candidate benchmark JSONL
  before changing a route role. This policy checks route-specific terms and
  known bad patterns; it supports human review but does not replace it.

Acceptance:

- End-to-end latency claims report both decode/request latency and input
  pipeline overhead.
- A route improves average or p95 fake-stream latency without introducing guard
  failures in the routed cases.
- Candidate routes pass the structured quality review policy for the benchmark
  cases they claim to serve, or the failure is explicitly documented as a
  `route_only`/`failed_artifact` limitation.
- Routing changes are measured against the current three-frame fake-stream
  fixture before any real-camera work.

### 6. Memory, LFB, EMC, And Fixedness

Purpose: make repeated results comparable instead of memory-state accidents.

Formal comparison runs must:

- Confirm max clocks through the remote wrapper's root `jetson_clocks --show`
  capture when available.
- Drop page cache before each comparison variant.
- Record preflight `lfb` plus `/proc/buddyinfo` fragmentation state; skip and
  label variants below threshold instead of mixing fragmented-memory failures
  into parameter rankings.
- When prepare commands such as cache-drop plus `compact_memory` are enabled,
  record both before-prepare and after-prepare preflight samples plus a delta so
  fragmentation changes are measured directly instead of guessed from a single
  sample.
- Capture `EMC_FREQ`, `GR3D_FREQ`, CPU frequencies, RAM/SWAP, power, and
  temperatures across the request window.
- Record whether a failure happened before server ready, during first image
  request, or during fake-stream.

Acceptance:

- A startup OOM row states whether `lfb` was below threshold, whether
  `/proc/buddyinfo` still had higher-order free blocks, whether the mmproj
  allocation was on GPU, and whether the row is memory-state-sensitive or
  parameter/artifact-incompatible.
- A comparison doc cannot rank rows that were run under different clock policy
  unless the comparison is explicitly about dynamic clocks.

### 7. Artifact And llama.cpp Build Infra

Purpose: keep runtime images reproducible and minimal.

Build rules:

- Keep the artifact-copy image path. Build llama.cpp in the builder container,
  copy only the install tree into the runtime image, and avoid unrelated
  packages, caches, model weights, `.env` files, SSH files, outputs, or local
  benchmark artifacts.
- Use the self-built official llama.cpp image as the default multimodal Jetson
  runtime. dusty-nv `llama_cpp` image selection is opt-in and cannot support
  VLM claims until it accepts the actual `--mmproj` flag: direct Jetson smoke on
  June 9, 2026 for `dustynv/llama_cpp:b5283-r36.4-cu128-24.04` exited with
  `error: invalid argument: --mmproj` even though `llama-server --help`
  contained a generic `mmproj` marker.
- Capture a runtime probe in Jetson sweep plans so image-capable rows record
  whether `llama-server --help` exposes the exact `--mmproj` flag; skip rows with
  `runtime_missing_mmproj_support` rather than spending download/startup time on
  an image that cannot satisfy this repo's multimodal path.
- Record OCI labels for source revision, llama.cpp ref, build date, and base
  image.
- Record the copied file list and `llama-server --version` or help excerpt in
  ignored build outputs.
- If the pinned image lacks HTTPS support, keep the host-side Hub download path
  for GGUF candidates rather than expanding the runtime image with unrelated
  Python or CLI tooling.

Replacement gate:

- New image first passes a one-trial current-default smoke on MiniCPM and
  Gemma.
- Replacement over the canonical tag requires a 10-trial current-default suite
  with locked clocks, cache drop, `--min-lfb-blocks 150`, and three fake-stream
  frames.
- After replacement, remove old local dangling images only after the pushed tag
  and digest are verified.

## Ranking Policy

Use these thresholds until the profiling harness produces better variance
estimates:

- Same-model infra/default promotion requires at least a 3% improvement in the
  primary metric and no more than 1% regression in fake-stream latency. Startup
  cannot regress by more than 5% unless the row is explicitly marked
  long-lived-only.
- New model ranking can use larger tradeoffs, but the row must be labeled by
  role: `latency_floor`, `balanced_candidate`, `quality_reference`,
  `route_only`, or `failed_artifact`.
- Power or temperature regressions above 10% require a workload-specific reason
  before promotion.
- Any guard failure blocks ranking regardless of speed.

## Execution Order

1. Keep the profiling harness as the next-gate source of truth: structured
   `tegrastats` samples now include UTC capture timestamps and elapsed seconds
   when collected through the formal wrapper.
2. Treat `lightweight-repeat5-20260531T134554Z` as the completed fixed-policy
   5-trial repeat for SmolVLM2 256M, Qwen3-VL 2B Thinking, and Youtu Q4
   third-party CPU-mmproj. Qwen3-VL 2B Instruct Q4 is a newly configured
   smoke candidate, not part of that historical repeat. HunyuanOCR Q8 and
   official Youtu-VL Q8 stay excluded from the default repeat queue until their
   current guard/OOM causes change.
3. Run the dedicated Tencent Hy-MT1.5/Hy-MT2/Youtu-LLM text/router suite
   separately if a text-only route is useful; keep all eleven default rows out
   of VLM ranking tables,
   use `tencent-text-smoke64-20260531T120054Z` only as one-trial Q4/Q6
   full-suite evidence, use `tencent-hy-mt2-q6-smoke64-cached-20260531a` and
   `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` only as one-trial
   cached evidence, use `tencent-text-repeat5-20260531a` as Q4/Q6/Q8 5-trial
   text/router repeat evidence, and treat low-bit failures as
   runtime-compatibility evidence.
4. Run the route-specific review CLI, `edge_vlm.quality_review`, with
   `configs/benchmark/quality_review_policy.json` on repeat JSONL before any
   model is promoted beyond its current role label.
5. Add artifact A/B rows only for candidates that survive repeat and quality
   review. Start with Qwen role-quality checks and memory/runtime
   characterization; run Youtu Q4 GPU-mmproj/offload only if it answers a
   scoped artifact question.
6. Use profiling evidence to choose exactly one runtime lane for a smoke. The
   repeat already separates `gpu_compute` rows from Gemma/Youtu
   `runtime_overhead` rows with minimum profiled `lfb` of one block.
7. Add route-policy experiments after baseline profiling can separate input
   overhead from server latency; real camera/live input remains a later
   validation lane after folder-based fake-stream controls are understood.

## Source References

Repository docs:

- `docs/specs/jetson_optimization_loop.md`
- `docs/benchmarks/jetson_isolated_repeats_20260531.md`
- `docs/benchmarks/jetson_lightweight_models_20260531.md`
- `docs/benchmarks/jetson_remote_smoke_20260531.md`
- `docs/runtime_matrix.md`
- `docs/design/edge_vlm_architecture.md`

External references checked:

- NVIDIA tegrastats utility: https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html
- NVIDIA Jetson Platform Services VLM service: https://docs.nvidia.com/jetson/jps/inference-services/vlm.html
- NVIDIA TensorRT-LLM docs: https://docs.nvidia.com/tensorrt-llm/
- llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Tencent Youtu-VL-4B GGUF: https://hf.co/tencent/Youtu-VL-4B-Instruct-GGUF
- HunyuanOCR GGUF: https://hf.co/ggml-org/HunyuanOCR-GGUF
- Qwen3-VL 2B Instruct GGUF: https://hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF
- Tencent HunyuanOCR base: https://hf.co/tencent/HunyuanOCR
- Tencent Penguin-VL-2B: https://hf.co/tencent/Penguin-VL-2B
- Tencent HY-Embodied-0.5: https://hf.co/tencent/HY-Embodied-0.5
- Tencent HY-Embodied-0.5-X: https://hf.co/tencent/HY-Embodied-0.5-X
- Tencent Youtu-Parsing: https://hf.co/tencent/Youtu-Parsing
- Tencent Hy-MT1.5 1.8B 1.25bit GGUF: https://hf.co/tencent/Hy-MT1.5-1.8B-1.25bit-GGUF
- Tencent Hy-MT1.5 1.8B 2bit GGUF: https://hf.co/tencent/Hy-MT1.5-1.8B-2bit-GGUF
- Tencent HY-MT1.5 1.8B GGUF: https://hf.co/tencent/HY-MT1.5-1.8B-GGUF
- Tencent Hy-MT2 1.8B GGUF: https://hf.co/tencent/Hy-MT2-1.8B-GGUF
- Tencent Hy-MT2 1.8B 1.25Bit GGUF: https://hf.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF
- Tencent Hy-MT2 1.8B 2Bit GGUF: https://hf.co/tencent/Hy-MT2-1.8B-2Bit-GGUF
- Tencent Youtu-LLM 2B GGUF: https://hf.co/tencent/Youtu-LLM-2B-GGUF
