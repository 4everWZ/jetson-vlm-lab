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
| SmolVLM2 256M Q8 | Jetson smoke passed and is a latency floor; visible output is weaker and repetitive on simple frames. | 5-trial formal repeat, raw excerpt review, classify as latency-floor or route-only. |
| Qwen3-VL 2B Thinking Q4 | Jetson cached smoke passed and is much faster than Gemma. | 5-trial formal repeat, raw excerpt review, compare against MiniCPM and Gemma references. |
| Tencent Youtu-VL-4B official Q8/BF16-mmproj | Downloaded but failed before server ready with CUDA OOM allocating the mmproj buffer. | Defer until lower-bit official artifact or different runtime path exists. |
| Youtu-VL-4B third-party Q4 | Jetson cached smoke passed with CPU mmproj, but it is slow and not official Tencent support. | Separate CPU-mmproj and GPU-mmproj artifact A/B before any ranking. |

Tencent Hub note from the 2026-05-31 refresh: the current official small GGUF
Tencent rows such as `tencent/Hy-MT2-1.8B-GGUF` are text/translation models, not
VLM candidates. `tencent/HY-Embodied-0.5-X` is a VLM-like Transformers/custom
code release, not a low-friction GGUF path. It stays in the deferred runtime
lane unless a GGUF or bounded Transformers runtime is selected.

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
- `jetson_clocks --show`, `nvpmodel -q`, `uname -a`, Docker version, runtime
  image tag/id/digest, and llama.cpp ref.
- Host memory snapshot before startup and before each variant.
- Startup timing separated from benchmark request timing.
- First-run download time separated from cached startup time for Hub GGUF
  candidates.

Profiling output should be machine-readable under ignored output paths, for
example:

- `outputs/profiles/<run_prefix>/profile.jsonl` for time-aligned samples.
- `outputs/profiles/<run_prefix>/summary.json` for derived bottleneck labels.
- Existing sweep manifests should link to those profile files rather than copy
  sensitive environment state.

Acceptance:

- A profile can support one of these bottleneck labels with evidence:
  `gpu_compute`, `emc_memory_bandwidth`, `cpu_prepost`, `power_or_thermal`,
  `startup_or_download`, `input_payload`, `runtime_overhead`, or
  `not_identified`.
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
- Secondary lane: slightly larger models only when they fill a specific
  comparison role, such as official Tencent/Youtu evidence or quality anchor.
- Text-only small models are allowed only for a separate text/router study; they
  must not be mixed into VLM rankings.

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
- Fake-stream scheduling delay and backpressure. Implemented in
  `stream_timing.schedule_delay_s`, `stream_timing.backpressure_s`, and
  `stream_timing.pre_frame_sleep_s` using fixed-cadence frame scheduling.

Routing policy candidates:

- Use a fast low-end VLM for simple frame descriptions or low-risk triage.
- Fall back to MiniCPM or another reference model when guard terms fail, output
  is too short/repetitive, or the route is quality-sensitive.
- Add frame skipping or interval adaptation when fake-stream latency exceeds
  the target frame interval.
- Keep text-only small models out of image routes unless a text-only stage is
  explicitly added after image understanding.

Acceptance:

- End-to-end latency claims report both decode/request latency and input
  pipeline overhead.
- A route improves average or p95 fake-stream latency without introducing guard
  failures in the routed cases.
- Routing changes are measured against the current three-frame fake-stream
  fixture before any real-camera work.

### 6. Memory, LFB, EMC, And Fixedness

Purpose: make repeated results comparable instead of memory-state accidents.

Formal comparison runs must:

- Confirm max clocks through the remote wrapper's root `jetson_clocks --show`
  capture when available.
- Drop page cache before each comparison variant.
- Record preflight `lfb`; skip and label variants below threshold instead of
  mixing fragmented-memory failures into parameter rankings.
- Capture `EMC_FREQ`, `GR3D_FREQ`, CPU frequencies, RAM/SWAP, power, and
  temperatures across the request window.
- Record whether a failure happened before server ready, during first image
  request, or during fake-stream.

Acceptance:

- A startup OOM row states whether `lfb` was below threshold, whether the mmproj
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

1. Add the profiling harness and richer structured tegrastats parser.
2. Run 5-trial formal repeats for SmolVLM2 256M, Qwen3-VL 2B Thinking, and
   Youtu Q4 third-party CPU-mmproj under the fixed comparison policy.
3. Add artifact A/B rows only for candidates that survive repeat, starting with
   mmproj placement and available quantization differences.
4. Use profiling evidence to choose exactly one runtime lane for a smoke.
5. Add input-pipeline timing and route-policy experiments after baseline
   profiling can separate input overhead from server latency.

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
- Tencent HY-Embodied-0.5-X: https://hf.co/tencent/HY-Embodied-0.5-X
- Tencent Hy-MT2 1.8B GGUF: https://hf.co/tencent/Hy-MT2-1.8B-GGUF
