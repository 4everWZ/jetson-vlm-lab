# Edge VLM Handoff Status

Last updated: 2026-06-08
Current code baseline: branch `bench/formal-jetson-infra`; latest committed
baseline before this update is `ad42239` (`docs: refresh edge vlm handoff`).

## Objective

Stop broad MiniCPM/Gemma llama.cpp parameter tuning and hand off the current
spec-driven Jetson phase. The ongoing goal is to iterate on LLM/VLM candidates
at or below the 2B class, prefer pre-built Q4 GGUF artifacts, fall back to Q8
when Q4 is unavailable, and use deeper infra evidence before runtime,
quantization, pipeline, or lower-level optimization work.

Scope boundary: the repo remains a thin OpenAI-compatible client and benchmark
harness. Inference stays in llama.cpp or another explicitly selected backend.
Text-only models are not VLM ranking rows.

## State Summary

Current position:

- Quality review gate is implemented in `src/edge_vlm/quality_review.py` with
  `configs/benchmark/quality_review_policy.json` and docs/matrix coverage.
- `lightweight-repeat5-20260531T134554Z` is the fixed-policy lightweight repeat
  evidence: MiniCPM and third-party Youtu Q4 pass 30/30 quality-review excerpts;
  SmolVLM2 and Qwen pass 20/30 and are narrowed to route-specific roles.
- Formal `tegrastats` capture now prefixes lines with UTC timestamps in
  `scripts/jetson/run_formal_benchmark.sh`.
- `src/edge_vlm/jetson_profile.py` parses timestamped samples and emits
  `captured_at`, `elapsed_s`, `first_captured_at`, `last_captured_at`, and
  `captured_duration_s` for phase-window profiling.
- Tencent text/router lane now includes 11 default GGUF rows:
  Hy-MT1.5 1.25bit/2bit/Q4/Q6/Q8, Hy-MT2 1.25Bit/2Bit/Q4/Q6/Q8, and
  Youtu-LLM 2B Q8. Youtu now has both strict first-download evidence
  (`tencent-text-repeat5-prepctx-20260609T131412Z`) and cached rerun evidence
  (`youtu-llm-q8-cached-20260609T133339Z`, `artifact_check_or_download =
  cached`, `Startup s = 5.015`). These stay out of VLM ranking.
- Jetson launcher defaults now point to the self-built official llama.cpp image
  used by the observed multimodal smoke runs. dusty-nv `llama_cpp` is not the
  default VLM runtime path because it has not provided the required multimodal
  server path in this repo.
- Handoff point: current stage is closed. Do not continue broad flag tuning
  without a profile-backed bottleneck claim.

Requirement coverage:

- Profiling harness: implemented for structured samples, lifecycle/input timing
  summaries, conservative bottleneck labels, and timestamp alignment.
- 2B model expansion: implemented for existing VLM candidates and Tencent
  text/router configs; new HY-MT1.5 Q4/Q6/Q8 and Youtu-LLM Q8 still need Jetson
  evidence.
- Runtime/build infra: canonical self-built official llama.cpp image evidence
  exists; no new runtime lane has been promoted.
- Pipeline/routing: input timing and quality-review gates exist; full router
  policy and camera/live input are not implemented.

## Verification

Verified:

- `PYTHONPATH=src python3 -m unittest discover -s tests -v` -> 95 tests OK.
- `python3 -m compileall -q src` -> OK.
- `find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n` -> OK.
- `JETSON_DRY_RUN=1` launcher checks for Gemma and MiniCPM resolve the default
  Jetson image to `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`.
- JSONL parse check for benchmark configs and prompt cases -> OK.
- `python3 -m json.tool configs/benchmark/quality_review_policy.json` -> OK.
- `PYTHONPATH=src python3 -m edge_vlm.quality_review --help` -> OK.
- `git diff --check` -> OK.
- Secret scan for the provided SSH password patterns in tracked files -> no
  matches.

Not verified in this final stop:

- No new real Jetson benchmark was run for HY-MT1.5 Q4/Q6/Q8 or Youtu-LLM Q8.
- No new llama.cpp image was built in this stage.
- No runtime A/B beyond the existing documented evidence was executed.

## Blockers / Risks

- Gemma 4 E2B remains a larger reference model, not a promising speed target on
  Jetson; further flag tuning is unlikely to pay off without new profiling.
- SmolVLM2 is fast but semantically weak in current review; keep it as
  `latency_floor`.
- Qwen3-VL 2B is the current image/fake-stream balanced candidate, but text/code
  route use needs human review or prompt/policy changes.
- Third-party Youtu Q4 passes quality review but is slow, memory-stressed, and
  not official Tencent artifact evidence.
- New Tencent text/router rows are executable defaults only; they need locked
  clocks, cache-drop, min-lfb, and quality review before route use.

## Next Steps

1. Run one scoped Jetson text/router suite for the new HY-MT1.5 Q4/Q6/Q8 and
   Youtu-LLM Q8 rows, then apply `edge_vlm.quality_review` before route claims.
2. Use timestamped profile JSONL to align `tegrastats` samples with
   `server_startup`, formal text/image windows, fake-stream windows, and
   shutdown before choosing runtime or input-pipeline work.
3. Pick exactly one next infra lane from evidence: runtime A/B for
   `runtime_overhead`, input pipeline for payload/request bottlenecks, or
   artifact/mmproj/offload A/B for memory/lfb pressure.
4. Keep remote Jetson connection settings in ignored `.env.jetson`; do not copy
   SSH hosts, passwords, tokens, or private paths into tracked docs.
5. Keep `docs/specs/next_phase_infra_and_model_strategy.md` and
   `docs/matrix_edge_vlm_workflow.md` as the source of truth for scope and
   role labels.

## References

- Matrix: `docs/matrix_edge_vlm_workflow.md`
- Strategy spec: `docs/specs/next_phase_infra_and_model_strategy.md`
- Benchmark/model spec: `docs/specs/next_phase_benchmark_and_models.md`
- Optimization loop: `docs/specs/jetson_optimization_loop.md`
- Benchmark protocol: `docs/benchmark_protocol.md`
- Profiling plan: `docs/plans/2026-05-31-jetson-profiling-harness.md`
- Lightweight evidence: `docs/benchmarks/jetson_lightweight_models_20260531.md`
