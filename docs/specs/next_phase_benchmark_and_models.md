# Next Phase: Benchmark Infra And Model Expansion

## Goal

Build a reproducible Jetson benchmark loop before adding more model families. New models must be compared against the same run metadata, repeated trials, power/thermal context, and prompt cases as the existing MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 smoke paths.

Current follow-on direction: `docs/specs/next_phase_infra_and_model_strategy.md`
narrows the next phase after the MiniCPM/Gemma parameter sweeps. Treat this
document as the benchmark/model expansion foundation, and use the infra strategy
spec for profiling gates, runtime gates, artifact comparison rules, and the
decision to stop broad llama.cpp flag sweeps for the current defaults.

## Phase 1: Formal Benchmark Infra

Implement and use a formal benchmark path around the existing `edge_vlm.benchmark` runner.

Required fields:

- `run_id` on every JSONL record.
- `trial_index` and `case_index` on every JSONL record.
- JSON manifest sidecar with model/config paths, benchmark arguments, device label, runtime environment, success/failure counts, and Jetson profile pointers.
- Optimization sweep manifest with per-variant preflight memory state and server startup timing.
- Optional Markdown summary remains a readable sidecar; JSONL plus manifest remain the raw source of truth.
- Jetson wrapper script captures `tegrastats` when available and records `nvpmodel`, `jetson_clocks`, `uname`, and Docker version outputs under a profile directory.

Acceptance:

- Dry-run wrapper works without Jetson hardware or `tegrastats`.
- Real Jetson runs record `tegrastats` next to benchmark JSONL.
- Existing benchmark cases remain compatible.
- Existing smoke output is not reinterpreted as formal performance.

## Phase 2: Lightweight Model Expansion

After Phase 1 is usable, add new models only when they satisfy the repo's low-friction rule:

- GGUF model artifact is available.
- Multimodal projector / `mmproj` path is available or the repo documents equivalent llama.cpp multimodal loading.
- `llama-server` or `llama-mtmd-cli` can load it without local conversion on the memory-constrained WSL host.
- WSL dry-run and payload checks pass before Jetson smoke.
- Jetson support is claimed only after real JSONL plus manifest exists.

Initial candidate order:

| Candidate | Purpose | Source | Initial status |
|---|---|---|---|
| SmolVLM2 256M | Lowest-resource image baseline and latency floor | `ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0` | Jetson 1-trial smoke passed in `smolvlm2-256m-smoke64-20260531c`; 5-trial repeat `lightweight-repeat5-20260531T134554Z` passed at 199.847 text tok/s and 164.551 image tok/s, but structured excerpt review passed only 20/30 and keeps it as latency_floor rather than default replacement or text-reasoning route |
| Qwen3-VL-2B Thinking | New small Qwen VLM quality/speed comparison | `Qwen/Qwen3-VL-2B-Thinking-GGUF:Q4_K_M` | Jetson 1-trial cached smoke passed in `qwen3vl-2b-thinking-smoke64-cached-20260531c`; 5-trial repeat `lightweight-repeat5-20260531T134554Z` passed at 34.761 text tok/s, 34.290 image tok/s, and 1.942 s fake latency, but structured excerpt review passed only 20/30 and narrows it to image/fake-stream balanced_candidate rather than text/code route |
| HunyuanOCR 1B Q8 | Tencent-base OCR/VLM route candidate with low model size | `ggml-org/HunyuanOCR-GGUF:Q8_0` | Jetson smoke `hunyuanocr-q8-smoke64-20260531c` loaded after the launcher-resume fix and completed benchmark/fake-stream records, but failed the guard with repeated exclamation-mark outputs. It is a ggml-org GGUF quantization of Tencent HunyuanOCR, not an official Tencent-owned GGUF artifact; do not rank or repeat until quality triage changes the artifact, prompt/template handling, or runtime path. |
| Tencent Youtu-VL-4B | Tencent small VLM candidate for Chinese/image reasoning comparison | `tencent/Youtu-VL-4B-Instruct-GGUF:Q8_0` | Downloaded official Q8/BF16-mmproj artifacts, but Jetson smoke `youtu-vl-4b-q8-smoke64-20260531a` failed before server ready with CUDA OOM while allocating the 893MB mmproj buffer; defer unless a lower-bit official artifact or different backend is available |
| Youtu-VL-4B third-party Q4 | Clearly separated low-bit experiment for Tencent Youtu base-model behavior | `mradermacher/Youtu-VL-4B-Instruct-GGUF:Q4_K_M` | Jetson 1-trial cached smoke passed in `youtu-vl-4b-q4-thirdparty-smoke64-cached-20260531b` with CPU mmproj via `--no-mmproj-offload`; 5-trial repeat `lightweight-repeat5-20260531T134554Z` passed but was slow at 7.502 text tok/s and 9.186 s fake latency, with `runtime_overhead` and min `lfb` 1, so it is not a default ranking row |
| SmolVLM2 500M | Slightly larger latency/quality point if 256M is too weak | `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF` | Watchlist; add after 256M establishes the path |
| InternVL3 1B / 2B | Compact OpenGVLab comparison point | `ggml-org/InternVL3-1B-Instruct-GGUF`, `ggml-org/InternVL3-2B-Instruct-GGUF` | Candidate, not observed in this repo |
| Moondream2 | Very small VLM behavior/latency check | `ggml-org/moondream2-20250414-GGUF` | Candidate, not observed in this repo |
| Tencent HY-Embodied-0.5-X | Latest Tencent edge-oriented VLM watchlist item | `tencent/HY-Embodied-0.5-X` | Deferred: current release is Transformers/Safetensors/custom-code, not a low-friction llama.cpp GGUF candidate |
| Tencent Penguin-VL-2B | Tencent small custom-code VLM watchlist item | `tencent/Penguin-VL-2B` | Deferred: current release is Transformers/Safetensors/custom-code, not a low-friction llama.cpp GGUF candidate |
| Tencent HY-Embodied-0.5 | Tencent embodied VLM watchlist item | `tencent/HY-Embodied-0.5` | Deferred: current release is Transformers/Safetensors/custom-code, not a low-friction llama.cpp GGUF candidate |
| Tencent Youtu-Parsing | Tencent 2.5B-ish Youtu VLM/custom-code watchlist item | `tencent/Youtu-Parsing` | Deferred: current release is Transformers/Safetensors/custom-code, not a low-friction llama.cpp GGUF candidate |

Use the generic `scripts/jetson/run_hf_gguf_vlm_llama_docker.sh` launcher for
Hub-hosted GGUF candidates. The launcher downloads the named GGUF and mmproj
files to `MODEL_DIR`, then starts the pinned container with local
`-m`/`--mmproj` paths. This avoids direct `llama-server -hf` downloads because
the pinned Jetson llama.cpp container currently reports no HTTPS support. Add
model-specific launchers only when a candidate needs confirmed nonstandard
server flags. Do not broaden runtime claims from one candidate to another.

First smoke order:

1. `smolvlm2-256m-q8-smoke`
2. `qwen3-vl-2b-thinking-q4-smoke`
3. `hunyuanocr-q8-smoke` loaded but failed the guard with repetitive output;
   keep it out of ranking and formal repeat until quality triage fixes the
   current artifact/runtime path.
4. `youtu-vl-4b-q8-smoke` did not become a promotable smoke candidate because
   official Youtu-VL Q8 plus BF16 mmproj failed startup on Jetson with CUDA OOM.
5. `youtu-vl-4b-q4-thirdparty-smoke` passed as a separate CPU-mmproj smoke path
   and also passed `lightweight-repeat5-20260531T134554Z`, but the repeat
   confirmed it is slow and memory-stressed. Keep any GPU-mmproj/offload canary
   as a separate artifact/runtime variant before comparing throughput.

Each smoke must pass the same preflight, startup timing, formal benchmark, and
three-frame fake-stream path before any tuning sweep is added.

### Text/Router Lane

Tencent's newest small official GGUF rows are text/translation models, not
VLMs. The executable Hy-MT1.5, Hy-MT2, and Youtu-LLM rows are configured
separately so they can be tested as text/router candidates without polluting VLM
rankings:

| Candidate | Config | Source file |
|---|---|---|
| Hy-MT1.5 1.8B 1.25bit | `configs/models/tencent_hy_mt1p5_1p8b_1p25bit.yaml` | `tencent/Hy-MT1.5-1.8B-1.25bit-GGUF` / `Hy-MT1.5-1.8B-1.25bit.gguf`; default text-suite runtime canary failed on pinned llama.cpp with invalid ggml type 42 |
| Hy-MT1.5 1.8B 2bit | `configs/models/tencent_hy_mt1p5_1p8b_2bit.yaml` | `tencent/Hy-MT1.5-1.8B-2bit-GGUF` / `Hy-MT1.5-1.8B-2bit.gguf`; default text-suite runtime canary failed on pinned llama.cpp with tensor offset `203248672` |
| HY-MT1.5 1.8B Q4_K_M | `configs/models/tencent_hy_mt1p5_1p8b_q4.yaml` | `tencent/HY-MT1.5-1.8B-GGUF` / `HY-MT1.5-1.8B-Q4_K_M.gguf`; added as an official text/router default, no Jetson evidence yet |
| HY-MT1.5 1.8B Q6_K | `configs/models/tencent_hy_mt1p5_1p8b_q6.yaml` | `tencent/HY-MT1.5-1.8B-GGUF` / `HY-MT1.5-1.8B-Q6_K.gguf`; added as an official text/router default, no Jetson evidence yet |
| HY-MT1.5 1.8B Q8_0 | `configs/models/tencent_hy_mt1p5_1p8b_q8.yaml` | `tencent/HY-MT1.5-1.8B-GGUF` / `HY-MT1.5-1.8B-Q8_0.gguf`; added as an official text/router default, no Jetson evidence yet |
| Hy-MT2 1.8B 1.25Bit | `configs/models/tencent_hy_mt2_1p8b_1p25bit.yaml` | `tencent/Hy-MT2-1.8B-1.25Bit-GGUF` / `Hy-MT2-1.8B-1.25Bit.gguf`; default text-suite runtime canary |
| Hy-MT2 1.8B 2Bit | `configs/models/tencent_hy_mt2_1p8b_2bit.yaml` | `tencent/Hy-MT2-1.8B-2Bit-GGUF` / `Hy-MT2-1.8B-2Bit.gguf`; default text-suite runtime canary failed on pinned llama.cpp with tensor offset `203248672` |
| Hy-MT2 1.8B Q4_K_M | `configs/models/tencent_hy_mt2_1p8b_q4.yaml` | `tencent/Hy-MT2-1.8B-GGUF` / `Hy-MT2-1.8B-Q4_K_M.gguf`; cached Jetson smoke `tencent-hy-mt2-q4-smoke64-cached-20260531b` passed text guard at 19.759 tok/s, full-suite run `tencent-text-smoke64-20260531T120054Z` passed at 33.541 tok/s, and repeat `tencent-text-repeat5-20260531a` passed 20/20 at 34.528 tok/s |
| Hy-MT2 1.8B Q6_K | `configs/models/tencent_hy_mt2_1p8b_q6.yaml` | `tencent/Hy-MT2-1.8B-GGUF` / `Hy-MT2-1.8B-Q6_K.gguf`; full-suite run `tencent-text-smoke64-20260531T120054Z` passed at 26.287 tok/s, cached run `tencent-hy-mt2-q6-smoke64-cached-20260531a` passed at 26.298 tok/s, and repeat `tencent-text-repeat5-20260531a` passed 20/20 at 26.778 tok/s |
| Hy-MT2 1.8B Q8_0 | `configs/models/tencent_hy_mt2_1p8b_q8.yaml` | `tencent/Hy-MT2-1.8B-GGUF` / `Hy-MT2-1.8B-Q8_0.gguf`; Q8 full-suite row is invalidated by the pre-fix launcher cleanup/stale-port issue, but cached rerun `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` passed at 30.756 tok/s with `terminate_group` shutdown, and repeat `tencent-text-repeat5-20260531a` passed 20/20 at 31.395 tok/s |
| Youtu-LLM 2B Q8_0 | `configs/models/tencent_youtu_llm_2b_q8.yaml` | `tencent/Youtu-LLM-2B-GGUF` / `Youtu-LLM-2B-Q8_0.gguf`; added as an official text/router default, no Jetson evidence yet; F16 sibling is deferred from the default suite due higher memory pressure |

These variants use `scripts/jetson/run_hf_gguf_llama_docker.sh`,
`configs/benchmark/text_prompt_cases.jsonl`, and `capabilities.image=false`.
The dedicated wrapper `scripts/jetson/run_remote_tencent_text_suite.sh`
defaults to all eleven configured Tencent text GGUF rows under locked clocks,
cache drop, `--min-lfb-blocks`, and a mechanical comparison report with
`--fake-stream-max-frames 0`. Use `JETSON_TENCENT_TEXT_VARIANTS` to narrow the
default set or `JETSON_TENCENT_TEXT_EXTRA_VARIANTS` to append scoped canaries.
The sweep planner also skips fake-stream for them because the configs are
text-only. Treat their results as a separate text/router study; the current Q4
smoke, `tencent-text-smoke64-20260531T120054Z` Q4/Q6 rows, the
`tencent-hy-mt2-q6-smoke64-cached-20260531a` cached Q6 row, and
`tencent-text-repeat5-20260531a` Q4/Q6/Q8 rows are useful evidence for the lane,
not a VLM ranking row. Q8 full-suite row is invalidated but the cached rerun and
repeat Q8 row are valid text/router evidence with the fixed harness.

Deferred latest non-GGUF Tencent text rows checked in the same refresh:

| Candidate | Source | Status |
|---|---|---|
| Hy-MT1.5 1.8B Safetensors 1.25bit/2bit | `tencent/Hy-MT1.5-1.8B-1.25bit`, `tencent/Hy-MT1.5-1.8B-2bit` | Deferred because the current bench lane is GGUF/llama.cpp only; do not add configs until a bounded Transformers or conversion path is selected. |

## Phase 3: llama.cpp Acceleration Sweep

This phase has enough evidence for MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4:
the tracked sweeps and max-clocks repeats did not produce a clean default
promotion from batch/ubatch, Flash Attention, KV-cache precision, memory
mapping, locking, prompt-cache, host-buffer, repack, DirectIO, warmup, or higher
Gemma GPU offload. Do not continue broad llama.cpp flag sweeps for these two
current defaults. New llama.cpp probes should start from a profiling claim or a
new confirmed container capability and should follow
`docs/specs/next_phase_infra_and_model_strategy.md`.

When a scoped probe is justified, use `docs/specs/jetson_optimization_loop.md`
and `scripts/jetson/run_optimization_sweep.sh` for reproducible sweeps. A
faster candidate is not promotable unless the optimization report marks its
sanity guard as passing.

Sweep variables:

- `N_GPU_LAYERS`
- `CTX_SIZE`
- `LLAMA_BATCH_SIZE`
- `LLAMA_UBATCH_SIZE`
- KV cache type
- mmproj offload on/off
- warmup on/off and cold-start separation
- pinned container image / llama.cpp artifact version, recorded through sweep
  `server_runtime` image id/digest/ref metadata before comparing rows

TensorRT, TensorRT-LLM, NanoLLM, Ollama, vLLM, and custom kernels stay deferred until the formal llama.cpp records show a specific bottleneck worth paying integration cost for.

## References Checked

- llama.cpp multimodal documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- SmolVLM2 256M GGUF: https://hf.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF
- SmolVLM2 500M GGUF: https://hf.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF
- Qwen3-VL 2B Instruct GGUF: https://hf.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF
- Qwen3-VL 2B Thinking GGUF: https://hf.co/Qwen/Qwen3-VL-2B-Thinking-GGUF
- HunyuanOCR GGUF: https://hf.co/ggml-org/HunyuanOCR-GGUF
- Tencent HunyuanOCR base: https://hf.co/tencent/HunyuanOCR
- Tencent Youtu-VL-4B Instruct GGUF: https://hf.co/tencent/Youtu-VL-4B-Instruct-GGUF
- Tencent Penguin-VL-2B: https://hf.co/tencent/Penguin-VL-2B
- Tencent HY-Embodied-0.5: https://hf.co/tencent/HY-Embodied-0.5
- Tencent HY-Embodied-0.5-X: https://hf.co/tencent/HY-Embodied-0.5-X
- Tencent Youtu-Parsing: https://hf.co/tencent/Youtu-Parsing
- Tencent Hy-MT1.5 1.8B 1.25bit GGUF: https://hf.co/tencent/Hy-MT1.5-1.8B-1.25bit-GGUF
- Tencent Hy-MT1.5 1.8B 2bit GGUF: https://hf.co/tencent/Hy-MT1.5-1.8B-2bit-GGUF
- Tencent HY-MT1.5 1.8B GGUF: https://hf.co/tencent/HY-MT1.5-1.8B-GGUF
- Tencent Hy-MT1.5 1.8B 1.25bit Safetensors: https://hf.co/tencent/Hy-MT1.5-1.8B-1.25bit
- Tencent Hy-MT1.5 1.8B 2bit Safetensors: https://hf.co/tencent/Hy-MT1.5-1.8B-2bit
- Tencent Hy-MT2 1.8B GGUF: https://hf.co/tencent/Hy-MT2-1.8B-GGUF
- Tencent Hy-MT2 1.8B 1.25Bit GGUF: https://hf.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF
- Tencent Hy-MT2 1.8B 2Bit GGUF: https://hf.co/tencent/Hy-MT2-1.8B-2Bit-GGUF
- Tencent Youtu-LLM 2B GGUF: https://hf.co/tencent/Youtu-LLM-2B-GGUF
- InternVL3 1B GGUF: https://hf.co/ggml-org/InternVL3-1B-Instruct-GGUF
- InternVL3 2B GGUF: https://hf.co/ggml-org/InternVL3-2B-Instruct-GGUF
- Moondream2 GGUF: https://hf.co/ggml-org/moondream2-20250414-GGUF
