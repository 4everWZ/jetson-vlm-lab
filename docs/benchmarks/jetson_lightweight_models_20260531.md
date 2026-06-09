# Jetson Lightweight Model Expansion - 2026-05-31

This document records the first lightweight-model checks after the MiniCPM-V
4.6 Q4 and Gemma 4 E2B-it Q4 infra sweeps. Raw JSONL, fake-stream JSONL,
manifests, server logs, and downloaded model artifacts stayed under ignored
Jetson paths.

## Environment

| Field | Value |
|---|---|
| Local branch / commit | `bench/formal-jetson-infra` / through `2831f8f` for VLM repeat evidence and `24481dd` for Tencent text repeat evidence; HunyuanOCR smoke used `158f0e4`, Hy-MT1.5 text canary used `f91712b` |
| Jetson branch / commit | `bench/formal-jetson-infra` / through `2831f8f` for VLM repeat evidence and `24481dd` for Tencent text repeat evidence; HunyuanOCR smoke used `158f0e4`, Hy-MT1.5 text canary used `f91712b` |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Model root | `/home/weizheng/code/jetson-vlm-lab/models` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Trial count | 1 for smoke rows; 5 for `lightweight-repeat5-20260531T134554Z` and `tencent-text-repeat5-20260531a` |
| Max tokens | 64 for accepted smoke; 32 failed the guard |
| Temperature | 0 |
| Fake-stream frames | 3 |

## HF GGUF Launcher Finding

The first SmolVLM2 attempt used `llama-server -hf` through the generic launcher
from commit `ae954b1`. The pinned Jetson container exited before server ready
with:

```text
HTTPS is not supported. Please rebuild with one of:
  -DLLAMA_BUILD_BORINGSSL=ON
  -DLLAMA_BUILD_LIBRESSL=ON
  -DLLAMA_OPENSSL=ON
```

Commit `94154a4` changed the generic launcher to download the named GGUF and
mmproj files on the Jetson host with `curl`, then start the container with
local `-m` and `--mmproj` paths.

Commit `158f0e4` added a recovery path for interrupted host-side downloads: if
`curl --continue-at -` receives HTTP 416 while a `.partial` file already starts
with the GGUF magic bytes, the launcher accepts that completed partial and
renames it to the final artifact path. This unblocked the HunyuanOCR mmproj
file after the first run timed out during artifact download.

Downloaded SmolVLM2 artifacts:

| File | Size |
|---|---:|
| `models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/SmolVLM2-256M-Video-Instruct-Q8_0.gguf` | 175,056,352 bytes |
| `models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf` | 103,771,680 bytes |

## SmolVLM2 256M Q8 Smoke

Variant: `smolvlm2-256m-q8-smoke`

The first real run with `--max-tokens 32` completed but failed the lightweight
guard because `text_cn_short` was shorter than the report threshold. The
accepted smoke reran the same variant with `--max-tokens 64`.

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `smolvlm2-256m-smoke64-20260531c` | 234x4MB | 3.020 | yes | 6/6 | 3/3 | 143.761 | 70.655 | 0.463 | 0.514 | 0.525 |

Per-frame fake-stream latency:

| Frame | Latency s |
|---|---:|
| `frame_001.png` | 0.554 |
| `frame_002.png` | 0.510 |
| `frame_003.png` | 0.511 |

Decision: SmolVLM2 256M Q8 is now a valid Jetson smoke baseline for the generic
HF GGUF launcher path. It is much faster than the MiniCPM/Gemma baselines, but
its output is visibly weaker and repetitive on the simple sample frames. Treat
it as a latency floor, not as a replacement default.

## Qwen3-VL 2B Thinking Q4 Smoke

Variant: `qwen3-vl-2b-thinking-q4-smoke`

The first Qwen run with the default 180s server wait timed out while downloading
the 1.056GB model file; no server startup had happened yet. Rerunning with
`--wait-timeout-s 900` resumed the partial download and completed the smoke. A
third run used the cached files to measure startup without download time.

Downloaded Qwen artifacts:

| File | Size |
|---|---:|
| `models/Qwen/Qwen3-VL-2B-Thinking-GGUF/Qwen3VL-2B-Thinking-Q4_K_M.gguf` | 1,130,724,320 bytes |
| `models/Qwen/Qwen3-VL-2B-Thinking-GGUF/mmproj-Qwen3VL-2B-Thinking-Q8_0.gguf` | 445,053,216 bytes |

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3vl-2b-thinking-smoke64-20260531b` | 231x4MB | 201.450 | yes | 6/6 | 3/3 | 33.057 | 30.774 | 1.940 | 2.081 | 1.983 |
| `qwen3vl-2b-thinking-smoke64-cached-20260531c` | 239x4MB | 5.067 | yes | 6/6 | 3/3 | 32.947 | 30.837 | 1.947 | 2.077 | 1.978 |

Decision: Qwen3-VL 2B Thinking Q4 is a valid Jetson smoke candidate. It is much
slower than SmolVLM2 256M but still substantially faster than the current
Gemma Q4 baseline, and its sample outputs are more deliberate than SmolVLM2.
The formal repeat below keeps it as the current 2B-class balanced candidate,
pending human output review and route-specific quality checks.

## HunyuanOCR 1B Q8 Smoke

Variant: `hunyuanocr-q8-smoke`

The first HunyuanOCR attempt (`hunyuanocr-q8-smoke64-20260531a`) did not reach
server readiness because it spent the wait window downloading artifacts. The
second attempt (`hunyuanocr-q8-smoke64-20260531b`) immediately failed with
`curl` HTTP 416 while resuming a completed-looking `mmproj` partial. Commit
`158f0e4` changed both HF GGUF launchers to accept an existing `.partial` as
complete only when the server returned HTTP 416 and the local file has GGUF
magic bytes. The third run pulled that launcher fix on the Jetson and completed
the smoke.

Downloaded HunyuanOCR artifacts:

| File | Size |
|---|---:|
| `models/ggml-org/HunyuanOCR-GGUF/HunyuanOCR-Q8_0.gguf` | 583,134,944 bytes |
| `models/ggml-org/HunyuanOCR-GGUF/mmproj-HunyuanOCR-Q8_0.gguf` | 732,938,240 bytes |

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `hunyuanocr-q8-smoke64-20260531a` | 242x4MB | n/a | n/a | 0/0 | 0/0 | n/a | n/a | n/a | n/a | n/a |
| `hunyuanocr-q8-smoke64-20260531b` | 242x4MB | n/a | n/a | 0/0 | 0/0 | n/a | n/a | n/a | n/a | n/a |
| `hunyuanocr-q8-smoke64-20260531c` | 245x4MB | 5.022 | no | 6/6 | 3/3 | 67.745 | 35.006 | 0.945 | 1.830 | 1.738 |

Failed-run causes:

| Run prefix | Server ready | Server return code | Wait s | Failure |
|---|---|---:|---:|---|
| `hunyuanocr-q8-smoke64-20260531a` | no | -15 | 600.951 | Timeout while downloading artifacts; `mmproj` remained as `.partial`. |
| `hunyuanocr-q8-smoke64-20260531b` | no | 22 | 1.000 | `curl --continue-at -` returned HTTP 416 for the completed `.partial`. |

Profile summary for the successful run recorded six `tegrastats` samples,
minimum profiled `lfb` of 186 blocks, average GR3D utilization of 95.333%,
average CPU utilization of 3.028%, max temperature 51.437 C, average `VDD_IN`
power 18.975 W, and bottleneck label `gpu_compute`. The launcher lifecycle
marked `artifact_check_or_download` as `downloaded_or_checked` with 0.689s
duration.

The model loaded and returned HTTP responses, but every text, image, and
fake-stream output excerpt was a repeated exclamation-mark string. The report
therefore marked the guard as failed for repetitive output and quality-term
misses across all benchmark and fake-stream records.

Decision: HunyuanOCR Q8 is a valid load-path smoke for the Tencent-base
ggml-org artifact, but it is not a usable VLM/OCR candidate under the current
llama.cpp runtime, prompt path, and Q8 artifact. Do not spend 5-trial formal
repeat budget on it until a bounded quality triage changes the artifact,
prompt/template handling, or runtime path.

## Youtu-VL 4B Q8 Smoke

Variant: `youtu-vl-4b-q8-smoke`

The official Tencent GGUF artifacts downloaded successfully, but the pinned
llama.cpp container failed before server ready while loading the BF16 multimodal
projector. The failed run used conservative GPU offload (`N_GPU_LAYERS=12`) and
had a clean preflight memory state.

Downloaded Youtu artifacts:

| File | Size |
|---|---:|
| `models/tencent/Youtu-VL-4B-Instruct-GGUF/Youtu-VL-4B-Instruct-Q8_0.gguf` | 5,211,323,488 bytes |
| `models/tencent/Youtu-VL-4B-Instruct-GGUF/mmproj-Youtu-VL-4b-Instruct-BF16.gguf` | 893,397,344 bytes |

| Run prefix | Preflight `lfb` | Server ready | Server return code | Wait s | Failure |
|---|---:|---|---:|---:|---|
| `youtu-vl-4b-q8-smoke64-20260531a` | 245x4MB | no | 133 | 1818.172 | CUDA OOM allocating 851.99 MiB for the mmproj buffer |

Relevant server log tail:

```text
device_info: CUDA0 : Orin (7619 MiB, 6558 MiB free)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 851.99 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 893373568
```

Decision: the official Tencent Youtu-VL-4B Q8 plus BF16 mmproj package is not a
viable default smoke candidate on this Jetson with the current pinned llama.cpp
image/config. Keep the downloaded artifacts as evidence, but defer promotion
until a lower-bit official artifact or different backend is available. A
third-party Q4 GGUF would be a separate research choice, not the same official
Tencent candidate.

## Youtu-VL 4B Q4 Third-Party Smoke

Variant: `youtu-vl-4b-q4-thirdparty-smoke`

This is not an official Tencent GGUF artifact. It uses
`mradermacher/Youtu-VL-4B-Instruct-GGUF:Q4_K_M`, a third-party quantization of
Tencent's Youtu-VL-4B-Instruct base model, with the Q8_0 mmproj from the same
third-party repo. The first smoke kept the mmproj on CPU with
`--no-mmproj-offload` because the official Tencent Q8/BF16-mmproj run failed
during CUDA mmproj allocation. Runtime image metadata in the manifest records
canonical image id `36f3398b7885` and llama.cpp ref `d749821db3bd`.

Downloaded third-party Youtu artifacts:

| File | Size |
|---|---:|
| `models/mradermacher/Youtu-VL-4B-Instruct-GGUF/Youtu-VL-4B-Instruct.Q4_K_M.gguf` | 3,089,819,584 bytes |
| `models/mradermacher/Youtu-VL-4B-Instruct-GGUF/Youtu-VL-4B-Instruct.mmproj-Q8_0.gguf` | 602,557,888 bytes |

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `youtu-vl-4b-q4-thirdparty-smoke64-20260531a` | 233x4MB | 1124.738 | yes | 6/6 | 3/3 | 6.724 | 5.930 | 9.577 | 8.111 | 9.365 |
| `youtu-vl-4b-q4-thirdparty-smoke64-cached-20260531b` | 239x4MB | 7.027 | yes | 6/6 | 3/3 | 6.434 | 5.884 | 9.979 | 8.162 | 9.487 |

The first startup includes host-side HF downloads. The cached run is the useful
startup datapoint for this CPU-mmproj smoke path.

Sample outputs were coherent on the simple prompts and sample frames. The image
cases correctly identified the white/dark square scene and reported no hazards
for the safety prompt.

Decision: the third-party Youtu Q4 path is a valid Jetson smoke candidate under
CPU mmproj, but it is slow and must stay separate from the official Tencent Q8
result. The formal repeat below confirms it is not competitive as a default
runtime row; revisit only for an explicit artifact/runtime A/B.

## Fixed-Policy 5-Trial Repeat

Run prefix: `lightweight-repeat5-20260531T134554Z`.

The repeat used locked clocks, cache drop before each variant,
`--trial-count 5`, `--max-tokens 64`, three fake-stream frames,
`--min-lfb-blocks 150`, and the canonical llama.cpp image
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`
with image id `36f3398b7885` and llama.cpp ref `d749821db3bd`. The default
repeat set intentionally excluded HunyuanOCR Q8 and official Youtu-VL Q8 after
their guard-failure/OOM evidence; official Youtu-VL Q8 can only be rerun by
passing it through `JETSON_LIGHTWEIGHT_EXTRA_VARIANTS`.

| Model | Variant | Preflight `lfb` | Startup s | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Avg power W | Avg GR3D % | Min lfb blocks | Bottleneck |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8` | 209x4MB | 6.019 | yes | 30/30 | 3/3 | 48.598 | 46.530 | 1.317 | 1.390 | 1.655 | 19.268 | 93.727 | 71 | `gpu_compute` |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8` | 210x4MB | 6.019 | yes | 30/30 | 3/3 | 12.072 | 12.616 | 5.307 | 5.158 | 6.163 | 16.464 | 25.508 | 1 | `runtime_overhead` |
| SmolVLM2 256M Q8 | `smolvlm2-256m-q8-smoke` | 220x4MB | 2.008 | yes | 30/30 | 3/3 | 199.847 | 164.551 | 0.321 | 0.227 | 0.462 | 16.958 | 89.143 | 220 | `gpu_compute` |
| Qwen3-VL 2B Thinking Q4 | `qwen3-vl-2b-thinking-q4-smoke` | 220x4MB | 5.016 | yes | 30/30 | 3/3 | 34.761 | 34.290 | 1.841 | 1.869 | 1.942 | 21.750 | 96.457 | 154 | `gpu_compute` |
| Youtu-VL 4B Q4 third-party | `youtu-vl-4b-q4-thirdparty-smoke` | 224x4MB | 7.021 | yes | 30/30 | 3/3 | 7.502 | 7.249 | 8.570 | 6.705 | 9.186 | 17.047 | 19.546 | 1 | `runtime_overhead` |

Repeat conclusions:

- SmolVLM2 256M Q8 is the latency floor. It is fast enough for cheap routing,
  liveness, and harness stress, but the raw text excerpts still show weak
  semantics such as interpreting WSL as a generic web-services layer.
- Qwen3-VL 2B Thinking Q4 is the balanced_candidate for image/fake-stream
  experiments: it is slower than MiniCPM-V 4.6 Q4 but far ahead of Gemma and
  keeps enough `lfb` headroom for continued pipeline/routing experiments.
- MiniCPM-V 4.6 Q4 remains the practical default reference row for quality and
  speed balance.
- Gemma 4 E2B-it Q4 is not a 2B-class target in practice. The repeat reinforces
  the decision to stop broad llama.cpp flag tuning for it: throughput is low,
  GR3D is not saturated, and profiled `lfb` falls to one block.
- Third-party Youtu Q4 is coherent on simple prompts but too slow, nonofficial,
  and also reaches one profiled `lfb` block. Keep it out of default ranking
  unless an artifact/runtime A/B has a specific reason.
- Profiling currently separates two lanes: MiniCPM, SmolVLM2, and Qwen are
  mostly `gpu_compute`; Gemma and Youtu Q4 show `runtime_overhead` with very
  low profiled `lfb`, so deeper infra work should focus on memory/runtime
  characterization rather than more llama.cpp flag sweeps.

Structured quality review:

Policy: `configs/benchmark/quality_review_policy.json`, runner:
`python -m edge_vlm.quality_review`. The review below applies route-specific
checks to recorded `output_excerpt` fields from
`lightweight-repeat5-20260531T134554Z`; it is a mechanical gate before human
review, not a substitute for full-output inspection.

| Model | Passed records | Policy pass | Failed policy cases | Route implication |
|---|---:|---|---|---|
| MiniCPM-V 4.6 Q4 | 30/30 | yes | none | Keep as quality_reference/default row. |
| Gemma 4 E2B-it Q4 | 20/30 | no | `text_cn_short` 0/5, `text_code_short` 0/5 | Keep only as quality/reference evidence where explicitly useful; not a speed or route candidate. |
| SmolVLM2 256M Q8 | 20/30 | no | `text_cn_short` 0/5, `text_en_reasoning_short` 0/5 | Keep as latency_floor and harness stress row; do not use for text reasoning routes. |
| Qwen3-VL 2B Thinking Q4 | 20/30 | no | `text_cn_short` 0/5, `text_code_short` 0/5 | Keep as image/fake-stream balanced_candidate only; do not use for text/code routes until prompt/output evidence changes. |
| Youtu-VL 4B Q4 third-party | 30/30 | yes | none | Quality policy passes, but it remains a failed_artifact for default ranking because it is slow, nonofficial, and memory-stressed. |

The failures narrow route roles rather than invalidate the raw repeat. SmolVLM2
still proves the low-latency floor; Qwen still has useful image/fake-stream
behavior but failed the current text/code excerpt policy; Youtu Q4 still cannot
be promoted because artifact provenance and runtime cost dominate the simple
quality pass.

## Tencent Small-Model Refresh

The 2026-05-31 Hugging Face refresh found additional small Tencent models, but
they split into two lanes:

| Model | HF source | Repo status |
|---|---|---|
| Hy-MT1.5 1.8B 1.25bit GGUF | `tencent/Hy-MT1.5-1.8B-1.25bit-GGUF` / `Hy-MT1.5-1.8B-1.25bit.gguf` | Added as a default text/router runtime canary; first Jetson smoke failed before server ready with `invalid ggml type 42` on the pinned llama.cpp image. |
| Hy-MT1.5 1.8B 2bit GGUF | `tencent/Hy-MT1.5-1.8B-2bit-GGUF` / `Hy-MT1.5-1.8B-2bit.gguf` | Added as a default text/router runtime canary; failures are runtime-support evidence, not VLM ranking evidence. |
| HY-MT1.5 1.8B Q4/Q6/Q8 GGUF | `tencent/HY-MT1.5-1.8B-GGUF` / `HY-MT1.5-1.8B-{Q4_K_M,Q6_K,Q8_0}.gguf` | Added as default text/router configs and variants after the 2026-05-31 refresh. Still no Jetson evidence; keep out of VLM ranking. |
| Hy-MT1.5 1.8B Safetensors 1.25bit/2bit | `tencent/Hy-MT1.5-1.8B-1.25bit`, `tencent/Hy-MT1.5-1.8B-2bit` | Deferred; latest non-GGUF quant rows, no selected Transformers/conversion path in the current GGUF bench lane. |
| Hy-MT2 1.8B 1.25Bit GGUF | `tencent/Hy-MT2-1.8B-1.25Bit-GGUF` / `Hy-MT2-1.8B-1.25Bit.gguf` | Added as a default text/router runtime canary; failures are runtime-support evidence, not VLM ranking evidence. |
| Hy-MT2 1.8B 2Bit GGUF | `tencent/Hy-MT2-1.8B-2Bit-GGUF` / `Hy-MT2-1.8B-2Bit.gguf` | Added as a default text/router runtime canary; failures are runtime-support evidence, not VLM ranking evidence. |
| Hy-MT2 1.8B Q4/Q6/Q8 GGUF | `tencent/Hy-MT2-1.8B-GGUF` / `Hy-MT2-1.8B-{Q4_K_M,Q6_K,Q8_0}.gguf` | Added as default text/router configs and variants; Q4/Q6 passed the full text-suite smoke, Q6 cached run `tencent-hy-mt2-q6-smoke64-cached-20260531a` passed at 26.298 tok/s, Q8 cached rerun `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` passed at 30.756 tok/s, and 5-trial repeat `tencent-text-repeat5-20260531a` passed Q4/Q6/Q8 at 34.528/26.778/31.395 tok/s. Not VLM candidates. |
| Hy-MT2 1.8B FP8 | `tencent/Hy-MT2-1.8B-FP8` | Deferred; Safetensors/compressed-tensors path, no low-friction GGUF launcher row. |
| Youtu-LLM 2B Q8 GGUF | `tencent/Youtu-LLM-2B-GGUF` / `Youtu-LLM-2B-Q8_0.gguf` | Added as a default text/router config and variant. Prepared repeat `tencent-text-repeat5-prepctx-20260609T131412Z` passed 20/20 records, `Quality review = yes (20/20)`, and `Promotion precheck = yes` under locked clocks, cache drop, and `Required lfb = 150`; cached rerun `youtu-llm-q8-cached-20260609T133339Z` then recorded `artifact_check_or_download = cached (0.002 s)`, `Startup s = 5.015`, and `Text tok/s = 24.621`. F16 sibling is deferred from the default suite due higher memory pressure. |
| HunyuanOCR 1B Q8 GGUF | `ggml-org/HunyuanOCR-GGUF` / `HunyuanOCR-Q8_0.gguf`, `mmproj-HunyuanOCR-Q8_0.gguf` | Jetson smoke loaded after the launcher-resume fix and completed benchmark/fake-stream records, but failed the guard with repeated exclamation-mark outputs; not an official Tencent-owned GGUF artifact and not ranked. |
| Penguin-VL-2B | `tencent/Penguin-VL-2B` | Deferred; Transformers/Safetensors/custom-code, no low-friction GGUF path in this repo yet. |
| HY-Embodied-0.5 / HY-Embodied-0.5-X | `tencent/HY-Embodied-0.5`, `tencent/HY-Embodied-0.5-X` | Deferred; Transformers/Safetensors/custom-code, no low-friction GGUF path in this repo yet. |
| Youtu-Parsing | `tencent/Youtu-Parsing` | Deferred; Transformers/Safetensors/custom-code, no low-friction GGUF path in this repo yet. |

The executable Hy-MT1.5 and Hy-MT2 rows use
`scripts/jetson/run_hf_gguf_llama_docker.sh`,
`configs/benchmark/text_prompt_cases.jsonl`, and `capabilities.image=false`.
`scripts/jetson/run_remote_tencent_text_suite.sh` defaults to all eleven
configured Tencent text GGUF rows with the same locked-clocks/cache-drop/min-lfb
policy and `--fake-stream-max-frames 0`. Low-bit failures are runtime
compatibility evidence; use `JETSON_TENCENT_TEXT_VARIANTS` only when a run needs
to narrow the default set. The sweep planner also skips fake-stream for these
rows because their configs are text-only. Do not compare them against
SmolVLM2/Qwen/HunyuanOCR/Youtu image or fake-stream metrics.

Hy-MT1.5 1.25bit canary evidence:

| Run prefix | Preflight `lfb` | Server ready | Server return code | Wait s | Failure |
|---|---:|---|---:|---:|---|
| `tencent-hy-mt1p5-1p25bit-smoke64-20260531a` | 251x4MB | no | 1 | 2.012 | pinned llama.cpp rejected tensor type 42 while loading `blk.0.attn_k.weight` |

Hy-MT2 Q4 cached smoke evidence:

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Text tok/s | Text latency s | Max temp C | Avg power W | Avg GR3D % | Min lfb blocks |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `tencent-hy-mt2-q4-smoke64-cached-20260531b` | 233x4MB | 4.029 | yes | 4/4 | 19.759 | 3.070 | 53.812 | 21.731 | 95.333 | 160 |

The first Q4 attempt was interrupted while downloading the artifact. A
background `curl --continue-at -` resume completed
`models/tencent/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q4_K_M.gguf`
at 1,133,080,448 bytes; the cached smoke then loaded the local artifact, skipped
fake-stream because the config is text-only, and produced coherent text outputs
for the four text prompt cases. Treat this as a text/router smoke only.

Tencent text-suite smoke evidence:

Run prefix:
`tencent-text-smoke64-20260531T120054Z`. The suite used the then-configured
seven Hy-MT1.5/Hy-MT2 text rows, `--trial-count 1`, `--max-tokens 64`,
`--fake-stream-max-frames 0`, locked clocks, cache drop, and
`--min-lfb-blocks 150`. Q4 and Q6 include first-run host-side artifact download
time in startup; do not use those startup values as cached-startup evidence.

| Variant | Preflight `lfb` | Server ready | Guard | Success | Startup s | Text tok/s | Text latency s | Avg power W | Avg GR3D % | Min lfb blocks | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tencent-hy-mt1p5-1p8b-1p25bit-text-smoke` | 248x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | Failed before server ready: `invalid ggml type 42` while loading `blk.0.attn_k.weight`. |
| `tencent-hy-mt1p5-1p8b-2bit-text-smoke` | 250x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | Failed before server ready: tensor `blk.0.attn_k_norm.weight` offset 203248672, expected `203129888`. |
| `tencent-hy-mt2-1p8b-1p25bit-text-smoke` | 250x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | Failed before server ready: `invalid ggml type 42` while loading `blk.0.attn_k.weight`. |
| `tencent-hy-mt2-1p8b-2bit-text-smoke` | 249x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | Failed before server ready: tensor `blk.0.attn_k_norm.weight` offset 203248672, expected `203572256`. |
| `tencent-hy-mt2-1p8b-q4-text-smoke` | 249x4MB | yes | yes | 4/4 | 529.771 | 33.541 | 1.688 | 21.415 | 95.333 | 188 | Valid text/router smoke. |
| `tencent-hy-mt2-1p8b-q6-text-smoke` | 247x4MB | yes | yes | 4/4 | 923.418 | 26.287 | 2.145 | 21.613 | 96.875 | 156 | Valid text/router smoke. |
| `tencent-hy-mt2-1p8b-q8-text-smoke` | 234x4MB | yes | no | 3/4 | 183.282 | 17.119 | 32.806 | 10.089 | 8.885 | 160 | Q8 full-suite row is invalidated: the artifact was still `Hy-MT2-1.8B-Q8_0.gguf.partial` and the launcher download child outlived the parent in the pre-fix harness, so the row may have hit a stale server. Rerun cached after the launcher cleanup and port-guard fix before reporting Q8. |

These rows are runtime and text/router evidence only. The low-bit failures feed
the runtime/build compatibility backlog, not VLM model ranking.

Hy-MT2 cached rerun evidence:

| Run prefix | Preflight `lfb` | Startup s | Guard | Success | Text tok/s | Text latency s | Max temp C | Avg power W | Avg GR3D % | Min lfb blocks | Shutdown |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tencent-hy-mt2-q6-smoke64-cached-20260531a` | 196x4MB | 4.021 | yes | 4/4 | 26.298 | 2.144 | 52.843 | 21.662 | 97.000 | 92 | `terminate_group` |
| `tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z` | 177x4MB | 6.016 | yes | 4/4 | 30.756 | 1.841 | 52.812 | 22.247 | 96.857 | 52 | `terminate_group`, port closed |

The Q6 cached run used the completed `Hy-MT2-1.8B-Q6_K.gguf` artifact and
recorded `server_shutdown_method=terminate_group`. The Q8 cached rerun used the
fixed sweep harness at commit `eed03f8`, normalized the completed Q8 artifact to
`Hy-MT2-1.8B-Q8_0.gguf`, dropped caches before the variant, and recorded
`server_port_closed_after_shutdown=true`. The code prompt outputs still need
human review before route use, so treat these as valid text/router smokes rather
than promotions.

Tencent text-suite 5-trial repeat evidence:

Run prefix: `tencent-text-repeat5-20260531a`. The repeat used the
then-configured seven Hy-MT1.5/Hy-MT2 text rows, locked clocks, cache drop
before each variant, `--trial-count 5`, `--max-tokens 64`,
`--fake-stream-max-frames 0`,
`--min-lfb-blocks 150`, and the canonical llama.cpp image
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`
with image id `36f3398b7885` and llama.cpp ref `d749821db3bd`.

| Variant | Preflight `lfb` | Server ready | Guard | Success | Trials | Startup s | Text tok/s | Text latency s | Max temp C | Avg power W | Avg GR3D % | Min lfb blocks | Shutdown | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `tencent-hy-mt1p5-1p8b-1p25bit-text-smoke` | 234x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `already_exited`, port closed | Failed before server ready: `invalid ggml type 42`. |
| `tencent-hy-mt1p5-1p8b-2bit-text-smoke` | 233x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `already_exited`, port closed | Failed before server ready: tensor offset `203248672`, expected `203129888`. |
| `tencent-hy-mt2-1p8b-1p25bit-text-smoke` | 232x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `already_exited`, port closed | Failed before server ready: `invalid ggml type 42`. |
| `tencent-hy-mt2-1p8b-2bit-text-smoke` | 232x4MB | no | n/a | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `already_exited`, port closed | Failed before server ready: tensor offset `203248672`, expected `203572256`. |
| `tencent-hy-mt2-1p8b-q4-text-smoke` | 232x4MB | yes | yes | 20/20 | 5 | 4.085 | 34.528 | 1.643 | 56.562 | 21.833 | 96.719 | 231 | `terminate_group`, port closed | Valid text/router repeat; fastest Hy-MT2 row. |
| `tencent-hy-mt2-1p8b-q6-text-smoke` | 234x4MB | yes | yes | 20/20 | 5 | 4.014 | 26.778 | 2.108 | 59.812 | 21.987 | 97.429 | 169 | `terminate_group`, port closed | Valid text/router repeat. |
| `tencent-hy-mt2-1p8b-q8-text-smoke` | 232x4MB | yes | yes | 20/20 | 5 | 4.013 | 31.395 | 1.806 | 60.656 | 22.541 | 95.472 | 101 | `terminate_group`, port closed | Valid text/router repeat. |

The repeat keeps Q4 as the fastest Hy-MT2 text row, Q8 second, and Q6 third
under this pinned llama.cpp image. All three profiled valid rows are
`gpu_compute` bottlenecked. This remains text/router evidence only; route use
still needs human output review and route-specific translation/router checks.

Tencent text-suite prepared repeat evidence:

Run prefix: `tencent-text-repeat5-prepctx-20260609T131412Z`. This refresh used
the strict remote Tencent text suite defaults on current `main`: locked clocks,
cache drop before each variant, `--trial-count 5`, `--max-tokens 64`,
`--fake-stream-max-frames 0`, `--min-lfb-blocks 150`, `Prepare ctx =
max_clocks, drop_caches`, and structured `Quality review` plus `Promotion
precheck`. All four scoped rows carried `Required lfb = 150`. The Hy-MT2 rows
passed formal and ranking checks but still failed promotion on output quality;
Youtu-LLM 2B Q8 is the first Tencent text row in this suite to clear both the
quality gate and `Promotion precheck = yes`.

| Variant | Prepare ctx | Preflight `lfb` | Required lfb | Ranking precheck | Promotion precheck | Quality review | Success | Startup s | Text tok/s | Text latency s | Prepare lfb delta | Prepare avail MB delta | Status |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `tencent-hy-mt2-1p8b-q4-text-smoke` | `max_clocks, drop_caches` | 190x4MB | 150 | yes | no (`quality_review_failed 15/20 text_en_reasoning_short`) | no (15/20; `text_en_reasoning_short`) | 20/20 | 4.013 | 34.511 | 1.643 | +93 | +958.605 | Fastest Hy-MT2 row, but not promotable under the current quality policy. |
| `tencent-hy-mt2-1p8b-q6-text-smoke` | `max_clocks, drop_caches` | 187x4MB | 150 | yes | no (`quality_review_failed 10/20 text_en_reasoning_short,text_code_short`) | no (10/20; `text_en_reasoning_short`, `text_code_short`) | 20/20 | 4.014 | 26.754 | 2.111 | +116 | +949.539 | Valid strict repeat; promotion blocked by reasoning and code excerpts. |
| `tencent-hy-mt2-1p8b-q8-text-smoke` | `max_clocks, drop_caches` | 183x4MB | 150 | yes | no (`quality_review_failed 10/20 text_en_reasoning_short,text_code_short`) | no (10/20; `text_en_reasoning_short`, `text_code_short`) | 20/20 | 5.015 | 31.408 | 1.806 | +114 | +954.520 | Second-fastest Hy-MT2 row; promotion blocked by the same review failures as Q6. |
| `tencent-youtu-llm-2b-q8-text-smoke` | `max_clocks, drop_caches` | 192x4MB | 150 | yes | yes (`Promotion precheck = yes`) | yes (20/20) | 20/20 | 354.568 | 24.577 | 1.865 | +121 | +954.090 | First strict Jetson evidence for Youtu-LLM 2B Q8. Startup includes first artifact download; do not compare it against cached-start rows. |

Youtu-LLM 2B Q8 cached rerun evidence:

Run prefix: `youtu-llm-q8-cached-20260609T133339Z`. This rerun kept the same
strict remote Tencent text suite policy as the prepared repeat above, but
reused the already-downloaded artifact. The lifecycle phase timing recorded
`artifact_check_or_download = cached (0.002 s)`, so the `5.015 s` startup below
is the useful cached-start datapoint for this route.

| Variant | Prepare ctx | Preflight `lfb` | Required lfb | Ranking precheck | Promotion precheck | Quality review | Success | Startup s | Text tok/s | Text latency s | Prepare lfb delta | Prepare avail MB delta | Artifact phase | Status |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `tencent-youtu-llm-2b-q8-text-smoke` | `max_clocks, drop_caches` | 198x4MB | 150 | yes | yes (`Promotion precheck = yes`) | yes (20/20) | 20/20 | 5.015 | 24.621 | 1.861 | +118 | +989.227 | `artifact_check_or_download = cached (0.002 s)` | Cached strict text/router evidence; still separate from VLM ranking because the row is text-only. |

## Next Model Checks

1. Do human full-output review for `lightweight-repeat5-20260531T134554Z`;
   structured quality review already keeps SmolVLM2 as latency_floor and narrows
   Qwen3-VL 2B to image/fake-stream balanced_candidate rather than text/code.
2. Use profiling next: capture memory/`lfb`, CPU/GPU/EMC, phase, and input
   pipeline evidence before selecting any deeper runtime or routing change.
3. Run the dedicated Tencent Hy-MT1.5/Hy-MT2/Youtu-LLM text suite only as a
   separate text/router study if it becomes useful for routing or translation
   pre/post-processing; Hy-MT2 Q4/Q6 have one valid full-suite smoke, Hy-MT2
   Q6/Q8 have cached single-run evidence, Hy-MT2 Q4/Q6/Q8 have valid 5-trial
   text repeat evidence, `tencent-text-repeat5-prepctx-20260609T131412Z`
   refreshes Hy-MT2 Q4/Q6/Q8 under strict `Required lfb = 150` plus
   `Prepare ctx = max_clocks, drop_caches`, Youtu-LLM 2B Q8 now has both the
   first-download strict row and cached rerun
   `youtu-llm-q8-cached-20260609T133339Z` with
   `artifact_check_or_download = cached (0.002 s)`, `Startup s = 5.015`, and
   `Promotion precheck = yes`, HY-MT1.5 Q4/Q6/Q8 rows still need Jetson
   evidence, and low-bit failures should feed the runtime/build lane, not VLM
   ranking.
4. Add a separate Youtu Q4 GPU-mmproj/offload canary only if memory allows and
   the result answers an artifact/runtime question; keep it
   distinct from the CPU-mmproj smoke and the official Tencent Q8 failure.
5. Revisit HunyuanOCR only through a bounded quality triage of artifact,
   prompt/template handling, or runtime path; do not rank the current Q8 smoke.
6. Promote none of these candidates until formal repeat evidence and output
   review agree on the intended role.
