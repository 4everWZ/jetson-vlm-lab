# Jetson Lightweight Model Expansion - 2026-05-31

This document records the first lightweight-model checks after the MiniCPM-V
4.6 Q4 and Gemma 4 E2B-it Q4 infra sweeps. Raw JSONL, fake-stream JSONL,
manifests, server logs, and downloaded model artifacts stayed under ignored
Jetson paths.

## Environment

| Field | Value |
|---|---|
| Local branch / commit | `bench/formal-jetson-infra` / `94154a4` |
| Jetson branch / commit | `bench/formal-jetson-infra` / `94154a4` |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Model root | `/home/weizheng/code/jetson-vlm-lab/models` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Trial count | 1 |
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

## Next Model Checks

1. Run `qwen3-vl-2b-thinking-q4-smoke` with the same 1-trial, 64-token,
   three-frame fake-stream protocol.
2. Run `youtu-vl-4b-q8-smoke` only after confirming enough contiguous memory,
   because the Q8 model is much larger than SmolVLM2 256M.
3. Promote none of these candidates until a 3- or 5-trial formal repeat passes
   the guard and preserves acceptable output quality.
