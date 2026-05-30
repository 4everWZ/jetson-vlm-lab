# Jetson Optimization Sweep - 2026-05-30

This document records the first optimization sweep using
`scripts/jetson/run_optimization_sweep.sh` from commit `aa3a11a`.

Raw benchmark JSONL, fake-stream JSONL, server logs, manifests, and tegrastats
profiles stayed under ignored `outputs/optimization_sweeps/` paths on the
Jetson worktree.

## Environment

| Field | Value |
|---|---|
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Git branch / commit | `bench/formal-jetson-infra` / `aa3a11a` |
| Model root | `~/code/jetson-vlm-lab/models` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Sweep wrapper | `scripts/jetson/run_optimization_sweep.sh` |
| Trial count | 3 |
| Max tokens | 64 |
| Temperature | 0 |
| Fake-stream check | 1 frame from `data/sample_stream` |

Before the Gemma sweep, page cache was dropped because `tegrastats` showed
`lfb 80x4MB` after the MiniCPM sweep. After `sync; echo 3 >
/proc/sys/vm/drop_caches`, `lfb` improved to `150x4MB`.

## MiniCPM-V 4.6 Q4 Sweep

Run prefix: `minicpm-opt-20260530a`

| Rank | Variant | Batch | UBatch | Guard | Formal success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake-stream latency s |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `minicpm-q4-b512-u128-kvq8` | 512 | 128 | yes | 18/18 | 44.231 | 41.780 | 1.450 | 1.551 | 1.711 |
| 2 | `minicpm-q4-b256-u64-kvq8` | 256 | 64 | yes | 18/18 | 44.106 | 41.753 | 1.455 | 1.551 | 1.761 |
| 3 | `minicpm-q4-baseline-b128-u32-kvq8` | 128 | 32 | yes | 18/18 | 44.066 | 41.486 | 1.457 | 1.565 | 1.754 |

The `b512/u128` MiniCPM candidate is the current fastest observed MiniCPM
setting in this repo. The gain over the `b128/u32` baseline is small but
consistent in this sweep: text throughput +0.37%, image throughput +0.71%, and
one-frame fake-stream latency improved from 1.754s to 1.711s.

Do not treat this as final saturation yet; the margin is small enough that the
candidate should be repeated in a longer thermal run before replacing the
documented baseline everywhere.

## Gemma 4 E2B-it Q4 Sweep

Run prefix: `gemma-opt-20260530a`

| Rank | Variant | GPU layers | Batch | UBatch | Guard | Formal success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake-stream latency s |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `gemma-q4-gpu12-b256-u256-kvq8` | 12 | 256 | 256 | yes | 18/18 | 7.063 | 7.008 | 9.066 | 9.277 | 10.864 |
| 2 | `gemma-q4-baseline-gpu12-b512-u512-kvq8` | 12 | 512 | 512 | yes | 18/18 | 7.073 | 6.951 | 9.056 | 9.347 | 9.664 |
| - | `gemma-q4-gpu16-b512-u512-kvq8` | 16 | 512 | 512 | no | not started |  |  |  |  |  |

The formal optimization report ranked `b256/u256` first because it slightly
improved image throughput, but it also made one-frame fake-stream latency worse
than the `b512/u512` baseline. It should not replace the baseline yet.

`N_GPU_LAYERS=16` failed during server startup with llama.cpp scheduler assert:

```text
GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

This is not a promotable Gemma path on the pinned container image.

## Current Promotion State

| Model | Current best promotable candidate | Reason |
|---|---|---|
| MiniCPM-V 4.6 Q4 | `batch=512`, `ubatch=128`, `N_GPU_LAYERS=32`, `CTX_SIZE=512`, q8_0 KV cache | Passed formal benchmark and fake-stream sanity with small speed gain over baseline |
| Gemma 4 E2B-it Q4 | keep `batch=512`, `ubatch=512`, `N_GPU_LAYERS=12`, `CTX_SIZE=512`, q8_0 KV cache | `b256/u256` is mixed, and `gpu16` fails startup |

## Next Optimization Work

1. Repeat MiniCPM `b128/u32` vs `b512/u128` with a longer run and thermal notes.
2. Use the updated sweep preflight JSON for future runs so `tegrastats` `lfb`
   is recorded before each Gemma variant and failed starts can be labeled as
   memory-state-sensitive or parameter-incompatible.
3. Check the pinned llama.cpp server help inside the container before adding
   speculative acceleration flags such as flash attention or mmap/mlock changes.
4. For Gemma, next useful variants are still within `N_GPU_LAYERS=12`; test
   prompt/cache/batch behavior before raising GPU layers again.

## Follow-Up Validation

Commit `eaed9e6` added per-variant preflight JSON and fake-stream-aware
optimization reports. A focused MiniCPM validation on the Jetson worktree showed
why this matters:

| Run prefix | Variant | Max tokens | Preflight `lfb` | Server ready | Report guard | Evidence |
|---|---|---:|---|---|---|---|
| `preflight-validate-20260530` | `minicpm-q4-b512-u128-kvq8` | 16 | `71x4MB` | no | no report | server log shows CUDA allocation failure while loading 1057.36 MiB mmproj buffer |
| `preflight-validate-20260530b` | `minicpm-q4-b512-u128-kvq8` | 16 | `197x4MB` | yes | no | quality guard rejected `text_cn_short:short_output` |
| `preflight-validate-20260530c` | `minicpm-q4-b512-u128-kvq8` | 64 | `216x4MB` | yes | yes | report includes text/image throughput, fake-stream latency 1.773s, and `Fake success` 1/1 |

This confirms two constraints for the next optimization pass:

1. Low `lfb` can make even MiniCPM fail startup, so preflight memory state must
   be recorded for every promoted run.
2. Too-small `max_tokens` can inflate throughput while producing unusably short
   answers, so 64 tokens remains the minimum current benchmark setting for
   promotion candidates.
