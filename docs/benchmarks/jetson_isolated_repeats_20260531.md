# Jetson Isolated Optimization Repeats - 2026-05-31

This document records isolated remote repeats for MiniCPM-V 4.6 Q4 and Gemma 4
E2B-it Q4. Each run used `scripts/jetson/run_remote_optimization_sweep.sh` on
the Jetson bench worktree and started after a page-cache clear, so each variant
had a fresh contiguous-memory preflight instead of inheriting fragmentation from
the previous variant.

Raw JSONL, manifests, fake-stream sidecars, server logs, and reports stayed
under ignored `outputs/optimization_sweeps/` paths on the Jetson worktree.

## Environment

| Field | Value |
|---|---|
| Local branch / commit | `bench/formal-jetson-infra` / `c7c6ce3` |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Jetson branch / commit | `bench/formal-jetson-infra` / `c7c6ce3` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Max tokens | 64 |
| Temperature | 0 |
| Memory gate | `--min-lfb-blocks 150` |
| Cache clearing | `sync; echo 3 > /proc/sys/vm/drop_caches` before each isolated run |

The non-interactive `sudo -n` pre-variant command still fails on this Jetson
because sudo requires a password. For these isolated repeats, cache clearing was
performed before each one-variant run, then the sweep used the `lfb` gate to
avoid running under low contiguous-memory conditions.

## MiniCPM-V 4.6 Q4

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `minicpm-q4-baseline-b128-u32-kvq8` | `minicpm-iso-20260531a` | 229x4MB | 5 | yes | 30/30 | 1/1 | 44.624 | 43.047 | 1.437 | 1.503 | 1.752 |
| `minicpm-q4-b512-u128-kvq8` | `minicpm-iso-20260531b` | 235x4MB | 5 | yes | 30/30 | 1/1 | 44.486 | 42.951 | 1.442 | 1.505 | 1.739 |

Delta for `b512/u128` versus baseline:

| Metric | Delta |
|---|---:|
| Text throughput | -0.31% |
| Image throughput | -0.22% |
| Text latency | +0.35% |
| Image latency | +0.13% |
| Fake-stream latency | -0.74% |

Decision: keep `batch=128`, `ubatch=32` as the MiniCPM default. The
`batch=512`, `ubatch=128` candidate did not beat the baseline on formal text or
image throughput in the isolated 5-trial repeat.

## Gemma 4 E2B-it Q4

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-iso-20260531a` | 242x4MB | 3 | yes | 18/18 | 1/1 | 7.016 | 6.803 | 9.129 | 9.525 | 10.662 |
| `gemma-q4-gpu12-b256-u256-kvq8` | `gemma-iso-20260531b` | 242x4MB | 3 | yes | 18/18 | 1/1 | 7.048 | 6.866 | 9.090 | 9.384 | 10.671 |

Delta for `b256/u256` versus baseline:

| Metric | Delta |
|---|---:|
| Text throughput | +0.46% |
| Image throughput | +0.93% |
| Text latency | -0.43% |
| Image latency | -1.48% |
| Fake-stream latency | +0.08% |

Decision: keep `batch=512`, `ubatch=512` as the documented Gemma default for
now, but keep `batch=256`, `ubatch=256` as the next promotion candidate. This
isolated 3-trial repeat favors `b256/u256` on formal text and image latency, but
the margin is under 1% on throughput and earlier sweep evidence showed a larger
fake-stream latency regression. Promote only after a longer isolated repeat
confirms the gain and fake-stream latency remains effectively flat.

## Current Promotion State

| Model | Default after this repeat | Candidate to keep testing | Reason |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `batch=128`, `ubatch=32`, `N_GPU_LAYERS=32`, q8_0 KV cache | none ahead of baseline yet | isolated 5-trial repeat did not show a `b512/u128` throughput win |
| Gemma 4 E2B-it Q4 | `batch=512`, `ubatch=512`, `N_GPU_LAYERS=12`, q8_0 KV cache | `batch=256`, `ubatch=256`, `N_GPU_LAYERS=12` | isolated repeat slightly favors lower batch/ubatch, but margin is small and needs a longer fake-stream-stable repeat |
