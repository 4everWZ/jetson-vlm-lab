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
| Local branch / commit | `bench/formal-jetson-infra` / `d01e905`, then `3ea75f5` for `b384/u384`, `c322312` for Flash Attention variants, `135900c` for memory mapping variants, `433c718` for `mlock` plus Docker memlock ulimit variants, `643d63c`/`316f999` for quality canaries, and `f1c219e` for cache/continuous-batching variants |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Jetson branch / commit | `bench/formal-jetson-infra` / `d01e905`, then `3ea75f5` for `b384/u384`, `c322312` for Flash Attention variants, `135900c` for memory mapping variants, `433c718` for `mlock` plus Docker memlock ulimit variants, `643d63c`/`316f999` for quality canaries, and `f1c219e` for cache/continuous-batching variants |
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

## Pinned llama-server Flag Check

The pinned image help was captured from the Jetson container into ignored
`outputs/jetson_inspect/llama-server-help-20260531.txt`. Relevant supported
flags include:

```text
-fa, --flash-attn [on|off|auto]
--mlock
--mmap, --no-mmap
--cache-type-k TYPE
--cache-type-v TYPE
--cont-batching, --no-cont-batching
--cache-ram N
--cache-prompt, --no-cache-prompt
```

Flash Attention candidates were added only after this help output confirmed the
flag exists in the pinned Jetson image.

## Quality Canary Guard

Commit `643d63c` added optional `quality_terms_any` terms to prompt cases and
to the optimization report guard. Commit `316f999` broadened the Chinese prompt
case with English resource terms after Gemma's visible excerpt started with an
English thinking trace. The canary is intentionally weak: it rejects obviously
off-topic or collapsed answers before speed ranking, but it is not a full
quality evaluation.

## Gemma 4 E2B-it Q4 - 3-Trial Repeat

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

## Gemma 4 E2B-it Q4 - 5-Trial Batch Search

The longer repeat tested the baseline, the faster formal-latency candidate, and
a midpoint candidate added in commit `3ea75f5`.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-iso5-20260531a` | 247x4MB | 5 | yes | 30/30 | 1/1 | 7.139 | 7.095 | 8.970 | 9.119 | 10.196 |
| `gemma-q4-gpu12-b384-u384-kvq8` | `gemma-iso5-20260531c` | 258x4MB | 5 | yes | 30/30 | 1/1 | 7.108 | 7.052 | 9.010 | 9.148 | 10.094 |
| `gemma-q4-gpu12-b256-u256-kvq8` | `gemma-iso5-20260531b` | 255x4MB | 5 | yes | 30/30 | 1/1 | 7.155 | 7.193 | 8.952 | 8.973 | 10.465 |

Delta versus baseline:

| Variant | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---:|---:|---:|---:|---:|
| `b384/u384` | -0.43% | -0.61% | +0.45% | +0.32% | -1.00% |
| `b256/u256` | +0.22% | +1.38% | -0.20% | -1.60% | +2.64% |

Decision: keep `batch=512`, `ubatch=512` as the Gemma default. The lower
`b256/u256` setting is best for formal text/image throughput but hurts
fake-stream latency. The midpoint `b384/u384` improves fake-stream latency but
regresses formal text/image throughput. Neither is a clean promotion candidate.

## Flash Attention Candidates

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-faon` | `minicpm-faon-20260531a` | 257x4MB | 5 | yes | 30/30 | 1/1 | 44.541 | 42.892 | 1.440 | 1.508 | 1.750 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-faon` | `gemma-faon5-20260531a` | 283x4MB | 5 | yes | 30/30 | 1/1 | 7.055 | 7.211 | 9.081 | 9.008 | 10.256 |

Delta versus each model's current default:

| Model | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---:|---:|---:|---:|---:|
| MiniCPM Flash Attention on | -0.19% | -0.36% | +0.21% | +0.33% | -0.11% |
| Gemma Flash Attention on | -1.18% | +1.64% | +1.24% | -1.22% | +0.59% |

Decision: do not promote `--flash-attn on` as a default. MiniCPM regresses
formal text/image throughput. Gemma improves image throughput and image latency
but regresses text throughput and fake-stream latency, so it is a tradeoff
rather than a clean acceleration.

## Memory Mapping and Locking Candidates

These exploratory runs tested `--mlock`, `--no-mmap`, and `--mlock` with Docker
`--ulimit memlock=-1:-1`. The plain `--mlock` runs started, but the server logs
showed `RLIMIT_MEMLOCK` failures, so those rows are not valid measurements of
actual locked-memory behavior.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Note |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-mlock` | `minicpm-mlock3-20260531b` | 283x4MB | 3 | yes | 18/18 | 1/1 | 44.286 | 41.508 | 1.448 | 1.565 | 1.738 | `mlock` failed due `RLIMIT_MEMLOCK` |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-mlock-ulimit` | `minicpm-mlockulimit3-20260531b` | 310x4MB | 3 | yes | 18/18 | 1/1 | 44.317 | 41.591 | 1.448 | 1.563 | 1.751 | no `mlock` warning |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-nommap` | `minicpm-nommap3-20260531b` | 278x4MB | 3 | yes | 18/18 | 1/1 | 44.111 | 41.558 | 1.454 | 1.563 | 1.741 | started cleanly |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock` | `gemma-mlock3-20260531b` | 279x4MB | 3 | yes | 18/18 | 1/1 | 6.989 | 6.907 | 9.161 | 9.365 | 10.174 | `mlock` failed due `RLIMIT_MEMLOCK` |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock-ulimit` | `gemma-mlockulimit3-20260531b` | 296x4MB | 3 | yes | 18/18 | 1/1 | 6.856 | 6.898 | 9.337 | 9.457 | 9.731 | 3-trial fake latency improved, then failed to repeat |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock-ulimit` | `gemma-mlockulimit5-20260531b` | 312x4MB | 5 | yes | 30/30 | 1/1 | 6.985 | 7.005 | 9.169 | 9.223 | 10.656 | no `mlock` warning |

Gemma `--no-mmap` is not viable on this pinned image and memory state:

| Variant | Run prefix | Preflight `lfb` | Server ready | Server return code | Evidence |
|---|---|---:|---|---:|---|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8-nommap` | `gemma-nommap3-20260531b` | 286x4MB | no | 133 | server log shows `cudaMalloc failed: out of memory` while allocating a 317.05 MiB CUDA buffer |

Delta for the valid `mlock+ulimit` rows versus each model's current default:

| Model | Run prefix | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM `mlock+ulimit` 3-trial | `minicpm-mlockulimit3-20260531b` | -0.69% | -3.38% | +0.77% | +3.99% | -0.06% |
| Gemma `mlock+ulimit` 5-trial | `gemma-mlockulimit5-20260531b` | -2.16% | -1.27% | +2.22% | +1.14% | +4.51% |

Decision: do not promote `--mlock`, `--mlock` with raised Docker memlock
ulimit, or `--no-mmap` as defaults. Plain `--mlock` was not a valid test until
the Docker ulimit was raised; with the ulimit raised it still failed to improve
formal throughput. Gemma `--no-mmap` fails startup under high `lfb`, so keep it
out of promotion sweeps unless a future memory/layout change makes it relevant.

## Cache Precision and Continuous Batching Candidates

These runs tested lower-precision KV cache and disabling continuous batching.
The `kq4-vq8` variants failed during server startup on both models with the
pinned image reporting that a quantized V cache requires Flash Attention.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Note |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kq4-vq8` | `minicpm-kq4vq8-3-20260531c` | 314x4MB | 0 | n/a | n/a | n/a |  |  |  |  |  | server rc 139 |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq4` | `minicpm-kvq4-3-20260531c` | 304x4MB | 3 | yes | 18/18 | 1/1 | 44.153 | 41.330 | 1.452 | 1.571 | 1.781 | lower K/V cache precision |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-nocb` | `minicpm-nocb-3-20260531c` | 298x4MB | 3 | yes | 18/18 | 1/1 | 44.258 | 41.637 | 1.450 | 1.560 | 1.754 | continuous batching disabled |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kq4-vq8` | `gemma-kq4vq8-3-20260531c` | 295x4MB | 0 | n/a | n/a | n/a |  |  |  |  |  | server rc 139 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq4` | `gemma-kvq4-3-20260531d` | 294x4MB | 3 | yes | 18/18 | 1/1 | 7.120 | 7.174 | 8.997 | 9.055 | 10.611 | lower K/V cache precision |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-nocb` | `gemma-nocb-3-20260531d` | 298x4MB | 3 | yes | 18/18 | 1/1 | 6.930 | 6.743 | 9.239 | 9.629 | 11.080 | continuous batching disabled |

Delta versus each model's current default:

| Model | Variant | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM | `kvq4` | -1.06% | -3.99% | +1.04% | +4.52% | +1.66% |
| MiniCPM | `nocb` | -0.82% | -3.28% | +0.90% | +3.79% | +0.11% |
| Gemma | `kvq4` | -0.27% | +1.11% | +0.30% | -0.70% | +4.07% |
| Gemma | `nocb` | -2.93% | -4.96% | +3.00% | +5.59% | +8.67% |

Decision: do not promote lower-precision KV cache or `--no-cont-batching` as
defaults. MiniCPM regresses across formal metrics. Gemma `kvq4` improves image
throughput in a 3-trial repeat but slows fake-stream latency more than the
existing Flash Attention image-only tradeoff, so it is not a better candidate.
Disabling continuous batching is negative for Gemma and not useful for MiniCPM.

## Current Promotion State

| Model | Default after this repeat | Candidate to keep testing | Reason |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `batch=128`, `ubatch=32`, `N_GPU_LAYERS=32`, q8_0 KV cache | none ahead of baseline yet | isolated 5-trial repeat did not show a `b512/u128` throughput win |
| Gemma 4 E2B-it Q4 | `batch=512`, `ubatch=512`, `N_GPU_LAYERS=12`, q8_0 KV cache | `batch=256`, `ubatch=256` for formal throughput; `batch=384`, `ubatch=384` for fake-stream latency; Flash Attention for image-only workloads | Batch and Flash Attention variants are tradeoffs; memory mapping, locking, cache precision, and no-continuous-batching variants are not promotion candidates |
