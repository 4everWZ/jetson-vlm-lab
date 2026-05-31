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
| Local branch / commit | `bench/formal-jetson-infra` / `d01e905`, then `3ea75f5` for `b384/u384`, `c322312` for Flash Attention variants, `135900c` for memory mapping variants, `433c718` for `mlock` plus Docker memlock ulimit variants, `643d63c`/`316f999` for quality canaries, `f1c219e` for cache/continuous-batching variants, `7cee0f2` for prompt-cache variants, `ee8b604` for host/repack variants, `9a8b4e8` for startup timing capture, `ddb76ad` for DirectIO variants, and `62382e5` for max-clocks repeats |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Jetson branch / commit | `bench/formal-jetson-infra` / `d01e905`, then `3ea75f5` for `b384/u384`, `c322312` for Flash Attention variants, `135900c` for memory mapping variants, `433c718` for `mlock` plus Docker memlock ulimit variants, `643d63c`/`316f999` for quality canaries, `f1c219e` for cache/continuous-batching variants, `7cee0f2` for prompt-cache variants, `ee8b604` for host/repack variants, `9a8b4e8` for startup timing capture, `ddb76ad` for DirectIO variants, and `62382e5` for max-clocks repeats |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Max tokens | 64 |
| Temperature | 0 |
| Memory gate | `--min-lfb-blocks 150` |
| Cache clearing | `sync; echo 3 > /proc/sys/vm/drop_caches` before each isolated run |

The non-interactive `sudo -n` pre-variant command still fails on this Jetson
because sudo requires a password. For these isolated repeats, cache clearing was
performed before each one-variant run, then the sweep used the `lfb` gate to
avoid running under low contiguous-memory conditions.

The max-clocks repeat first ran `sudo jetson_clocks` and confirmed:

```text
cpu0-5 MinFreq=1728000 MaxFreq=1728000 CurrentFreq=1728000
GPU MinFreq=1020000000 MaxFreq=1020000000 CurrentFreq=1020000000
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=3199000000 FreqOverride=1
NV Power Mode: MAXN_SUPER
```

The formal wrapper's non-root `jetson_clocks --show` profile file still records
a permission error on this Jetson, so the root `jetson_clocks --show` command
above is the authoritative max-clocks confirmation for the max-clocks rows.

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
--repack, --no-repack
--no-host
--direct-io, --no-direct-io
```

Flash Attention and later infra candidates were added only after this help
output confirmed the relevant flags exist in the pinned Jetson image.

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

## Prompt Cache Candidates

These runs tested prompt-cache controls after server logs showed repeated prompt
cache updates. `--cache-ram 0` really disabled the prompt cache according to the
server log. `--no-cache-prompt` did not disable prompt-cache RAM updates in this
server path, so it should not be treated as equivalent to `--cache-ram 0`.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Note |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-cache-ram0` | `minicpm-cacheram0-3-20260531e` | 291x4MB | 3 | yes | 18/18 | 1/1 | 43.936 | 36.112 | 1.460 | 1.773 | 1.748 | prompt cache disabled |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-nocacheprompt` | `minicpm-nocacheprompt-3-20260531e` | 273x4MB | 3 | yes | 18/18 | 1/1 | 43.488 | 35.852 | 1.475 | 1.786 | 1.746 | prompt cache still updated |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-cache-ram0` | `gemma-cacheram0-3-20260531e` | 253x4MB | 3 | yes | 18/18 | 1/1 | 7.103 | 5.964 | 9.015 | 10.738 | 10.410 | prompt cache disabled |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-nocacheprompt` | `gemma-nocacheprompt-3-20260531e` | 270x4MB | 3 | yes | 18/18 | 1/1 | 6.783 | 5.857 | 9.448 | 10.944 | 10.649 | prompt cache still updated |

Delta versus each model's current default:

| Model | Variant | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM | `cache-ram0` | -1.54% | -16.11% | +1.60% | +17.96% | -0.23% |
| MiniCPM | `nocacheprompt` | -2.55% | -16.71% | +2.64% | +18.83% | -0.34% |
| Gemma | `cache-ram0` | -0.50% | -15.94% | +0.50% | +17.75% | +2.10% |
| Gemma | `nocacheprompt` | -4.99% | -17.45% | +5.33% | +20.01% | +4.44% |

Decision: keep prompt caching enabled. The cache update overhead is much smaller
than the image-path slowdown from disabling prompt cache RAM, and
`--no-cache-prompt` is not the right knob for disabling the observed cache
updates in this server build.

## Host Buffer and Repack Candidates

These runs tested the next confirmed pinned-image infra flags after
prompt-cache controls were negative. `--no-host` bypasses the host buffer, and
`--no-repack` disables llama.cpp weight repacking.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Note |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-nohost` | `minicpm-nohost-3-20260531f` | 268x4MB | 3 | yes | 18/18 | 1/1 | 44.181 | 41.416 | 1.452 | 1.567 | 1.794 | host buffer bypass |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-norepack` | `minicpm-norepack-3-20260531f` | 248x4MB | 3 | yes | 18/18 | 1/1 | 44.125 | 41.629 | 1.454 | 1.561 | 1.745 | weight repacking disabled |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-nohost` | `gemma-nohost-3-20260531f` | 246x4MB | 3 | no | 6/18 | 0/1 | n/a | n/a | n/a | n/a | n/a | CUDA OOM during request; partial report metrics are not comparable |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-norepack` | `gemma-norepack-3-20260531f` | 250x4MB | 3 | yes | 18/18 | 1/1 | 6.947 | 7.016 | 9.219 | 9.262 | 9.683 | single-frame fake latency looked better, then failed to repeat |

Delta versus each model's current default:

| Model | Variant | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM | `nohost` | -0.99% | -3.79% | +1.04% | +4.26% | +2.40% |
| MiniCPM | `norepack` | -1.12% | -3.29% | +1.18% | +3.86% | -0.40% |
| Gemma | `nohost` | guard failed | guard failed | guard failed | guard failed | guard failed |
| Gemma | `norepack` 3-trial | -2.69% | -1.11% | +2.78% | +1.57% | -5.03% |

Because the Gemma `--no-repack` 3-trial run only improved the single-frame
fake-stream metric, a focused 5-trial repeat compared the baseline and
`--no-repack`. The command requested `--fake-stream-max-frames 3`, but both the
local and Jetson `data/sample_stream` directories then contained only
`frame_001.png`, so fake-stream evidence is still one frame per variant.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-baseline-5fake3-20260531f` | 246x4MB | 5 | yes | 30/30 | 1/1 | 6.960 | 7.024 | 9.201 | 9.180 | 9.955 |
| `gemma-q4-baseline-gpu12-b512-u512-kvq8-norepack` | `gemma-norepack-5fake3-20260531f` | 243x4MB | 5 | yes | 30/30 | 1/1 | 6.926 | 7.045 | 9.248 | 9.168 | 10.236 |

5-trial delta for `--no-repack` versus the same-run baseline:

| Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---:|---:|---:|---:|---:|
| -0.49% | +0.30% | +0.51% | -0.13% | +2.82% |

Decision: do not promote `--no-host` or `--no-repack`. MiniCPM regresses on
formal throughput and latency. Gemma `--no-host` fails the quality guard after a
CUDA OOM during request processing. Gemma `--no-repack` is at best an image-only
micro-tradeoff in the 5-trial repeat and no longer improves fake-stream latency.

## Startup Timing Capture Validation

Commit `9a8b4e8` added per-variant server startup timing fields to the sweep
manifest:

- `server_started_at`
- `server_ready_at`
- `server_wait_seconds`
- `server_startup_seconds`

The validation run below only checks that the manifest records real timing on
the Jetson. It used one formal trial and is not a promotion-performance sample.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Server startup s | Text tok/s | Image tok/s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `minicpm-q4-baseline-b128-u32-kvq8` | `timing-validate-minicpm-20260531g` | 242x4MB | 1 | yes | 6/6 | 1/1 | 7.036 | 42.275 | 34.593 | 1.764 |

Decision: startup timing is now available for future load-path experiments such
as `--direct-io` / `--no-direct-io`. Those flags should be judged on
`server_startup_seconds` separately from steady-state benchmark throughput and
latency.

## DirectIO Candidates

These runs used the startup timing fields from commit `9a8b4e8` to separate
load-path behavior from steady-state benchmark metrics. The 3-trial run tested
the baseline, explicit `--direct-io`, and explicit `--no-direct-io` for both
models. A follow-up 5-trial run repeated Gemma baseline versus `--direct-io`
because the 3-trial Gemma signal was large enough to require confirmation.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8` | `minicpm-baseline-directio3-20260531h` | 189x4MB | 3 | yes | 18/18 | 1/1 | 7.037 | 44.189 | 41.414 | 1.451 | 1.569 | 1.734 |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-directio` | `minicpm-directio-3-20260531h` | 187x4MB | 3 | yes | 18/18 | 1/1 | 7.034 | 44.191 | 41.550 | 1.453 | 1.564 | 1.738 |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8-nodirectio` | `minicpm-nodirectio-3-20260531h` | 200x4MB | 3 | yes | 18/18 | 1/1 | 7.037 | 44.200 | 41.413 | 1.452 | 1.570 | 1.744 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-baseline-directio3-20260531h` | 206x4MB | 3 | yes | 18/18 | 1/1 | 7.032 | 6.961 | 6.697 | 9.199 | 9.672 | 10.132 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` | `gemma-directio-3-20260531h` | 244x4MB | 3 | yes | 18/18 | 1/1 | 8.041 | 7.020 | 7.159 | 9.121 | 9.084 | 9.497 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-nodirectio` | `gemma-nodirectio-3-20260531h` | 257x4MB | 3 | yes | 18/18 | 1/1 | 7.036 | 7.033 | 6.886 | 9.101 | 9.419 | 10.130 |

3-trial delta versus each same-run baseline:

| Model | Variant | Startup | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniCPM | `directio` | -0.04% | +0.00% | +0.33% | +0.14% | -0.32% | +0.23% |
| MiniCPM | `nodirectio` | -0.01% | +0.02% | -0.00% | +0.07% | +0.06% | +0.58% |
| Gemma | `directio` | +14.35% | +0.85% | +6.90% | -0.85% | -6.08% | -6.27% |
| Gemma | `nodirectio` | +0.06% | +1.03% | +2.82% | -1.07% | -2.62% | -0.02% |

The 5-trial Gemma repeat reduced the formal throughput delta, but it kept a
single-frame fake-stream latency improvement. Startup stayed about one second
slower with `--direct-io`.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-baseline-directio5-20260531h` | 262x4MB | 5 | yes | 30/30 | 1/1 | 7.034 | 7.135 | 7.058 | 8.976 | 9.149 | 10.356 |
| `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` | `gemma-directio-5-20260531h` | 252x4MB | 5 | yes | 30/30 | 1/1 | 8.040 | 7.149 | 7.073 | 8.958 | 9.078 | 9.897 |

5-trial delta for Gemma `--direct-io` versus the same-run baseline:

| Startup | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---:|---:|---:|---:|---:|---:|
| +14.30% | +0.20% | +0.21% | -0.20% | -0.78% | -4.43% |

Decision: do not change MiniCPM; DirectIO is noise-level there. Keep Gemma
`--direct-io` as a long-lived-server streaming candidate, not a default yet. It
passes the guard and does not hurt formal throughput in the 5-trial repeat, but
it increases startup time by about one second and the fake-stream evidence is
still one frame because that run used the old one-frame `data/sample_stream`
fixture. The next confirmation run should use the tracked three-frame fixture
with `--fake-stream-max-frames 3`.

The three-frame confirmation used commit `e189965`, which added
`data/sample_stream/frame_002.png` and `frame_003.png`. Each variant was run
separately after dropping page cache, with `--fake-stream-max-frames 3` and
`--min-lfb-blocks 150`.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-baseline-fake3-20260531i` | 250x4MB | 5 | yes | 30/30 | 3/3 | 7.035 | 6.975 | 7.142 | 9.183 | 9.060 | 9.889 |
| `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` | `gemma-directio-fake3-20260531i` | 242x4MB | 5 | yes | 30/30 | 3/3 | 8.038 | 6.968 | 7.090 | 9.198 | 9.064 | 9.691 |

Three-frame delta for Gemma `--direct-io` versus the same-fixture baseline:

| Startup | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---:|---:|---:|---:|---:|---:|
| +14.26% | -0.10% | -0.72% | +0.16% | +0.04% | -2.00% |

Per-frame fake-stream latencies:

| Variant | `frame_001.png` | `frame_002.png` | `frame_003.png` |
|---|---:|---:|---:|
| Baseline | 9.790 | 9.736 | 10.142 |
| DirectIO | 9.774 | 9.776 | 9.525 |

Decision: the multi-frame check weakens the DirectIO case. It still passes the
guard and improves average fake-stream latency by about 2%, but formal
throughput is slightly lower and startup remains about one second slower. Keep
Gemma `--direct-io` as an optional long-lived-server streaming candidate only;
do not promote it to the default runtime.

## Gemma GPU Offload And Flash Attention/KV Cache Combinations

Commit `7a7c070` added two Gemma combination candidates that pair Flash
Attention with lower-precision KV cache. This pass also rechecked the existing
`N_GPU_LAYERS=16` upper-offload probe with the current three-frame fixture path.

The `N_GPU_LAYERS=16` run failed before server ready even after dropping page
cache. This reproduced the earlier scheduler assertion under a clean preflight,
so this is a pinned-llama.cpp parameter incompatibility rather than a
contiguous-memory miss.

| Variant | Run prefix | Preflight `lfb` | Server ready | Server return code | Startup s | Evidence |
|---|---|---:|---|---:|---:|---|
| `gemma-q4-gpu16-b512-u512-kvq8` | `gemma-gpu16-3fake3-20260531a` | 272x4MB | no | 133 | n/a | `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed` |

The Flash Attention plus KV-cache-precision combinations both loaded and passed
the guard, but neither is a promotion candidate.

| Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma-q4-baseline-gpu12-b512-u512-kq4-vq8-faon` | `gemma-kq4vq8-faon-3fake3-20260531a` | 261x4MB | 3 | yes | 18/18 | 3/3 | 7.036 | 6.038 | 5.647 | 10.606 | 11.575 | 13.400 |
| `gemma-q4-baseline-gpu12-b512-u512-kvq4-faon` | `gemma-kvq4-faon-3fake3-20260531a` | 261x4MB | 3 | yes | 18/18 | 3/3 | 7.035 | 7.053 | 6.862 | 9.079 | 9.500 | 10.054 |

Delta versus the same three-frame baseline
`gemma-baseline-fake3-20260531i`:

| Variant | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---|---:|---:|---:|---:|---:|
| `kq4/vq8 + flash-attn` | -13.43% | -20.93% | +15.50% | +27.76% | +35.51% |
| `kvq4 + flash-attn` | +1.12% | -3.92% | -1.13% | +4.86% | +1.67% |

Decision: do not promote Flash Attention plus lower-precision KV cache as a
Gemma default. The K-only cache reduction combined with Flash Attention is
clearly slower. The full K/V q4 combination improves text latency slightly but
regresses image and fake-stream latency, which is the wrong tradeoff for the VLM
use case.

## Max-Clocks Repeat

This pass checked whether the earlier optimization rankings were limited by
Jetson dynamic clocks rather than llama.cpp flags. It ran after `sudo
jetson_clocks`, with cache dropped before each one-variant sweep and
`--min-lfb-blocks 150`.

| Model | Variant | Run prefix | Preflight `lfb` | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Max temp C | Avg power W |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8` | `minicpm-baseline-clocks10-20260531a` | 258x4MB | 10 | yes | 60/60 | 3/3 | 6.023 | 48.878 | 47.742 | 1.310 | 1.349 | 1.651 | 56.906 | 19.381 |
| MiniCPM-V 4.6 Q4 | `minicpm-q4-b512-u128-kvq8` | `minicpm-b512-clocks10-20260531a` | 246x4MB | 10 | yes | 60/60 | 3/3 | 6.024 | 48.761 | 47.666 | 1.313 | 1.350 | 1.640 | 57.468 | 19.396 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8` | `gemma-baseline-clocks5-20260531a` | 235x4MB | 5 | yes | 30/30 | 3/3 | 6.022 | 12.199 | 12.637 | 5.251 | 5.138 | 6.230 | 55.750 | 16.455 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` | `gemma-directio-clocks5-20260531a` | 247x4MB | 5 | yes | 30/30 | 3/3 | 7.025 | 12.149 | 12.721 | 5.273 | 5.102 | 5.771 | 56.125 | 16.444 |

Delta for MiniCPM `b512/u128` versus the max-clocks baseline:

| Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---:|---:|---:|---:|---:|
| -0.24% | -0.16% | +0.23% | +0.07% | -0.67% |

Delta for Gemma `--direct-io` versus the max-clocks baseline:

| Startup | Text tok/s | Image tok/s | Text latency | Image latency | Fake-stream latency |
|---:|---:|---:|---:|---:|---:|
| +16.65% | -0.41% | +0.66% | +0.42% | -0.70% | -7.37% |

Decision: max clocks are a benchmark prerequisite, not an optional tuning flag.
They materially improve both models, especially Gemma. Under max clocks, MiniCPM
still keeps `b128/u32` as the default because `b512/u128` does not improve
formal text or image throughput. Gemma keeps the baseline as the default for
cold-start or mixed workloads, while `--direct-io` becomes stronger as an
optional long-lived streaming flag: it improves three-frame fake-stream latency
by 7.37% with nearly flat formal throughput, but still adds about one second of
startup.

## Current Promotion State

| Model | Default after this repeat | Candidate to keep testing | Reason |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `sudo jetson_clocks` first, then `batch=128`, `ubatch=32`, `N_GPU_LAYERS=32`, q8_0 KV cache | none ahead of baseline yet | isolated 5-trial and max-clocks 10-trial repeats did not show a `b512/u128` formal throughput win |
| Gemma 4 E2B-it Q4 | `sudo jetson_clocks` first, then `batch=512`, `ubatch=512`, `N_GPU_LAYERS=12`, q8_0 KV cache | `--direct-io` only for optional long-lived streaming workloads | Batch, Flash Attention, DirectIO, and Flash Attention plus lower-precision KV are tradeoffs; `N_GPU_LAYERS=16`, memory mapping, locking, cache precision, no-continuous-batching, prompt-cache, host-buffer, and repack variants are not default-promotion candidates |
