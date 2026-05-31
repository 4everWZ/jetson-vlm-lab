# Jetson Remote Sweep Smoke - 2026-05-31

This document records the first end-to-end remote sweep run through
`scripts/jetson/run_remote_optimization_sweep.sh`. Raw outputs stayed under
ignored `outputs/optimization_sweeps/` paths on the Jetson worktree.

## Environment

| Field | Value |
|---|---|
| Local branch / commit | `bench/formal-jetson-infra` / `81d315c` |
| Jetson worktree | `~/code/jetson-vlm-lab-bench` |
| Jetson branch / commit | `bench/formal-jetson-infra` / `81d315c` |
| Remote helper | `scripts/jetson/remote_exec.sh` with ignored `.env.jetson` |
| Remote sweep wrapper | `scripts/jetson/run_remote_optimization_sweep.sh` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |

## Remote Dry-Run Validation

Run prefix: `remote-smoke-plan-20260531b`

The remote dry-run wrote:

```text
outputs/optimization_sweeps/remote-smoke-plan-20260531b/plan.json
```

The plan recorded the inherited launcher image in each variant `server_env`:

```text
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87
```

This validates that remote sweep plans now carry the pinned server image instead
of relying only on an implicit process environment.

## Smoke Attempts

| Run prefix | Variant | Gate / setup | Result |
|---|---|---|---|
| `remote-smoke-real-20260531a` | `minicpm-q4-baseline-b128-u32-kvq8` | `--pre-variant-command "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"` | skipped before preflight; `sudo: a password is required`, `preflight_reason=pre_variant_command_failed returncode 1` |
| `remote-smoke-real-20260531b` | `minicpm-q4-baseline-b128-u32-kvq8` | `--min-lfb-blocks 150` | skipped before Docker; `lfb_free_blocks 88 < required 150` |
| `remote-smoke-real-20260531c` | `minicpm-q4-baseline-b128-u32-kvq8` | no `lfb` gate, 1 trial, 64 tokens | completed formal benchmark and fake-stream sidecar |
| `dropcache-validate-minicpm-20260531c` | `minicpm-q4-baseline-b128-u32-kvq8` | `JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`, `JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, `--min-lfb-blocks 150` | completed formal benchmark and three-frame fake-stream sidecar; manifest records `pre_variant_command_passed=true` |
| `current-defaults-wrapper-smoke64-20260531a` | MiniCPM/Gemma current defaults | `scripts/jetson/run_remote_current_defaults_suite.sh`, `JETSON_REMOTE_SYNC=0`, 1 trial, 64 tokens, one fake-stream frame | completed both variants, generated `comparison.md`, and passed both guards |

## Successful Smoke Result

Run id: `remote-smoke-real-20260531c-minicpm-q4-baseline-b128-u32-kvq8`

Preflight memory state:

```text
lfb 88x4MB
```

| Guard | Formal success | Fake success | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s |
|---|---:|---:|---:|---:|---:|---:|---:|
| yes | 6/6 | 1/1 | 42.133 | 34.776 | 1.526 | 1.841 | 1.747 |

This is a remote execution smoke, not a promotion run. It used a low-contiguous
memory state and only one formal trial, so it should not replace the documented
MiniCPM baseline or be used to rank batch/ubatch candidates.

## Current Defaults Wrapper Smoke

Run prefix: `current-defaults-wrapper-smoke64-20260531a`

This validated `scripts/jetson/run_remote_current_defaults_suite.sh` end to
end on the Jetson bench worktree. The wrapper ran both current default variants
with `JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, `--min-lfb-blocks 150`, one
formal trial, 64 tokens, one fake-stream frame, then generated the comparison
report from the sweep manifest.

| Model | Variant | Preflight `lfb` | Guard | Formal success | Fake success | Startup s | Text tok/s | Image tok/s | Fake latency s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpm-q4-baseline-b128-u32-kvq8` | 269x4MB | yes | 6/6 | 1/1 | 6.027 | 47.466 | 37.840 | 1.657 |
| Gemma 4 E2B-it Q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8` | 260x4MB | yes | 6/6 | 1/1 | 6.019 | 11.773 | 9.418 | 6.257 |

This is wrapper validation, not a promotion run. A shorter 32-token probe
completed but failed the lightweight quality guard due missing canary terms, so
64 tokens should remain the minimum smoke setting when the guard result matters.

## llama.cpp Runtime Image Replacement Smoke

Run prefixes:

- `runtime-b4c0549-smoke64-20260531a`
- `runtime-d749821-smoke64-20260531a`

This checked whether a newer artifact-copy llama.cpp runtime image can replace
the current canonical Jetson image. Both runs used
`scripts/jetson/run_remote_current_defaults_suite.sh`, `JETSON_REMOTE_SYNC=0`,
`JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, `--min-lfb-blocks 150`, one
formal trial, 64 tokens, and one fake-stream frame.

Runtime images:

| Runtime | Docker tag | Image id | llama.cpp ref | Version output |
|---|---|---|---|---|
| Previous canonical | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` | `52a8ad644e41` | `b4c0549a49be` | `9352 (b4c0549a4)` |
| Candidate | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-d749821` | `36f3398b7885` | `d749821db3bd` | `9438 (d749821db)` |

The candidate image was built from the existing artifact-copy path. The Docker
build context was 137.69MB, matching the 132MB `artifacts/llama.cpp-install`
tree, and the runtime install tree contained only:

```text
/opt/llama.cpp/LLAMA_CPP_REF
/opt/llama.cpp/bin/llama-mtmd-cli
/opt/llama.cpp/bin/llama-server
```

The latest llama.cpp CMake configure still reported HTTPS disabled because
OpenSSL was not found, so the host-side HF download path remains necessary for
Hub-hosted GGUF candidates.

Cross-runtime comparison:

| Model | Runtime ref | Preflight `lfb` | Guard | Formal success | Fake success | Startup s | Text tok/s | Image tok/s | Fake latency s | Text delta | Image delta | Fake delta |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `b4c0549a49be` | 192x4MB | yes | 6/6 | 1/1 | 6.024 | 47.399 | 37.650 | 1.659 | +0.00% | +0.00% | +0.00% |
| MiniCPM-V 4.6 Q4 | `d749821db3bd` | 211x4MB | yes | 6/6 | 1/1 | 6.024 | 47.071 | 37.812 | 1.660 | -0.69% | +0.43% | +0.06% |
| Gemma 4 E2B-it Q4 | `b4c0549a49be` | 209x4MB | yes | 6/6 | 1/1 | 6.020 | 12.314 | 9.374 | 6.379 | +0.00% | +0.00% | +0.00% |
| Gemma 4 E2B-it Q4 | `d749821db3bd` | 214x4MB | yes | 6/6 | 1/1 | 6.019 | 12.383 | 9.812 | 6.074 | +0.56% | +4.67% | -4.78% |

Decision: the candidate passed the current-default smoke guard for both default
models and is eligible to replace the canonical Jetson llama.cpp tag. Treat the
speed deltas as smoke evidence only; run a 10-trial current-defaults suite
before making a performance-promotion claim.

Post-smoke registry action: the candidate image was retagged and pushed over
the canonical tag. Both
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` and
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-d749821` now resolve
locally on the Jetson to image id `36f3398b7885` and repo digest
`sha256:86dd1f9dd7bd0f42940c591142279cdf6c5659317486e0b12167571c2046bffa`.
The previous local image id `52a8ad644e41` was removed after the push.

## New Canonical 10-Trial Current Defaults

Run prefix: `current-defaults-d749821-clocks10-20260531a`

This reran the current-defaults wrapper after the canonical image tag was
overwritten and pulled from GHCR. The manifest records canonical runtime
metadata:

```text
image: ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87
image id: sha256:36f3398b7885ac62c5fcd335bde1d428a15d458997764135a555e16806e77036
repo digest: ghcr.io/4everwz/jetson-llama-cpp@sha256:86dd1f9dd7bd0f42940c591142279cdf6c5659317486e0b12167571c2046bffa
llama.cpp ref: d749821db3bd587932d1ed57d43626cd552c9909
```

Conditions: `JETSON_REMOTE_SYNC=0`, `JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, `--min-lfb-blocks 150`, ten
formal trials, 64 tokens, and three fake-stream frames.

| Model | Preflight `lfb` | Guard | Formal success | Fake success | Startup s | Text tok/s | Image tok/s | Fake latency s | Max temp C | Avg power W |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | 218x4MB | yes | 60/60 | 3/3 | 6.026 | 48.782 | 47.645 | 1.657 | 57.281 | 19.375 |
| Gemma 4 E2B-it Q4 | 223x4MB | yes | 60/60 | 3/3 | 6.020 | 12.016 | 12.948 | 5.820 | 56.937 | 16.409 |

Delta versus the earlier `current-defaults-clocks10-20260531a` canonical run:

| Model | Startup | Text tok/s | Image tok/s | Fake latency | Avg power |
|---|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | +0.02% | -0.29% | -0.30% | +0.18% | +0.03% |
| Gemma 4 E2B-it Q4 | +0.02% | -2.22% | -2.60% | -0.85% | -0.45% |

Decision: the new canonical image is valid for continued work and model
expansion, but the 10-trial results are not a performance promotion over the
previous canonical run. Keep the current MiniCPM/Gemma runtime parameters, and
use this image mainly for the newer llama.cpp multimodal/model support.

## Follow-Up

Before the next promotion comparison, run the remote wrapper with
`JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, and `--min-lfb-blocks 150` or
stricter. The earlier direct `sudo -n` pre-variant command failed on this
Jetson because sudo requires a password; the wrapper now feeds sudo over stdin
through a per-run FIFO so the manifest records the cache-drop command without
recording the password.
