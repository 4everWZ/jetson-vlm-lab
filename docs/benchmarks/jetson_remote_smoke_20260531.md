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

## Follow-Up

Before the next promotion comparison, run the remote wrapper with
`JETSON_REMOTE_PREPARE_MAX_CLOCKS=1`,
`JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1`, and `--min-lfb-blocks 150` or
stricter. The earlier direct `sudo -n` pre-variant command failed on this
Jetson because sudo requires a password; the wrapper now feeds sudo over stdin
through a per-run FIFO so the manifest records the cache-drop command without
recording the password.
