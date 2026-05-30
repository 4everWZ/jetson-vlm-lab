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

## Follow-Up

Before the next promotion comparison, clear page cache or otherwise restore
contiguous free blocks, then run the remote wrapper with `--min-lfb-blocks 150`
or stricter. The current non-interactive `sudo -n` pre-variant command fails on
this Jetson because sudo requires a password.
