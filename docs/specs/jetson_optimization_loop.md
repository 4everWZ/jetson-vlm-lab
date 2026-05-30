# Jetson Optimization Loop

## Objective

Find the fastest usable MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 settings on the
Jetson without accepting obviously degraded output. The loop optimizes server
parameters, records formal benchmark evidence, and ranks only runs that pass a
lightweight output sanity guard.

## Guardrail

`python -m edge_vlm.optimization report` ranks benchmark JSONL files, and can
also read matching fake-stream JSONL files, only after checking:

- every non-placeholder benchmark record succeeded
- output excerpts are not empty
- output excerpts meet a minimum length
- output excerpts are not dominated by obvious repetition
- fake-stream frame outputs pass the same lightweight output checks when
  provided

This is a sanity guard, not a full quality evaluation. A high-speed candidate
that fails the guard is kept in the report but excluded from ranked candidates.

## Variant Source

Variants live in `configs/benchmark/jetson_optimization_variants.jsonl`.

Current sweep knobs are deliberately narrow:

- `CTX_SIZE`
- `N_GPU_LAYERS`
- `LLAMA_BATCH_SIZE`
- `LLAMA_UBATCH_SIZE`
- llama.cpp server args already observed in the repo: `--parallel`,
  `--batch-size`, `--ubatch-size`, `--cache-type-k`, `--cache-type-v`,
  `--no-warmup`, and Gemma `-fit off`

Do not add speculative llama.cpp flags until the container help or a dry-run
command confirms the flag exists in the pinned Jetson image.

## Dry Run

Use dry-run mode before launching Docker:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --dry-run \
  --plan-output outputs/optimization_sweeps/plan.json \
  --run-prefix opt-plan \
  --model minicpmv46-q4
```

The plan records server commands, benchmark output paths, preflight output
paths, environment overrides, and fake-stream commands. The sweep sets
`DOCKER_TTY=0` so Docker can run under background automation instead of
requiring an interactive terminal.

## Real Sweep

Run a small sweep first:

```bash
PYTHON_BIN=python3 scripts/jetson/run_optimization_sweep.sh \
  --run-prefix minicpm-opt-001 \
  --model minicpmv46-q4 \
  --trial-count 3 \
  --max-tokens 64 \
  --temperature 0
```

Outputs stay under ignored `outputs/optimization_sweeps/<run-prefix>/`:

- `benchmarks/*.jsonl`
- `benchmarks/*.md`
- `benchmarks/*.manifest.json`
- `fake_stream/*.jsonl`
- `preflight/*.preflight.json`
- `server_logs/*.server.log`
- `optimization_report.md`
- `<run-prefix>.manifest.json`

Each variant captures a preflight JSON file before server startup. On Jetson,
this includes `/proc/meminfo` and a short `tegrastats` sample with parsed
`lfb` when available. Use this to distinguish memory-state-sensitive startup
failures from parameter-incompatible failures.

## Promotion Rule

A candidate can become the new baseline only when:

1. the formal benchmark completes successfully
2. the one-frame fake-stream check completes successfully
3. the optimization report marks the candidate guard as `yes`
4. its throughput or latency improves over the prior baseline
5. the fake-stream latency is not worse enough to invalidate the use case
6. the exact server parameters and Jetson memory notes are documented in a
   tracked benchmark result file

If a faster run fails the sanity guard, keep it as a failed optimization
candidate. Do not promote it.
