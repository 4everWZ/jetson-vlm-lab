# Jetson Profiling Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first profiling harness layer that turns Jetson `tegrastats`
logs into structured profile JSONL samples and summaries linked from sweep
manifests and comparison reports.

**Architecture:** Keep runtime inference outside Python. Add a focused parser
module for Jetson profile logs, wire it into the existing sweep/optimization
reporting path, and preserve JSONL/manifests as source of truth. The benchmark
runner continues to collect raw logs; the new code parses, summarizes, and
links derived artifacts without moving inference into Python.

**Tech Stack:** Python standard library, `unittest`, existing
`edge_vlm.jetson_sweep`, `edge_vlm.optimization`, Jetson `tegrastats` logs.

---

## File Structure

- Create `src/edge_vlm/jetson_profile.py`: parse individual `tegrastats` lines,
  summarize log files, derive conservative bottleneck labels, and write profile
  JSONL plus summary JSON artifacts.
- Modify `src/edge_vlm/optimization.py`: replace the local temp/power-only
  parser with `edge_vlm.jetson_profile`, and add GR3D/EMC/lfb/bottleneck fields
  to sweep comparison rows.
- Modify `src/edge_vlm/jetson_sweep.py`: add per-variant profile summary paths
  under `outputs/optimization_sweeps/<run-prefix>/profiles/`, generate profile
  summaries after benchmark runs, and link them from the sweep result manifest.
- Modify `tests/test_edge_vlm.py`: add parser, summary, manifest-linking, and
  comparison-report coverage.
- Modify `docs/specs/jetson_optimization_loop.md` and
  `docs/benchmark_protocol.md`: document the profile summary artifact and the
  new comparison columns.

## Scope Corrections From Review

The first implementation slice covers structured `tegrastats` parsing,
`profile.jsonl`, summary JSON, available phase timings, and comparison report
columns. It does not yet fully satisfy the whole strategy spec. These
requirements remain explicit follow-up tasks:

- Instrument `artifact_check_or_download` inside launchers so first-run
  downloads and cached startup are separated.
- Instrument `warmup` and `shutdown` instead of leaving them as `not_recorded`.
- Input-pipeline timing for image read, MIME detection, base64 encoding, JSON
  serialization, and HTTP request/response is implemented in the client path;
  fake-stream scheduling/backpressure timing remains open.
- Add evidence-backed `input_payload` and `runtime_overhead` labels after input
  and server-side timing data exist.
- Add time stamps to profile JSONL records when the raw `tegrastats` line does
  not include a parseable timestamp.

## Task 1: Structured Tegrastats Parser

**Files:**
- Create: `src/edge_vlm/jetson_profile.py`
- Modify: `tests/test_edge_vlm.py`

- [x] **Step 1: Write the failing parser test**

Add this test near the existing Jetson sweep parser tests in
`tests/test_edge_vlm.py`:

```python
def test_jetson_profile_parses_core_tegrastats_fields(self):
    from edge_vlm.jetson_profile import parse_tegrastats_line

    sample = parse_tegrastats_line(
        "05-31-2026 RAM 2100/7620MB (lfb 180x4MB) "
        "SWAP 12/3810MB (cached 4MB) CPU [10%@1728,off,35%@1728] "
        "GR3D_FREQ 89%@[1020] EMC_FREQ 76%@3199 "
        "cpu@52.0C gpu@54.5C tj@55.0C "
        "VDD_IN 17400mW/16800mW VDD_CPU_GPU_CV 8900mW/8200mW"
    )

    self.assertEqual(sample["ram"], {"used_mb": 2100, "total_mb": 7620})
    self.assertEqual(sample["swap"], {"used_mb": 12, "total_mb": 3810, "cached_mb": 4})
    self.assertEqual(sample["lfb"], {"free_blocks": 180, "block_mb": 4})
    self.assertEqual(sample["cpu"]["cores"][0], {"state": "online", "util_pct": 10, "freq_mhz": 1728})
    self.assertEqual(sample["cpu"]["cores"][1], {"state": "off", "util_pct": None, "freq_mhz": None})
    self.assertEqual(sample["gr3d"], {"util_pct": 89, "freq_mhz": 1020})
    self.assertEqual(sample["emc"], {"util_pct": 76, "freq_mhz": 3199})
    self.assertEqual(sample["temps_c"]["gpu"], 54.5)
    self.assertEqual(sample["power_mw"]["VDD_IN"], {"instant": 17400, "average": 16800})
    self.assertEqual(sample["power_mw"]["VDD_CPU_GPU_CV"], {"instant": 8900, "average": 8200})
```

- [x] **Step 2: Run the single test to verify RED**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k jetson_profile_parses_core_tegrastats_fields
```

Expected: fail with `ModuleNotFoundError: No module named
'edge_vlm.jetson_profile'` or an import error for `parse_tegrastats_line`.

- [x] **Step 3: Implement the parser**

Create `src/edge_vlm/jetson_profile.py` with regex-based parsers for:

```python
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator

RAM_RE = re.compile(r"\bRAM\s+(?P<used>\d+)/(?P<total>\d+)MB(?:\s+\(lfb\s+(?P<lfb_blocks>\d+)x(?P<lfb_mb>\d+)MB\))?")
SWAP_RE = re.compile(r"\bSWAP\s+(?P<used>\d+)/(?P<total>\d+)MB(?:\s+\(cached\s+(?P<cached>\d+)MB\))?")
CPU_RE = re.compile(r"\bCPU\s+\[(?P<cores>[^\]]*)\]")
ENGINE_RE = re.compile(r"\b(?P<name>GR3D_FREQ|EMC_FREQ)\s+(?P<util>\d+)%?(?:@\[?(?P<freq>\d+)\]?)?")
TEMP_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_]+)@(?P<temp>[0-9]+(?:\.[0-9]+)?)C\b")
POWER_RE = re.compile(r"\b(?P<rail>VDD_[A-Z0-9_]+)\s+(?P<instant>\d+)mW/(?P<average>\d+)mW\b")

def parse_tegrastats_line(line: str) -> dict[str, Any]:
    sample: dict[str, Any] = {"raw": line}
    return sample
```

The implementation must return `None` values for unavailable per-core CPU
fields and omit top-level keys only when the source line does not contain that
metric.

- [x] **Step 4: Run the single test to verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k jetson_profile_parses_core_tegrastats_fields
```

Expected: one matching test passes.

- [x] **Step 5: Commit Task 1**

Run:

```bash
git add src/edge_vlm/jetson_profile.py tests/test_edge_vlm.py
git commit -m "feat: parse jetson profile samples"
```

## Task 2: Profile Log Summary And Bottleneck Labels

**Files:**
- Modify: `src/edge_vlm/jetson_profile.py`
- Modify: `tests/test_edge_vlm.py`

- [x] **Step 1: Write the failing summary test**

Add this test after the Task 1 test:

```python
def test_jetson_profile_summarizes_log_and_labels_bottlenecks(self):
    from edge_vlm.jetson_profile import summarize_tegrastats_log

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "tegrastats.log"
        log.write_text(
            "\n".join(
                [
                    "RAM 2000/7620MB (lfb 200x4MB) CPU [40%@1728] GR3D_FREQ 92%@[1020] EMC_FREQ 81%@3199 gpu@54.0C VDD_IN 18000mW/17000mW",
                    "RAM 2200/7620MB (lfb 160x4MB) CPU [45%@1728] GR3D_FREQ 88%@[1020] EMC_FREQ 86%@3199 gpu@58.0C VDD_IN 19000mW/18000mW",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = summarize_tegrastats_log(log)

    self.assertEqual(summary["samples"], 2)
    self.assertEqual(summary["min_lfb_free_blocks"], 160)
    self.assertEqual(summary["max_temp_c"], 58.0)
    self.assertEqual(summary["avg_power_w"], 18.5)
    self.assertEqual(summary["avg_gr3d_util_pct"], 90.0)
    self.assertEqual(summary["avg_emc_util_pct"], 83.5)
    self.assertIn("gpu_compute", summary["bottleneck_labels"])
    self.assertIn("emc_memory_bandwidth", summary["bottleneck_labels"])
```

- [x] **Step 2: Run the single test to verify RED**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k jetson_profile_summarizes_log_and_labels_bottlenecks
```

Expected: fail because `summarize_tegrastats_log` is not implemented.

- [x] **Step 3: Implement summary helpers**

Add `iter_tegrastats_samples(path)`, `summarize_tegrastats_samples(samples)`,
`summarize_tegrastats_log(path)`, `write_profile_summary(log_path,
output_path)`, and `write_profile_artifacts(...)` to
`src/edge_vlm/jetson_profile.py`.

Conservative label thresholds:

- `gpu_compute` when average `GR3D_FREQ` utilization is at least `85`.
- `emc_memory_bandwidth` when average `EMC_FREQ` utilization is at least `80`.
- `power_or_thermal` when max temperature is at least `80` Celsius.
- `cpu_prepost` when average CPU utilization is at least `80` and average GR3D
  utilization is below `50`.
- `startup_or_download` when recorded server startup or artifact/download phase
  duration is at least `30` seconds.
- `not_identified` when no other label applies.

- [x] **Step 4: Run summary tests to verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k jetson_profile
```

Expected: Task 1 and Task 2 tests pass.

- [x] **Step 5: Commit Task 2**

Run:

```bash
git add src/edge_vlm/jetson_profile.py tests/test_edge_vlm.py
git commit -m "feat: summarize jetson profile logs"
```

## Task 3: Link Profile Summaries From Sweep Manifests

**Files:**
- Modify: `src/edge_vlm/jetson_sweep.py`
- Modify: `tests/test_edge_vlm.py`

- [x] **Step 1: Write the failing sweep-link test**

Extend `test_jetson_sweep_run_records_preflight_and_reports_fake_stream` so the
plan includes:

```python
"manifest_json": str(tmp_path / "benchmarks" / "unit-run.manifest.json"),
"profile_jsonl": str(tmp_path / "profiles" / "unit-run.profile.jsonl"),
"profile_summary_json": str(tmp_path / "profiles" / "unit-run.summary.json"),
```

inside `paths`, and the fake benchmark manifest includes:

```python
"jetson": {"tegrastats_log": str(tmp_path / "tegrastats" / "unit-run.log")}
```

with a two-line log. Assert:

```python
self.assertTrue(Path(result["results"][0]["profile_summary_path"]).is_file())
self.assertTrue(Path(result["results"][0]["profile_jsonl_path"]).is_file())
self.assertEqual(result["results"][0]["profile_summary"]["samples"], 2)
self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["formal_text"]["duration_s"], 1.0)
self.assertIn("gpu_compute", result["results"][0]["profile_summary"]["bottleneck_labels"])
```

- [x] **Step 2: Run the sweep-link test to verify RED**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k jetson_sweep_run_records_preflight
```

Expected: fail because `profile_summary_path` is missing.

- [x] **Step 3: Implement manifest linking**

In `build_sweep_plan`, add:

```python
profile_jsonl = output_base / "profiles" / f"{run_id}.profile.jsonl"
profile_summary_json = output_base / "profiles" / f"{run_id}.summary.json"
```

and include both paths in `paths`.

In `run_sweep`, after the benchmark command succeeds, read the benchmark
manifest path from `paths["manifest_json"]`, get `jetson.tegrastats_log`, call
`write_profile_artifacts`, and add `profile_jsonl_path`,
`profile_summary_path`, and `profile_summary` to the result entry. If the
benchmark manifest or log is missing, set the planned output paths and
`profile_summary` to `{"available": False, "reason": "missing_tegrastats_log"}`.
Derive available phase durations for `server_startup`, `formal_text`,
`formal_image`, and `fake_stream`; leave unmeasured phases explicitly marked as
`not_recorded`.

- [x] **Step 4: Run sweep tests to verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k "jetson_sweep"
```

Expected: Jetson sweep tests pass.

- [x] **Step 5: Commit Task 3**

Run:

```bash
git add src/edge_vlm/jetson_sweep.py tests/test_edge_vlm.py
git commit -m "feat: link jetson profile summaries"
```

## Task 4: Add Profile Metrics To Comparison Reports

**Files:**
- Modify: `src/edge_vlm/optimization.py`
- Modify: `tests/test_edge_vlm.py`

- [x] **Step 1: Write the failing comparison test**

Extend `test_optimization_comparison_report_adds_manifest_context_and_deltas`
so its synthetic tegrastats log includes `GR3D_FREQ 90%@[1020] EMC_FREQ
82%@3199`, then assert the report contains:

```python
self.assertIn("Avg GR3D %", report_text)
self.assertIn("Avg EMC %", report_text)
self.assertIn("Bottlenecks", report_text)
self.assertIn("gpu_compute", report_text)
```

- [x] **Step 2: Run comparison test to verify RED**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k optimization_comparison_report_adds_manifest_context
```

Expected: fail because the report has no GR3D/EMC/bottleneck columns.

- [x] **Step 3: Implement comparison columns**

Import `summarize_tegrastats_log` from `edge_vlm.jetson_profile`, extend
`SweepComparisonRow` with `avg_gr3d_util_pct`, `avg_emc_util_pct`,
`min_lfb_free_blocks`, and `bottleneck_labels`, and update
`_format_sweep_comparison_report` to add columns:

```markdown
| Avg GR3D % | Avg EMC % | Min lfb blocks | Bottlenecks |
```

Keep existing `Max temp C` and `Avg power W` values sourced from the new
summary.

- [x] **Step 4: Run comparison tests to verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k optimization_comparison
```

Expected: comparison tests pass.

- [x] **Step 5: Commit Task 4**

Run:

```bash
git add src/edge_vlm/optimization.py tests/test_edge_vlm.py
git commit -m "feat: report jetson profile bottlenecks"
```

## Task 5: Docs And Full Verification

**Files:**
- Modify: `docs/specs/jetson_optimization_loop.md`
- Modify: `docs/benchmark_protocol.md`

- [ ] **Step 1: Update docs**

Document that sweep manifests now include `profiles/*.profile.jsonl` and
`profiles/*.summary.json`, comparison reports include
GR3D/EMC/min-lfb/bottleneck columns, available phase timings are recorded in
summary JSON, and raw `tegrastats` logs remain ignored output artifacts.

- [ ] **Step 2: Run full verification**

Run:

```bash
git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests
PATTERN='TB''D|TO''DO|PLACE''HOLDER|X''XX|FIX''ME'
rg -n "$PATTERN" docs/plans/2026-05-31-jetson-profiling-harness.md docs/specs/jetson_optimization_loop.md docs/benchmark_protocol.md src/edge_vlm tests/test_edge_vlm.py
```

Expected: `git diff --check` exits `0`; unittest reports all tests `OK`; `rg`
exits `1` with no placeholder matches.

- [ ] **Step 3: Commit docs and final verification state**

Run:

```bash
git add docs/specs/jetson_optimization_loop.md docs/benchmark_protocol.md docs/plans/2026-05-31-jetson-profiling-harness.md
git commit -m "docs: describe jetson profiling harness"
```

- [ ] **Step 4: Push branch**

Run:

```bash
git push origin bench/formal-jetson-infra
```

## Task 6: Remaining Phase Timing Instrumentation

**Files:**
- Modify: `scripts/jetson/run_hf_gguf_vlm_llama_docker.sh`
- Modify: `scripts/jetson/run_minicpmv46_llama_docker.sh`
- Modify: `scripts/jetson/run_gemma4_e2b_llama_docker.sh`
- Modify: `src/edge_vlm/jetson_sweep.py`
- Modify: `tests/test_edge_vlm.py`

- [ ] **Step 1: Add launcher phase-event tests**

Add dry-run shell tests that set `EDGE_VLM_PHASE_EVENTS` to a temp file and
verify launchers write JSONL events for `artifact_check_or_download` with
`cached=true` when artifacts already exist.

- [ ] **Step 2: Implement launcher phase events**

Use a shell helper named `record_phase_event` in each launcher. Each event must
be a one-line JSON object with `phase`, `event`, `time`, and optional `cached`.
For launchers that only check local files, record the check around the existing
`[[ -f ... ]]` validations.

- [ ] **Step 3: Merge launcher phase events into profile summary**

Teach `run_sweep` to read `EDGE_VLM_PHASE_EVENTS` after server startup and pass
the measured `artifact_check_or_download` duration into
`write_profile_artifacts`.

- [ ] **Step 4: Verify phase-event path**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -k phase
```

Expected: launcher phase-event tests and sweep phase merge tests pass.
