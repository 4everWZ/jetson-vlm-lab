"""Jetson sweep preflight and skip-path contract tests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _build_single_variant_plan(tmp_path):
    return {
        "run_prefix": "unit",
        "port": 18080,
        "variants": [
            {
                "variant": {"id": "unit-variant"},
                "run_id": "unit-run",
                "server_command": ["bash", "server.sh"],
                "server_env": {},
                "benchmark_command": ["bash", "bench.sh"],
                "benchmark_env": {},
                "fake_stream_command": None,
                "fake_stream_env": {},
                "paths": {
                    "benchmark_jsonl": str(tmp_path / "benchmarks" / "unit-run.jsonl"),
                    "fake_stream_jsonl": str(tmp_path / "fake_stream" / "unit-run.jsonl"),
                    "server_log": str(tmp_path / "logs" / "server.log"),
                    "preflight_json": str(tmp_path / "preflight" / "unit-run.preflight.json"),
                },
            }
        ],
    }


def _write_preflight_sample(path, *, free_blocks):
    sample = {
        "captured_at": "2026-05-30T00:00:00+00:00",
        "tegrastats": {
            "available": True,
            "raw": f"RAM 716/7620MB (lfb {free_blocks}x4MB)",
            "lfb": {"free_blocks": free_blocks, "block_mb": 4},
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sample), encoding="utf-8")
    return sample


class JetsonSweepPreflightContractsTest(unittest.TestCase):
    def test_jetson_sweep_skips_variant_when_lfb_is_below_minimum(self):
        from edge_vlm.jetson_sweep import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = tmp_path / "report.md"
            plan = _build_single_variant_plan(tmp_path)

            def fake_preflight(path):
                return _write_preflight_sample(path, free_blocks=71)

            with patch("edge_vlm.jetson_sweep.capture_preflight_sample", side_effect=fake_preflight):
                with patch("edge_vlm.jetson_sweep.subprocess.Popen") as popen:
                    result = run_sweep(
                        plan,
                        wait_timeout_s=1.0,
                        report_output=report,
                        min_lfb_blocks=150,
                    )

        self.assertEqual(result["report_output"], None)
        self.assertFalse(report.exists())
        self.assertFalse(popen.called)
        skipped = result["results"][0]
        self.assertFalse(skipped["preflight_passed"])
        self.assertEqual(skipped["preflight_reason"], "lfb_free_blocks 71 < required 150")
        self.assertEqual(skipped["server_ready"], False)
        self.assertEqual(skipped["benchmark_returncode"], None)

    def test_jetson_sweep_skips_variant_when_server_port_is_already_open(self):
        from edge_vlm.jetson_sweep import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = tmp_path / "report.md"
            plan = _build_single_variant_plan(tmp_path)

            def fake_preflight(path):
                return _write_preflight_sample(path, free_blocks=180)

            with patch("edge_vlm.jetson_sweep.capture_preflight_sample", side_effect=fake_preflight):
                with patch("edge_vlm.jetson_sweep._wait_for_server_port_closed", return_value=False):
                    with patch("edge_vlm.jetson_sweep.subprocess.Popen") as popen:
                        result = run_sweep(
                            plan,
                            wait_timeout_s=1.0,
                            report_output=report,
                            min_lfb_blocks=150,
                        )

        self.assertEqual(result["report_output"], None)
        self.assertFalse(report.exists())
        self.assertFalse(popen.called)
        skipped = result["results"][0]
        self.assertTrue(skipped["preflight_passed"])
        self.assertEqual(skipped["preflight_reason"], "server_port_still_open_before_start")
        self.assertEqual(skipped["server_ready"], False)
        self.assertEqual(skipped["benchmark_returncode"], None)

    def test_jetson_sweep_runs_pre_variant_command_before_preflight(self):
        from edge_vlm.jetson_sweep import run_sweep

        class FakeProcess:
            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_jsonl = tmp_path / "benchmarks" / "unit-run.jsonl"
            report = tmp_path / "report.md"
            events = []
            plan = _build_single_variant_plan(tmp_path)

            def fake_preflight(path):
                events.append("preflight")
                sample = {
                    "captured_at": "2026-05-31T00:00:00+00:00",
                    "tegrastats": {
                        "available": True,
                        "raw": "RAM 645/7620MB (lfb 180x4MB)",
                        "lfb": {"free_blocks": 180, "block_mb": 4},
                    },
                }
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(sample), encoding="utf-8")
                return sample

            def fake_run(command, **_kwargs):
                if command == "sync; echo 3 > /proc/sys/vm/drop_caches":
                    events.append("cleanup")
                    return subprocess.CompletedProcess(command, 0, stdout="clean\n", stderr="")
                events.append("benchmark")
                benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    json.dumps(
                        {
                            "model": "local-model",
                            "run_id": "unit-run",
                            "prompt_case_id": "text_case",
                            "input_type": "text",
                            "success": True,
                            "latency_s": 1.0,
                            "tokens": 64,
                            "tokens_per_sec": 64.0,
                            "output_excerpt": "A usable answer with enough detail.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with patch("edge_vlm.jetson_sweep.capture_preflight_sample", side_effect=fake_preflight):
                with patch("edge_vlm.jetson_sweep._wait_for_server", return_value=True):
                    with patch("edge_vlm.jetson_sweep.subprocess.Popen", return_value=FakeProcess()):
                        with patch("edge_vlm.jetson_sweep.subprocess.run", side_effect=fake_run):
                            result = run_sweep(
                                plan,
                                wait_timeout_s=1.0,
                                report_output=report,
                                min_lfb_blocks=150,
                                pre_variant_command="sync; echo 3 > /proc/sys/vm/drop_caches",
                            )

        self.assertEqual(events[:2], ["cleanup", "preflight"])
        self.assertEqual(result["results"][0]["pre_variant_command_returncode"], 0)
        self.assertEqual(result["results"][0]["preflight"]["tegrastats"]["lfb"]["free_blocks"], 180)

    def test_jetson_sweep_skips_variant_when_pre_variant_command_fails(self):
        from edge_vlm.jetson_sweep import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = tmp_path / "report.md"
            plan = _build_single_variant_plan(tmp_path)

            with patch(
                "edge_vlm.jetson_sweep.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                    1,
                    stdout="",
                    stderr="sudo: a password is required\n",
                ),
            ):
                with patch("edge_vlm.jetson_sweep.capture_preflight_sample") as preflight:
                    with patch("edge_vlm.jetson_sweep.subprocess.Popen") as popen:
                        result = run_sweep(
                            plan,
                            wait_timeout_s=1.0,
                            report_output=report,
                            pre_variant_command="sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                        )

        self.assertFalse(preflight.called)
        self.assertFalse(popen.called)
        self.assertEqual(result["report_output"], None)
        failed = result["results"][0]
        self.assertFalse(failed["pre_variant_command_passed"])
        self.assertEqual(failed["pre_variant_command_returncode"], 1)
        self.assertEqual(failed["preflight_reason"], "pre_variant_command_failed returncode 1")

    def test_jetson_sweep_parses_tegrastats_lfb(self):
        from edge_vlm.jetson_sweep import parse_tegrastats_lfb

        parsed = parse_tegrastats_lfb(
            "05-30-2026 RAM 645/7620MB (lfb 150x4MB) CPU [1%@729] GR3D_FREQ 0%"
        )

        self.assertEqual(parsed, {"free_blocks": 150, "block_mb": 4})


if __name__ == "__main__":
    unittest.main()
