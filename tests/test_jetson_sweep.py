"""Jetson sweep runner execution contract tests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JetsonSweepRunnerContractsTest(unittest.TestCase):
    def test_jetson_sweep_server_processes_are_group_terminated(self):
        sweep = Path("src/edge_vlm/jetson_sweep.py").read_text(encoding="utf-8")

        self.assertIn("start_new_session=True", sweep)
        self.assertIn("os.killpg", sweep)
        self.assertIn("signal.SIGTERM", sweep)
        self.assertIn("signal.SIGKILL", sweep)

    def test_jetson_sweep_run_records_preflight_and_reports_fake_stream(self):
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
            fake_stream_jsonl = tmp_path / "fake_stream" / "unit-run.jsonl"
            benchmark_manifest = tmp_path / "benchmarks" / "unit-run.manifest.json"
            tegrastats_log = tmp_path / "tegrastats" / "unit-run.log"
            profile_jsonl = tmp_path / "profiles" / "unit-run.profile.jsonl"
            profile_summary_json = tmp_path / "profiles" / "unit-run.summary.json"
            lifecycle_jsonl = tmp_path / "lifecycle" / "unit-run.lifecycle.jsonl"
            preflight_json = tmp_path / "preflight" / "unit-run.preflight.json"
            report = tmp_path / "report.md"
            plan = {
                "run_prefix": "unit",
                "port": 18080,
                "variants": [
                    {
                        "variant": {"id": "unit-variant", "args": []},
                        "run_id": "unit-run",
                        "server_command": ["bash", "server.sh"],
                        "server_env": {},
                        "benchmark_command": ["bash", "bench.sh"],
                        "benchmark_env": {},
                        "fake_stream_command": ["python3", "-m", "edge_vlm.fake_stream"],
                        "fake_stream_env": {},
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_stream_jsonl),
                            "profile_jsonl": str(profile_jsonl),
                            "profile_summary_json": str(profile_summary_json),
                            "lifecycle_jsonl": str(lifecycle_jsonl),
                            "server_log": str(tmp_path / "logs" / "server.log"),
                            "preflight_json": str(preflight_json),
                        },
                    }
                ],
            }

            def fake_preflight(path):
                sample = {
                    "captured_at": "2026-05-30T00:00:00+00:00",
                    "tegrastats": {
                        "available": True,
                        "raw": "RAM 645/7620MB (lfb 150x4MB)",
                        "lfb": {"free_blocks": 150, "block_mb": 4},
                    },
                }
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(sample), encoding="utf-8")
                lifecycle_jsonl.parent.mkdir(parents=True, exist_ok=True)
                lifecycle_jsonl.write_text(
                    json.dumps(
                        {
                            "phase": "artifact_check_or_download",
                            "available": True,
                            "duration_s": 3.25,
                            "reason": None,
                            "source": "launcher",
                            "details": {"status": "downloaded_or_checked"},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return sample

            def fake_run(command, **_kwargs):
                if command == ["bash", "bench.sh"]:
                    benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    benchmark_jsonl.write_text(
                        "\n".join(
                            [
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
                                        "input_timing": {
                                            "image_bytes": 0,
                                            "payload_build_s": 0.05,
                                            "json_serialize_s": 0.02,
                                            "http_request_s": 0.90,
                                            "response_parse_s": 0.01,
                                            "request_body_bytes": 500,
                                        },
                                    }
                                ),
                                json.dumps(
                                    {
                                        "model": "local-model",
                                        "run_id": "unit-run",
                                        "prompt_case_id": "image_case",
                                        "input_type": "image",
                                        "success": True,
                                        "latency_s": 2.0,
                                        "tokens": 64,
                                        "tokens_per_sec": 32.0,
                                        "output_excerpt": "The image contains simple contrasting shapes.",
                                        "input_timing": {
                                            "image_bytes": 1000,
                                            "payload_build_s": 0.20,
                                            "json_serialize_s": 0.05,
                                            "http_request_s": 1.80,
                                            "response_parse_s": 0.02,
                                            "request_body_bytes": 1500,
                                        },
                                    }
                                ),
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                    tegrastats_log.write_text(
                        "\n".join(
                            [
                                "RAM 2000/7620MB (lfb 200x4MB) CPU [40%@1728] GR3D_FREQ 92%@[1020] EMC_FREQ 81%@3199 gpu@54.0C VDD_IN 18000mW/17000mW",
                                "RAM 2200/7620MB (lfb 160x4MB) CPU [45%@1728] GR3D_FREQ 88%@[1020] EMC_FREQ 86%@3199 gpu@58.0C VDD_IN 19000mW/18000mW",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    benchmark_manifest.write_text(
                        json.dumps(
                            {
                                "run_id": "unit-run",
                                "jetson": {
                                    "tegrastats_log": str(tegrastats_log),
                                    "power_mode": str(tmp_path / "profile" / "nvpmodel.txt"),
                                    "jetson_clocks": str(tmp_path / "profile" / "jetson-clocks.txt"),
                                },
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    fake_stream_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    fake_stream_jsonl.write_text(
                        json.dumps(
                            {
                                "frame_index": 0,
                                "frame_id": "frame_001.png",
                                "success": True,
                                "latency_s": 1.5,
                                "output_excerpt": "The frame shows two contrasting square shapes.",
                                "input_timing": {
                                    "image_bytes": 900,
                                    "payload_build_s": 0.10,
                                    "json_serialize_s": 0.03,
                                    "http_request_s": 1.30,
                                    "response_parse_s": 0.01,
                                    "request_body_bytes": 1400,
                                },
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
                            with patch("edge_vlm.jetson_sweep.time.monotonic", side_effect=[10.0, 12.5, 20.0, 20.75]):
                                result = run_sweep(plan, wait_timeout_s=1.0, report_output=report)

            report_text = report.read_text(encoding="utf-8")
            profile_jsonl_exists = profile_jsonl.is_file()
            profile_summary_json_exists = profile_summary_json.is_file()

        self.assertEqual(result["results"][0]["preflight"]["tegrastats"]["lfb"]["free_blocks"], 150)
        self.assertEqual(result["results"][0]["preflight_path"], str(preflight_json))
        self.assertEqual(result["results"][0]["fake_stream_returncode"], 0)
        self.assertEqual(result["results"][0]["server_wait_seconds"], 2.5)
        self.assertEqual(result["results"][0]["server_startup_seconds"], 2.5)
        self.assertEqual(result["results"][0]["server_shutdown_seconds"], 0.75)
        self.assertIsInstance(result["results"][0]["server_started_at"], str)
        self.assertIsInstance(result["results"][0]["server_ready_at"], str)
        self.assertEqual(result["results"][0]["profile_summary_path"], str(profile_summary_json))
        self.assertEqual(result["results"][0]["profile_jsonl_path"], str(profile_jsonl))
        self.assertTrue(profile_jsonl_exists)
        self.assertTrue(profile_summary_json_exists)
        self.assertEqual(result["results"][0]["profile_summary"]["samples"], 2)
        self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["server_startup"]["duration_s"], 2.5)
        self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["formal_text"]["duration_s"], 1.0)
        self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["formal_image"]["duration_s"], 2.0)
        self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["fake_stream"]["duration_s"], 1.5)
        self.assertEqual(result["results"][0]["profile_summary"]["phase_timings"]["shutdown"]["duration_s"], 0.75)
        warmup_phase = result["results"][0]["profile_summary"]["phase_timings"]["warmup"]
        self.assertFalse(warmup_phase["available"])
        self.assertEqual(warmup_phase["reason"], "included_in_server_startup")
        self.assertEqual(warmup_phase["source"], "sweep")
        self.assertEqual(warmup_phase["details"]["status"], "enabled_not_separated")
        artifact_phase = result["results"][0]["profile_summary"]["phase_timings"]["artifact_check_or_download"]
        self.assertTrue(artifact_phase["available"])
        self.assertEqual(artifact_phase["duration_s"], 3.25)
        self.assertEqual(artifact_phase["source"], "launcher")
        self.assertEqual(artifact_phase["details"]["status"], "downloaded_or_checked")
        input_summary = result["results"][0]["profile_summary"]["input_timing_summary"]
        self.assertTrue(input_summary["available"])
        self.assertEqual(input_summary["records"], 3)
        self.assertEqual(input_summary["records_with_latency"], 3)
        self.assertEqual(input_summary["sources"], {"benchmark_jsonl": 2, "fake_stream_jsonl": 1})
        self.assertEqual(input_summary["avg_payload_overhead_s"], 0.15)
        self.assertEqual(input_summary["avg_e2e_latency_s"], 1.65)
        self.assertEqual(input_summary["max_request_body_bytes"], 1500)
        self.assertIn("gpu_compute", result["results"][0]["profile_summary"]["bottleneck_labels"])
        self.assertIn("emc_memory_bandwidth", result["results"][0]["profile_summary"]["bottleneck_labels"])
        self.assertIn("1.500", report_text)
        self.assertIn("Fake latency s", report_text)


if __name__ == "__main__":
    unittest.main()
