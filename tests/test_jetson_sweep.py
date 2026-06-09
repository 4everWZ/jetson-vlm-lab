"""Jetson sweep plan, preflight, and runner contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JetsonSweepContractsTest(unittest.TestCase):

    def test_jetson_sweep_server_processes_are_group_terminated(self):
        sweep = Path("src/edge_vlm/jetson_sweep.py").read_text(encoding="utf-8")

        self.assertIn("start_new_session=True", sweep)
        self.assertIn("os.killpg", sweep)
        self.assertIn("signal.SIGTERM", sweep)
        self.assertIn("signal.SIGKILL", sweep)

    def test_jetson_sweep_dry_run_writes_reproducible_variant_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants = tmp_path / "variants.jsonl"
            plan = tmp_path / "plan.json"
            variants.write_text(
                json.dumps(
                    {
                        "id": "minicpm-unit",
                        "model": "minicpmv46-q4",
                        "config": "configs/models/minicpmv46_q4.yaml",
                        "launcher": "scripts/jetson/run_minicpmv46_llama_docker.sh",
                        "env": {
                            "MODEL_DIR": str(tmp_path / "models"),
                            "MODEL_PATH": "${MODEL_DIR}/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf",
                            "MODEL_ALIAS": "minicpmv46-q4",
                            "CTX_SIZE": 512,
                            "N_GPU_LAYERS": 32,
                            "LLAMA_BATCH_SIZE": 128,
                            "LLAMA_UBATCH_SIZE": 32,
                        },
                        "args": [
                            "--parallel",
                            "1",
                            "--batch-size",
                            "128",
                            "--ubatch-size",
                            "32",
                            "--cache-type-k",
                            "q8_0",
                            "--cache-type-v",
                            "q8_0",
                            "--no-warmup",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.jetson_sweep",
                    "--variants",
                    str(variants),
                    "--model",
                    "minicpmv46-q4",
                    "--run-prefix",
                    "unit-sweep",
                    "--output-root",
                    str(tmp_path / "outputs"),
                    "--server-log-dir",
                    str(tmp_path / "logs"),
                    "--trial-count",
                    "1",
                    "--max-tokens",
                    "16",
                    "--temperature",
                    "0",
                    "--fake-stream-interval-s",
                    "1.0",
                    "--fake-stream-skip-late-frames",
                    "--fake-stream-skip-threshold-s",
                    "0.5",
                    "--fake-stream-adaptive-interval",
                    "--fake-stream-adaptive-interval-scale",
                    "1.25",
                    "--fake-stream-adaptive-interval-max-s",
                    "3.0",
                    "--min-lfb-blocks",
                    "150",
                    "--pre-variant-command",
                    "sync; echo 3 > /proc/sys/vm/drop_caches",
                    "--dry-run",
                    "--plan-output",
                    str(plan),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )
            plan_data = json.loads(plan.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(plan_data["variants"]), 1)
        self.assertEqual(plan_data["pre_variant_command"], "sync; echo 3 > /proc/sys/vm/drop_caches")
        variant_plan = plan_data["variants"][0]
        self.assertEqual(variant_plan["run_id"], "unit-sweep-minicpm-unit")
        self.assertEqual(variant_plan["server_env"]["DOCKER_TTY"], "0")
        self.assertEqual(
            variant_plan["server_env"]["MODEL_PATH"],
            str(tmp_path / "models" / "MiniCPM-V-4.6-gguf" / "MiniCPM-V-4_6-Q4_K_M.gguf"),
        )
        self.assertEqual(variant_plan["benchmark_env"]["EDGE_VLM_TRIAL_COUNT"], "1")
        self.assertEqual(variant_plan["benchmark_env"]["EDGE_VLM_MAX_TOKENS"], "16")
        self.assertIn("scripts/jetson/run_minicpmv46_llama_docker.sh", variant_plan["server_command"])
        self.assertIn("--cache-type-k", variant_plan["server_command"])
        self.assertTrue(variant_plan["paths"]["benchmark_jsonl"].endswith("unit-sweep-minicpm-unit.jsonl"))
        self.assertTrue(variant_plan["paths"]["preflight_json"].endswith("unit-sweep-minicpm-unit.preflight.json"))
        self.assertTrue(variant_plan["paths"]["lifecycle_jsonl"].endswith("unit-sweep-minicpm-unit.lifecycle.jsonl"))
        self.assertEqual(
            variant_plan["server_env"]["EDGE_VLM_LAUNCH_PHASE_LOG"],
            variant_plan["paths"]["lifecycle_jsonl"],
        )
        warmup_phase = variant_plan["phase_defaults"]["warmup"]
        self.assertFalse(warmup_phase["available"])
        self.assertEqual(warmup_phase["reason"], "disabled_by_variant")
        self.assertEqual(warmup_phase["source"], "sweep")
        self.assertEqual(warmup_phase["details"]["flag"], "--no-warmup")
        fake_command = variant_plan["fake_stream_command"]
        self.assertEqual(fake_command[fake_command.index("--max-frames") + 1], "3")
        self.assertEqual(fake_command[fake_command.index("--interval-s") + 1], "1.0")
        self.assertIn("--skip-late-frames", fake_command)
        self.assertEqual(fake_command[fake_command.index("--skip-threshold-s") + 1], "0.5")
        self.assertIn("--adaptive-interval", fake_command)
        self.assertEqual(fake_command[fake_command.index("--adaptive-interval-scale") + 1], "1.25")
        self.assertEqual(fake_command[fake_command.index("--adaptive-interval-max-s") + 1], "3.0")

    def test_jetson_sweep_plan_records_inherited_launcher_environment(self):
        from edge_vlm.jetson_sweep import build_sweep_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants = tmp_path / "variants.jsonl"
            variants.write_text(
                json.dumps(
                    {
                        "id": "minicpm-unit",
                        "model": "minicpmv46-q4",
                        "config": "configs/models/minicpmv46_q4.yaml",
                        "launcher": "scripts/jetson/run_minicpmv46_llama_docker.sh",
                        "env": {
                            "MODEL_DIR": str(tmp_path / "models"),
                            "MODEL_ALIAS": "minicpmv46-q4",
                            "CTX_SIZE": 512,
                            "N_GPU_LAYERS": 32,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            plan = build_sweep_plan(
                variants_path=variants,
                run_prefix="unit",
                output_root=tmp_path / "outputs",
                server_log_dir=tmp_path / "logs",
                port=18080,
                trial_count=1,
                max_tokens=16,
                temperature=0,
                python_bin="python3",
                base_env={
                    "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:test",
                    "LLAMA_SERVER_CMD": "/usr/local/bin/llama-server",
                    "DOCKER_GPU_ARGS": "--runtime nvidia",
                },
            )

        server_env = plan["variants"][0]["server_env"]
        self.assertEqual(server_env["LLAMA_CPP_DOCKER_IMAGE"], "ghcr.io/4everwz/jetson-llama-cpp:test")
        self.assertEqual(server_env["LLAMA_SERVER_CMD"], "/usr/local/bin/llama-server")
        self.assertEqual(server_env["DOCKER_GPU_ARGS"], "--runtime nvidia")

    def test_jetson_sweep_plan_skips_fake_stream_for_text_only_configs(self):
        from edge_vlm.jetson_sweep import build_sweep_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "text-model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: text-only-local",
                        "  backend: llama.cpp",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  text: true",
                        "  image: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            variants = tmp_path / "variants.jsonl"
            variants.write_text(
                json.dumps(
                    {
                        "id": "text-unit",
                        "model": "text-only-local",
                        "config": str(config),
                        "launcher": "scripts/jetson/run_hf_gguf_llama_docker.sh",
                        "env": {
                            "MODEL_REF": "tencent/example-GGUF:Q4_K_M",
                            "MODEL_FILE": "example.gguf",
                            "MODEL_ALIAS": "text-only-local",
                            "EDGE_VLM_CASES": "configs/benchmark/text_prompt_cases.jsonl",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            plan = build_sweep_plan(
                variants_path=variants,
                run_prefix="unit",
                output_root=tmp_path / "outputs",
                server_log_dir=tmp_path / "logs",
                port=18080,
                trial_count=1,
                max_tokens=16,
                temperature=0,
                python_bin="python3",
                include_fake_stream=True,
                base_env={
                    "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:test",
                },
            )

        variant_plan = plan["variants"][0]
        self.assertIsNone(variant_plan["fake_stream_command"])
        self.assertEqual(variant_plan["benchmark_env"]["EDGE_VLM_CASES"], "configs/benchmark/text_prompt_cases.jsonl")

    def test_jetson_sweep_plan_records_docker_image_metadata(self):
        from edge_vlm.jetson_sweep import build_sweep_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants = tmp_path / "variants.jsonl"
            variants.write_text(
                json.dumps(
                    {
                        "id": "minicpm-unit",
                        "model": "minicpmv46-q4",
                        "config": "configs/models/minicpmv46_q4.yaml",
                        "launcher": "scripts/jetson/run_minicpmv46_llama_docker.sh",
                        "env": {
                            "MODEL_DIR": str(tmp_path / "models"),
                            "MODEL_ALIAS": "minicpmv46-q4",
                            "CTX_SIZE": 512,
                            "N_GPU_LAYERS": 32,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            inspect_payload = [
                {
                    "Id": "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
                    "Created": "2026-05-27T13:47:04.31937282+09:30",
                    "RepoDigests": [
                        "ghcr.io/4everwz/jetson-llama-cpp@sha256:c39cdc50c4564f29490c69b30f601da23f4d086f99d5c4b562426dbf65fec263"
                    ],
                    "Config": {
                        "Labels": {
                            "org.opencontainers.image.version": "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774",
                            "org.opencontainers.image.revision": "735d6e569bf8",
                            "org.opencontainers.image.base.name": "dustynv/cuda-python:r36.4.0-cu128-24.04",
                        }
                    },
                }
            ]

            with patch(
                "edge_vlm.jetson_sweep.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["docker", "image", "inspect", "ghcr.io/4everwz/jetson-llama-cpp:test"],
                    0,
                    stdout=json.dumps(inspect_payload),
                    stderr="",
                ),
            ) as docker_inspect:
                plan = build_sweep_plan(
                    variants_path=variants,
                    run_prefix="unit",
                    output_root=tmp_path / "outputs",
                    server_log_dir=tmp_path / "logs",
                    port=18080,
                    trial_count=1,
                    max_tokens=16,
                    temperature=0,
                    python_bin="python3",
                    base_env={
                        "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:test",
                    },
                )

        docker_inspect.assert_called_once_with(
            ["docker", "image", "inspect", "ghcr.io/4everwz/jetson-llama-cpp:test"],
            check=False,
            capture_output=True,
            text=True,
        )
        runtime = plan["variants"][0]["server_runtime"]
        self.assertEqual(runtime["image"], "ghcr.io/4everwz/jetson-llama-cpp:test")
        self.assertTrue(runtime["inspect_ok"])
        self.assertEqual(
            runtime["image_id"],
            "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
        )
        self.assertEqual(runtime["repo_digests"], inspect_payload[0]["RepoDigests"])
        self.assertEqual(runtime["llama_cpp_ref"], "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774")
        self.assertEqual(runtime["source_revision"], "735d6e569bf8")
        self.assertEqual(runtime["base_image"], "dustynv/cuda-python:r36.4.0-cu128-24.04")

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

    def test_jetson_sweep_skips_variant_when_lfb_is_below_minimum(self):
        from edge_vlm.jetson_sweep import run_sweep

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preflight_json = tmp_path / "preflight" / "unit-run.preflight.json"
            report = tmp_path / "report.md"
            plan = {
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
                        "raw": "RAM 716/7620MB (lfb 71x4MB)",
                        "lfb": {"free_blocks": 71, "block_mb": 4},
                    },
                }
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(sample), encoding="utf-8")
                return sample

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
            preflight_json = tmp_path / "preflight" / "unit-run.preflight.json"
            report = tmp_path / "report.md"
            plan = {
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
                        "raw": "RAM 716/7620MB (lfb 180x4MB)",
                        "lfb": {"free_blocks": 180, "block_mb": 4},
                    },
                }
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(sample), encoding="utf-8")
                return sample

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
            preflight_json = tmp_path / "preflight" / "unit-run.preflight.json"
            report = tmp_path / "report.md"
            events = []
            plan = {
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
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "fake_stream_jsonl": str(tmp_path / "fake_stream" / "unit-run.jsonl"),
                            "server_log": str(tmp_path / "logs" / "server.log"),
                            "preflight_json": str(preflight_json),
                        },
                    }
                ],
            }

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
            plan = {
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
