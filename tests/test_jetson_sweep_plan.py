"""Jetson sweep plan and dry-run contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JetsonSweepPlanContractsTest(unittest.TestCase):
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
        self.assertTrue(
            variant_plan["paths"]["preflight_before_prepare_json"].endswith(
                "unit-sweep-minicpm-unit.preflight-before-prepare.json"
            )
        )
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

    def test_jetson_sweep_dry_run_preserves_selection_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants = tmp_path / "variants.jsonl"
            plan = tmp_path / "plan.json"
            selection_context = tmp_path / "qwen3-selection-context.json"
            variants.write_text(
                json.dumps(
                    {
                        "id": "qwen3-vl-2b-instruct-q4-smoke",
                        "model": "qwen3-vl-2b-instruct-q4",
                        "config": "configs/models/qwen3_vl_2b_instruct_q4.yaml",
                        "launcher": "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                        "env": {
                            "MODEL_DIR": str(tmp_path / "models"),
                            "MODEL_ALIAS": "qwen3-vl-2b-instruct-q4",
                            "CTX_SIZE": 1024,
                            "N_GPU_LAYERS": 99,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selection_context.write_text(
                json.dumps(
                    {
                        "selection_id": "qwen3-vl-2b-instruct-auto",
                        "comparison_group": "qwen3-vl-2b-instruct",
                        "selected_variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                        "selected_reason": "primary_usable",
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
                    "--variant",
                    "qwen3-vl-2b-instruct-q4-smoke",
                    "--run-prefix",
                    "unit-selection-context",
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
                    "--selection-context-json",
                    str(selection_context),
                    "--plan-output",
                    str(plan),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )
            plan_data = json.loads(plan.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            plan_data["selection_contexts"],
            [
                {
                    "selection_id": "qwen3-vl-2b-instruct-auto",
                    "comparison_group": "qwen3-vl-2b-instruct",
                    "selected_variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                    "selected_reason": "primary_usable",
                }
            ],
        )

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
                "edge_vlm.jetson_sweep._docker_image_runtime_probe",
                return_value={
                    "llama_server_probe_ok": False,
                    "llama_server_probe_error": "not part of this test",
                    "llama_server_found": None,
                    "llama_server_path": None,
                    "llama_server_help_ok": None,
                    "llama_server_supports_mmproj": None,
                    "llama_server_multimodal_markers": [],
                },
            ):
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

    def test_jetson_sweep_plan_records_llama_server_runtime_probe(self):
        from edge_vlm.jetson_sweep import build_sweep_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants = tmp_path / "variants.jsonl"
            variants.write_text(
                json.dumps(
                    {
                        "id": "qwen-unit",
                        "model": "qwen3-vl-2b-instruct-q4",
                        "config": "configs/models/qwen3_vl_2b_instruct_q4.yaml",
                        "launcher": "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                        "env": {
                            "MODEL_DIR": str(tmp_path / "models"),
                            "MODEL_ALIAS": "qwen3-vl-2b-instruct-q4",
                            "CTX_SIZE": 1024,
                            "N_GPU_LAYERS": 99,
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
                    "RepoDigests": [],
                    "Config": {
                        "Labels": {
                            "org.opencontainers.image.version": "d749821db3bd8c5c52360f55ca1a1b76ae46d8e7",
                        }
                    },
                }
            ]
            probe_stdout = "\n".join(
                [
                    "llama_server_found=1",
                    "llama_server_path=/usr/local/bin/llama-server",
                    "llama_server_help_ok=1",
                    "llama_server_supports_mmproj=1",
                    "llama_server_multimodal_markers=--mmproj,mmproj",
                ]
            )

            def fake_run(command, **kwargs):
                if command[:3] == ["docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(inspect_payload),
                        stderr="",
                    )
                if command[:3] == ["docker", "run", "--rm"]:
                    self.assertIn("--runtime", command)
                    runtime_index = command.index("--runtime")
                    self.assertLess(runtime_index + 1, len(command))
                    self.assertEqual(command[runtime_index + 1], "nvidia")
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=probe_stdout,
                        stderr="",
                    )
                raise AssertionError(f"unexpected command: {command}")

            with patch("edge_vlm.jetson_sweep.subprocess.run", side_effect=fake_run):
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

        runtime = plan["variants"][0]["server_runtime"]
        self.assertTrue(runtime["llama_server_probe_ok"])
        self.assertTrue(runtime["llama_server_found"])
        self.assertEqual(runtime["llama_server_path"], "/usr/local/bin/llama-server")
        self.assertTrue(runtime["llama_server_help_ok"])
        self.assertTrue(runtime["llama_server_supports_mmproj"])
        self.assertEqual(runtime["llama_server_multimodal_markers"], ["--mmproj", "mmproj"])
        self.assertTrue(plan["variants"][0]["supports_images"])

    def test_docker_runtime_probe_requires_explicit_mmproj_flag(self):
        from edge_vlm.jetson_sweep import _docker_image_runtime_probe

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_server = fake_bin / "llama-server"
            fake_server.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "if [[ \"${1:-}\" == \"--help\" ]]; then",
                        "  printf '%s\\n' 'usage: llama-server [options]'",
                        "  printf '%s\\n' 'multimodal projector: mmproj file path required'",
                        "  exit 0",
                        "fi",
                        "exit 9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_server, 0o755)
            original_run = subprocess.run

            def fake_run(command, **kwargs):
                if command[:3] == ["docker", "run", "--rm"]:
                    probe_script = command[-1]
                    local_env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
                    local_result = original_run(
                        ["bash", "-lc", probe_script],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=local_env,
                    )
                    return subprocess.CompletedProcess(
                        command,
                        local_result.returncode,
                        stdout=local_result.stdout,
                        stderr=local_result.stderr,
                    )
                raise AssertionError(f"unexpected command: {command}")

            with patch("edge_vlm.jetson_sweep.subprocess.run", side_effect=fake_run):
                runtime = _docker_image_runtime_probe("dustynv/llama_cpp:test")

        self.assertTrue(runtime["llama_server_probe_ok"])
        self.assertTrue(runtime["llama_server_found"])
        self.assertTrue(runtime["llama_server_help_ok"])
        self.assertFalse(runtime["llama_server_supports_mmproj"])
        self.assertEqual(runtime["llama_server_multimodal_markers"], ["mmproj"])


if __name__ == "__main__":
    unittest.main()
