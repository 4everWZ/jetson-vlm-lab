"""Benchmark, client payload, image payload, and formal wrapper contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch



class BenchmarkPayloadContractsTest(unittest.TestCase):

    def test_image_payload_uses_data_url_content_part(self):
        from edge_vlm.image_payload import build_user_content

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")

            content = build_user_content("Describe the image.", image)

        self.assertEqual(content[0], {"type": "text", "text": "Describe the image."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_image_payload_reports_input_timing_breakdown(self):
        from edge_vlm.image_payload import build_user_content_with_timing

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")

            content, timing = build_user_content_with_timing("Describe the image.", image)

        self.assertEqual(content[0], {"type": "text", "text": "Describe the image."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(timing["image_bytes"], 8)
        self.assertGreaterEqual(timing["mime_detect_s"], 0.0)
        self.assertGreaterEqual(timing["image_read_s"], 0.0)
        self.assertGreaterEqual(timing["base64_encode_s"], 0.0)
        self.assertGreaterEqual(timing["data_url_build_s"], 0.0)

    def test_client_dry_run_builds_openai_chat_payload(self):
        from edge_vlm.client import OpenAICompatClient

        client = OpenAICompatClient(base_url="http://127.0.0.1:8080/v1", model="local-model")
        result = client.complete(prompt="Say hi.", dry_run=True, max_tokens=16, temperature=0.0)

        self.assertTrue(result.ok)
        self.assertEqual(result.request["model"], "local-model")
        self.assertEqual(result.request["max_tokens"], 16)
        self.assertEqual(result.request["messages"][0]["content"], "Say hi.")
        self.assertIn("dry run", result.text)
        self.assertIn("payload_build_s", result.timings)
        self.assertIn("json_serialize_s", result.timings)

    def test_client_extracts_reasoning_content_when_final_content_is_empty(self):
        from edge_vlm.client import _extract_text

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "thinking text from llama-server",
                    }
                }
            ]
        }

        self.assertEqual(_extract_text(response), "thinking text from llama-server")

    def test_benchmark_dry_run_writes_jsonl_records(self):
        from edge_vlm.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "id": "text_en_reasoning_short",
                        "input_type": "text",
                        "prompt": "Give one reason edge devices are memory constrained.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  backend: llama.cpp",
                        "  model_ref: ggml-org/example-GGUF",
                        "  quantization: Q4_K_M",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  image: false",
                    ]
                ),
                encoding="utf-8",
            )

            count = run_benchmark(config_path=config, cases_path=cases, output_path=output, dry_run=True)

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(record["prompt_case_id"], "text_en_reasoning_short")
        self.assertEqual(record["success"], True)
        self.assertEqual(record["device"], "wsl")
        self.assertIsInstance(record["latency_s"], float)
        self.assertIn("input_timing", record)
        self.assertIn("payload_build_s", record["input_timing"])
        self.assertEqual(record["quality_terms_any"], ["memory", "bandwidth"])

    def test_benchmark_end_time_follows_monotonic_latency_when_wall_clock_moves_backward(self):
        from edge_vlm.benchmark import run_benchmark
        from edge_vlm.client import CompletionResult

        class FakeClient:
            def complete(self, **_kwargs):
                return CompletionResult(
                    ok=True,
                    text="ok",
                    error=None,
                    latency_s=0.01,
                    request={"model": "local-model"},
                    response={"usage": {"completion_tokens": 1}},
                )

        class BackwardClock:
            calls = [
                datetime(2026, 5, 26, 8, 2, 59, 800000, tzinfo=timezone.utc),
                datetime(2026, 5, 26, 8, 2, 59, 100000, tzinfo=timezone.utc),
            ]

            @classmethod
            def now(cls, tz=None):
                value = cls.calls.pop(0)
                if tz is not None:
                    return value.astimezone(tz)
                return value

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            cases.write_text(
                json.dumps({"id": "text_case", "input_type": "text", "prompt": "Say hi."}) + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  backend: llama.cpp",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  image: false",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("edge_vlm.benchmark.OpenAICompatClient.from_config", return_value=FakeClient()):
                with patch("edge_vlm.benchmark.datetime", BackwardClock):
                    with patch("edge_vlm.benchmark.time.perf_counter", side_effect=[10.0, 10.5]):
                        count = run_benchmark(config_path=config, cases_path=cases, output_path=output, dry_run=True)

            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(count, 1)
        start = datetime.fromisoformat(record["start_time"])
        end = datetime.fromisoformat(record["end_time"])
        self.assertEqual(record["latency_s"], 0.5)
        self.assertGreaterEqual(end, start)
        self.assertEqual(end, start + timedelta(seconds=0.5))

    def test_benchmark_logs_missing_image_case_without_crashing(self):
        from edge_vlm.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "id": "image_caption_single",
                        "input_type": "image",
                        "prompt": "Describe this image.",
                        "image_path": str(tmp_path / "missing.jpg"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  backend: llama.cpp",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  image: true",
                    ]
                ),
                encoding="utf-8",
            )

            count = run_benchmark(config_path=config, cases_path=cases, output_path=output, dry_run=True)

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertFalse(record["success"])
        self.assertIn("image not found", record["error"])

    def test_benchmark_marks_fake_stream_case_as_separate_runner(self):
        from edge_vlm.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "id": "fake_stream_folder_sample",
                        "input_type": "fake_stream",
                        "prompt": "Describe this frame.",
                        "image_dir": str(tmp_path / "frames"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  backend: llama.cpp",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  image: true",
                    ]
                ),
                encoding="utf-8",
            )

            count = run_benchmark(config_path=config, cases_path=cases, output_path=output, dry_run=True)

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertTrue(record["success"])
        self.assertIn("fake stream", record["output_excerpt"])
        self.assertEqual(record["image_path"], str(tmp_path / "frames"))

    def test_benchmark_writes_markdown_summary_for_current_run(self):
        from edge_vlm.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = tmp_path / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            summary = tmp_path / "bench.md"
            cases.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "text_case",
                                "input_type": "text",
                                "prompt": "Say one short sentence.",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "image_case",
                                "input_type": "image",
                                "prompt": "Describe this image.",
                                "image_path": str(image),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  backend: llama.cpp",
                        "  quantization: Q8_0",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "capabilities:",
                        "  image: true",
                    ]
                ),
                encoding="utf-8",
            )

            count = run_benchmark(
                config_path=config,
                cases_path=cases,
                output_path=output,
                summary_path=summary,
                dry_run=True,
            )

            summary_text = summary.read_text(encoding="utf-8")

        self.assertEqual(count, 2)
        self.assertIn("# Edge VLM Benchmark Summary", summary_text)
        self.assertIn("- Cases written: 2", summary_text)
        self.assertIn("- Successful: 2", summary_text)
        self.assertIn("| text_case | text | yes |", summary_text)
        self.assertIn("| image_case | image | yes |", summary_text)

    def test_benchmark_writes_run_metadata_and_repeats_trials(self):
        from edge_vlm.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            metadata = tmp_path / "bench.manifest.json"
            cases.write_text(
                json.dumps(
                    {
                        "id": "text_case",
                        "input_type": "text",
                        "prompt": "Say one short sentence.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = tmp_path / "model.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: local-model",
                        "  family: Local",
                        "  backend: llama.cpp",
                        "  model_ref: local/example",
                        "  quantization: Q4_K_M",
                        "server:",
                        "  base_url: http://127.0.0.1:8080/v1",
                        "runtime:",
                        "  ctx_size: 512",
                        "capabilities:",
                        "  image: false",
                    ]
                ),
                encoding="utf-8",
            )

            count = run_benchmark(
                config_path=config,
                cases_path=cases,
                output_path=output,
                metadata_path=metadata,
                run_id="formal-unit-run",
                trial_count=2,
                dry_run=True,
                max_tokens=16,
                temperature=0.0,
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(metadata.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual([record["trial_index"] for record in records], [1, 2])
        self.assertEqual([record["case_index"] for record in records], [1, 1])
        self.assertTrue(all(record["run_id"] == "formal-unit-run" for record in records))
        self.assertEqual(manifest["run_id"], "formal-unit-run")
        self.assertEqual(manifest["cases_written"], 2)
        self.assertEqual(manifest["successful"], 2)
        self.assertEqual(manifest["failed"], 0)
        self.assertEqual(manifest["benchmark"]["trial_count"], 2)
        self.assertEqual(manifest["benchmark"]["max_tokens"], 16)
        self.assertEqual(manifest["model"]["name"], "local-model")
        self.assertEqual(manifest["model"]["quantization"], "Q4_K_M")
        self.assertIn("started_at", manifest)
        self.assertIn("ended_at", manifest)
        self.assertIn("runtime_env", manifest)

    def test_formal_jetson_benchmark_wrapper_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            summary = tmp_path / "bench.md"
            metadata = tmp_path / "bench.manifest.json"
            cases.write_text(
                json.dumps({"id": "text_case", "input_type": "text", "prompt": "Say hi."}) + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PYTHONPATH": "src",
                "EDGE_VLM_FORMAL_RUN_ID": "formal-wrapper-unit",
                "EDGE_VLM_CONFIG": "configs/models/minicpmv46_q4.yaml",
                "EDGE_VLM_CASES": str(cases),
                "EDGE_VLM_OUTPUT": str(output),
                "EDGE_VLM_SUMMARY_OUTPUT": str(summary),
                "EDGE_VLM_METADATA_OUTPUT": str(metadata),
                "EDGE_VLM_TRIAL_COUNT": "2",
                "EDGE_VLM_MAX_TOKENS": "8",
                "EDGE_VLM_TEMPERATURE": "0",
                "EDGE_VLM_FORMAL_DRY_RUN": "1",
                "EDGE_VLM_SKIP_TEGRASTATS": "1",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(metadata.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 2)
        self.assertEqual(manifest["run_id"], "formal-wrapper-unit")
        self.assertEqual(manifest["cases_written"], 2)
        self.assertEqual(manifest["jetson"]["tegrastats_log"], None)
        self.assertEqual(manifest["jetson"]["tegrastats_status"], "skipped")

    def test_formal_jetson_benchmark_wrapper_prefixes_tegrastats_with_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            fake_tegrastats = bin_dir / "tegrastats"
            fake_tegrastats.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "printf '%s\\n' 'RAM 1000/7620MB (lfb 200x4MB) CPU [10%@1000] GR3D_FREQ 20%@[1020] EMC_FREQ 30%@3199 gpu@40.0C VDD_IN 8000mW/7000mW'",
                        "sleep 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_tegrastats, 0o755)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            summary = tmp_path / "bench.md"
            metadata = tmp_path / "bench.manifest.json"
            tegrastats_log = tmp_path / "tegrastats.log"
            cases.write_text(
                json.dumps({"id": "text_case", "input_type": "text", "prompt": "Say hi."}) + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": "src",
                "EDGE_VLM_FORMAL_RUN_ID": "formal-wrapper-timestamp-unit",
                "EDGE_VLM_CONFIG": "configs/models/minicpmv46_q4.yaml",
                "EDGE_VLM_CASES": str(cases),
                "EDGE_VLM_OUTPUT": str(output),
                "EDGE_VLM_SUMMARY_OUTPUT": str(summary),
                "EDGE_VLM_METADATA_OUTPUT": str(metadata),
                "EDGE_VLM_TEGRASTATS_LOG": str(tegrastats_log),
                "EDGE_VLM_TRIAL_COUNT": "1",
                "EDGE_VLM_MAX_TOKENS": "8",
                "EDGE_VLM_TEMPERATURE": "0",
                "EDGE_VLM_FORMAL_DRY_RUN": "1",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )
            log_text = tegrastats_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(log_text, r"^\d{4}-\d{2}-\d{2}T.*Z RAM 1000/7620MB")


if __name__ == "__main__":
    unittest.main()
