"""Benchmark runner JSONL, timing, summary, and metadata tests."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


class BenchmarkRunnerOutputsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
