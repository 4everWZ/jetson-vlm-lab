import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


class EdgeVlmContractsTest(unittest.TestCase):
    def test_gemma_q8_config_is_available_for_low_memory_wsl_path(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/gemma4_e2b_q8.yaml")

        self.assertEqual(config["model"]["name"], "gemma4-e2b-it-q8")
        self.assertEqual(config["model"]["quantization"], "Q8_0")
        self.assertTrue(config["capabilities"]["image"])

    def test_gemma_q8_wsl_defaults_stay_within_observed_low_memory_smoke_path(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/gemma4_e2b_q8.yaml")
        launch_script = Path("scripts/wsl/run_gemma4_e2b_llama.sh").read_text(encoding="utf-8")

        self.assertEqual(config["runtime"]["ctx_size"], 512)
        self.assertEqual(config["runtime"]["n_gpu_layers"], 0)
        self.assertIn('ctx_size="${CTX_SIZE:-512}"', launch_script)
        self.assertIn('llama_threads="${LLAMA_THREADS:-2}"', launch_script)
        self.assertIn('llama_parallel="${LLAMA_PARALLEL:-1}"', launch_script)
        self.assertIn('llama_batch_size="${LLAMA_BATCH_SIZE:-128}"', launch_script)
        self.assertIn('llama_ubatch_size="${LLAMA_UBATCH_SIZE:-32}"', launch_script)

    def test_gemma_q4_prepare_downloads_prequantized_artifacts(self):
        prepare_script = Path("scripts/wsl/prepare_gemma4_e2b_q4.sh").read_text(encoding="utf-8")
        config_text = Path("configs/models/gemma4_e2b_q4.yaml").read_text(encoding="utf-8")

        self.assertIn("mradermacher/gemma-4-E2B-it-GGUF", prepare_script)
        self.assertIn("gemma-4-E2B-it.Q4_K_M.gguf", prepare_script)
        self.assertIn("gemma-4-E2B-it.mmproj-Q8_0.gguf", prepare_script)
        self.assertIn("hf_hub_download", prepare_script)
        self.assertNotIn("llama-quantize", prepare_script)
        self.assertNotIn("ALLOW_HIGH_MEMORY_QUANTIZE", prepare_script)
        self.assertNotIn("ALLOW_Q8_REQUANTIZE", prepare_script)
        self.assertIn("prequantized_gguf", config_text)

    def test_wsl_cuda_wrappers_use_separate_cuda_build_and_gpu_layers(self):
        cuda_build_script = Path("scripts/wsl/build_llama_cpp_cuda.sh").read_text(encoding="utf-8")
        gemma_cuda_script = Path("scripts/wsl/run_gemma4_e2b_llama_cuda.sh").read_text(encoding="utf-8")
        minicpm_cuda_script = Path("scripts/wsl/run_minicpmv46_llama_cuda.sh").read_text(encoding="utf-8")

        self.assertIn('ENABLE_CUDA="${ENABLE_CUDA:-1}"', cuda_build_script)
        self.assertIn('LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_dir}/build-cuda}"', cuda_build_script)
        self.assertIn('BUILD_JOBS="${BUILD_JOBS:-8}"', cuda_build_script)
        self.assertIn('CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"', cuda_build_script)
        self.assertIn("nvcc", cuda_build_script)
        self.assertIn('LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build-cuda/bin/llama-server}"', gemma_cuda_script)
        self.assertIn('N_GPU_LAYERS="${N_GPU_LAYERS:-32}"', gemma_cuda_script)
        self.assertIn('LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-512}"', gemma_cuda_script)
        self.assertIn('LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}"', gemma_cuda_script)
        self.assertIn('scripts/wsl/run_gemma4_e2b_llama.sh', gemma_cuda_script)
        self.assertIn('LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build-cuda/bin/llama-server}"', minicpm_cuda_script)
        self.assertIn('N_GPU_LAYERS="${N_GPU_LAYERS:-32}"', minicpm_cuda_script)
        self.assertIn('LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-128}"', minicpm_cuda_script)
        self.assertIn('LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-32}"', minicpm_cuda_script)
        self.assertIn('scripts/wsl/run_minicpmv46_llama.sh', minicpm_cuda_script)

    def test_minicpm_prepare_downloads_official_prebuilt_artifacts(self):
        prepare_script = Path("scripts/wsl/prepare_minicpmv46_q4.sh").read_text(encoding="utf-8")
        config_text = Path("configs/models/minicpmv46_q4.yaml").read_text(encoding="utf-8")

        self.assertIn("openbmb/MiniCPM-V-4.6-gguf", prepare_script)
        self.assertIn("MiniCPM-V-4_6-Q4_K_M.gguf", prepare_script)
        self.assertIn("mmproj-model-f16.gguf", prepare_script)
        self.assertIn("hf_hub_download", prepare_script)
        self.assertNotIn("ALLOW_MINICPM_FULL_PREPARE", prepare_script)
        self.assertNotIn("convert_hf_to_gguf", prepare_script)
        self.assertNotIn("llama-quantize", prepare_script)
        self.assertIn("prequantized_gguf", config_text)

    def test_minicpm_inspection_script_does_not_download_model_files(self):
        inspect_script = Path("scripts/wsl/inspect_minicpmv46_hf.sh").read_text(encoding="utf-8")

        self.assertIn("model_info", inspect_script)
        self.assertIn("files_metadata=True", inspect_script)
        self.assertIn("openbmb/MiniCPM-V-4.6-gguf", inspect_script)
        self.assertIn("scripts/wsl/prepare_minicpmv46_q4.sh", inspect_script)
        self.assertNotIn("snapshot_download", inspect_script)
        self.assertNotIn("hf_hub_download", inspect_script)
        self.assertNotIn("convert_hf_to_gguf", inspect_script)

    def test_minicpm_config_uses_prebuilt_wsl_defaults(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/minicpmv46_q4.yaml")
        launch_script = Path("scripts/wsl/run_minicpmv46_llama.sh").read_text(encoding="utf-8")

        self.assertEqual(config["runtime"]["ctx_size"], 512)
        self.assertEqual(config["runtime"]["n_gpu_layers"], 0)
        self.assertIn('ctx_size="${CTX_SIZE:-512}"', launch_script)
        self.assertIn('llama_threads="${LLAMA_THREADS:-2}"', launch_script)
        self.assertIn('llama_parallel="${LLAMA_PARALLEL:-1}"', launch_script)
        self.assertIn('llama_batch_size="${LLAMA_BATCH_SIZE:-128}"', launch_script)
        self.assertIn('llama_ubatch_size="${LLAMA_UBATCH_SIZE:-32}"', launch_script)
        self.assertIn("pre-built", config["notes"]["status"].lower())

    def test_image_payload_uses_data_url_content_part(self):
        from edge_vlm.image_payload import build_user_content

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")

            content = build_user_content("Describe the image.", image)

        self.assertEqual(content[0], {"type": "text", "text": "Describe the image."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_client_dry_run_builds_openai_chat_payload(self):
        from edge_vlm.client import OpenAICompatClient

        client = OpenAICompatClient(base_url="http://127.0.0.1:8080/v1", model="local-model")
        result = client.complete(prompt="Say hi.", dry_run=True, max_tokens=16, temperature=0.0)

        self.assertTrue(result.ok)
        self.assertEqual(result.request["model"], "local-model")
        self.assertEqual(result.request["max_tokens"], 16)
        self.assertEqual(result.request["messages"][0]["content"], "Say hi.")
        self.assertIn("dry run", result.text)

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

    def test_optimization_report_ranks_only_sanity_passing_runs(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fast_but_bad = tmp_path / "fast_bad.jsonl"
            steady_good = tmp_path / "steady_good.jsonl"
            report = tmp_path / "report.md"
            fast_but_bad.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-fast",
                                "run_id": "fast-bad",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 0.5,
                                "tokens": 64,
                                "tokens_per_sec": 128.0,
                                "output_excerpt": "ok ok ok ok ok ok ok ok ok ok ok ok",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-fast",
                                "run_id": "fast-bad",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 0.5,
                                "tokens": 64,
                                "tokens_per_sec": 128.0,
                                "output_excerpt": "",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            steady_good.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-steady",
                                "run_id": "steady-good",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A concise answer that mentions memory, bandwidth, and thermal limits.",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-steady",
                                "run_id": "steady-good",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image contains two high-contrast squares on a simple background.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[fast_but_bad, steady_good],
                output_path=report,
                min_output_chars=24,
                max_repeat_ratio=0.6,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([summary.run_id for summary in summaries], ["steady-good", "fast-bad"])
        self.assertTrue(summaries[0].guard_passed)
        self.assertFalse(summaries[1].guard_passed)
        self.assertIn("| 1 | steady-good | local-steady | yes |", report_text)
        self.assertIn("| - | fast-bad | local-fast | no |", report_text)
        self.assertIn("repetitive_output", report_text)
        self.assertIn("empty_output", report_text)

    def test_optimization_report_includes_fake_stream_guard_and_latency(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bench = tmp_path / "benchmarks" / "run-a.jsonl"
            fake_stream = tmp_path / "fake_stream" / "run-a.jsonl"
            report = tmp_path / "report.md"
            bench.parent.mkdir()
            fake_stream.parent.mkdir()
            bench.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-model",
                                "run_id": "run-a",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 1.0,
                                "tokens": 64,
                                "tokens_per_sec": 64.0,
                                "output_excerpt": "A useful answer with enough detail to pass the sanity guard.",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-model",
                                "run_id": "run-a",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "The image shows two contrasting square shapes.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_stream.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 3.25,
                        "output_excerpt": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[bench],
                fake_stream_paths=[fake_stream],
                output_path=report,
                min_output_chars=24,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(len(summaries), 1)
        self.assertFalse(summaries[0].guard_passed)
        self.assertEqual(summaries[0].fake_stream_records, 1)
        self.assertEqual(summaries[0].fake_stream_successful, 1)
        self.assertEqual(summaries[0].fake_stream_avg_latency_s, 3.25)
        self.assertIn("Fake latency s", report_text)
        self.assertIn("fake_stream:frame_001.png:empty_output", report_text)

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
                        "fake_stream_command": ["python3", "-m", "edge_vlm.fake_stream"],
                        "fake_stream_env": {},
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "fake_stream_jsonl": str(fake_stream_jsonl),
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
                                    }
                                ),
                            ]
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
                            result = run_sweep(plan, wait_timeout_s=1.0, report_output=report)

            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(result["results"][0]["preflight"]["tegrastats"]["lfb"]["free_blocks"], 150)
        self.assertEqual(result["results"][0]["preflight_path"], str(preflight_json))
        self.assertEqual(result["results"][0]["fake_stream_returncode"], 0)
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

    def test_next_phase_spec_orders_infra_before_model_expansion_and_lists_tencent_youtu_vl(self):
        spec = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")

        self.assertIn("Phase 1: Formal Benchmark Infra", spec)
        self.assertIn("Phase 2: Lightweight Model Expansion", spec)
        self.assertLess(
            spec.index("Phase 1: Formal Benchmark Infra"),
            spec.index("Phase 2: Lightweight Model Expansion"),
        )
        self.assertIn("tencent/Youtu-VL-4B-Instruct-GGUF", spec)
        self.assertIn("SmolVLM2", spec)
        self.assertIn("Qwen3-VL-2B", spec)

    def test_jetson_optimization_variants_include_gemma_mid_batch_candidate(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        candidate = by_id["gemma-q4-gpu12-b384-u384-kvq8"]

        self.assertEqual(candidate["model"], "gemma4-e2b-it-q4")
        self.assertEqual(candidate["env"]["N_GPU_LAYERS"], 12)
        self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], 384)
        self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], 384)
        self.assertIn("--batch-size", candidate["args"])
        self.assertIn("384", candidate["args"])
        self.assertIn("--cache-type-k", candidate["args"])
        self.assertIn("q8_0", candidate["args"])

    def test_shared_prompt_case_assets_exist_for_out_of_box_dry_runs(self):
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        cases = [
            json.loads(line)
            for line in Path("configs/benchmark/prompt_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for case in cases:
            input_type = case.get("input_type")
            if input_type == "image":
                image_path = Path(case["image_path"])
                self.assertTrue(image_path.is_file(), f"missing sample image: {image_path}")
                self.assertIn(image_path.suffix.lower(), image_suffixes)
                self.assertTrue(
                    image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                    or image_path.read_bytes().startswith(b"\xff\xd8\xff"),
                    f"sample image is not a PNG or JPEG: {image_path}",
                )
            elif input_type == "fake_stream":
                image_dir = Path(case["image_dir"])
                self.assertTrue(image_dir.is_dir(), f"missing fake-stream directory: {image_dir}")
                frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in image_suffixes)
                self.assertGreater(len(frames), 0, f"no sample stream frames in: {image_dir}")

    def test_jetson_gemma_launcher_can_dry_run_without_docker_or_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "VLM_SERVER_PORT": "19090",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("dustynv/llama_cpp", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("ggml-org/gemma-4-E2B-it-GGUF:Q8_0", result.stdout)
        self.assertIn("-p 19090:8080", result.stdout)

    def test_jetson_launcher_allows_explicit_llama_cpp_image_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "LLAMA_CPP_DOCKER_IMAGE": "dustynv/llama_cpp:b5283-r36.4-cu128-24.04",
                "LLAMA_SERVER_CMD": "/usr/local/bin/llama-server",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dustynv/llama_cpp:b5283-r36.4-cu128-24.04", result.stdout)
        self.assertIn("/usr/local/bin/llama-server", result.stdout)

    def test_jetson_gemma_launcher_dry_run_allows_explicit_missing_model_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(model_dir),
                "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87",
                "MODEL_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.Q4_K_M.gguf"),
                "MMPROJ_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.mmproj-Q8_0.gguf"),
                "MODEL_ALIAS": "gemma4-e2b-it-q4",
                "CTX_SIZE": "512",
                "N_GPU_LAYERS": "12",
            }
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
                    "-fit",
                    "off",
                    "--parallel",
                    "1",
                    "--batch-size",
                    "512",
                    "--ubatch-size",
                    "512",
                    "--cache-type-k",
                    "q8_0",
                    "--cache-type-v",
                    "q8_0",
                    "--no-warmup",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("-m /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf", result.stdout)
        self.assertIn("--batch-size 512", result.stdout)

    def test_jetson_minicpm_launcher_can_dry_run_without_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(model_dir),
                "VLM_SERVER_PORT": "19091",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_minicpmv46_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("dustynv/llama_cpp", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("-m /models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf", result.stdout)

    def test_jetson_launchers_can_disable_docker_tty_for_automation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_minicpmv46_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_jetson_remote_exec_dry_run_sources_ignored_env_without_exposing_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.jetson"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/remote_exec.sh",
                    "git",
                    "status",
                    "--short",
                    "--branch",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "JETSON_REMOTE_DRY_RUN": "1",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("weizheng@192.168.1.12", result.stdout)
        self.assertIn("cd ~/code/jetson-vlm-lab", result.stdout)
        self.assertIn("git status --short --branch", result.stdout)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_jetson_remote_exec_requires_host_and_user(self):
        result = subprocess.run(
            ["bash", "scripts/jetson/remote_exec.sh", "git", "status"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                **os.environ,
                "JETSON_ENV_FILE": "/tmp/edge-vlm-missing-env-file",
                "JETSON_REMOTE_DRY_RUN": "1",
                "JETSON_SSH_HOST": "",
                "JETSON_SSH_USER": "",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_SSH_HOST and JETSON_SSH_USER are required", result.stderr)

    def test_jetson_remote_exec_can_use_askpass_without_sshpass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            log_file = tmp_path / "ssh.log"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            (fake_bin / "setsid").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "if [[ \"${1:-}\" == \"-w\" ]]; then shift; fi",
                        "exec \"$@\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (fake_bin / "ssh").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "password=\"$(${SSH_ASKPASS:?})\"",
                        "if [[ \"${password}\" != \"${EXPECTED_JETSON_PASSWORD:?}\" ]]; then",
                        "  echo 'askpass password mismatch' >&2",
                        "  exit 3",
                        "fi",
                        "printf 'ASKPASS_OK\\n' > \"${FAKE_SSH_LOG:?}\"",
                        "printf 'SSH_ASKPASS_REQUIRE=%s\\n' \"${SSH_ASKPASS_REQUIRE:-}\" >> \"${FAKE_SSH_LOG}\"",
                        "printf 'ARGS=%s\\n' \"$*\" >> \"${FAKE_SSH_LOG}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_bin / "setsid", 0o755)
            os.chmod(fake_bin / "ssh", 0o755)
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab",
                        "JETSON_SSH_PASSWORD=secret-password",
                        "JETSON_SSH_PASSWORD_HELPER=askpass",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/remote_exec.sh",
                    "git",
                    "pull",
                    "--ff-only",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "EXPECTED_JETSON_PASSWORD": "secret-password",
                    "FAKE_SSH_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("ASKPASS_OK", log_text)
        self.assertIn("SSH_ASKPASS_REQUIRE=force", log_text)
        self.assertIn("weizheng@192.168.1.12", log_text)
        self.assertIn("cd ~/code/jetson-vlm-lab && git pull --ff-only", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", log_text)

    def test_remote_optimization_sweep_syncs_branch_and_forwards_pinned_sweep_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-remote",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                    "--min-lfb-blocks",
                    "150",
                    "--pre-variant-command",
                    "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=git\nARG=pull\nARG=--ff-only\n", log_text)
        self.assertIn("ARG=env\n", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertIn("ARG=--run-prefix\nARG=unit-remote\n", log_text)
        self.assertIn("ARG=--variant\nARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("ARG=--min-lfb-blocks\nARG=150\n", log_text)
        self.assertIn("ARG=--pre-variant-command\nARG=sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n", log_text)

    def test_remote_optimization_sweep_can_skip_git_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 1)
        self.assertNotIn("ARG=pull\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertIn("ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)

    def test_fake_stream_dry_run_continues_after_missing_frame(self):
        from edge_vlm.fake_stream import run_fake_stream

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            (image_dir / "001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            output = tmp_path / "stream.jsonl"
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

            count = run_fake_stream(
                config_path=config,
                image_dir=image_dir,
                output_path=output,
                prompt="Describe this frame.",
                interval_s=0,
                max_frames=1,
                dry_run=True,
                stop_on_error=False,
            )

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(record["frame_id"], "001.jpg")
        self.assertEqual(record["success"], True)
        self.assertIn("dry run", record["output_excerpt"])

    def test_fake_stream_end_time_follows_client_latency_when_wall_clock_moves_backward(self):
        from edge_vlm.client import CompletionResult
        from edge_vlm.fake_stream import run_fake_stream

        class FakeClient:
            def complete(self, **_kwargs):
                return CompletionResult(
                    ok=True,
                    text="frame ok",
                    request={},
                    response={"dry_run": True},
                    latency_s=0.25,
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
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            (image_dir / "001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            output = tmp_path / "stream.jsonl"
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

            with patch("edge_vlm.fake_stream.OpenAICompatClient.from_config", return_value=FakeClient()):
                with patch("edge_vlm.fake_stream.datetime", BackwardClock):
                    count = run_fake_stream(
                        config_path=config,
                        image_dir=image_dir,
                        output_path=output,
                        prompt="Describe this frame.",
                        interval_s=0,
                        max_frames=1,
                    )

            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(count, 1)
        start = datetime.fromisoformat(record["start_time"])
        end = datetime.fromisoformat(record["end_time"])
        self.assertEqual(record["latency_s"], 0.25)
        self.assertGreaterEqual(end, start)
        self.assertEqual(end, start + timedelta(seconds=0.25))

    def test_fake_stream_continues_after_individual_frame_failure(self):
        from edge_vlm.client import CompletionResult
        from edge_vlm.fake_stream import run_fake_stream

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def complete(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("bad frame")
                return CompletionResult(
                    ok=True,
                    text="second frame ok",
                    request={},
                    response={"dry_run": True},
                    latency_s=0.01,
                )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            (image_dir / "001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (image_dir / "002.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            output = tmp_path / "stream.jsonl"
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

            with patch("edge_vlm.fake_stream.OpenAICompatClient.from_config", return_value=FakeClient()):
                count = run_fake_stream(
                    config_path=config,
                    image_dir=image_dir,
                    output_path=output,
                    prompt="Describe this frame.",
                    interval_s=0,
                    max_frames=2,
                    dry_run=False,
                    stop_on_error=False,
                )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 2)
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["error"], "bad frame")
        self.assertTrue(records[1]["success"])


if __name__ == "__main__":
    unittest.main()
