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

    def test_llama_cpp_artifact_builder_prefers_direct_docker_when_available(self):
        artifact_builder = Path("scripts/build_llama_cpp_artifacts.sh").read_text(encoding="utf-8")

        self.assertIn('BUILD_JOBS="${BUILD_JOBS:-6}"', artifact_builder)
        self.assertIn('BUILD_LOG_DIR="${BUILD_LOG_DIR:-$PWD/outputs/build/llama-cpp}"', artifact_builder)
        self.assertIn('DOCKER_BIN="${DOCKER_BIN:-}"', artifact_builder)
        self.assertIn("docker ps >/dev/null 2>&1", artifact_builder)
        self.assertIn("DOCKER_CMD=(docker)", artifact_builder)
        self.assertIn("DOCKER_CMD=(sudo docker)", artifact_builder)
        self.assertIn('-v "$BUILD_LOG_DIR:/build-logs"', artifact_builder)
        self.assertIn("llama-server.help.txt", artifact_builder)
        self.assertIn("copied-files.txt", artifact_builder)
        self.assertIn('"${DOCKER_CMD[@]}" run --rm', artifact_builder)

    def test_llama_cpp_image_builder_records_ref_in_tag_and_labels(self):
        image_builder = Path("scripts/build_llama_cpp_image.sh").read_text(encoding="utf-8")

        self.assertIn('LLAMA_CPP_REF="${LLAMA_CPP_REF:-', image_builder)
        self.assertIn('BUILD_LOG_DIR="${BUILD_LOG_DIR:-$PWD/outputs/build/llama-cpp}"', image_builder)
        self.assertIn('IMAGE_TAG="${IMAGE_TAG:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-${LLAMA_CPP_REF:0:7}}"', image_builder)
        self.assertIn("--build-arg", image_builder)
        self.assertIn("LLAMA_CPP_REF=${LLAMA_CPP_REF}", image_builder)
        self.assertIn("image-manifest", image_builder)
        self.assertIn("artifact-files", image_builder)
        self.assertIn("-f docker/llama-cpp/Dockerfile", image_builder)

    def test_llama_cpp_image_builder_rejects_unexpected_artifact_files(self):
        image_builder = Path("scripts/build_llama_cpp_image.sh").read_text(encoding="utf-8")

        self.assertIn("allowed_artifact_files=(", image_builder)
        self.assertIn("LLAMA_CPP_REF", image_builder)
        self.assertIn("bin/llama-server", image_builder)
        self.assertIn("bin/llama-mtmd-cli", image_builder)
        self.assertIn("Unexpected file in artifacts/llama.cpp-install", image_builder)

    def test_llama_cpp_docker_context_excludes_local_state_and_keeps_artifacts(self):
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

        self.assertIn("**", dockerignore)
        self.assertIn("!.dockerignore", dockerignore)
        self.assertIn("!docker/llama-cpp/Dockerfile", dockerignore)
        self.assertIn("!artifacts/llama.cpp-install/**", dockerignore)
        self.assertNotIn("!.env", dockerignore)
        self.assertNotIn("!models/", dockerignore)
        self.assertNotIn("!outputs/", dockerignore)

    def test_llama_cpp_build_artifacts_are_not_git_tracked(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("artifacts/", gitignore)

    def test_jetson_sweep_server_processes_are_group_terminated(self):
        sweep = Path("src/edge_vlm/jetson_sweep.py").read_text(encoding="utf-8")

        self.assertIn("start_new_session=True", sweep)
        self.assertIn("os.killpg", sweep)
        self.assertIn("signal.SIGTERM", sweep)
        self.assertIn("signal.SIGKILL", sweep)

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

    def test_optimization_report_rejects_quality_term_miss(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fast_off_topic = tmp_path / "fast_off_topic.jsonl"
            steady_relevant = tmp_path / "steady_relevant.jsonl"
            report = tmp_path / "report.md"
            fast_off_topic.write_text(
                json.dumps(
                    {
                        "model": "local-fast",
                        "run_id": "fast-off-topic",
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 0.5,
                        "tokens": 64,
                        "tokens_per_sec": 128.0,
                        "output_excerpt": "A fluent answer that is long enough but avoids the required subject.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            steady_relevant.write_text(
                json.dumps(
                    {
                        "model": "local-steady",
                        "run_id": "steady-relevant",
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 2.0,
                        "tokens": 64,
                        "tokens_per_sec": 32.0,
                        "output_excerpt": "A useful answer that names memory pressure and bandwidth limits.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[fast_off_topic, steady_relevant],
                output_path=report,
                min_output_chars=24,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([summary.run_id for summary in summaries], ["steady-relevant", "fast-off-topic"])
        self.assertTrue(summaries[0].guard_passed)
        self.assertFalse(summaries[1].guard_passed)
        self.assertIn("quality_terms_miss", report_text)

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

    def test_optimization_comparison_report_adds_manifest_context_and_deltas(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "gemma-compare"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(run_prefix, variant_id, *, text_tps, image_tps, fake_latency, startup_s, lfb_blocks, power_w):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": "gemma4-e2b-it-q4",
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": "gemma4-e2b-it-q4",
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
                                    "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                    "quality_terms_any": ["square", "background"],
                                }
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                fake_jsonl.write_text(
                    json.dumps(
                        {
                            "frame_id": "frame_001.png",
                            "success": True,
                            "latency_s": fake_latency,
                            "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tegrastats_log.write_text(
                    "\n".join(
                        [
                            (
                                "05-31-2026 RAM 2000/7620MB (lfb 200x4MB) "
                                "GR3D_FREQ 80%@[1020] EMC_FREQ 82%@3199 cpu@50.0C gpu@51.0C tj@51.0C "
                                f"VDD_IN {int(power_w * 1000)}mW/{int(power_w * 1000)}mW"
                            ),
                            (
                                "05-31-2026 RAM 2100/7620MB (lfb 180x4MB) "
                                "GR3D_FREQ 90%@[1020] EMC_FREQ 84%@3199 cpu@52.0C gpu@54.5C tj@54.5C "
                                f"VDD_IN {int((power_w + 1) * 1000)}mW/{int((power_w + 1) * 1000)}mW"
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                benchmark_manifest.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "benchmark": {"trial_count": 1},
                            "cases_written": 2,
                            "successful": 2,
                            "failed": 0,
                            "jetson": {"tegrastats_log": str(tegrastats_log)},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "server_runtime": {
                            "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                            "image_id": "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
                            "llama_cpp_ref": "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774",
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        }
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": lfb_blocks, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "gemma-baseline",
                "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                text_tps=10.0,
                image_tps=8.0,
                fake_latency=9.0,
                startup_s=5.0,
                lfb_blocks=180,
                power_w=10.0,
            )
            candidate = write_run(
                "gemma-directio",
                "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio",
                text_tps=12.0,
                image_tps=7.0,
                fake_latency=8.0,
                startup_s=6.0,
                lfb_blocks=190,
                power_w=12.0,
            )
            manifest = output_root / "gemma-compare.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {"run_prefix": "gemma-compare", "variants": [baseline["plan"], candidate["plan"]]},
                        "result": {"results": [baseline["result"], candidate["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["gemma-q4-baseline-gpu12-b512-u512-kvq8"],
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([row.variant_id for row in rows], [
            "gemma-q4-baseline-gpu12-b512-u512-kvq8",
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio",
        ])
        self.assertEqual(rows[0].server_image, "ghcr.io/4everwz/jetson-llama-cpp:test")
        self.assertEqual(rows[0].server_image_id, "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844")
        self.assertEqual(rows[0].llama_cpp_ref, "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774")
        self.assertEqual(rows[1].delta_text_tokens_per_s_pct, 20.0)
        self.assertEqual(rows[1].delta_image_tokens_per_s_pct, -12.5)
        self.assertAlmostEqual(rows[1].delta_fake_stream_latency_pct, -11.111111, places=5)
        self.assertIn("| Model | Variant | Run prefix | Runtime | Preflight lfb |", report_text)
        self.assertIn("Avg GR3D %", report_text)
        self.assertIn("Avg EMC %", report_text)
        self.assertIn("Bottlenecks", report_text)
        self.assertIn("gpu_compute", report_text)
        self.assertEqual(rows[0].avg_gr3d_util_pct, 85.0)
        self.assertEqual(rows[0].avg_emc_util_pct, 83.0)
        self.assertEqual(rows[0].min_lfb_free_blocks, 180)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:test / 52a8ad644e41 / b4c0549a49be", report_text)
        self.assertIn("| gemma4-e2b-it-q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` | gemma-directio | ghcr.io/4everwz/jetson-llama-cpp:test / 52a8ad644e41 / b4c0549a49be | 190x4MB | 1 | yes | 2/2 | 1/1 | 6.000 | 12.000 | 7.000 | 5.333 | 9.143 | 8.000 | 54.500 | 12.500 | 85.000 | 83.000 | 180 | gpu_compute, emc_memory_bandwidth | +20.00% | -12.50% | +20.00% | -11.11% |", report_text)
        self.assertIn("Baseline rows use `0.00%` deltas", report_text)

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

    def test_jetson_profile_labels_input_payload_and_runtime_overhead_from_input_timing(self):
        from edge_vlm.jetson_profile import summarize_input_timing_records, summarize_tegrastats_samples

        payload_input_summary = summarize_input_timing_records(
            [
                {
                    "latency_s": 1.0,
                    "input_timing": {
                        "image_bytes": 1024,
                        "payload_build_s": 0.30,
                        "json_serialize_s": 0.10,
                        "http_request_s": 0.80,
                        "response_parse_s": 0.02,
                        "request_body_bytes": 2048,
                    },
                },
                {
                    "latency_s": 1.4,
                    "input_timing": {
                        "image_bytes": 512,
                        "payload_build_s": 0.20,
                        "json_serialize_s": 0.10,
                        "http_request_s": 1.10,
                        "response_parse_s": 0.02,
                        "request_body_bytes": 1024,
                    },
                },
            ],
            sources={"benchmark_jsonl": 2},
        )
        payload_summary = summarize_tegrastats_samples(
            [
                {
                    "cpu": {"cores": [{"util_pct": 20}]},
                    "gr3d": {"util_pct": 10},
                    "emc": {"util_pct": 25},
                }
            ],
            input_timing_summary=payload_input_summary,
        )

        runtime_input_summary = summarize_input_timing_records(
            [
                {
                    "latency_s": 2.5,
                    "input_timing": {
                        "payload_build_s": 0.02,
                        "json_serialize_s": 0.01,
                        "http_request_s": 2.40,
                        "response_parse_s": 0.01,
                        "request_body_bytes": 640,
                    },
                }
            ],
            sources={"benchmark_jsonl": 1},
        )
        runtime_summary = summarize_tegrastats_samples(
            [
                {
                    "cpu": {"cores": [{"util_pct": 25}]},
                    "gr3d": {"util_pct": 12},
                    "emc": {"util_pct": 30},
                }
            ],
            input_timing_summary=runtime_input_summary,
        )

        self.assertTrue(payload_input_summary["available"])
        self.assertEqual(payload_input_summary["records"], 2)
        self.assertEqual(payload_input_summary["avg_payload_overhead_s"], 0.35)
        self.assertEqual(payload_input_summary["avg_payload_overhead_ratio"], 0.226)
        self.assertIn("input_payload", payload_summary["bottleneck_labels"])
        self.assertNotIn("runtime_overhead", payload_summary["bottleneck_labels"])
        self.assertIn("runtime_overhead", runtime_summary["bottleneck_labels"])
        self.assertNotIn("input_payload", runtime_summary["bottleneck_labels"])

    def test_jetson_profile_writes_profile_jsonl_and_phase_summary(self):
        from edge_vlm.jetson_profile import write_profile_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log = tmp_path / "tegrastats.log"
            profile_jsonl = tmp_path / "profile.jsonl"
            summary_json = tmp_path / "summary.json"
            log.write_text(
                "\n".join(
                    [
                        "RAM 2000/7620MB (lfb 200x4MB) CPU [40%@1728] GR3D_FREQ 12%@[1020] EMC_FREQ 31%@3199 gpu@54.0C VDD_IN 8000mW/7000mW",
                        "RAM 2200/7620MB (lfb 160x4MB) CPU [45%@1728] GR3D_FREQ 18%@[1020] EMC_FREQ 36%@3199 gpu@58.0C VDD_IN 9000mW/8000mW",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_profile_artifacts(
                tegrastats_log=log,
                profile_jsonl_path=profile_jsonl,
                summary_path=summary_json,
                phase_timings={"server_startup": {"available": True, "duration_s": 35.0}},
                profile_files={"jetson_clocks": "outputs/benchmarks/unit.profile/jetson-clocks.txt"},
            )
            profile_records = [json.loads(line) for line in profile_jsonl.read_text(encoding="utf-8").splitlines()]
            summary_record = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(len(profile_records), 2)
        self.assertEqual(profile_records[0]["sample_index"], 0)
        self.assertEqual(profile_records[0]["sample"]["ram"]["used_mb"], 2000)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["phase_timings"]["server_startup"]["duration_s"], 35.0)
        self.assertFalse(summary["phase_timings"]["artifact_check_or_download"]["available"])
        self.assertIn("startup_or_download", summary["bottleneck_labels"])
        self.assertEqual(
            summary["profile_files"]["jetson_clocks"],
            "outputs/benchmarks/unit.profile/jetson-clocks.txt",
        )
        self.assertEqual(summary_record, summary)

    def test_quality_review_applies_route_specific_policy_to_benchmark_jsonl(self):
        from edge_vlm.quality_review import format_markdown_report, review_benchmark_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_jsonl = tmp_path / "benchmark.jsonl"
            benchmark_jsonl.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 1,
                            "prompt_case_id": "text_code_short",
                            "input_type": "text",
                            "success": True,
                            "output_excerpt": "def tokens_per_second(latency_s, token_count):\n    return token_count / latency_s",
                        },
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 2,
                            "prompt_case_id": "text_code_short",
                            "input_type": "text",
                            "success": True,
                            "output_excerpt": "return latency * token_count",
                        },
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 3,
                            "prompt_case_id": "image_safety_scene_single",
                            "input_type": "image",
                            "success": True,
                            "output_excerpt": "No visible hazards are present in the simple square scene.",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            policy = {
                "default": {"min_output_chars": 12, "max_repeat_ratio": 0.8},
                "cases": {
                    "text_code_short": {
                        "must_include_all": ["def ", "tokens_per_second"],
                        "must_not_include_any": ["latency * token_count"],
                    },
                    "image_safety_scene_single": {
                        "must_include_any": ["no visible", "none"],
                        "must_not_include_any": ["fire", "knife"],
                    },
                },
            }

            report = review_benchmark_jsonl(benchmark_jsonl, policy)
            markdown = format_markdown_report(report)

        self.assertFalse(report["passed"])
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["failed_records"], 1)
        self.assertEqual(report["case_summaries"]["text_code_short"]["failed"], 1)
        self.assertIn("text_code_short", markdown)
        self.assertIn("missing_all:def ", json.dumps(report, ensure_ascii=False))
        self.assertIn("forbidden:latency * token_count", json.dumps(report, ensure_ascii=False))

    def test_quality_review_cli_and_docs_are_wired(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_jsonl = tmp_path / "benchmark.jsonl"
            report_json = tmp_path / "quality.json"
            report_md = tmp_path / "quality.md"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "run_id": "quality-cli-unit",
                        "model": "candidate",
                        "trial_index": 1,
                        "case_index": 1,
                        "prompt_case_id": "text_translation_zh_to_en_short",
                        "input_type": "text",
                        "success": True,
                        "output_excerpt": "Jetson Orin runs a vision language model on the edge with memory bandwidth, power, and latency constraints.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.quality_review",
                    "--input",
                    str(benchmark_jsonl),
                    "--policy",
                    "configs/benchmark/quality_review_policy.json",
                    "--output",
                    str(report_json),
                    "--markdown-output",
                    str(report_md),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src", "PYTHONPYCACHEPREFIX": "/tmp/edge-vlm-pycache"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            markdown = report_md.read_text(encoding="utf-8")

        self.assertTrue(report["passed"])
        self.assertIn("quality-cli-unit", report["run_ids"])
        self.assertIn("text_translation_zh_to_en_short", markdown)

        protocol = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")
        for text in (protocol, strategy, matrix):
            self.assertIn("edge_vlm.quality_review", text)
            self.assertIn("configs/benchmark/quality_review_policy.json", text)

    def test_next_phase_spec_orders_infra_before_model_expansion_and_lists_tencent_youtu_vl(self):
        spec = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")

        self.assertIn("Phase 1: Formal Benchmark Infra", spec)
        self.assertIn("Phase 2: Lightweight Model Expansion", spec)
        self.assertLess(
            spec.index("Phase 1: Formal Benchmark Infra"),
            spec.index("Phase 2: Lightweight Model Expansion"),
        )
        self.assertIn("tencent/Youtu-VL-4B-Instruct-GGUF", spec)
        self.assertIn("ggml-org/HunyuanOCR-GGUF", spec)
        self.assertIn("Hy-MT1.5 1.8B Safetensors", spec)
        self.assertIn("Hy-MT1.5", spec)
        self.assertIn("SmolVLM2", spec)
        self.assertIn("Qwen3-VL-2B", spec)

    def test_workflow_matrix_tracks_next_phase_infra_and_model_specs(self):
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        self.assertIn("next_phase_infra_and_model_strategy.md", matrix)
        self.assertIn("jetson_optimization_loop.md", matrix)
        self.assertIn("src/edge_vlm/jetson_profile.py", matrix)
        self.assertIn("scripts/jetson/run_optimization_sweep.sh", matrix)
        self.assertIn("scripts/jetson/run_remote_current_defaults_suite.sh", matrix)
        self.assertIn("run_hf_gguf_vlm_llama_docker.sh", matrix)
        self.assertIn("run_remote_tencent_text_suite.sh", matrix)
        self.assertIn("SmolVLM2", matrix)
        self.assertIn("Qwen3-VL", matrix)
        self.assertIn("HunyuanOCR", matrix)
        self.assertIn("Hy-MT1.5", matrix)
        self.assertIn("Youtu-VL", matrix)

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

    def test_jetson_optimization_variants_include_flash_attention_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        minicpm = by_id["minicpm-q4-baseline-b128-u32-kvq8-faon"]
        gemma = by_id["gemma-q4-baseline-gpu12-b512-u512-kvq8-faon"]

        for candidate in (minicpm, gemma):
            self.assertIn("--flash-attn", candidate["args"])
            flag_index = candidate["args"].index("--flash-attn")
            self.assertEqual(candidate["args"][flag_index + 1], "on")
            self.assertIn("--cache-type-k", candidate["args"])
            self.assertIn("--cache-type-v", candidate["args"])

        self.assertEqual(minicpm["env"]["LLAMA_BATCH_SIZE"], 128)
        self.assertEqual(minicpm["env"]["LLAMA_UBATCH_SIZE"], 32)
        self.assertEqual(gemma["env"]["LLAMA_BATCH_SIZE"], 512)
        self.assertEqual(gemma["env"]["LLAMA_UBATCH_SIZE"], 512)

    def test_jetson_optimization_variants_include_memory_mapping_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-mlock": ("minicpmv46-q4", 128, 32, "--mlock"),
            "minicpm-q4-baseline-b128-u32-kvq8-nommap": ("minicpmv46-q4", 128, 32, "--no-mmap"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock": ("gemma4-e2b-it-q4", 512, 512, "--mlock"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nommap": ("gemma4-e2b-it-q4", 512, 512, "--no-mmap"),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, candidate["args"])
                self.assertIn("--cache-type-k", candidate["args"])
                self.assertIn("--cache-type-v", candidate["args"])

    def test_jetson_optimization_variants_include_mlock_ulimit_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-mlock-ulimit": ("minicpmv46-q4", 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock-ulimit": ("gemma4-e2b-it-q4", 512, 512),
        }

        for variant_id, (model, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertEqual(candidate["env"]["DOCKER_GPU_ARGS"], "--runtime nvidia --ulimit memlock=-1:-1")
                self.assertIn("--mlock", candidate["args"])

    def test_jetson_optimization_variants_include_cache_precision_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kq4-vq8": ("minicpmv46-q4", 128, 32, "q4_0", "q8_0"),
            "minicpm-q4-baseline-b128-u32-kvq4": ("minicpmv46-q4", 128, 32, "q4_0", "q4_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kq4-vq8": ("gemma4-e2b-it-q4", 512, 512, "q4_0", "q8_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq4": ("gemma4-e2b-it-q4", 512, 512, "q4_0", "q4_0"),
        }

        for variant_id, (model, batch_size, ubatch_size, cache_k, cache_v) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertEqual(args[args.index("--cache-type-k") + 1], cache_k)
                self.assertEqual(args[args.index("--cache-type-v") + 1], cache_v)

    def test_jetson_optimization_variants_include_gemma_flash_attention_cache_precision_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "gemma-q4-baseline-gpu12-b512-u512-kq4-vq8-faon": ("q4_0", "q8_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq4-faon": ("q4_0", "q4_0"),
        }

        for variant_id, (cache_k, cache_v) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], "gemma4-e2b-it-q4")
                self.assertEqual(candidate["env"]["N_GPU_LAYERS"], 12)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], 512)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], 512)
                self.assertEqual(args[args.index("--cache-type-k") + 1], cache_k)
                self.assertEqual(args[args.index("--cache-type-v") + 1], cache_v)
                self.assertEqual(args[args.index("--flash-attn") + 1], "on")

    def test_jetson_optimization_variants_include_no_cont_batching_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-nocb": ("minicpmv46-q4", 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nocb": ("gemma4-e2b-it-q4", 512, 512),
        }

        for variant_id, (model, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn("--no-cont-batching", candidate["args"])
                self.assertIn("--cache-type-k", candidate["args"])
                self.assertIn("--cache-type-v", candidate["args"])

    def test_jetson_optimization_variants_include_prompt_cache_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-cache-ram0": ("minicpmv46-q4", 128, 32, "--cache-ram", "0"),
            "minicpm-q4-baseline-b128-u32-kvq8-nocacheprompt": ("minicpmv46-q4", 128, 32, "--no-cache-prompt", None),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-cache-ram0": ("gemma4-e2b-it-q4", 512, 512, "--cache-ram", "0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nocacheprompt": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--no-cache-prompt",
                None,
            ),
        }

        for variant_id, (model, batch_size, ubatch_size, flag, value) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                if value is not None:
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_host_repack_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-nohost": ("minicpmv46-q4", 128, 32, "--no-host"),
            "minicpm-q4-baseline-b128-u32-kvq8-norepack": ("minicpmv46-q4", 128, 32, "--no-repack"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nohost": ("gemma4-e2b-it-q4", 512, 512, "--no-host"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-norepack": ("gemma4-e2b-it-q4", 512, 512, "--no-repack"),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_direct_io_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-directio": ("minicpmv46-q4", 128, 32, "--direct-io"),
            "minicpm-q4-baseline-b128-u32-kvq8-nodirectio": ("minicpmv46-q4", 128, 32, "--no-direct-io"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--direct-io",
            ),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nodirectio": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--no-direct-io",
            ),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_warmup_candidates(self):
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-warmup": ("minicpmv46-q4", 32, 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-warmup": ("gemma4-e2b-it-q4", 12, 512, 512),
        }

        for variant_id, (model, n_gpu_layers, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["N_GPU_LAYERS"], n_gpu_layers)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)
                self.assertNotIn("--no-warmup", args)

    def test_lightweight_hf_gguf_vlm_configs_and_variants_exist(self):
        from edge_vlm.config import config_supports_images, load_model_config

        expected = {
            "smolvlm2-256m-q8": {
                "config": "configs/models/smolvlm2_256m_q8.yaml",
                "model_ref": "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0",
                "model_file": "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "mmproj_file": "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "ctx_size": 512,
            },
            "qwen3-vl-2b-thinking-q4": {
                "config": "configs/models/qwen3_vl_2b_thinking_q4.yaml",
                "model_ref": "Qwen/Qwen3-VL-2B-Thinking-GGUF:Q4_K_M",
                "model_file": "Qwen3VL-2B-Thinking-Q4_K_M.gguf",
                "mmproj_file": "mmproj-Qwen3VL-2B-Thinking-Q8_0.gguf",
                "ctx_size": 1024,
            },
            "hunyuanocr-q8": {
                "config": "configs/models/hunyuanocr_q8.yaml",
                "model_ref": "ggml-org/HunyuanOCR-GGUF:Q8_0",
                "model_file": "HunyuanOCR-Q8_0.gguf",
                "mmproj_file": "mmproj-HunyuanOCR-Q8_0.gguf",
                "ctx_size": 1024,
                "batch_size": 128,
                "ubatch_size": 32,
            },
            "youtu-vl-4b-q8": {
                "config": "configs/models/youtu_vl_4b_q8.yaml",
                "model_ref": "tencent/Youtu-VL-4B-Instruct-GGUF:Q8_0",
                "model_file": "Youtu-VL-4B-Instruct-Q8_0.gguf",
                "mmproj_file": "mmproj-Youtu-VL-4b-Instruct-BF16.gguf",
                "ctx_size": 2048,
            },
            "youtu-vl-4b-q4-thirdparty": {
                "config": "configs/models/youtu_vl_4b_q4_thirdparty.yaml",
                "model_ref": "mradermacher/Youtu-VL-4B-Instruct-GGUF:Q4_K_M",
                "model_file": "Youtu-VL-4B-Instruct.Q4_K_M.gguf",
                "mmproj_file": "Youtu-VL-4B-Instruct.mmproj-Q8_0.gguf",
                "ctx_size": 1024,
                "n_gpu_layers": 8,
                "batch_size": 256,
                "ubatch_size": 256,
                "required_arg": "--no-mmproj-offload",
            },
        }
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        def assert_arg_value(args: list[str], flag: str, expected_value: object) -> None:
            self.assertIn(flag, args)
            flag_index = args.index(flag)
            self.assertLess(flag_index + 1, len(args), f"{flag} has no value")
            self.assertEqual(args[flag_index + 1], str(expected_value))

        for model_name, expected_values in expected.items():
            with self.subTest(model_name=model_name):
                config = load_model_config(expected_values["config"])
                self.assertEqual(config["model"]["name"], model_name)
                self.assertEqual(config["model"]["model_ref"], expected_values["model_ref"])
                self.assertTrue(config_supports_images(config))
                self.assertEqual(config["runtime"]["model_file"], expected_values["model_file"])
                self.assertEqual(config["runtime"]["mmproj_file"], expected_values["mmproj_file"])
                self.assertEqual(
                    config["runtime"]["jetson_script"],
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                )

                variant_id = f"{model_name}-smoke"
                variant = by_id[variant_id]
                self.assertEqual(variant["model"], model_name)
                self.assertEqual(variant["config"], expected_values["config"])
                self.assertEqual(variant["launcher"], "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh")
                self.assertEqual(variant["env"]["MODEL_REF"], expected_values["model_ref"])
                self.assertEqual(variant["env"]["MODEL_FILE"], expected_values["model_file"])
                self.assertEqual(variant["env"]["MMPROJ_FILE"], expected_values["mmproj_file"])
                self.assertEqual(variant["env"]["CTX_SIZE"], expected_values["ctx_size"])
                self.assertEqual(variant["env"]["MODEL_ALIAS"], model_name)
                if "n_gpu_layers" in expected_values:
                    self.assertEqual(variant["env"]["N_GPU_LAYERS"], expected_values["n_gpu_layers"])
                if "batch_size" in expected_values:
                    self.assertEqual(variant["env"]["LLAMA_BATCH_SIZE"], expected_values["batch_size"])
                    assert_arg_value(variant["args"], "--batch-size", expected_values["batch_size"])
                if "ubatch_size" in expected_values:
                    self.assertEqual(variant["env"]["LLAMA_UBATCH_SIZE"], expected_values["ubatch_size"])
                    assert_arg_value(variant["args"], "--ubatch-size", expected_values["ubatch_size"])
                if "required_arg" in expected_values:
                    self.assertIn(expected_values["required_arg"], variant["args"])
                self.assertIn("--parallel", variant["args"])
                self.assertIn("--no-warmup", variant["args"])
                if model_name.endswith("-thirdparty"):
                    status = config["notes"]["status"]
                    self.assertIn("third-party", status)
                    self.assertIn("not an official Tencent GGUF artifact", status)

    def test_tencent_text_gguf_configs_and_variants_exist(self):
        from edge_vlm.config import config_supports_images, load_model_config

        expected = {
            "tencent-hy-mt1p5-1p8b-1p25bit": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_1p25bit.yaml",
                "model_ref": "tencent/Hy-MT1.5-1.8B-1.25bit-GGUF:1.25bit",
                "model_file": "Hy-MT1.5-1.8B-1.25bit.gguf",
                "quantization": "1.25bit",
            },
            "tencent-hy-mt1p5-1p8b-2bit": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_2bit.yaml",
                "model_ref": "tencent/Hy-MT1.5-1.8B-2bit-GGUF:2bit",
                "model_file": "Hy-MT1.5-1.8B-2bit.gguf",
                "quantization": "2bit",
            },
            "tencent-hy-mt2-1p8b-1p25bit": {
                "config": "configs/models/tencent_hy_mt2_1p8b_1p25bit.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit",
                "model_file": "Hy-MT2-1.8B-1.25Bit.gguf",
                "quantization": "1.25Bit",
            },
            "tencent-hy-mt2-1p8b-2bit": {
                "config": "configs/models/tencent_hy_mt2_1p8b_2bit.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-2Bit-GGUF:2Bit",
                "model_file": "Hy-MT2-1.8B-2Bit.gguf",
                "quantization": "2Bit",
            },
            "tencent-hy-mt2-1p8b-q4": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q4.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
                "model_file": "Hy-MT2-1.8B-Q4_K_M.gguf",
                "quantization": "Q4_K_M",
            },
            "tencent-hy-mt2-1p8b-q6": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q6.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q6_K",
                "model_file": "Hy-MT2-1.8B-Q6_K.gguf",
                "quantization": "Q6_K",
            },
            "tencent-hy-mt2-1p8b-q8": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q8.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q8_0",
                "model_file": "Hy-MT2-1.8B-Q8_0.gguf",
                "quantization": "Q8_0",
            },
        }
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}
        text_cases = [
            json.loads(line)
            for line in Path("configs/benchmark/text_prompt_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertGreaterEqual(len(text_cases), 4)
        self.assertTrue(all(case.get("input_type") == "text" for case in text_cases))
        self.assertIn("text_translation_zh_to_en_short", {case["id"] for case in text_cases})
        for model_name, expected_values in expected.items():
            with self.subTest(model_name=model_name):
                config = load_model_config(expected_values["config"])
                self.assertEqual(config["model"]["name"], model_name)
                self.assertEqual(config["model"]["model_ref"], expected_values["model_ref"])
                self.assertEqual(config["model"]["quantization"], expected_values["quantization"])
                self.assertFalse(config_supports_images(config))
                self.assertEqual(config["runtime"]["model_file"], expected_values["model_file"])
                self.assertEqual(
                    config["runtime"]["jetson_script"],
                    "scripts/jetson/run_hf_gguf_llama_docker.sh",
                )

                variant = by_id[f"{model_name}-text-smoke"]
                self.assertEqual(variant["model"], model_name)
                self.assertEqual(variant["config"], expected_values["config"])
                self.assertEqual(variant["launcher"], "scripts/jetson/run_hf_gguf_llama_docker.sh")
                self.assertEqual(variant["env"]["MODEL_REF"], expected_values["model_ref"])
                self.assertEqual(variant["env"]["MODEL_FILE"], expected_values["model_file"])
                self.assertEqual(variant["env"]["MODEL_ALIAS"], model_name)
                self.assertEqual(variant["env"]["EDGE_VLM_CASES"], "configs/benchmark/text_prompt_cases.jsonl")
                self.assertIn("--parallel", variant["args"])
                self.assertIn("--no-warmup", variant["args"])

    def test_tencent_text_suite_docs_match_all_variant_default_policy(self):
        checked_paths = (
            "docs/specs/next_phase_infra_and_model_strategy.md",
            "docs/specs/next_phase_benchmark_and_models.md",
            "docs/benchmarks/jetson_lightweight_models_20260531.md",
            "docs/benchmark_protocol.md",
            "docs/matrix_edge_vlm_workflow.md",
            "configs/benchmark/jetson_optimization_variants.jsonl",
            "configs/models/tencent_hy_mt1p5_1p8b_1p25bit.yaml",
            "configs/models/tencent_hy_mt1p5_1p8b_2bit.yaml",
            "configs/models/tencent_hy_mt2_1p8b_1p25bit.yaml",
            "configs/models/tencent_hy_mt2_1p8b_2bit.yaml",
        )
        stale_phrases = (
            "defaults to Hy-MT2 Q4/Q6/Q8",
            "default-suite rows",
            "current-runtime-compatible Hy-MT2 Q4_K_M/Q6_K/Q8_0",
            "keep out of default text repeats",
            "stay out of default repeats",
            "not in the default text suite",
            "must be passed through `JETSON_TENCENT_TEXT_VARIANTS`",
            "Hy-MT2 Q4/Q6/Q8 rows only",
        )

        for path in checked_paths:
            text = Path(path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                for phrase in stale_phrases:
                    self.assertNotIn(phrase, text)

    def test_tencent_hy_mt2_q4_smoke_evidence_is_documented_as_text_only(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc):
            self.assertIn("tencent-hy-mt2-q4-smoke64-cached-20260531b", text)
            self.assertIn("19.759", text)
        self.assertIn("text/router", benchmark_doc)
        self.assertIn("not a VLM ranking row", model_doc)
        self.assertIn("guard-passing cached text smoke", matrix)

    def test_tencent_text_suite_full_smoke_evidence_is_documented_with_invalid_q8(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc, protocol_doc):
            self.assertIn("tencent-text-smoke64-20260531T120054Z", text)
            self.assertIn("33.541", text)
            self.assertIn("26.287", text)
            self.assertIn("tencent-hy-mt2-q6-smoke64-cached-20260531a", text)
            self.assertIn("26.298", text)
            self.assertIn("Q8 full-suite row is invalidated", text)
            self.assertIn("tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z", text)
            self.assertIn("30.756", text)
            self.assertIn("terminate_group", text)
            self.assertIn("tencent-text-repeat5-20260531a", text)
            self.assertIn("34.528", text)
            self.assertIn("31.395", text)
            self.assertIn("26.778", text)
            self.assertIn("20/20", text)
        self.assertIn("tencent-text-repeat5-20260531a", matrix)
        self.assertIn("invalid ggml type 42", benchmark_doc)
        self.assertIn("offset 203248672", benchmark_doc)
        self.assertIn("server_port_still_open_before_start", protocol_doc)

    def test_lightweight_formal_repeat_evidence_is_documented(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc, matrix):
            self.assertIn("lightweight-repeat5-20260531T134554Z", text)
        for metric in ("199.847", "48.598", "34.761", "7.502"):
            self.assertIn(metric, benchmark_doc)
        self.assertIn("latency_floor", strategy_doc)
        self.assertIn("balanced_candidate", strategy_doc)
        self.assertIn("runtime_overhead", strategy_doc)
        self.assertIn("official Youtu-VL Q8", model_doc)
        self.assertIn("Structured quality review", benchmark_doc)
        self.assertIn("MiniCPM-V 4.6 Q4 | 30/30", benchmark_doc)
        self.assertIn("SmolVLM2 256M Q8 | 20/30", benchmark_doc)
        self.assertIn("Qwen3-VL 2B Thinking Q4 | 20/30", benchmark_doc)
        self.assertIn("Youtu-VL 4B Q4 third-party | 30/30", benchmark_doc)
        self.assertIn("Qwen remains image/fake-stream balanced_candidate", strategy_doc)

    def test_jetson_hf_gguf_vlm_launcher_can_dry_run_model_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "MODEL_REF": "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0",
                "MODEL_FILE": "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "MMPROJ_FILE": "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "MODEL_ALIAS": "smolvlm2-256m-q8",
                "CTX_SIZE": "512",
                "N_GPU_LAYERS": "99",
                "VLM_SERVER_PORT": "19101",
            }
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                    "--parallel",
                    "1",
                    "--batch-size",
                    "128",
                    "--ubatch-size",
                    "32",
                    "--no-warmup",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("-m /models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/SmolVLM2-256M-Video-Instruct-Q8_0.gguf", result.stdout)
        self.assertIn("--mmproj /models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf", result.stdout)
        self.assertNotIn("-hf ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0", result.stdout)
        self.assertIn("--alias smolvlm2-256m-q8", result.stdout)
        self.assertIn("-p 19101:8080", result.stdout)
        self.assertIn("-c 512", result.stdout)
        self.assertIn("--n-gpu-layers 99", result.stdout)
        self.assertIn("--batch-size 128", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_hf_gguf_launcher_accepts_completed_partial_after_resume_416(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            fake_curl = bin_dir / "curl"
            fake_docker = bin_dir / "docker"
            fake_curl.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'curl: (22) The requested URL returned error: 416\\n' >&2",
                        "exit 22",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_docker.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf '%s\\n' \"$*\" > \"${FAKE_DOCKER_LOG:?}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_curl, 0o755)
            os.chmod(fake_docker, 0o755)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "ggml-org" / "HunyuanOCR-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "HunyuanOCR-Q8_0.gguf").write_bytes(b"GGUFmodel")
            partial = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf.partial"
            partial.write_bytes(b"GGUFmmproj")

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                    "--parallel",
                    "1",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_DOCKER_LOG": str(docker_log),
                    "DOCKER_TTY": "0",
                    "DOCKER_GPU_ARGS": "",
                    "LLAMA_CPP_DOCKER_IMAGE": "unit/llama-cpp:test",
                    "MODEL_DIR": str(model_dir),
                    "MODEL_REF": "ggml-org/HunyuanOCR-GGUF:Q8_0",
                    "MODEL_FILE": "HunyuanOCR-Q8_0.gguf",
                    "MMPROJ_FILE": "mmproj-HunyuanOCR-Q8_0.gguf",
                    "MODEL_ALIAS": "hunyuanocr-q8",
                },
            )

            completed = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf"
            docker_command = docker_log.read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(completed.is_file())
            self.assertFalse(partial.exists())
            self.assertIn("HTTP 416", result.stderr)
            self.assertIn("--mmproj /models/ggml-org/HunyuanOCR-GGUF/mmproj-HunyuanOCR-Q8_0.gguf", docker_command)

    def test_jetson_hf_gguf_text_launcher_can_dry_run_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "MODEL_REF": "tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit",
                "MODEL_FILE": "Hy-MT2-1.8B-1.25Bit.gguf",
                "MODEL_ALIAS": "tencent-hy-mt2-1p8b-1p25bit",
                "CTX_SIZE": "1024",
                "N_GPU_LAYERS": "99",
                "VLM_SERVER_PORT": "19102",
            }
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_llama_docker.sh",
                    "--parallel",
                    "1",
                    "--batch-size",
                    "128",
                    "--ubatch-size",
                    "32",
                    "--no-warmup",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("-m /models/tencent/Hy-MT2-1.8B-1.25Bit-GGUF/Hy-MT2-1.8B-1.25Bit.gguf", result.stdout)
        self.assertNotIn("--mmproj", result.stdout)
        self.assertNotIn("-hf tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit", result.stdout)
        self.assertIn("--alias tencent-hy-mt2-1p8b-1p25bit", result.stdout)
        self.assertIn("-p 19102:8080", result.stdout)
        self.assertIn("-c 1024", result.stdout)
        self.assertIn("--n-gpu-layers 99", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_jetson_hf_gguf_text_launcher_defaults_to_verified_hy_mt2_q4(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", "scripts/jetson/run_hf_gguf_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_DRY_RUN": "1",
                    "DOCKER_TTY": "0",
                    "MODEL_DIR": str(Path(tmp) / "models"),
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tencent/Hy-MT2-1.8B-GGUF", result.stdout)
        self.assertIn("Hy-MT2-1.8B-Q4_K_M.gguf", result.stdout)
        self.assertIn("--alias tencent-hy-mt2-1p8b-q4", result.stdout)

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
                self.assertGreaterEqual(
                    len(frames),
                    3,
                    f"fake-stream sample should include at least three frames: {image_dir}",
                )

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

    def test_jetson_launchers_record_optional_artifact_phase_logs(self):
        phase_helper = Path("scripts/jetson/phase_logging.sh").read_text(encoding="utf-8")
        self.assertIn("EDGE_VLM_LAUNCH_PHASE_LOG", phase_helper)
        self.assertIn("write_launch_phase", phase_helper)
        launcher_paths = [
            Path("scripts/jetson/run_minicpmv46_llama_docker.sh"),
            Path("scripts/jetson/run_gemma4_e2b_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_vlm_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_llama_docker.sh"),
        ]

        for launcher_path in launcher_paths:
            launcher = launcher_path.read_text(encoding="utf-8")
            self.assertIn("phase_logging.sh", launcher, str(launcher_path))
            self.assertIn("artifact_check_or_download", launcher, str(launcher_path))
            self.assertIn("write_launch_phase", launcher, str(launcher_path))

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

    def test_remote_optimization_sweep_can_prepare_max_clocks_without_logging_password(self):
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
                        "if [[ \"${1:-}\" == \"sudo\" ]]; then",
                        "  IFS= read -r password_from_stdin || true",
                        "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                        "fi",
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
                    "minicpm-q4-baseline-b128-u32-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_PREPARE_MAX_CLOCKS": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=sudo\nARG=-S\nARG=sh\nARG=-c\n", log_text)
        self.assertIn("jetson_clocks && jetson_clocks --show > outputs/jetson_inspect/jetson-clocks-max-", log_text)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_prepare_max_clocks_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

            env = {
                **os.environ,
                "JETSON_REMOTE_EXEC": str(fake_remote),
                "JETSON_REMOTE_SYNC": "0",
                "JETSON_REMOTE_PREPARE_MAX_CLOCKS": "1",
            }
            env.pop("JETSON_SSH_PASSWORD", None)
            env.pop("JETSON_REMOTE_SUDO_PASSWORD", None)
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
                env=env,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_REMOTE_PREPARE_MAX_CLOCKS requires", result.stderr)

    def test_remote_optimization_sweep_can_drop_caches_before_variants_without_logging_password(self):
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
                        "if [[ \"${1:-}\" == \"bash\" && \"${2:-}\" == \"-lc\" ]]; then",
                        "  IFS= read -r password_from_stdin || true",
                        "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                        "fi",
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
                    "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 1)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=bash\nARG=-lc\n", log_text)
        self.assertIn("mkfifo", log_text)
        self.assertIn("trap '' PIPE", log_text)
        self.assertIn('> "${pw_fifo}" 2>/dev/null', log_text)
        self.assertIn("--pre-variant-command", log_text)
        self.assertIn("sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertNotIn("sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertNotIn("|| exit 0", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_drop_caches_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

            env = {
                **os.environ,
                "JETSON_REMOTE_EXEC": str(fake_remote),
                "JETSON_REMOTE_SYNC": "0",
                "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
            }
            env.pop("JETSON_SSH_PASSWORD", None)
            env.pop("JETSON_REMOTE_SUDO_PASSWORD", None)
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT requires", result.stderr)

    def test_remote_optimization_sweep_drop_caches_rejects_existing_pre_variant_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                    "--pre-variant-command",
                    "true",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined with --pre-variant-command", result.stderr)

    def test_remote_current_defaults_suite_runs_locked_sweep_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_current_defaults_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_CURRENT_DEFAULTS_RUN_PREFIX": "defaults-unit",
                    "JETSON_CURRENT_DEFAULTS_TRIAL_COUNT": "7",
                    "JETSON_CURRENT_DEFAULTS_MAX_TOKENS": "33",
                    "JETSON_CURRENT_DEFAULTS_FAKE_STREAM_MAX_FRAMES": "2",
                    "JETSON_CURRENT_DEFAULTS_MIN_LFB_BLOCKS": "199",
                    "JETSON_CURRENT_DEFAULTS_WAIT_TIMEOUT_S": "123",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=defaults-unit\n", log_text)
        self.assertIn("SWEEP_ARG=--variant\nSWEEP_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("SWEEP_ARG=--variant\nSWEEP_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=7\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=33\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=2\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=199\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=123\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/defaults-unit/defaults-unit.manifest.json\n",
            log_text,
        )
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/defaults-unit/comparison.md\n",
            log_text,
        )

    def test_remote_lightweight_model_suite_runs_baselines_and_candidates_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_lightweight_model_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_LIGHTWEIGHT_RUN_PREFIX": "light-unit",
                    "JETSON_LIGHTWEIGHT_TRIAL_COUNT": "6",
                    "JETSON_LIGHTWEIGHT_MAX_TOKENS": "44",
                    "JETSON_LIGHTWEIGHT_FAKE_STREAM_MAX_FRAMES": "4",
                    "JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS": "177",
                    "JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S": "321",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=light-unit\n", log_text)
        for variant_id in (
            "minicpm-q4-baseline-b128-u32-kvq8",
            "gemma-q4-baseline-gpu12-b512-u512-kvq8",
            "smolvlm2-256m-q8-smoke",
            "qwen3-vl-2b-thinking-q4-smoke",
            "youtu-vl-4b-q4-thirdparty-smoke",
        ):
            self.assertIn(f"SWEEP_ARG=--variant\nSWEEP_ARG={variant_id}\n", log_text)
        self.assertNotIn("SWEEP_ARG=--variant\nSWEEP_ARG=youtu-vl-4b-q8-smoke\n", log_text)
        self.assertNotIn("SWEEP_ARG=--variant\nSWEEP_ARG=hunyuanocr-q8-smoke\n", log_text)
        protocol_text = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        self.assertIn("excludes HunyuanOCR and official Youtu-VL Q8", protocol_text)
        self.assertIn("JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=youtu-vl-4b-q8-smoke", protocol_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=6\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=44\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=4\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=177\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=321\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/light-unit/light-unit.manifest.json\n",
            log_text,
        )
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/light-unit/comparison.md\n",
            log_text,
        )

    def test_remote_tencent_text_suite_runs_text_candidates_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_tencent_text_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_TENCENT_TEXT_RUN_PREFIX": "tencent-text-unit",
                    "JETSON_TENCENT_TEXT_TRIAL_COUNT": "4",
                    "JETSON_TENCENT_TEXT_MAX_TOKENS": "55",
                    "JETSON_TENCENT_TEXT_MIN_LFB_BLOCKS": "188",
                    "JETSON_TENCENT_TEXT_WAIT_TIMEOUT_S": "222",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=tencent-text-unit\n", log_text)
        for variant_id in (
            "tencent-hy-mt1p5-1p8b-1p25bit-text-smoke",
            "tencent-hy-mt1p5-1p8b-2bit-text-smoke",
            "tencent-hy-mt2-1p8b-1p25bit-text-smoke",
            "tencent-hy-mt2-1p8b-2bit-text-smoke",
            "tencent-hy-mt2-1p8b-q4-text-smoke",
            "tencent-hy-mt2-1p8b-q6-text-smoke",
            "tencent-hy-mt2-1p8b-q8-text-smoke",
        ):
            self.assertIn(f"SWEEP_ARG=--variant\nSWEEP_ARG={variant_id}\n", log_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=4\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=55\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=0\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=188\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=222\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/tencent-text-unit/tencent-text-unit.manifest.json\n",
            log_text,
        )
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/tencent-text-unit/comparison.md\n",
            log_text,
        )

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

    def test_fake_stream_records_schedule_delay_and_backpressure(self):
        from edge_vlm.client import CompletionResult
        from edge_vlm.fake_stream import run_fake_stream

        class FakeClock:
            def __init__(self):
                self.now = 100.0
                self.sleeps: list[float] = []

            def perf_counter(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds

        class FakeClient:
            def __init__(self, clock):
                self.clock = clock
                self.latencies = [0.25, 1.25, 0.1]
                self.calls = 0

            def complete(self, **_kwargs):
                latency = self.latencies[self.calls]
                self.calls += 1
                self.clock.now += latency
                return CompletionResult(
                    ok=True,
                    text=f"frame {self.calls} ok with enough detail",
                    request={},
                    response={"dry_run": True},
                    latency_s=latency,
                    timings={"http_request_s": latency},
                )

        clock = FakeClock()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            for frame_id in ("001.jpg", "002.jpg", "003.jpg"):
                (image_dir / frame_id).write_bytes(b"\xff\xd8\xff\xd9")
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

            with patch("edge_vlm.fake_stream.OpenAICompatClient.from_config", return_value=FakeClient(clock)):
                with patch("edge_vlm.fake_stream.time.perf_counter", side_effect=clock.perf_counter):
                    with patch("edge_vlm.fake_stream.time.sleep", side_effect=clock.sleep):
                        count = run_fake_stream(
                            config_path=config,
                            image_dir=image_dir,
                            output_path=output,
                            prompt="Describe this frame.",
                            interval_s=1.0,
                            max_frames=3,
                        )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(clock.sleeps, [0.75])
        self.assertEqual(records[0]["stream_timing"]["scheduled_offset_s"], 0.0)
        self.assertEqual(records[0]["stream_timing"]["pre_frame_sleep_s"], 0.0)
        self.assertEqual(records[0]["stream_timing"]["schedule_delay_s"], 0.0)
        self.assertEqual(records[1]["stream_timing"]["scheduled_offset_s"], 1.0)
        self.assertEqual(records[1]["stream_timing"]["pre_frame_sleep_s"], 0.75)
        self.assertEqual(records[1]["stream_timing"]["schedule_delay_s"], 0.0)
        self.assertEqual(records[2]["stream_timing"]["scheduled_offset_s"], 2.0)
        self.assertEqual(records[2]["stream_timing"]["pre_frame_sleep_s"], 0.0)
        self.assertEqual(records[2]["stream_timing"]["schedule_delay_s"], 0.25)
        self.assertEqual(records[2]["stream_timing"]["backpressure_s"], 0.25)

    def test_fake_stream_can_skip_late_frames_without_contacting_model(self):
        from edge_vlm.client import CompletionResult
        from edge_vlm.fake_stream import run_fake_stream

        class FakeClock:
            def __init__(self):
                self.now = 100.0
                self.sleeps: list[float] = []

            def perf_counter(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds

        class FakeClient:
            def __init__(self, clock):
                self.clock = clock
                self.calls: list[str] = []
                self.latencies = [2.4, 0.1, 0.1]

            def complete(self, **kwargs):
                frame_name = Path(kwargs["image_path"]).name
                self.calls.append(frame_name)
                latency = self.latencies[len(self.calls) - 1]
                self.clock.now += latency
                return CompletionResult(
                    ok=True,
                    text=f"{frame_name} processed with enough detail",
                    request={},
                    response={"dry_run": True},
                    latency_s=latency,
                    timings={"http_request_s": latency},
                )

        clock = FakeClock()
        fake_client = FakeClient(clock)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            for frame_id in ("001.jpg", "002.jpg", "003.jpg", "004.jpg"):
                (image_dir / frame_id).write_bytes(b"\xff\xd8\xff\xd9")
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

            with patch("edge_vlm.fake_stream.OpenAICompatClient.from_config", return_value=fake_client):
                with patch("edge_vlm.fake_stream.time.perf_counter", side_effect=clock.perf_counter):
                    with patch("edge_vlm.fake_stream.time.sleep", side_effect=clock.sleep):
                        count = run_fake_stream(
                            config_path=config,
                            image_dir=image_dir,
                            output_path=output,
                            prompt="Describe this frame.",
                            interval_s=1.0,
                            max_frames=4,
                            skip_late_frames=True,
                            skip_threshold_s=1.0,
                        )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(fake_client.calls, ["001.jpg", "003.jpg", "004.jpg"])
        self.assertEqual([record["frame_id"] for record in records], ["001.jpg", "003.jpg", "004.jpg"])
        self.assertEqual(records[0]["stream_timing"]["skipped_frames_before"], 0)
        self.assertEqual(records[1]["frame_index"], 2)
        self.assertEqual(records[1]["stream_timing"]["skipped_frames_before"], 1)
        self.assertAlmostEqual(records[1]["stream_timing"]["schedule_delay_s"], 0.4)
        self.assertEqual(records[2]["stream_timing"]["skipped_frames_before"], 0)
        self.assertEqual(clock.sleeps, [0.5])

    def test_fake_stream_can_adapt_interval_from_previous_frame_elapsed_time(self):
        from edge_vlm.client import CompletionResult
        from edge_vlm.fake_stream import run_fake_stream

        class FakeClock:
            def __init__(self):
                self.now = 100.0
                self.sleeps: list[float] = []

            def perf_counter(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds

        class FakeClient:
            def __init__(self, clock):
                self.clock = clock
                self.latencies = [1.5, 0.2, 0.2]
                self.calls = 0

            def complete(self, **_kwargs):
                latency = self.latencies[self.calls]
                self.calls += 1
                self.clock.now += latency
                return CompletionResult(
                    ok=True,
                    text=f"frame {self.calls} ok with enough detail",
                    request={},
                    response={"dry_run": True},
                    latency_s=latency,
                    timings={"http_request_s": latency},
                )

        clock = FakeClock()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "frames"
            image_dir.mkdir()
            for frame_id in ("001.jpg", "002.jpg", "003.jpg"):
                (image_dir / frame_id).write_bytes(b"\xff\xd8\xff\xd9")
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

            with patch("edge_vlm.fake_stream.OpenAICompatClient.from_config", return_value=FakeClient(clock)):
                with patch("edge_vlm.fake_stream.time.perf_counter", side_effect=clock.perf_counter):
                    with patch("edge_vlm.fake_stream.time.sleep", side_effect=clock.sleep):
                        count = run_fake_stream(
                            config_path=config,
                            image_dir=image_dir,
                            output_path=output,
                            prompt="Describe this frame.",
                            interval_s=1.0,
                            max_frames=3,
                            adaptive_interval=True,
                            adaptive_interval_scale=1.0,
                        )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], 0.8)
        self.assertEqual(records[0]["stream_timing"]["scheduled_offset_s"], 0.0)
        self.assertEqual(records[0]["stream_timing"]["effective_interval_s"], 1.0)
        self.assertEqual(records[1]["stream_timing"]["scheduled_offset_s"], 1.5)
        self.assertEqual(records[1]["stream_timing"]["effective_interval_s"], 1.5)
        self.assertEqual(records[1]["stream_timing"]["schedule_delay_s"], 0.0)
        self.assertEqual(records[2]["stream_timing"]["scheduled_offset_s"], 2.5)
        self.assertEqual(records[2]["stream_timing"]["effective_interval_s"], 1.0)
        self.assertAlmostEqual(records[2]["stream_timing"]["pre_frame_sleep_s"], 0.8)


if __name__ == "__main__":
    unittest.main()
