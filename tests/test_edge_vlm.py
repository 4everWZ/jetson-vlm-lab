import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EdgeVlmContractsTest(unittest.TestCase):
    def test_gemma_q8_config_is_available_for_low_memory_wsl_path(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/gemma4_e2b_q8.yaml")

        self.assertEqual(config["model"]["name"], "gemma4-e2b-it-q8")
        self.assertEqual(config["model"]["quantization"], "Q8_0")
        self.assertTrue(config["capabilities"]["image"])

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
