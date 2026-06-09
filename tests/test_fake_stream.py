"""Fake stream runner contract tests."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch



class FakeStreamContractsTest(unittest.TestCase):

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
