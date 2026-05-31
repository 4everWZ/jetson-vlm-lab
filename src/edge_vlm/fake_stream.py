"""Fake image stream runner for folder-based VLM experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .client import OpenAICompatClient
from .config import config_supports_images, load_model_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _wall_time_pair_from_latency(started_wall: datetime, latency_s: float) -> tuple[str, str]:
    ended_wall = started_wall + timedelta(seconds=latency_s)
    return started_wall.isoformat(), ended_wall.isoformat()


def _stream_timing_for_frame(
    *,
    stream_started: float,
    frame_index: int,
    interval_s: float,
    scheduled_offset_s: float | None = None,
    effective_interval_s: float | None = None,
) -> tuple[float, dict[str, Any]]:
    actual_scheduled_offset_s = (
        scheduled_offset_s
        if scheduled_offset_s is not None
        else (frame_index * interval_s if interval_s > 0 else 0.0)
    )
    actual_effective_interval_s = effective_interval_s if effective_interval_s is not None else interval_s
    scheduled_start = stream_started + actual_scheduled_offset_s
    before_sleep = time.perf_counter()
    pre_frame_sleep_s = 0.0
    if interval_s > 0 and before_sleep < scheduled_start:
        sleep_for = scheduled_start - before_sleep
        time.sleep(sleep_for)
        pre_frame_sleep_s = time.perf_counter() - before_sleep
    actual_start = time.perf_counter()
    schedule_delay_s = max(0.0, actual_start - scheduled_start) if interval_s > 0 else 0.0
    return actual_start, {
        "interval_s": interval_s,
        "effective_interval_s": actual_effective_interval_s,
        "scheduled_offset_s": actual_scheduled_offset_s,
        "pre_frame_sleep_s": pre_frame_sleep_s,
        "schedule_delay_s": schedule_delay_s,
        "backpressure_s": schedule_delay_s,
    }


def _current_schedule_delay(
    *,
    stream_started: float,
    frame_index: int,
    interval_s: float,
    scheduled_offset_s: float | None = None,
) -> float:
    if interval_s <= 0:
        return 0.0
    actual_scheduled_offset_s = (
        scheduled_offset_s
        if scheduled_offset_s is not None
        else frame_index * interval_s
    )
    scheduled_start = stream_started + actual_scheduled_offset_s
    return max(0.0, time.perf_counter() - scheduled_start)


def _should_skip_late_frame(
    *,
    stream_started: float,
    frame_index: int,
    interval_s: float,
    has_future_frame: bool,
    skip_late_frames: bool,
    skip_threshold_s: float | None,
    scheduled_offset_s: float | None = None,
) -> bool:
    if not skip_late_frames or interval_s <= 0 or not has_future_frame:
        return False
    threshold = skip_threshold_s if skip_threshold_s is not None else interval_s
    if threshold < 0:
        raise ValueError("skip_threshold_s must be >= 0")
    return _current_schedule_delay(
        stream_started=stream_started,
        frame_index=frame_index,
        interval_s=interval_s,
        scheduled_offset_s=scheduled_offset_s,
    ) >= threshold


def _next_adaptive_interval(
    *,
    base_interval_s: float,
    frame_elapsed_s: float,
    scale: float,
    max_interval_s: float | None,
) -> float:
    if base_interval_s <= 0:
        return 0.0
    if scale <= 0:
        raise ValueError("adaptive_interval_scale must be > 0")
    next_interval_s = max(base_interval_s, frame_elapsed_s * scale)
    if max_interval_s is not None:
        if max_interval_s < base_interval_s:
            raise ValueError("adaptive_interval_max_s must be >= interval_s")
        next_interval_s = min(next_interval_s, max_interval_s)
    return next_interval_s


def run_fake_stream(
    *,
    config_path: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
    prompt: str,
    interval_s: float = 1.0,
    max_frames: int | None = None,
    dry_run: bool = False,
    stop_on_error: bool = False,
    max_tokens: int = 128,
    temperature: float = 0.2,
    skip_late_frames: bool = False,
    skip_threshold_s: float | None = None,
    adaptive_interval: bool = False,
    adaptive_interval_scale: float = 1.0,
    adaptive_interval_max_s: float | None = None,
) -> int:
    config = load_model_config(config_path)
    if not config_supports_images(config):
        raise ValueError("fake stream requires a model config with capabilities.image=true")
    client = OpenAICompatClient.from_config(config)
    frames = _iter_images(Path(image_dir))
    if max_frames is not None:
        frames = frames[:max_frames]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    stream_started = time.perf_counter()
    skipped_frames_before = 0
    next_scheduled_offset_s = 0.0
    effective_interval_s = interval_s
    with output.open("a", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            scheduled_offset_s = next_scheduled_offset_s if adaptive_interval else None
            current_effective_interval_s = effective_interval_s if adaptive_interval else None
            if _should_skip_late_frame(
                stream_started=stream_started,
                frame_index=index,
                interval_s=interval_s,
                has_future_frame=index < len(frames) - 1,
                skip_late_frames=skip_late_frames,
                skip_threshold_s=skip_threshold_s,
                scheduled_offset_s=scheduled_offset_s,
            ):
                skipped_frames_before += 1
                if adaptive_interval:
                    next_scheduled_offset_s += effective_interval_s
                continue
            frame_started_mono, stream_timing = _stream_timing_for_frame(
                stream_started=stream_started,
                frame_index=index,
                interval_s=interval_s,
                scheduled_offset_s=scheduled_offset_s,
                effective_interval_s=current_effective_interval_s,
            )
            stream_timing["skipped_frames_before"] = skipped_frames_before
            skipped_frames_before = 0
            started_wall = datetime.now(timezone.utc)
            try:
                result = client.complete(
                    prompt=prompt,
                    image_path=frame,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    dry_run=dry_run,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                frame_elapsed_s = time.perf_counter() - frame_started_mono
                started, ended = _wall_time_pair_from_latency(started_wall, 0.0)
                record: dict[str, Any] = {
                    "frame_index": index,
                    "frame_id": frame.name,
                    "image_path": str(frame),
                    "start_time": started,
                    "end_time": ended,
                    "success": False,
                    "error": str(exc),
                    "latency_s": 0.0,
                    "output_excerpt": "",
                    "input_timing": {},
                    "stream_timing": {
                        **stream_timing,
                        "frame_elapsed_s": frame_elapsed_s,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                count += 1
                if adaptive_interval:
                    effective_interval_s = _next_adaptive_interval(
                        base_interval_s=interval_s,
                        frame_elapsed_s=frame_elapsed_s,
                        scale=adaptive_interval_scale,
                        max_interval_s=adaptive_interval_max_s,
                    )
                    next_scheduled_offset_s += effective_interval_s
                if stop_on_error:
                    break
                continue
            frame_elapsed_s = time.perf_counter() - frame_started_mono
            started, ended = _wall_time_pair_from_latency(started_wall, result.latency_s)
            record: dict[str, Any] = {
                "frame_index": index,
                "frame_id": frame.name,
                "image_path": str(frame),
                "start_time": started,
                "end_time": ended,
                "success": result.ok,
                "error": result.error,
                "latency_s": result.latency_s,
                "output_excerpt": result.text[:500],
                "input_timing": dict(result.timings),
                "stream_timing": {
                    **stream_timing,
                    "frame_elapsed_s": frame_elapsed_s,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if adaptive_interval:
                effective_interval_s = _next_adaptive_interval(
                    base_interval_s=interval_s,
                    frame_elapsed_s=frame_elapsed_s,
                    scale=adaptive_interval_scale,
                    max_interval_s=adaptive_interval_max_s,
                )
                next_scheduled_offset_s += effective_interval_s
            if stop_on_error and not result.ok:
                break
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send a sorted folder of images to a local VLM server.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default="outputs/fake_stream/run.jsonl")
    parser.add_argument("--prompt", default="Describe this frame.")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--skip-late-frames",
        action="store_true",
        help="Drop non-final source frames when backpressure exceeds the skip threshold.",
    )
    parser.add_argument(
        "--skip-threshold-s",
        type=float,
        default=None,
        help="Backpressure threshold for --skip-late-frames; defaults to --interval-s.",
    )
    parser.add_argument(
        "--adaptive-interval",
        action="store_true",
        help="Adapt the next frame interval from the previous processed frame elapsed time.",
    )
    parser.add_argument("--adaptive-interval-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-interval-max-s", type=float, default=None)
    args = parser.parse_args(argv)
    count = run_fake_stream(
        config_path=args.config,
        image_dir=args.image_dir,
        output_path=args.output,
        prompt=args.prompt,
        interval_s=args.interval_s,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        skip_late_frames=args.skip_late_frames,
        skip_threshold_s=args.skip_threshold_s,
        adaptive_interval=args.adaptive_interval,
        adaptive_interval_scale=args.adaptive_interval_scale,
        adaptive_interval_max_s=args.adaptive_interval_max_s,
    )
    print(json.dumps({"frames_written": count, "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
