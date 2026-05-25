"""Fake image stream runner for folder-based VLM experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import OpenAICompatClient
from .config import config_supports_images, load_model_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


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
    with output.open("a", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            started = datetime.now(timezone.utc).isoformat()
            try:
                result = client.complete(
                    prompt=prompt,
                    image_path=frame,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    dry_run=dry_run,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                ended = datetime.now(timezone.utc).isoformat()
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
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                count += 1
                if stop_on_error:
                    break
                if interval_s > 0 and index < len(frames) - 1:
                    time.sleep(interval_s)
                continue
            ended = datetime.now(timezone.utc).isoformat()
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
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if stop_on_error and not result.ok:
                break
            if interval_s > 0 and index < len(frames) - 1:
                time.sleep(interval_s)
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
    )
    print(json.dumps({"frames_written": count, "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
