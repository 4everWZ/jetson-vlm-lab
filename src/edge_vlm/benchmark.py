"""Benchmark harness for WSL and Jetson llama-server experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .client import OpenAICompatClient
from .config import config_supports_images, load_model_config


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: each JSONL line must be an object")
            yield record


def _device_label() -> str:
    override = os.environ.get("EDGE_VLM_DEVICE")
    if override:
        return override
    machine = platform.machine().lower()
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return "wsl"
    if machine in {"aarch64", "arm64"}:
        return "jetson-or-arm64"
    return machine or "unknown"


def _usage_tokens(response: dict[str, Any] | None) -> int | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        total = usage.get("completion_tokens") or usage.get("total_tokens")
        if isinstance(total, int):
            return total
    return None


def _build_record(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    image_path: str | None,
    result_ok: bool,
    output_text: str,
    error: str | None,
    response: dict[str, Any] | None,
    started_wall: str,
    ended_wall: str,
    latency_s: float,
) -> dict[str, Any]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    backend_cfg = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    tokens = _usage_tokens(response)
    return {
        "model": model_cfg.get("name"),
        "backend": model_cfg.get("backend") or backend_cfg.get("name"),
        "quantization": model_cfg.get("quantization"),
        "model_ref": model_cfg.get("model_ref"),
        "device": _device_label(),
        "prompt_case_id": case.get("id"),
        "input_type": case.get("input_type"),
        "image_path": image_path,
        "start_time": started_wall,
        "end_time": ended_wall,
        "latency_s": latency_s,
        "tokens": tokens,
        "tokens_per_sec": (tokens / latency_s) if tokens and latency_s > 0 else None,
        "success": result_ok,
        "error": error,
        "output_excerpt": output_text[:500],
    }


def run_benchmark(
    *,
    config_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    dry_run: bool = False,
    max_tokens: int = 128,
    temperature: float = 0.2,
    stream: bool = False,
    stop_on_error: bool = False,
) -> int:
    config = load_model_config(config_path)
    supports_images = config_supports_images(config)
    client = OpenAICompatClient.from_config(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("a", encoding="utf-8") as handle:
        for case in _iter_jsonl(Path(cases_path)):
            input_type = str(case.get("input_type", "text"))
            image_path = case.get("image_path")
            if input_type == "fake_stream":
                started_wall = ended_wall = datetime.now(timezone.utc).isoformat()
                result_ok = True
                output_text = (
                    "[fake stream case] run this case with "
                    "python -m edge_vlm.fake_stream using the image_dir value"
                )
                error = None
                response = {"fake_stream_case": True, "image_dir": case.get("image_dir")}
                latency_s = 0.0
                image_path = case.get("image_dir")
            elif input_type.startswith("image") and not supports_images:
                result_ok = False
                output_text = ""
                error = "case requires image input but selected config has capabilities.image=false"
                response = None
                latency_s = 0.0
                started_wall = ended_wall = datetime.now(timezone.utc).isoformat()
            else:
                started_wall = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                try:
                    result = client.complete(
                        prompt=str(case.get("prompt", "")),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                        image_path=image_path,
                        dry_run=dry_run,
                    )
                except (FileNotFoundError, ValueError, OSError) as exc:
                    ended_wall = datetime.now(timezone.utc).isoformat()
                    latency_s = time.perf_counter() - started
                    result_ok = False
                    output_text = ""
                    error = str(exc)
                    response = None
                    result = None
                else:
                    ended_wall = datetime.now(timezone.utc).isoformat()
                    latency_s = time.perf_counter() - started
                    result_ok = result.ok
                    output_text = result.text
                    error = result.error
                    response = result.response
            record = _build_record(
                config=config,
                case=case,
                image_path=image_path,
                result_ok=result_ok,
                output_text=output_text,
                error=error,
                response=response,
                started_wall=started_wall,
                ended_wall=ended_wall,
                latency_s=latency_s,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if stop_on_error and not result_ok:
                break
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run JSONL prompt cases against a local OpenAI-compatible server.")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--cases", default="configs/benchmark/prompt_cases.jsonl")
    parser.add_argument("--output", default="outputs/benchmarks/run.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    count = run_benchmark(
        config_path=args.config,
        cases_path=args.cases,
        output_path=args.output,
        dry_run=args.dry_run,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps({"cases_written": count, "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
