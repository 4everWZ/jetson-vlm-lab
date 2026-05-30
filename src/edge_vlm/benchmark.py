"""Benchmark harness for WSL and Jetson llama-server experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .client import OpenAICompatClient
from .config import config_supports_images, load_model_config

RUNTIME_ENV_KEYS = (
    "EDGE_VLM_DEVICE",
    "LLAMA_CPP_DOCKER_IMAGE",
    "LLAMA_SERVER_CMD",
    "MODEL_ALIAS",
    "MODEL_PATH",
    "MMPROJ_PATH",
    "MODEL_REF",
    "MODEL_DIR",
    "CTX_SIZE",
    "N_GPU_LAYERS",
    "LLAMA_BATCH_SIZE",
    "LLAMA_UBATCH_SIZE",
    "LLAMA_PARALLEL",
    "LLAMA_THREADS",
    "VLM_SERVER_HOST",
    "VLM_SERVER_PORT",
    "EDGE_VLM_TEGRASTATS_LOG",
    "EDGE_VLM_TEGRASTATS_STATUS",
    "EDGE_VLM_POWER_MODE",
    "EDGE_VLM_JETSON_CLOCKS",
)


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


def _wall_time_pair_from_latency(started_wall: datetime, latency_s: float) -> tuple[str, str]:
    ended_wall = started_wall + timedelta(seconds=latency_s)
    return started_wall.isoformat(), ended_wall.isoformat()


def _default_run_id(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%dT%H%M%SZ")


def _runtime_env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS}


def _build_record(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    run_id: str,
    trial_index: int,
    case_index: int,
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
        "run_id": run_id,
        "trial_index": trial_index,
        "case_index": case_index,
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


def _format_markdown_summary(records: list[dict[str, Any]], output_path: Path) -> str:
    successes = sum(1 for record in records if record.get("success") is True)
    failures = len(records) - successes
    latencies = [record.get("latency_s") for record in records if isinstance(record.get("latency_s"), (int, float))]
    total_latency = sum(float(latency) for latency in latencies)
    lines = [
        "# Edge VLM Benchmark Summary",
        "",
        f"- Output JSONL: `{output_path}`",
        f"- Cases written: {len(records)}",
        f"- Successful: {successes}",
        f"- Failed: {failures}",
        f"- Total latency seconds: {total_latency:.3f}",
        "",
        "| Case | Input | Success | Latency s | Tokens | Tokens/s | Error |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        latency = record.get("latency_s")
        latency_text = f"{latency:.3f}" if isinstance(latency, (int, float)) else ""
        tokens = record.get("tokens")
        tokens_text = str(tokens) if isinstance(tokens, int) else ""
        tokens_per_sec = record.get("tokens_per_sec")
        tokens_per_sec_text = f"{tokens_per_sec:.3f}" if isinstance(tokens_per_sec, (int, float)) else ""
        error = str(record.get("error") or "").replace("|", "\\|")
        lines.append(
            "| {case} | {input_type} | {success} | {latency} | {tokens} | {tokens_per_sec} | {error} |".format(
                case=record.get("prompt_case_id") or "",
                input_type=record.get("input_type") or "",
                success="yes" if record.get("success") else "no",
                latency=latency_text,
                tokens=tokens_text,
                tokens_per_sec=tokens_per_sec_text,
                error=error,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_markdown_summary(records: list[dict[str, Any]], summary_path: Path, output_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_format_markdown_summary(records, output_path), encoding="utf-8")


def _write_metadata_manifest(
    *,
    metadata_path: Path,
    run_id: str,
    config: dict[str, Any],
    config_path: Path,
    cases_path: Path,
    output_path: Path,
    summary_path: Path | None,
    records: list[dict[str, Any]],
    started_at: datetime,
    ended_at: datetime,
    dry_run: bool,
    max_tokens: int,
    temperature: float,
    stream: bool,
    trial_count: int,
) -> None:
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    backend_cfg = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    successes = sum(1 for record in records if record.get("success") is True)
    env_snapshot = _runtime_env_snapshot()
    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "device": _device_label(),
        "paths": {
            "config": str(config_path),
            "cases": str(cases_path),
            "output": str(output_path),
            "summary": str(summary_path) if summary_path is not None else None,
            "metadata": str(metadata_path),
        },
        "model": {
            "name": model_cfg.get("name"),
            "family": model_cfg.get("family"),
            "backend": model_cfg.get("backend") or backend_cfg.get("name"),
            "model_ref": model_cfg.get("model_ref"),
            "quantization": model_cfg.get("quantization"),
        },
        "runtime": runtime_cfg,
        "benchmark": {
            "dry_run": dry_run,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "trial_count": trial_count,
        },
        "cases_written": len(records),
        "successful": successes,
        "failed": len(records) - successes,
        "runtime_env": env_snapshot,
        "jetson": {
            "tegrastats_log": env_snapshot.get("EDGE_VLM_TEGRASTATS_LOG"),
            "tegrastats_status": env_snapshot.get("EDGE_VLM_TEGRASTATS_STATUS"),
            "power_mode": env_snapshot.get("EDGE_VLM_POWER_MODE"),
            "jetson_clocks": env_snapshot.get("EDGE_VLM_JETSON_CLOCKS"),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark(
    *,
    config_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    run_id: str | None = None,
    trial_count: int = 1,
    trial_delay_s: float = 0.0,
    dry_run: bool = False,
    max_tokens: int = 128,
    temperature: float = 0.2,
    stream: bool = False,
    stop_on_error: bool = False,
) -> int:
    if trial_count < 1:
        raise ValueError("trial_count must be >= 1")
    config_file = Path(config_path)
    cases_file = Path(cases_path)
    output = Path(output_path)
    summary_file = Path(summary_path) if summary_path is not None else None
    metadata_file = Path(metadata_path) if metadata_path is not None else None
    started_at = datetime.now(timezone.utc)
    actual_run_id = run_id or _default_run_id(started_at)
    config = load_model_config(config_file)
    supports_images = config_supports_images(config)
    client = OpenAICompatClient.from_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    records: list[dict[str, Any]] = []
    cases = list(_iter_jsonl(cases_file))
    stop_requested = False
    with output.open("a", encoding="utf-8") as handle:
        for trial_index in range(1, trial_count + 1):
            for case_index, case in enumerate(cases, start=1):
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
                    started_wall_dt = datetime.now(timezone.utc)
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
                        latency_s = time.perf_counter() - started
                        started_wall, ended_wall = _wall_time_pair_from_latency(started_wall_dt, latency_s)
                        result_ok = False
                        output_text = ""
                        error = str(exc)
                        response = None
                        result = None
                    else:
                        latency_s = time.perf_counter() - started
                        started_wall, ended_wall = _wall_time_pair_from_latency(started_wall_dt, latency_s)
                        result_ok = result.ok
                        output_text = result.text
                        error = result.error
                        response = result.response
                record = _build_record(
                    config=config,
                    case=case,
                    run_id=actual_run_id,
                    trial_index=trial_index,
                    case_index=case_index,
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
                records.append(record)
                count += 1
                if stop_on_error and not result_ok:
                    stop_requested = True
                    break
            if stop_requested:
                break
            if trial_delay_s > 0 and trial_index < trial_count:
                time.sleep(trial_delay_s)
    if summary_file is not None:
        _write_markdown_summary(records, summary_file, output)
    if metadata_file is not None:
        _write_metadata_manifest(
            metadata_path=metadata_file,
            run_id=actual_run_id,
            config=config,
            config_path=config_file,
            cases_path=cases_file,
            output_path=output,
            summary_path=summary_file,
            records=records,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            dry_run=dry_run,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            trial_count=trial_count,
        )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run JSONL prompt cases against a local OpenAI-compatible server.")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--cases", default="configs/benchmark/prompt_cases.jsonl")
    parser.add_argument("--output", default="outputs/benchmarks/run.jsonl")
    parser.add_argument("--summary-output", default=None, help="Optional Markdown summary path for this run")
    parser.add_argument("--metadata-output", default=None, help="Optional JSON manifest path for this run")
    parser.add_argument("--run-id", default=None, help="Stable identifier written to each record and manifest")
    parser.add_argument("--trial-count", type=int, default=1, help="Repeat the full case set this many times")
    parser.add_argument("--trial-delay-s", type=float, default=0.0, help="Delay between repeated trial passes")
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
        summary_path=args.summary_output,
        metadata_path=args.metadata_output,
        run_id=args.run_id,
        trial_count=args.trial_count,
        trial_delay_s=args.trial_delay_s,
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
