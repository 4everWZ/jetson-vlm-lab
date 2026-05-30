"""Run Jetson optimization sweeps from JSONL variant definitions."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .optimization import build_optimization_report


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
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            yield record


def _timestamp_run_prefix() -> str:
    return datetime.now(timezone.utc).strftime("jetson-opt-%Y%m%dT%H%M%SZ")


_ENV_REF_RE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")


def _expand_env_refs(value: str, env: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group("braced") or match.group("bare")
        return env.get(key, match.group(0))

    return _ENV_REF_RE.sub(replace, value)


def _string_env(raw_env: dict[str, Any], base_env: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, raw_value in raw_env.items():
        value = str(raw_value)
        merged = {**base_env, **env}
        expanded = _expand_env_refs(value, merged)
        if _ENV_REF_RE.search(expanded):
            raise ValueError(f"unresolved environment reference in {key}: {expanded}")
        env[str(key)] = expanded
    return env


def _load_variants(path: str | Path) -> list[dict[str, Any]]:
    variants = list(_iter_jsonl(Path(path)))
    for variant in variants:
        for key in ("id", "model", "config", "launcher"):
            if key not in variant:
                raise ValueError(f"variant missing required key {key}: {variant}")
        if "args" in variant and not isinstance(variant["args"], list):
            raise ValueError(f"variant args must be a list: {variant['id']}")
        if "env" in variant and not isinstance(variant["env"], dict):
            raise ValueError(f"variant env must be a mapping: {variant['id']}")
    return variants


def _matches_filters(variant: dict[str, Any], model_filters: set[str], variant_filters: set[str]) -> bool:
    if model_filters and str(variant["model"]) not in model_filters:
        return False
    if variant_filters and str(variant["id"]) not in variant_filters:
        return False
    return True


def build_sweep_plan(
    *,
    variants_path: str | Path,
    model_filters: Iterable[str] = (),
    variant_filters: Iterable[str] = (),
    run_prefix: str,
    output_root: str | Path,
    server_log_dir: str | Path,
    port: int,
    trial_count: int,
    max_tokens: int,
    temperature: float,
    python_bin: str,
    include_fake_stream: bool = True,
    fake_stream_image_dir: str = "data/sample_stream",
    fake_stream_prompt: str = "Describe this frame.",
    fake_stream_max_frames: int = 1,
    base_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_env = dict(os.environ if base_env is None else base_env)
    output_base = Path(output_root)
    log_base = Path(server_log_dir)
    selected_models = set(model_filters)
    selected_variants = set(variant_filters)
    planned: list[dict[str, Any]] = []
    for variant in _load_variants(variants_path):
        if not _matches_filters(variant, selected_models, selected_variants):
            continue
        variant_id = str(variant["id"])
        run_id = f"{run_prefix}-{variant_id}"
        variant_env = _string_env(dict(variant.get("env", {})), source_env)
        server_env = {
            **variant_env,
            "VLM_SERVER_PORT": str(port),
            "DOCKER_TTY": variant_env.get("DOCKER_TTY", "0"),
            "EDGE_VLM_DEVICE": variant_env.get("EDGE_VLM_DEVICE", "jetson-orin"),
        }
        benchmark_jsonl = output_base / "benchmarks" / f"{run_id}.jsonl"
        summary_md = output_base / "benchmarks" / f"{run_id}.md"
        manifest_json = output_base / "benchmarks" / f"{run_id}.manifest.json"
        profile_dir = output_base / "benchmarks" / f"{run_id}.profile"
        fake_stream_jsonl = output_base / "fake_stream" / f"{run_id}.jsonl"
        server_log = log_base / f"{run_id}.server.log"
        benchmark_env = {
            **server_env,
            "EDGE_VLM_FORMAL_RUN_ID": run_id,
            "EDGE_VLM_CONFIG": str(variant["config"]),
            "EDGE_VLM_OUTPUT": str(benchmark_jsonl),
            "EDGE_VLM_SUMMARY_OUTPUT": str(summary_md),
            "EDGE_VLM_METADATA_OUTPUT": str(manifest_json),
            "EDGE_VLM_PROFILE_DIR": str(profile_dir),
            "EDGE_VLM_TRIAL_COUNT": str(trial_count),
            "EDGE_VLM_MAX_TOKENS": str(max_tokens),
            "EDGE_VLM_TEMPERATURE": str(temperature),
            "PYTHON_BIN": python_bin,
        }
        fake_stream_command = [
            python_bin,
            "-m",
            "edge_vlm.fake_stream",
            "--config",
            str(variant["config"]),
            "--image-dir",
            fake_stream_image_dir,
            "--output",
            str(fake_stream_jsonl),
            "--prompt",
            fake_stream_prompt,
            "--interval-s",
            "0",
            "--max-frames",
            str(fake_stream_max_frames),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]
        planned.append(
            {
                "variant": variant,
                "run_id": run_id,
                "server_command": ["bash", str(variant["launcher"])] + [str(arg) for arg in variant.get("args", [])],
                "server_env": server_env,
                "benchmark_command": ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                "benchmark_env": benchmark_env,
                "fake_stream_command": fake_stream_command if include_fake_stream else None,
                "fake_stream_env": server_env,
                "paths": {
                    "benchmark_jsonl": str(benchmark_jsonl),
                    "summary_md": str(summary_md),
                    "manifest_json": str(manifest_json),
                    "profile_dir": str(profile_dir),
                    "fake_stream_jsonl": str(fake_stream_jsonl),
                    "server_log": str(server_log),
                },
            }
        )
    return {
        "variants_path": str(variants_path),
        "run_prefix": run_prefix,
        "port": port,
        "variants": planned,
    }


def _merged_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(overrides)
    repo_src = str(Path.cwd() / "src")
    env["PYTHONPATH"] = repo_src + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    return env


def _wait_for_server(port: int, process: subprocess.Popen[Any], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def run_sweep(plan: dict[str, Any], *, wait_timeout_s: float, report_output: str | Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    benchmark_paths: list[str] = []
    for variant_plan in plan["variants"]:
        paths = variant_plan["paths"]
        Path(paths["server_log"]).parent.mkdir(parents=True, exist_ok=True)
        server_log = open(paths["server_log"], "w", encoding="utf-8")
        server = subprocess.Popen(
            variant_plan["server_command"],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=_merged_env(variant_plan["server_env"]),
            text=True,
        )
        try:
            ready = _wait_for_server(plan["port"], server, wait_timeout_s)
            if not ready:
                results.append(
                    {
                        "run_id": variant_plan["run_id"],
                        "variant_id": variant_plan["variant"]["id"],
                        "server_ready": False,
                        "server_returncode": server.poll(),
                        "benchmark_returncode": None,
                        "fake_stream_returncode": None,
                    }
                )
                continue
            benchmark_result = subprocess.run(
                variant_plan["benchmark_command"],
                check=False,
                env=_merged_env(variant_plan["benchmark_env"]),
                text=True,
            )
            fake_stream_returncode = None
            if variant_plan["fake_stream_command"] is not None:
                fake_stream_result = subprocess.run(
                    variant_plan["fake_stream_command"],
                    check=False,
                    env=_merged_env(variant_plan["fake_stream_env"]),
                    text=True,
                )
                fake_stream_returncode = fake_stream_result.returncode
            if benchmark_result.returncode == 0 and Path(paths["benchmark_jsonl"]).is_file():
                benchmark_paths.append(paths["benchmark_jsonl"])
            results.append(
                {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_plan["variant"]["id"],
                    "server_ready": True,
                    "server_returncode": server.poll(),
                    "benchmark_returncode": benchmark_result.returncode,
                    "fake_stream_returncode": fake_stream_returncode,
                }
            )
        finally:
            _terminate_process(server)
            server_log.close()
    if benchmark_paths:
        build_optimization_report(input_paths=benchmark_paths, output_path=report_output)
    return {
        "run_prefix": plan["run_prefix"],
        "results": results,
        "report_output": str(report_output) if benchmark_paths else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run Jetson model optimization sweep variants.")
    parser.add_argument("--variants", default="configs/benchmark/jetson_optimization_variants.jsonl")
    parser.add_argument("--model", action="append", default=[], help="Model id to include; repeatable")
    parser.add_argument("--variant", action="append", default=[], help="Variant id to include; repeatable")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--server-log-dir", default=None)
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLM_SERVER_PORT", "8080")))
    parser.add_argument("--trial-count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--skip-fake-stream", action="store_true")
    parser.add_argument("--fake-stream-image-dir", default="data/sample_stream")
    parser.add_argument("--fake-stream-prompt", default="Describe this frame.")
    parser.add_argument("--fake-stream-max-frames", type=int, default=1)
    parser.add_argument("--wait-timeout-s", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args(argv)

    run_prefix = args.run_prefix or _timestamp_run_prefix()
    output_root = args.output_root or f"outputs/optimization_sweeps/{run_prefix}"
    server_log_dir = args.server_log_dir or f"{output_root}/server_logs"
    report_output = args.report_output or f"{output_root}/optimization_report.md"
    plan = build_sweep_plan(
        variants_path=args.variants,
        model_filters=args.model,
        variant_filters=args.variant,
        run_prefix=run_prefix,
        output_root=output_root,
        server_log_dir=server_log_dir,
        port=args.port,
        trial_count=args.trial_count,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        python_bin=args.python_bin,
        include_fake_stream=not args.skip_fake_stream,
        fake_stream_image_dir=args.fake_stream_image_dir,
        fake_stream_prompt=args.fake_stream_prompt,
        fake_stream_max_frames=args.fake_stream_max_frames,
    )
    if not plan["variants"]:
        print(json.dumps({"error": "no variants selected", "variants": args.variants}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.plan_output:
        plan_output = Path(args.plan_output)
        plan_output.parent.mkdir(parents=True, exist_ok=True)
        plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "variants": len(plan["variants"]), "plan_output": args.plan_output}, ensure_ascii=False))
        return 0
    result = run_sweep(plan, wait_timeout_s=args.wait_timeout_s, report_output=report_output)
    manifest = Path(output_root) / f"{run_prefix}.manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"plan": plan, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(plan["variants"]), "manifest": str(manifest), "report": result["report_output"]}, ensure_ascii=False))
    return 0 if result["report_output"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
