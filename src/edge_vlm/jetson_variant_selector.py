"""Select a preferred Jetson variant from an ordered primary/fallback pair."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .config import config_supports_images, load_model_config
from .gguf_artifacts import inspect_gguf_artifacts
from .jetson_sweep import (
    SERVER_ENV_PASSTHROUGH_KEYS,
    _preflight_block_reason,
    _runtime_metadata,
    capture_preflight_sample,
)
from .jetson_memory_diagnostics import capture_memory_diagnostics


_ENV_REF_RE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")


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


def _load_variant(variants_path: str | Path, variant_id: str, base_env: dict[str, str]) -> dict[str, Any]:
    source = Path(variants_path)
    for record in _iter_jsonl(source):
        if str(record.get("id")) != variant_id:
            continue
        resolved_env = _string_env(dict(record.get("env", {})), base_env)
        inherited_env = {
            key: base_env[key]
            for key in SERVER_ENV_PASSTHROUGH_KEYS
            if key in base_env and key not in resolved_env
        }
        config_path = str(record["config"])
        supports_images = config_supports_images(load_model_config(config_path))
        return {
            "variant": record,
            "variant_id": variant_id,
            "model": str(record.get("model") or ""),
            "config": config_path,
            "launcher": str(record.get("launcher") or ""),
            "args": [str(arg) for arg in record.get("args", [])],
            "env": {**inherited_env, **resolved_env},
            "supports_images": supports_images,
        }
    raise ValueError(f"{source}: variant not found: {variant_id}")


def _runtime_block_reason(*, supports_images: bool, runtime: dict[str, Any]) -> str | None:
    if not supports_images:
        return None
    if runtime.get("llama_server_found") is False:
        return "runtime_missing_llama_server"
    if runtime.get("llama_server_help_ok") is True and runtime.get("llama_server_supports_mmproj") is False:
        return "runtime_missing_mmproj_support"
    return None


def _artifact_paths(variant_env: dict[str, str], *, supports_images: bool) -> dict[str, str | None]:
    model_dir = variant_env.get("MODEL_DIR")
    model_path = variant_env.get("MODEL_PATH_ON_HOST")
    mmproj_path = variant_env.get("MMPROJ_PATH_ON_HOST")
    if not model_path:
        model_ref = variant_env.get("MODEL_REF", "")
        model_repo = variant_env.get("MODEL_REPO") or model_ref.split(":", 1)[0]
        model_subdir = variant_env.get("MODEL_SUBDIR") or model_repo
        model_file = variant_env.get("MODEL_FILE")
        if model_dir and model_subdir and model_file:
            model_path = str(Path(model_dir) / model_subdir / model_file)
    if supports_images and not mmproj_path:
        model_ref = variant_env.get("MODEL_REF", "")
        model_repo = variant_env.get("MODEL_REPO") or model_ref.split(":", 1)[0]
        model_subdir = variant_env.get("MODEL_SUBDIR") or model_repo
        mmproj_file = variant_env.get("MMPROJ_FILE")
        if model_dir and model_subdir and mmproj_file:
            mmproj_path = str(Path(model_dir) / model_subdir / mmproj_file)
    return {
        "model_path_on_host": model_path,
        "mmproj_path_on_host": mmproj_path if supports_images else None,
    }


def _artifact_block_reasons(
    paths: dict[str, str | None],
    *,
    supports_images: bool,
    artifact_manifest: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    model_path = paths.get("model_path_on_host")
    if not model_path:
        reasons.append("missing_model_path_config")
    elif not Path(model_path).is_file():
        reasons.append("missing_model_artifact")
    mmproj_path = paths.get("mmproj_path_on_host")
    if supports_images:
        if not mmproj_path:
            reasons.append("missing_mmproj_path_config")
        elif not Path(mmproj_path).is_file():
            reasons.append("missing_mmproj_artifact")
    for artifact in artifact_manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        status = artifact.get("status")
        if status in ("ok", "missing"):
            continue
        role = artifact.get("role")
        if role == "model":
            reasons.append("invalid_model_artifact")
        elif role == "mmproj":
            reasons.append("invalid_mmproj_artifact")
    return reasons


def _artifact_manifest(paths: dict[str, str | None], *, supports_images: bool) -> dict[str, Any]:
    artifacts: list[tuple[str, str | Path]] = []
    model_path = paths.get("model_path_on_host")
    if model_path:
        artifacts.append(("model", model_path))
    mmproj_path = paths.get("mmproj_path_on_host")
    if supports_images and mmproj_path:
        artifacts.append(("mmproj", mmproj_path))
    if not artifacts:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "artifact_count": 0,
            "failed_count": 0,
            "reason": "no_artifact_paths_resolved",
            "artifacts": [],
        }
    return inspect_gguf_artifacts(artifacts)


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    runtime: dict[str, Any],
    preflight: dict[str, Any],
    min_lfb_blocks: int | None,
) -> dict[str, Any]:
    artifact_paths = _artifact_paths(candidate["env"], supports_images=bool(candidate["supports_images"]))
    artifact_manifest = _artifact_manifest(artifact_paths, supports_images=bool(candidate["supports_images"]))
    block_reasons = _artifact_block_reasons(
        artifact_paths,
        supports_images=bool(candidate["supports_images"]),
        artifact_manifest=artifact_manifest,
    )
    runtime_reason = _runtime_block_reason(
        supports_images=bool(candidate["supports_images"]),
        runtime=runtime,
    )
    if runtime_reason is not None:
        block_reasons.append(runtime_reason)
    preflight_reason = _preflight_block_reason(preflight, min_lfb_blocks)
    if preflight_reason is not None:
        block_reasons.append(preflight_reason)
    return {
        "variant_id": candidate["variant_id"],
        "model": candidate["model"],
        "config": candidate["config"],
        "launcher": candidate["launcher"],
        "args": list(candidate["args"]),
        "launcher_env": dict(candidate["env"]),
        "supports_images": bool(candidate["supports_images"]),
        "min_lfb_blocks": min_lfb_blocks,
        "artifact_paths": artifact_paths,
        "artifact_manifest": artifact_manifest,
        "block_reasons": block_reasons,
        "usable": not block_reasons,
        "chosen": False,
    }


def select_preferred_variant(
    *,
    variants_path: str | Path,
    primary_variant_id: str,
    fallback_variant_id: str,
    min_lfb_blocks: int | None,
    fallback_min_lfb_blocks: int | None = None,
    memory_diagnostics_output: str | Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_env = dict(base_env or {})
    primary = _load_variant(variants_path, primary_variant_id, source_env)
    fallback = _load_variant(variants_path, fallback_variant_id, source_env)
    runtime = _runtime_metadata(
        source_env.get("LLAMA_CPP_DOCKER_IMAGE"),
        source_env.get("LLAMA_SERVER_CMD"),
        source_env.get("DOCKER_GPU_ARGS"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        preflight = capture_preflight_sample(Path(tmp) / "selector.preflight.json")
    effective_fallback_min_lfb_blocks = min_lfb_blocks if fallback_min_lfb_blocks is None else fallback_min_lfb_blocks
    candidates = [
        _evaluate_candidate(primary, runtime=runtime, preflight=preflight, min_lfb_blocks=min_lfb_blocks),
        _evaluate_candidate(
            fallback,
            runtime=runtime,
            preflight=preflight,
            min_lfb_blocks=effective_fallback_min_lfb_blocks,
        ),
    ]
    selected_reason = "no_usable_variant"
    selected_variant_id: str | None = None
    for index, candidate in enumerate(candidates):
        if not candidate["usable"]:
            continue
        candidate["chosen"] = True
        selected_variant_id = str(candidate["variant_id"])
        selected_reason = "primary_usable" if index == 0 else "primary_blocked_selected_fallback"
        break
    selection = {
        "selected_variant_id": selected_variant_id,
        "selected_reason": selected_reason,
        "primary_variant_id": primary_variant_id,
        "fallback_variant_id": fallback_variant_id,
        "runtime": runtime,
        "preflight": preflight,
        "candidates": candidates,
    }
    if memory_diagnostics_output is not None:
        diagnostics_path = Path(memory_diagnostics_output)
        diagnostics = capture_memory_diagnostics(diagnostics_path)
        selection["memory_diagnostics_path"] = str(diagnostics_path)
        selection["memory_diagnostics_summary"] = dict(diagnostics.get("summary", {}))
    return selection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="configs/benchmark/jetson_optimization_variants.jsonl",
        help="Path to the Jetson variant catalog JSONL",
    )
    parser.add_argument(
        "--primary-variant",
        default="qwen3-vl-2b-instruct-q4-smoke",
        help="Preferred primary variant id",
    )
    parser.add_argument(
        "--fallback-variant",
        default="qwen3-vl-2b-instruct-q8-smoke",
        help="Fallback variant id to use when the primary is unusable",
    )
    parser.add_argument(
        "--min-lfb-blocks",
        type=int,
        default=150,
        help="Default conservative lfb gate applied to the primary variant",
    )
    parser.add_argument(
        "--fallback-min-lfb-blocks",
        type=int,
        default=None,
        help="Optional separate lfb gate for the fallback variant; defaults to --min-lfb-blocks",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--memory-diagnostics-output",
        default=None,
        help="Optional JSON sidecar path for low-level Jetson memory diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    selection = select_preferred_variant(
        variants_path=args.variants,
        primary_variant_id=args.primary_variant,
        fallback_variant_id=args.fallback_variant,
        min_lfb_blocks=args.min_lfb_blocks,
        fallback_min_lfb_blocks=args.fallback_min_lfb_blocks,
        memory_diagnostics_output=args.memory_diagnostics_output,
        base_env=dict(os.environ),
    )
    text = json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
