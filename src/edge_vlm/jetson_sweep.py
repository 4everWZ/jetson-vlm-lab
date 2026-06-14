"""Run Jetson optimization sweeps from JSONL variant definitions."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import llama_cpp_runtime as _llama_cpp_runtime
from .config import config_supports_images, load_model_config
from .jetson_profile import REQUIRED_PHASES, summarize_input_timing_records, write_profile_artifacts
from .optimization import build_optimization_report


LFB_RE = re.compile(r"\blfb\s+(?P<free_blocks>\d+)x(?P<block_mb>\d+)MB\b")
BUDDYINFO_RE = re.compile(r"^Node\s+(?P<node>\d+),\s+zone\s+(?P<zone>\S+)\s+(?P<counts>.+)$")
SERVER_ENV_PASSTHROUGH_KEYS = (
    "LLAMA_CPP_DOCKER_IMAGE",
    "LLAMA_CPP_DOCKER_IMAGE_FALLBACK",
    "LLAMA_SERVER_CMD",
    "DOCKER_GPU_ARGS",
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
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            yield record


def _timestamp_run_prefix() -> str:
    return datetime.now(timezone.utc).strftime("jetson-opt-%Y%m%dT%H%M%SZ")


def parse_tegrastats_lfb(line: str) -> dict[str, int] | None:
    match = LFB_RE.search(line)
    if match is None:
        return None
    return {
        "free_blocks": int(match.group("free_blocks")),
        "block_mb": int(match.group("block_mb")),
    }


def _read_meminfo() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return meminfo
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if not parts:
            continue
        try:
            meminfo[key] = int(parts[0])
        except ValueError:
            continue
    return meminfo


def parse_buddyinfo_line(line: str) -> dict[str, Any] | None:
    match = BUDDYINFO_RE.match(line.strip())
    if match is None:
        return None
    try:
        counts = [int(part) for part in match.group("counts").split()]
    except ValueError:
        return None
    max_order = None
    for order, count in enumerate(counts):
        if count > 0:
            max_order = order
    return {
        "node": int(match.group("node")),
        "zone": match.group("zone"),
        "free_blocks_by_order": counts,
        "max_order_with_free_block": max_order,
    }


def _read_buddyinfo() -> dict[str, Any]:
    path = Path("/proc/buddyinfo")
    if not path.is_file():
        return {"available": False, "zones": [], "max_order_with_free_block": None}
    zones = []
    max_order = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_buddyinfo_line(line)
        if parsed is None:
            continue
        zones.append(parsed)
        zone_max_order = parsed.get("max_order_with_free_block")
        if isinstance(zone_max_order, int) and (max_order is None or zone_max_order > max_order):
            max_order = zone_max_order
    return {
        "available": bool(zones),
        "zones": zones,
        "max_order_with_free_block": max_order,
    }


def _sample_tegrastats(interval_ms: int = 1000, timeout_s: float = 2.5) -> dict[str, Any]:
    if shutil.which("tegrastats") is None:
        return {"available": False, "raw": None, "lfb": None}
    process = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=2.0)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    raw = lines[-1] if lines else None
    return {
        "available": True,
        "raw": raw,
        "lfb": parse_tegrastats_lfb(raw or ""),
    }


def capture_preflight_sample(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    sample = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "meminfo_kb": _read_meminfo(),
        "buddyinfo": _read_buddyinfo(),
        "tegrastats": _sample_tegrastats(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sample


def _preflight_before_prepare_path(paths: dict[str, Any]) -> str | None:
    explicit = paths.get("preflight_before_prepare_json")
    if isinstance(explicit, str) and explicit:
        return explicit
    preflight_path = paths.get("preflight_json")
    if not isinstance(preflight_path, str) or not preflight_path:
        return None
    if preflight_path.endswith(".preflight.json"):
        return preflight_path.removesuffix(".preflight.json") + ".preflight-before-prepare.json"
    return preflight_path + ".before-prepare"


def _extract_preflight_lfb_free_blocks(preflight: dict[str, Any] | None) -> int | None:
    if not isinstance(preflight, dict):
        return None
    tegrastats = preflight.get("tegrastats")
    if not isinstance(tegrastats, dict):
        return None
    lfb = tegrastats.get("lfb")
    if not isinstance(lfb, dict):
        return None
    free_blocks = lfb.get("free_blocks")
    return int(free_blocks) if isinstance(free_blocks, int) else None


def _extract_preflight_mem_available_kb(preflight: dict[str, Any] | None) -> int | None:
    if not isinstance(preflight, dict):
        return None
    meminfo = preflight.get("meminfo_kb")
    if not isinstance(meminfo, dict):
        return None
    mem_available = meminfo.get("MemAvailable")
    return int(mem_available) if isinstance(mem_available, int) else None


def _extract_preflight_buddyinfo_max_order(preflight: dict[str, Any] | None) -> int | None:
    if not isinstance(preflight, dict):
        return None
    buddyinfo = preflight.get("buddyinfo")
    if not isinstance(buddyinfo, dict):
        return None
    max_order = buddyinfo.get("max_order_with_free_block")
    return int(max_order) if isinstance(max_order, int) else None


def _delta_or_none(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _preflight_prepare_delta(
    before_prepare: dict[str, Any] | None,
    after_prepare: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if before_prepare is None or after_prepare is None:
        return None
    return {
        "lfb_free_blocks_delta": _delta_or_none(
            _extract_preflight_lfb_free_blocks(before_prepare),
            _extract_preflight_lfb_free_blocks(after_prepare),
        ),
        "mem_available_kb_delta": _delta_or_none(
            _extract_preflight_mem_available_kb(before_prepare),
            _extract_preflight_mem_available_kb(after_prepare),
        ),
        "buddyinfo_max_order_delta": _delta_or_none(
            _extract_preflight_buddyinfo_max_order(before_prepare),
            _extract_preflight_buddyinfo_max_order(after_prepare),
        ),
    }


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


def _docker_image_metadata(image: str | None) -> dict[str, Any]:
    return _llama_cpp_runtime.docker_image_metadata(image)


def _docker_image_runtime_probe(
    image: str | None,
    llama_server_cmd: str | None = None,
    docker_gpu_args: str | None = None,
) -> dict[str, Any]:
    return _llama_cpp_runtime.docker_image_runtime_probe(image, llama_server_cmd, docker_gpu_args)


def _runtime_metadata(
    image: str | None,
    llama_server_cmd: str | None = None,
    docker_gpu_args: str | None = None,
) -> dict[str, Any]:
    metadata = _docker_image_metadata(image)
    metadata.update(_docker_image_runtime_probe(image, llama_server_cmd, docker_gpu_args))
    metadata["multimodal_ready"] = _llama_cpp_runtime.runtime_is_multimodal_ready(metadata)
    return metadata


def _env_flag(source_env: dict[str, str], key: str) -> bool:
    return str(source_env.get(key, "")).strip() == "1"


def _env_positive_int(source_env: dict[str, str], key: str) -> int | None:
    raw_value = str(source_env.get(key, "")).strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


def _prepare_context(source_env: dict[str, str]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if _env_flag(source_env, "EDGE_VLM_PREPARE_MAX_CLOCKS_ENABLED"):
        context["max_clocks_enabled"] = True
    max_clocks_capture = str(source_env.get("EDGE_VLM_PREPARE_MAX_CLOCKS_CAPTURE", "")).strip()
    if max_clocks_capture:
        context["max_clocks_capture"] = max_clocks_capture
    if _env_flag(source_env, "EDGE_VLM_DROP_CACHES_BEFORE_VARIANT"):
        context["drop_caches_before_variant"] = True
    memory_prepare_attempts = _env_positive_int(source_env, "EDGE_VLM_MEMORY_PREPARE_ATTEMPTS")
    if memory_prepare_attempts is not None:
        context["memory_prepare_attempts"] = memory_prepare_attempts
    pre_variant_command_source = str(source_env.get("EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE", "")).strip()
    if pre_variant_command_source:
        context["pre_variant_command_source"] = pre_variant_command_source
    return context


def _read_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _variant_comparison_group(variant: dict[str, Any]) -> str:
    group = variant.get("comparison_group")
    if isinstance(group, str) and group.strip():
        return group.strip()
    model = variant.get("model")
    return str(model).strip() if isinstance(model, str) else ""


def _selector_selection_id(
    record: dict[str, Any],
    variant_groups: dict[str, str],
) -> str | None:
    explicit = record.get("selection_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for key in ("selected_variant_id", "primary_variant_id", "fallback_variant_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            group = variant_groups.get(value.strip())
            if group:
                return f"{group}-auto"
    return None


def _selection_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "variant_id",
        "model",
        "config",
        "launcher",
        "supports_images",
        "min_lfb_blocks",
        "usable",
        "chosen",
    ):
        if key not in candidate:
            continue
        value = candidate.get(key)
        if isinstance(value, (str, int, bool)) or value is None:
            summary[key] = value
    block_reasons = candidate.get("block_reasons")
    if isinstance(block_reasons, list):
        summary["block_reasons"] = [str(reason) for reason in block_reasons]
    artifact_paths = candidate.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        summary["artifact_paths"] = {
            str(key): str(value) if value is not None else None
            for key, value in artifact_paths.items()
        }
    artifact_manifest = candidate.get("artifact_manifest")
    if isinstance(artifact_manifest, dict):
        summary["artifact_manifest"] = artifact_manifest
    return summary


def _normalize_selection_context(
    record: dict[str, Any],
    *,
    source_path: str | Path,
    variant_groups: dict[str, str],
) -> dict[str, Any]:
    selection_id = _selector_selection_id(record, variant_groups)
    if not selection_id:
        raise ValueError(f"{source_path}: selection context requires selection_id or a known variant id")
    selected_variant_id = record.get("selected_variant_id")
    selected_variant_id = str(selected_variant_id).strip() if isinstance(selected_variant_id, str) else None
    primary_variant_id = record.get("primary_variant_id")
    primary_variant_id = str(primary_variant_id).strip() if isinstance(primary_variant_id, str) else None
    fallback_variant_id = record.get("fallback_variant_id")
    fallback_variant_id = str(fallback_variant_id).strip() if isinstance(fallback_variant_id, str) else None
    comparison_group = record.get("comparison_group")
    if isinstance(comparison_group, str) and comparison_group.strip():
        normalized_group = comparison_group.strip()
    else:
        normalized_group = ""
        for candidate in (selected_variant_id, primary_variant_id, fallback_variant_id):
            if candidate and variant_groups.get(candidate):
                normalized_group = variant_groups[candidate]
                break
    normalized: dict[str, Any] = {
        "selection_id": selection_id,
        "selected_variant_id": selected_variant_id,
    }
    if isinstance(record.get("selected_reason"), str) and str(record.get("selected_reason")).strip():
        normalized["selected_reason"] = str(record.get("selected_reason")).strip()
    if normalized_group:
        normalized["comparison_group"] = normalized_group
    if primary_variant_id:
        normalized["primary_variant_id"] = primary_variant_id
    if fallback_variant_id:
        normalized["fallback_variant_id"] = fallback_variant_id
    candidates = record.get("candidates")
    if isinstance(candidates, list):
        normalized_candidates = [
            _selection_candidate_summary(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        if normalized_candidates:
            normalized["candidates"] = normalized_candidates
    return normalized


def _load_selection_contexts(
    paths: Iterable[str | Path],
    *,
    variant_groups: dict[str, str],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        record = _read_json_object(source)
        if not record:
            raise ValueError(f"{source}: expected selection context JSON object")
        contexts.append(
            _normalize_selection_context(
                record,
                source_path=source,
                variant_groups=variant_groups,
            )
        )
    return contexts


def _parse_variant_int_override(raw_value: str, *, flag_name: str) -> tuple[str, int]:
    variant_id, separator, value_text = str(raw_value).partition("=")
    variant_id = variant_id.strip()
    value_text = value_text.strip()
    if separator != "=" or not variant_id or not value_text:
        raise ValueError(f"{flag_name} expects variant_id=value, got: {raw_value}")
    try:
        value = int(value_text)
    except ValueError as exc:
        raise ValueError(f"{flag_name} expects an integer value, got: {raw_value}") from exc
    if value < 0:
        raise ValueError(f"{flag_name} expects a non-negative integer, got: {raw_value}")
    return variant_id, value


def _parse_variant_int_overrides(values: Iterable[str], *, flag_name: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for raw_value in values:
        variant_id, value = _parse_variant_int_override(raw_value, flag_name=flag_name)
        overrides[variant_id] = value
    return overrides


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


def _config_supports_fake_stream(config_path: str | Path) -> bool:
    return config_supports_images(load_model_config(config_path))


def _runtime_mapping(config_path: str | Path) -> dict[str, Any]:
    config = load_model_config(config_path)
    runtime = config.get("runtime")
    return runtime if isinstance(runtime, dict) else {}


def _model_mapping(config_path: str | Path) -> dict[str, Any]:
    config = load_model_config(config_path)
    model = config.get("model")
    return model if isinstance(model, dict) else {}


def _expand_path_template(value: Any, server_env: dict[str, str]) -> str:
    expanded = _expand_env_refs(str(value), {"MODEL_DIR": "/mnt/nvme/models", **server_env})
    if _ENV_REF_RE.search(expanded):
        raise ValueError(f"unresolved environment reference in artifact path: {expanded}")
    return expanded


def _hf_artifact_path(server_env: dict[str, str], runtime: dict[str, Any], model: dict[str, Any], file_key: str) -> str | None:
    file_name = server_env.get(file_key.upper()) or runtime.get(file_key.lower())
    if not file_name:
        return None
    model_dir = server_env.get("MODEL_DIR", "/mnt/nvme/models")
    model_ref = (
        server_env.get("MODEL_REPO")
        or server_env.get("MODEL_REF")
        or model.get("model_ref")
        or ""
    )
    repo_id = str(model_ref).split(":", 1)[0]
    if not repo_id:
        return None
    model_subdir = server_env.get("MODEL_SUBDIR") or str(runtime.get("model_subdir") or repo_id)
    return str(Path(model_dir) / model_subdir / str(file_name))


def _artifact_preflight(variant: dict[str, Any], server_env: dict[str, str], output_path: Path, python_bin: str) -> dict[str, Any]:
    runtime = _runtime_mapping(str(variant["config"]))
    model = _model_mapping(str(variant["config"]))
    artifacts: list[dict[str, str]] = []
    explicit_model_path = (
        server_env.get("MODEL_PATH_ON_HOST")
        or server_env.get("MODEL_PATH")
        or runtime.get("model_path")
    )
    if explicit_model_path:
        artifacts.append({"role": "model", "path": _expand_path_template(explicit_model_path, server_env)})
    else:
        model_path = _hf_artifact_path(server_env, runtime, model, "MODEL_FILE")
        if model_path:
            artifacts.append({"role": "model", "path": model_path})
    explicit_mmproj_path = (
        server_env.get("MMPROJ_PATH_ON_HOST")
        or server_env.get("MMPROJ_PATH")
        or runtime.get("mmproj_path")
    )
    if explicit_mmproj_path:
        artifacts.append({"role": "mmproj", "path": _expand_path_template(explicit_mmproj_path, server_env)})
    else:
        mmproj_path = _hf_artifact_path(server_env, runtime, model, "MMPROJ_FILE")
        if mmproj_path:
            artifacts.append({"role": "mmproj", "path": mmproj_path})
    command: list[str] = []
    if artifacts:
        command = [
            python_bin,
            "-m",
            "edge_vlm.gguf_artifacts",
            "check",
        ]
        for artifact in artifacts:
            command.extend(["--artifact", artifact["role"], artifact["path"]])
        command.extend(["--output", str(output_path)])
    return {
        "available": bool(artifacts),
        "artifacts": artifacts,
        "command": command,
    }


def _runtime_support_block_reason(variant_plan: dict[str, Any]) -> str | None:
    if not bool(variant_plan.get("supports_images")):
        return None
    runtime = variant_plan.get("server_runtime")
    if not isinstance(runtime, dict):
        return None
    if runtime.get("llama_server_found") is False:
        return "runtime_missing_llama_server"
    if runtime.get("llama_server_help_ok") is True and runtime.get("llama_server_supports_mmproj") is False:
        return "runtime_missing_mmproj_support"
    return None


def _warmup_phase_from_args(args: Iterable[Any]) -> dict[str, Any]:
    normalized_args = [str(arg) for arg in args]
    if "--no-warmup" in normalized_args:
        return {
            "available": False,
            "duration_s": None,
            "reason": "disabled_by_variant",
            "source": "sweep",
            "details": {
                "status": "disabled",
                "flag": "--no-warmup",
            },
        }
    return {
        "available": False,
        "duration_s": None,
        "reason": "included_in_server_startup",
        "source": "sweep",
        "details": {
            "status": "enabled_not_separated",
        },
    }


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
    fake_stream_max_frames: int = 3,
    fake_stream_interval_s: float = 0.0,
    fake_stream_skip_late_frames: bool = False,
    fake_stream_skip_threshold_s: float | None = None,
    fake_stream_adaptive_interval: bool = False,
    fake_stream_adaptive_interval_scale: float = 1.0,
    fake_stream_adaptive_interval_max_s: float | None = None,
    min_lfb_blocks: int | None = None,
    pre_variant_command: str | None = None,
    selection_contexts: Iterable[dict[str, Any]] = (),
    variant_min_lfb_blocks: dict[str, int] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_env = dict(os.environ if base_env is None else base_env)
    output_base = Path(output_root)
    log_base = Path(server_log_dir)
    selected_models = set(model_filters)
    selected_variants = set(variant_filters)
    planned: list[dict[str, Any]] = []
    runtime_by_image: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    variants = _load_variants(variants_path)
    for variant in variants:
        if not _matches_filters(variant, selected_models, selected_variants):
            continue
        variant_id = str(variant["id"])
        run_id = f"{run_prefix}-{variant_id}"
        variant_env = _string_env(dict(variant.get("env", {})), source_env)
        inherited_server_env = {
            key: source_env[key]
            for key in SERVER_ENV_PASSTHROUGH_KEYS
            if key in source_env and key not in variant_env
        }
        benchmark_jsonl = output_base / "benchmarks" / f"{run_id}.jsonl"
        summary_md = output_base / "benchmarks" / f"{run_id}.md"
        manifest_json = output_base / "benchmarks" / f"{run_id}.manifest.json"
        profile_dir = output_base / "benchmarks" / f"{run_id}.profile"
        profile_jsonl = output_base / "profiles" / f"{run_id}.profile.jsonl"
        profile_summary_json = output_base / "profiles" / f"{run_id}.summary.json"
        lifecycle_jsonl = output_base / "lifecycle" / f"{run_id}.lifecycle.jsonl"
        gguf_artifacts_json = output_base / "artifacts" / f"{run_id}.gguf-artifacts.json"
        fake_stream_jsonl = output_base / "fake_stream" / f"{run_id}.jsonl"
        server_log = log_base / f"{run_id}.server.log"
        preflight_before_prepare_json = output_base / "preflight" / f"{run_id}.preflight-before-prepare.json"
        preflight_json = output_base / "preflight" / f"{run_id}.preflight.json"
        server_env = {
            **inherited_server_env,
            **variant_env,
            "VLM_SERVER_PORT": str(port),
            "DOCKER_TTY": variant_env.get("DOCKER_TTY", "0"),
            "EDGE_VLM_DEVICE": variant_env.get("EDGE_VLM_DEVICE", "jetson-orin"),
            "EDGE_VLM_LAUNCH_PHASE_LOG": str(lifecycle_jsonl),
        }
        image = server_env.get("LLAMA_CPP_DOCKER_IMAGE")
        llama_server_cmd = server_env.get("LLAMA_SERVER_CMD")
        docker_gpu_args = server_env.get("DOCKER_GPU_ARGS")
        runtime_key = (image, llama_server_cmd, docker_gpu_args)
        if runtime_key not in runtime_by_image:
            runtime_by_image[runtime_key] = _runtime_metadata(
                image,
                llama_server_cmd,
                docker_gpu_args,
            )
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
        supports_fake_stream = _config_supports_fake_stream(str(variant["config"]))
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
            str(fake_stream_interval_s),
            "--max-frames",
            str(fake_stream_max_frames),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]
        if fake_stream_skip_late_frames:
            fake_stream_command.append("--skip-late-frames")
        if fake_stream_skip_threshold_s is not None:
            fake_stream_command.extend(["--skip-threshold-s", str(fake_stream_skip_threshold_s)])
        if fake_stream_adaptive_interval:
            fake_stream_command.extend(
                [
                    "--adaptive-interval",
                    "--adaptive-interval-scale",
                    str(fake_stream_adaptive_interval_scale),
                ]
            )
            if fake_stream_adaptive_interval_max_s is not None:
                fake_stream_command.extend(["--adaptive-interval-max-s", str(fake_stream_adaptive_interval_max_s)])
        planned.append(
            {
                "variant": variant,
                "supports_images": supports_fake_stream,
                "run_id": run_id,
                "server_command": ["bash", str(variant["launcher"])] + [str(arg) for arg in variant.get("args", [])],
                "server_env": server_env,
                "server_runtime": runtime_by_image[runtime_key],
                "artifact_preflight": _artifact_preflight(variant, server_env, gguf_artifacts_json, python_bin),
                "benchmark_command": ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                "benchmark_env": benchmark_env,
                "fake_stream_command": fake_stream_command if include_fake_stream and supports_fake_stream else None,
                "fake_stream_env": server_env,
                "phase_defaults": {
                    "warmup": _warmup_phase_from_args(variant.get("args", [])),
                },
                "paths": {
                    "benchmark_jsonl": str(benchmark_jsonl),
                    "summary_md": str(summary_md),
                    "manifest_json": str(manifest_json),
                    "profile_dir": str(profile_dir),
                    "profile_jsonl": str(profile_jsonl),
                    "profile_summary_json": str(profile_summary_json),
                    "lifecycle_jsonl": str(lifecycle_jsonl),
                    "gguf_artifacts_json": str(gguf_artifacts_json),
                    "fake_stream_jsonl": str(fake_stream_jsonl),
                    "server_log": str(server_log),
                    "preflight_before_prepare_json": str(preflight_before_prepare_json),
                    "preflight_json": str(preflight_json),
                },
            }
        )
    return {
        "variants_path": str(variants_path),
        "run_prefix": run_prefix,
        "port": port,
        "min_lfb_blocks": min_lfb_blocks,
        "prepare_context": _prepare_context(source_env),
        "pre_variant_command": pre_variant_command,
        "selection_contexts": [dict(context) for context in selection_contexts],
        "variant_min_lfb_blocks": dict(variant_min_lfb_blocks or {}),
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


def _server_models_endpoint_open(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=1.0):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_server_port_closed(port: int, timeout_s: float = 5.0) -> bool:
    if not _server_models_endpoint_open(port):
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if not _server_models_endpoint_open(port):
            return True
    return False


def _terminate_process(process: subprocess.Popen[Any]) -> dict[str, Any]:
    shutdown_started = time.monotonic()
    if process.poll() is not None:
        return {
            "server_shutdown_seconds": 0.0,
            "server_shutdown_method": "already_exited",
        }
    pid = getattr(process, "pid", None)
    if isinstance(pid, int):
        method = "terminate_group"
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return {
                "server_shutdown_seconds": time.monotonic() - shutdown_started,
                "server_shutdown_method": "already_exited",
            }
    else:
        method = "terminate"
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if isinstance(pid, int):
            method = "kill_group"
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            method = "kill"
            process.kill()
        process.wait(timeout=20)
    return {
        "server_shutdown_seconds": time.monotonic() - shutdown_started,
        "server_shutdown_method": method,
    }


def _preflight_block_reason(preflight: dict[str, Any], min_lfb_blocks: int | None) -> str | None:
    if min_lfb_blocks is None:
        return None
    tegrastats = preflight.get("tegrastats")
    lfb = tegrastats.get("lfb") if isinstance(tegrastats, dict) else None
    if not isinstance(lfb, dict) or not isinstance(lfb.get("free_blocks"), int):
        return f"lfb_free_blocks unavailable; required {min_lfb_blocks}"
    free_blocks = int(lfb["free_blocks"])
    if free_blocks < min_lfb_blocks:
        return f"lfb_free_blocks {free_blocks} < required {min_lfb_blocks}"
    return None


def _text_tail(value: str | None, max_chars: int = 4000) -> str:
    if not value:
        return ""
    return value[-max_chars:]


def _run_pre_variant_command(command: str | None) -> dict[str, Any]:
    if not command:
        return {
            "pre_variant_command": None,
            "pre_variant_command_passed": None,
            "pre_variant_command_returncode": None,
            "pre_variant_command_stdout_tail": "",
            "pre_variant_command_stderr_tail": "",
        }
    result = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "pre_variant_command": command,
        "pre_variant_command_passed": result.returncode == 0,
        "pre_variant_command_returncode": result.returncode,
        "pre_variant_command_stdout_tail": _text_tail(result.stdout),
        "pre_variant_command_stderr_tail": _text_tail(result.stderr),
    }


def _not_started_server_timing() -> dict[str, Any]:
    return {
        "server_started_at": None,
        "server_ready_at": None,
        "server_wait_seconds": None,
        "server_startup_seconds": None,
    }


def _missing_profile_artifact(paths: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "profile_jsonl_path": str(paths.get("profile_jsonl") or ""),
        "profile_summary_path": str(paths.get("profile_summary_json") or ""),
        "profile_summary": {
            "available": False,
            "reason": reason,
        },
    }


def _write_run_profile_artifacts(
    paths: dict[str, Any],
    server_timing: dict[str, Any],
    phase_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_jsonl = paths.get("profile_jsonl")
    profile_summary_json = paths.get("profile_summary_json")
    manifest_json = paths.get("manifest_json")
    if not profile_jsonl or not profile_summary_json:
        return _missing_profile_artifact(paths, "missing_profile_output_path")
    if not manifest_json or not Path(str(manifest_json)).is_file():
        return _missing_profile_artifact(paths, "missing_benchmark_manifest")
    manifest = _read_json_object(str(manifest_json))
    jetson = manifest.get("jetson") if isinstance(manifest.get("jetson"), dict) else {}
    tegrastats_log = jetson.get("tegrastats_log") if isinstance(jetson, dict) else None
    if not isinstance(tegrastats_log, str) or not Path(tegrastats_log).is_file():
        return _missing_profile_artifact(paths, "missing_tegrastats_log")
    startup_seconds = server_timing.get("server_startup_seconds")
    shutdown_seconds = server_timing.get("server_shutdown_seconds")
    phase_timings = _profile_phase_timings(
        paths,
        startup_seconds,
        shutdown_seconds,
        phase_defaults=phase_defaults,
    )
    profile_summary = write_profile_artifacts(
        tegrastats_log=tegrastats_log,
        profile_jsonl_path=str(profile_jsonl),
        summary_path=str(profile_summary_json),
        phase_timings=phase_timings,
        input_timing_summary=_profile_input_timing_summary(paths),
        profile_files=_profile_files(paths, jetson),
    )
    return {
        "profile_jsonl_path": str(profile_jsonl),
        "profile_summary_path": str(profile_summary_json),
        "profile_summary": profile_summary,
    }


def _latency_sum_from_jsonl(path: str | Path, predicate: Any) -> float | None:
    source = Path(path)
    if not source.is_file():
        return None
    total = 0.0
    found = False
    for record in _iter_jsonl(source):
        if not predicate(record):
            continue
        latency = record.get("latency_s")
        if isinstance(latency, (int, float)):
            total += float(latency)
            found = True
    return total if found else None


def _phase_entry(duration: float | None, reason: str = "not_recorded") -> dict[str, Any]:
    return {
        "available": duration is not None,
        "duration_s": duration,
        "reason": None if duration is not None else reason,
    }


def _phase_timings_from_lifecycle(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    timings: dict[str, Any] = {}
    for record in _iter_jsonl(source):
        phase = record.get("phase")
        if phase not in REQUIRED_PHASES:
            continue
        duration = record.get("duration_s")
        entry: dict[str, Any] = {
            "available": bool(record.get("available")) if record.get("available") is not None else isinstance(duration, (int, float)),
            "duration_s": float(duration) if isinstance(duration, (int, float)) else None,
            "reason": str(record.get("reason") or "") or None,
        }
        source_name = record.get("source")
        if isinstance(source_name, str) and source_name:
            entry["source"] = source_name
        details = record.get("details")
        if isinstance(details, dict):
            entry["details"] = dict(details)
        timings[str(phase)] = entry
    return timings


def _profile_input_timing_summary(paths: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sources: dict[str, int] = {}
    for key in ("benchmark_jsonl", "fake_stream_jsonl"):
        path = paths.get(key)
        if not isinstance(path, str) or not Path(path).is_file():
            continue
        source_records = [
            record
            for record in _iter_jsonl(Path(path))
            if isinstance(record.get("input_timing"), dict)
        ]
        if source_records:
            records.extend(source_records)
            sources[key] = len(source_records)
    return summarize_input_timing_records(records, sources=sources)


def _profile_phase_timings(
    paths: dict[str, Any],
    startup_seconds: Any,
    shutdown_seconds: Any = None,
    *,
    phase_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_jsonl = paths.get("benchmark_jsonl")
    fake_stream_jsonl = paths.get("fake_stream_jsonl")
    lifecycle_jsonl = paths.get("lifecycle_jsonl")
    text_duration = (
        _latency_sum_from_jsonl(benchmark_jsonl, lambda record: record.get("input_type") == "text")
        if isinstance(benchmark_jsonl, str)
        else None
    )
    image_duration = (
        _latency_sum_from_jsonl(
            benchmark_jsonl,
            lambda record: str(record.get("input_type") or "").startswith("image"),
        )
        if isinstance(benchmark_jsonl, str)
        else None
    )
    fake_stream_duration = (
        _latency_sum_from_jsonl(fake_stream_jsonl, lambda _record: True)
        if isinstance(fake_stream_jsonl, str)
        else None
    )
    timings = dict(phase_defaults or {})
    timings.update(_phase_timings_from_lifecycle(lifecycle_jsonl if isinstance(lifecycle_jsonl, str) else None))
    timings.update(
        {
            "server_startup": _phase_entry(
                float(startup_seconds) if isinstance(startup_seconds, (int, float)) else None,
            ),
            "formal_text": _phase_entry(text_duration),
            "formal_image": _phase_entry(image_duration),
            "fake_stream": _phase_entry(fake_stream_duration),
            "shutdown": _phase_entry(
                float(shutdown_seconds) if isinstance(shutdown_seconds, (int, float)) else None,
            ),
        }
    )
    return timings


def _profile_files(paths: dict[str, Any], jetson: dict[str, Any]) -> dict[str, str]:
    profile_files: dict[str, str] = {}
    profile_dir = paths.get("profile_dir")
    if isinstance(profile_dir, str) and profile_dir:
        profile_files.update(
            {
                "uname": str(Path(profile_dir) / "uname.txt"),
                "docker_version": str(Path(profile_dir) / "docker-version.txt"),
                "nvpmodel": str(Path(profile_dir) / "nvpmodel.txt"),
                "jetson_clocks": str(Path(profile_dir) / "jetson-clocks.txt"),
            }
        )
    if isinstance(jetson.get("power_mode"), str):
        profile_files["nvpmodel"] = str(jetson["power_mode"])
    if isinstance(jetson.get("jetson_clocks"), str):
        profile_files["jetson_clocks"] = str(jetson["jetson_clocks"])
    return profile_files


def run_sweep(
    plan: dict[str, Any],
    *,
    wait_timeout_s: float,
    report_output: str | Path,
    min_lfb_blocks: int | None = None,
    pre_variant_command: str | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    benchmark_paths: list[str] = []
    fake_stream_paths: list[str] = []
    command = pre_variant_command if pre_variant_command is not None else plan.get("pre_variant_command")
    if min_lfb_blocks is None and isinstance(plan.get("min_lfb_blocks"), int):
        min_lfb_blocks = int(plan["min_lfb_blocks"])
    variant_min_lfb_blocks = plan.get("variant_min_lfb_blocks")
    if not isinstance(variant_min_lfb_blocks, dict):
        variant_min_lfb_blocks = {}
    for variant_plan in plan["variants"]:
        variant_id = str(variant_plan["variant"]["id"])
        paths = variant_plan["paths"]
        preflight_before_prepare_path = _preflight_before_prepare_path(paths)
        preflight_before_prepare = None
        if command and preflight_before_prepare_path:
            preflight_before_prepare = capture_preflight_sample(preflight_before_prepare_path)
        override_min_lfb_blocks = variant_min_lfb_blocks.get(variant_id)
        required_lfb_blocks = int(override_min_lfb_blocks) if isinstance(override_min_lfb_blocks, int) else min_lfb_blocks
        pre_variant_result = _run_pre_variant_command(command)
        if pre_variant_result["pre_variant_command_passed"] is False:
            results.append(
                {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_id,
                    "server_ready": False,
                    "server_returncode": None,
                    "benchmark_returncode": None,
                    "fake_stream_returncode": None,
                    "preflight_required_lfb_blocks": required_lfb_blocks,
                    "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                    "preflight_before_prepare": preflight_before_prepare,
                    "preflight_delta": None,
                    "preflight_path": paths["preflight_json"],
                    "preflight": None,
                    "preflight_passed": False,
                    "preflight_reason": (
                        "pre_variant_command_failed "
                        f"returncode {pre_variant_result['pre_variant_command_returncode']}"
                    ),
                    **_not_started_server_timing(),
                    **pre_variant_result,
                }
            )
            continue
        preflight = capture_preflight_sample(paths["preflight_json"])
        preflight_delta = _preflight_prepare_delta(preflight_before_prepare, preflight)
        preflight_reason = _preflight_block_reason(preflight, required_lfb_blocks)
        if preflight_reason is not None:
            results.append(
                {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_id,
                    "server_ready": False,
                    "server_returncode": None,
                    "benchmark_returncode": None,
                    "fake_stream_returncode": None,
                    "preflight_required_lfb_blocks": required_lfb_blocks,
                    "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                    "preflight_before_prepare": preflight_before_prepare,
                    "preflight_delta": preflight_delta,
                    "preflight_path": paths["preflight_json"],
                    "preflight": preflight,
                    "preflight_passed": False,
                    "preflight_reason": preflight_reason,
                    **_not_started_server_timing(),
                    **pre_variant_result,
                }
            )
            continue
        runtime_reason = _runtime_support_block_reason(variant_plan)
        if runtime_reason is not None:
            results.append(
                {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_id,
                    "server_ready": False,
                    "server_returncode": None,
                    "benchmark_returncode": None,
                    "fake_stream_returncode": None,
                    "preflight_required_lfb_blocks": required_lfb_blocks,
                    "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                    "preflight_before_prepare": preflight_before_prepare,
                    "preflight_delta": preflight_delta,
                    "preflight_path": paths["preflight_json"],
                    "preflight": preflight,
                    "preflight_passed": True,
                    "preflight_reason": runtime_reason,
                    **_not_started_server_timing(),
                    **pre_variant_result,
                }
            )
            continue
        if not _wait_for_server_port_closed(int(plan["port"]), timeout_s=5.0):
            results.append(
                {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_id,
                    "server_ready": False,
                    "server_returncode": None,
                    "benchmark_returncode": None,
                    "fake_stream_returncode": None,
                    "preflight_required_lfb_blocks": required_lfb_blocks,
                    "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                    "preflight_before_prepare": preflight_before_prepare,
                    "preflight_delta": preflight_delta,
                    "preflight_path": paths["preflight_json"],
                    "preflight": preflight,
                    "preflight_passed": True,
                    "preflight_reason": "server_port_still_open_before_start",
                    "server_port_available_before_start": False,
                    **_not_started_server_timing(),
                    **pre_variant_result,
                }
            )
            continue
        Path(paths["server_log"]).parent.mkdir(parents=True, exist_ok=True)
        server_log = open(paths["server_log"], "w", encoding="utf-8")
        server_started_at = datetime.now(timezone.utc).isoformat()
        server_wait_start = time.monotonic()
        server = subprocess.Popen(
            variant_plan["server_command"],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=_merged_env(variant_plan["server_env"]),
            text=True,
            start_new_session=True,
        )
        result_entry: dict[str, Any] | None = None
        server_timing: dict[str, Any] = _not_started_server_timing()
        try:
            ready = _wait_for_server(plan["port"], server, wait_timeout_s)
            server_wait_seconds = time.monotonic() - server_wait_start
            server_ready_at = datetime.now(timezone.utc).isoformat() if ready else None
            server_timing = {
                "server_started_at": server_started_at,
                "server_ready_at": server_ready_at,
                "server_wait_seconds": server_wait_seconds,
                "server_startup_seconds": server_wait_seconds if ready else None,
            }
            if not ready:
                result_entry = {
                    "run_id": variant_plan["run_id"],
                    "variant_id": variant_id,
                    "server_ready": False,
                    "server_returncode": server.poll(),
                    "benchmark_returncode": None,
                    "fake_stream_returncode": None,
                    "preflight_required_lfb_blocks": required_lfb_blocks,
                    "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                    "preflight_before_prepare": preflight_before_prepare,
                    "preflight_delta": preflight_delta,
                    "preflight_path": paths["preflight_json"],
                    "preflight": preflight,
                    "preflight_passed": True,
                    "preflight_reason": None,
                    **server_timing,
                    **pre_variant_result,
                }
                results.append(result_entry)
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
            if fake_stream_returncode == 0 and Path(paths["fake_stream_jsonl"]).is_file():
                fake_stream_paths.append(paths["fake_stream_jsonl"])
            result_entry = {
                "run_id": variant_plan["run_id"],
                "variant_id": variant_id,
                "server_ready": True,
                "server_returncode": server.poll(),
                "benchmark_returncode": benchmark_result.returncode,
                "fake_stream_returncode": fake_stream_returncode,
                "preflight_required_lfb_blocks": required_lfb_blocks,
                "preflight_before_prepare_path": preflight_before_prepare_path if command else None,
                "preflight_before_prepare": preflight_before_prepare,
                "preflight_delta": preflight_delta,
                "preflight_path": paths["preflight_json"],
                "preflight": preflight,
                "preflight_passed": True,
                "preflight_reason": None,
                **server_timing,
                **pre_variant_result,
            }
            results.append(result_entry)
        finally:
            shutdown_timing = _terminate_process(server)
            shutdown_timing["server_port_closed_after_shutdown"] = _wait_for_server_port_closed(
                int(plan["port"]),
                timeout_s=5.0,
            )
            server_log.close()
            if result_entry is not None:
                result_entry.update(shutdown_timing)
                result_entry["server_returncode"] = server.poll()
                if result_entry.get("server_ready") is True:
                    phase_defaults = variant_plan.get("phase_defaults")
                    if not isinstance(phase_defaults, dict):
                        variant = variant_plan.get("variant") if isinstance(variant_plan.get("variant"), dict) else {}
                        phase_defaults = {
                            "warmup": _warmup_phase_from_args(
                                variant.get("args", []) if isinstance(variant, dict) else []
                            ),
                        }
                    profile_artifact = _write_run_profile_artifacts(
                        paths,
                        {**server_timing, **shutdown_timing},
                        phase_defaults=phase_defaults,
                    )
                    result_entry.update(profile_artifact)
    if benchmark_paths:
        build_optimization_report(
            input_paths=benchmark_paths,
            fake_stream_paths=fake_stream_paths,
            output_path=report_output,
        )
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
    parser.add_argument("--fake-stream-max-frames", type=int, default=3)
    parser.add_argument("--fake-stream-interval-s", type=float, default=0.0)
    parser.add_argument("--fake-stream-skip-late-frames", action="store_true")
    parser.add_argument("--fake-stream-skip-threshold-s", type=float, default=None)
    parser.add_argument("--fake-stream-adaptive-interval", action="store_true")
    parser.add_argument("--fake-stream-adaptive-interval-scale", type=float, default=1.0)
    parser.add_argument("--fake-stream-adaptive-interval-max-s", type=float, default=None)
    parser.add_argument("--wait-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--min-lfb-blocks",
        type=int,
        default=None,
        help="Skip a variant before server startup when tegrastats lfb free blocks are below this threshold",
    )
    parser.add_argument(
        "--pre-variant-command",
        default=None,
        help="Shell command to run before each variant preflight; a non-zero exit skips that variant",
    )
    parser.add_argument(
        "--selection-context-json",
        action="append",
        default=[],
        help="JSON file describing an auto-selected lane; repeatable",
    )
    parser.add_argument(
        "--variant-min-lfb-blocks",
        action="append",
        default=[],
        help="Override min LFB blocks for a specific variant using variant_id=value; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args(argv)

    run_prefix = args.run_prefix or _timestamp_run_prefix()
    output_root = args.output_root or f"outputs/optimization_sweeps/{run_prefix}"
    server_log_dir = args.server_log_dir or f"{output_root}/server_logs"
    report_output = args.report_output or f"{output_root}/optimization_report.md"
    variants = _load_variants(args.variants)
    variant_groups = {
        str(variant["id"]): _variant_comparison_group(variant)
        for variant in variants
        if isinstance(variant, dict) and variant.get("id") is not None
    }
    try:
        variant_min_lfb_blocks = _parse_variant_int_overrides(
            args.variant_min_lfb_blocks,
            flag_name="--variant-min-lfb-blocks",
        )
    except ValueError as exc:
        parser.error(str(exc))
    selection_contexts = _load_selection_contexts(
        args.selection_context_json,
        variant_groups=variant_groups,
    )
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
        fake_stream_interval_s=args.fake_stream_interval_s,
        fake_stream_skip_late_frames=args.fake_stream_skip_late_frames,
        fake_stream_skip_threshold_s=args.fake_stream_skip_threshold_s,
        fake_stream_adaptive_interval=args.fake_stream_adaptive_interval,
        fake_stream_adaptive_interval_scale=args.fake_stream_adaptive_interval_scale,
        fake_stream_adaptive_interval_max_s=args.fake_stream_adaptive_interval_max_s,
        min_lfb_blocks=args.min_lfb_blocks,
        pre_variant_command=args.pre_variant_command,
        selection_contexts=selection_contexts,
        variant_min_lfb_blocks=variant_min_lfb_blocks,
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
    result = run_sweep(
        plan,
        wait_timeout_s=args.wait_timeout_s,
        report_output=report_output,
        min_lfb_blocks=args.min_lfb_blocks,
        pre_variant_command=args.pre_variant_command,
    )
    manifest = Path(output_root) / f"{run_prefix}.manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"plan": plan, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(plan["variants"]), "manifest": str(manifest), "report": result["report_output"]}, ensure_ascii=False))
    return 0 if result["report_output"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
