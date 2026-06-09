"""Optimization reporting helpers for Jetson VLM benchmark sweeps."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import config_supports_images, load_model_config
from .jetson_profile import summarize_tegrastats_log as summarize_jetson_profile_log


@dataclass(frozen=True)
class RunSummary:
    source: str
    run_id: str
    model: str
    records: int
    successful: int
    failed: int
    guard_passed: bool
    guard_failures: tuple[str, ...]
    text_avg_latency_s: float | None
    text_avg_tokens_per_s: float | None
    image_avg_latency_s: float | None
    image_avg_tokens_per_s: float | None
    fake_stream_records: int
    fake_stream_successful: int
    fake_stream_avg_latency_s: float | None
    speed_score: float | None


@dataclass
class SweepComparisonRow:
    source: str
    run_prefix: str
    run_id: str
    variant_id: str
    selection_id: str
    selection_reason: str
    model: str
    comparison_group: str
    server_image: str | None
    server_image_id: str | None
    llama_cpp_ref: str | None
    artifact_phase_status: str
    artifact_phase_duration_s: float | None
    prepare_context_summary: str
    prepare_max_clocks_enabled: bool
    prepare_drop_caches_before_variant: bool
    preflight_before_prepare_lfb: str
    preflight_lfb: str
    preflight_required_lfb_blocks: int | None
    preflight_prepare_lfb_delta: int | None
    preflight_prepare_mem_available_mb_delta: float | None
    preflight_prepare_buddyinfo_max_order_delta: int | None
    trials: int | None
    benchmark_max_tokens: int | None
    benchmark_temperature: float | None
    supports_images: bool
    quality_review_passed: bool | None
    quality_review_records: int | None
    quality_review_passed_records: int | None
    quality_review_failed_case_ids: tuple[str, ...]
    guard_passed: bool
    successful: int
    records: int
    fake_stream_successful: int
    fake_stream_records: int
    server_startup_seconds: float | None
    text_avg_tokens_per_s: float | None
    image_avg_tokens_per_s: float | None
    text_avg_latency_s: float | None
    image_avg_latency_s: float | None
    fake_stream_avg_latency_s: float | None
    max_temp_c: float | None
    avg_power_w: float | None
    avg_gr3d_util_pct: float | None
    avg_emc_util_pct: float | None
    min_lfb_free_blocks: int | None
    bottleneck_labels: tuple[str, ...]
    guard_failures: tuple[str, ...]
    delta_text_tokens_per_s_pct: float | None = None
    delta_image_tokens_per_s_pct: float | None = None
    delta_startup_pct: float | None = None
    delta_fake_stream_latency_pct: float | None = None
    ranking_precheck_passed: bool | None = None
    ranking_precheck_reason: str = ""
    promotion_precheck_passed: bool | None = None
    promotion_precheck_reason: str = ""


_PROMOTION_PRECHECK_STAGES: dict[str, dict[str, int]] = {
    "formal-repeat": {
        "min_trials": 5,
    },
    "promotion-reference": {
        "min_trials": 10,
    },
}


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _numeric_values(records: Iterable[dict[str, Any]], input_type: str, key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        record_input_type = str(record.get("input_type") or "")
        if input_type == "image":
            matches_type = record_input_type.startswith("image")
        else:
            matches_type = record_input_type == input_type
        value = record.get(key)
        if matches_type and isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _word_repeat_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 8:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def _char_repeat_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < 16:
        return 0.0
    counts = Counter(chars)
    return counts.most_common(1)[0][1] / len(chars)


def _quality_terms_miss(record: dict[str, Any], output: str) -> bool:
    terms = record.get("quality_terms_any")
    if not isinstance(terms, list):
        return False
    normalized_terms = [str(term).strip().casefold() for term in terms if str(term).strip()]
    if not normalized_terms:
        return False
    normalized_output = output.casefold()
    return not any(term in normalized_output for term in normalized_terms)


def _sanity_failures(
    records: Iterable[dict[str, Any]],
    *,
    min_output_chars: int,
    max_repeat_ratio: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    for record in records:
        input_type = str(record.get("input_type") or "")
        if input_type == "fake_stream":
            continue
        case_id = str(record.get("prompt_case_id") or "unknown_case")
        if record.get("success") is not True:
            failures.append(f"{case_id}:failed")
            continue
        output = str(record.get("output_excerpt") or "").strip()
        if not output:
            failures.append(f"{case_id}:empty_output")
            continue
        if len(output) < min_output_chars:
            failures.append(f"{case_id}:short_output")
        if max(_word_repeat_ratio(output), _char_repeat_ratio(output)) > max_repeat_ratio:
            failures.append(f"{case_id}:repetitive_output")
        if _quality_terms_miss(record, output):
            failures.append(f"{case_id}:quality_terms_miss")
    return tuple(failures)


def _fake_stream_sanity_failures(
    records: Iterable[dict[str, Any]],
    *,
    min_output_chars: int,
    max_repeat_ratio: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    for record in records:
        frame_id = str(record.get("frame_id") or record.get("frame_index") or "unknown_frame")
        prefix = f"fake_stream:{frame_id}"
        if record.get("success") is not True:
            failures.append(f"{prefix}:failed")
            continue
        output = str(record.get("output_excerpt") or "").strip()
        if not output:
            failures.append(f"{prefix}:empty_output")
            continue
        if len(output) < min_output_chars:
            failures.append(f"{prefix}:short_output")
        if max(_word_repeat_ratio(output), _char_repeat_ratio(output)) > max_repeat_ratio:
            failures.append(f"{prefix}:repetitive_output")
    return tuple(failures)


def summarize_run(
    path: str | Path,
    *,
    fake_stream_path: str | Path | None = None,
    min_output_chars: int = 32,
    max_repeat_ratio: float = 0.65,
) -> RunSummary:
    source = Path(path)
    records = list(_iter_jsonl(source))
    if not records:
        raise ValueError(f"no benchmark records found in {source}")
    first = records[0]
    text_latencies = _numeric_values(records, "text", "latency_s")
    text_tps = _numeric_values(records, "text", "tokens_per_sec")
    image_latencies = _numeric_values(records, "image", "latency_s")
    image_tps = _numeric_values(records, "image", "tokens_per_sec")
    avg_tps_values = [_mean_or_none(text_tps), _mean_or_none(image_tps)]
    speed_components = [value for value in avg_tps_values if value is not None]
    guard_failures = _sanity_failures(
        records,
        min_output_chars=min_output_chars,
        max_repeat_ratio=max_repeat_ratio,
    )
    fake_stream_records = list(_iter_jsonl(Path(fake_stream_path))) if fake_stream_path is not None else []
    fake_stream_failures = _fake_stream_sanity_failures(
        fake_stream_records,
        min_output_chars=min_output_chars,
        max_repeat_ratio=max_repeat_ratio,
    )
    all_guard_failures = guard_failures + fake_stream_failures
    successful = sum(1 for record in records if record.get("success") is True)
    failed = len(records) - successful
    fake_stream_successful = sum(1 for record in fake_stream_records if record.get("success") is True)
    fake_stream_latencies = [
        float(record["latency_s"])
        for record in fake_stream_records
        if isinstance(record.get("latency_s"), (int, float))
    ]
    return RunSummary(
        source=str(source),
        run_id=str(first.get("run_id") or source.stem),
        model=str(first.get("model") or "unknown"),
        records=len(records),
        successful=successful,
        failed=failed,
        guard_passed=failed == 0 and len(all_guard_failures) == 0,
        guard_failures=all_guard_failures,
        text_avg_latency_s=_mean_or_none(text_latencies),
        text_avg_tokens_per_s=_mean_or_none(text_tps),
        image_avg_latency_s=_mean_or_none(image_latencies),
        image_avg_tokens_per_s=_mean_or_none(image_tps),
        fake_stream_records=len(fake_stream_records),
        fake_stream_successful=fake_stream_successful,
        fake_stream_avg_latency_s=_mean_or_none(fake_stream_latencies),
        speed_score=_mean_or_none(speed_components),
    )


def _sort_summaries(summaries: Iterable[RunSummary]) -> list[RunSummary]:
    def key(summary: RunSummary) -> tuple[int, float, str]:
        score = summary.speed_score if summary.speed_score is not None else -1.0
        return (0 if summary.guard_passed else 1, -score, summary.run_id)

    return sorted(summaries, key=key)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:+.2f}%"


def _fmt_signed_int(value: int | None) -> str:
    return "" if value is None else f"{value:+d}"


def _fmt_signed_float(value: float | None) -> str:
    return "" if value is None else f"{value:+.3f}"


def _percent_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return ((value - baseline) / baseline) * 100.0


def _format_report(summaries: list[RunSummary]) -> str:
    lines = [
        "# Edge VLM Optimization Report",
        "",
        "Runs with failed sanity guards are kept in the report but excluded from ranked candidates.",
        "The guard checks for successful records, non-empty outputs, minimum output length, and obvious repetition; it is not a replacement for human or task-specific quality evaluation.",
        "",
        "| Rank | Run id | Model | Guard | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Success | Fake success | Guard failures | Source |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    rank = 0
    for summary in summaries:
        if summary.guard_passed:
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "-"
        failures = ", ".join(summary.guard_failures)
        lines.append(
            "| {rank} | {run_id} | {model} | {guard} | {text_tps} | {image_tps} | {text_latency} | {image_latency} | {fake_latency} | {success} | {fake_success} | {failures} | `{source}` |".format(
                rank=rank_text,
                run_id=summary.run_id,
                model=summary.model,
                guard="yes" if summary.guard_passed else "no",
                text_tps=_fmt(summary.text_avg_tokens_per_s),
                image_tps=_fmt(summary.image_avg_tokens_per_s),
                text_latency=_fmt(summary.text_avg_latency_s),
                image_latency=_fmt(summary.image_avg_latency_s),
                fake_latency=_fmt(summary.fake_stream_avg_latency_s),
                success=f"{summary.successful}/{summary.records}",
                fake_success=(
                    f"{summary.fake_stream_successful}/{summary.fake_stream_records}"
                    if summary.fake_stream_records
                    else ""
                ),
                failures=failures.replace("|", "\\|"),
                source=summary.source,
            )
        )
    lines.append("")
    return "\n".join(lines)


def summarize_tegrastats_log(path: str | Path) -> dict[str, Any]:
    return summarize_jetson_profile_log(path)


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _resolve_existing_path(path: Path | None, *, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    candidates = [path]
    if base is not None and not path.is_absolute():
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _format_lfb(preflight: Any) -> str:
    if not isinstance(preflight, dict):
        return ""
    tegrastats = preflight.get("tegrastats")
    if not isinstance(tegrastats, dict):
        return ""
    lfb = tegrastats.get("lfb")
    if not isinstance(lfb, dict):
        return ""
    free_blocks = lfb.get("free_blocks")
    block_mb = lfb.get("block_mb")
    if not isinstance(free_blocks, int) or not isinstance(block_mb, int):
        return ""
    return f"{free_blocks}x{block_mb}MB"


def _preflight_delta_value(entry: dict[str, Any], key: str) -> Any:
    delta = entry.get("preflight_delta")
    if isinstance(delta, dict) and key in delta:
        return delta.get(key)
    return None


def _preflight_prepare_lfb_delta(entry: dict[str, Any]) -> int | None:
    value = _preflight_delta_value(entry, "lfb_free_blocks_delta")
    return int(value) if isinstance(value, int) else None


def _preflight_prepare_mem_available_mb_delta(entry: dict[str, Any]) -> float | None:
    value = _preflight_delta_value(entry, "mem_available_kb_delta")
    if not isinstance(value, int):
        return None
    return float(value) / 1024.0


def _preflight_prepare_buddyinfo_max_order_delta(entry: dict[str, Any]) -> int | None:
    value = _preflight_delta_value(entry, "buddyinfo_max_order_delta")
    return int(value) if isinstance(value, int) else None


def _runtime_metadata(variant_plan: dict[str, Any]) -> dict[str, Any]:
    runtime = variant_plan.get("server_runtime")
    return runtime if isinstance(runtime, dict) else {}


def _profile_summary(entry: dict[str, Any], variant_plan: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    explicit = entry.get("profile_summary")
    if isinstance(explicit, dict):
        return explicit
    paths = variant_plan.get("paths") if isinstance(variant_plan, dict) else None
    paths = paths if isinstance(paths, dict) else {}
    summary_path = _resolve_existing_path(
        _path_or_none(paths.get("profile_summary_json")),
        base=manifest_path.parent,
    )
    if summary_path is None:
        return {}
    try:
        data = _read_json(summary_path)
    except (FileNotFoundError, ValueError):
        return {}
    return data


def _phase_timing(profile_summary: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_timings = profile_summary.get("phase_timings")
    if not isinstance(phase_timings, dict):
        return {}
    timing = phase_timings.get(phase)
    return timing if isinstance(timing, dict) else {}


def _artifact_phase_status(profile_summary: dict[str, Any]) -> str:
    timing = _phase_timing(profile_summary, "artifact_check_or_download")
    details = timing.get("details")
    if isinstance(details, dict):
        status = details.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    reason = timing.get("reason")
    if isinstance(reason, str) and reason.strip() and reason != "not_recorded":
        return reason.strip()
    if timing.get("available") is True:
        return "available"
    return ""


def _artifact_phase_duration_s(profile_summary: dict[str, Any]) -> float | None:
    timing = _phase_timing(profile_summary, "artifact_check_or_download")
    duration = timing.get("duration_s")
    return float(duration) if isinstance(duration, (int, float)) else None


def _comparison_group(variant_plan: dict[str, Any], model: str) -> str:
    variant = variant_plan.get("variant")
    if isinstance(variant, dict):
        group = variant.get("comparison_group")
        if isinstance(group, str) and group.strip():
            return group.strip()
    group = variant_plan.get("comparison_group")
    if isinstance(group, str) and group.strip():
        return group.strip()
    return model


def _selection_contexts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = plan.get("selection_contexts")
    if not isinstance(contexts, list):
        return []
    return [context for context in contexts if isinstance(context, dict)]


def _selection_context_for_variant(
    plan: dict[str, Any],
    *,
    variant_id: str,
    comparison_group: str,
) -> dict[str, Any]:
    for context in _selection_contexts(plan):
        selected_variant_id = context.get("selected_variant_id")
        if not isinstance(selected_variant_id, str) or selected_variant_id != variant_id:
            continue
        group = context.get("comparison_group")
        if isinstance(group, str) and group.strip() and group.strip() != comparison_group:
            continue
        return context
    return {}


def _short_ref(value: str | None, length: int = 12) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text[:length]


def _format_runtime(row: SweepComparisonRow) -> str:
    parts = [
        part
        for part in (
            row.server_image,
            _short_ref(row.server_image_id),
            _short_ref(row.llama_cpp_ref),
        )
        if part
    ]
    return " / ".join(parts)


def _prepare_context_summary(plan: dict[str, Any]) -> str:
    context = plan.get("prepare_context")
    if not isinstance(context, dict):
        return ""
    labels: list[str] = []
    if context.get("max_clocks_enabled") is True:
        labels.append("max_clocks")
    if context.get("drop_caches_before_variant") is True:
        labels.append("drop_caches")
    return ", ".join(labels)


def _prepare_context_flag(plan: dict[str, Any], key: str) -> bool:
    context = plan.get("prepare_context")
    if not isinstance(context, dict):
        return False
    return context.get(key) is True


def _format_selection(row: SweepComparisonRow) -> str:
    if not row.selection_id:
        return ""
    if row.selection_reason:
        return f"{row.selection_id} ({row.selection_reason})"
    return row.selection_id


def _preflight_required_lfb_blocks(entry: dict[str, Any]) -> int | None:
    value = entry.get("preflight_required_lfb_blocks")
    return int(value) if isinstance(value, int) else None


def _plan_min_lfb_blocks(plan: dict[str, Any]) -> int | None:
    value = plan.get("min_lfb_blocks")
    return int(value) if isinstance(value, int) else None


def _plan_variant_min_lfb_blocks(plan: dict[str, Any], variant_id: str) -> int | None:
    overrides = plan.get("variant_min_lfb_blocks")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get(variant_id)
    return int(value) if isinstance(value, int) else None


def _effective_required_lfb_blocks(
    entry: dict[str, Any],
    *,
    plan: dict[str, Any],
    variant_id: str,
) -> int | None:
    explicit_required_lfb_blocks = _preflight_required_lfb_blocks(entry)
    if explicit_required_lfb_blocks is not None:
        return explicit_required_lfb_blocks
    override_required_lfb_blocks = _plan_variant_min_lfb_blocks(plan, variant_id)
    if override_required_lfb_blocks is not None:
        return override_required_lfb_blocks
    return _plan_min_lfb_blocks(plan)


def _ranking_precheck(
    row: SweepComparisonRow,
    *,
    ranking_min_lfb_blocks: int | None,
) -> tuple[bool | None, str]:
    if ranking_min_lfb_blocks is None:
        return None, ""
    required_lfb_blocks = row.preflight_required_lfb_blocks
    if required_lfb_blocks is None:
        return False, "missing_required_lfb"
    if required_lfb_blocks < ranking_min_lfb_blocks:
        return False, f"required_lfb {required_lfb_blocks} < ranking {ranking_min_lfb_blocks}"
    return True, ""


def _format_ranking_precheck(row: SweepComparisonRow) -> str:
    if row.ranking_precheck_passed is None:
        return ""
    if row.ranking_precheck_passed:
        return "yes"
    if row.ranking_precheck_reason:
        return f"no ({row.ranking_precheck_reason})"
    return "no"


def _infer_run_prefix(run_id: str, variant_id: str, fallback: str) -> str:
    suffix = f"-{variant_id}"
    if run_id.endswith(suffix):
        return run_id[: -len(suffix)]
    return fallback


def _benchmark_metadata_from_manifest(path: Path | None) -> dict[str, Any]:
    metadata = {
        "trial_count": None,
        "max_tokens": None,
        "temperature": None,
        "tegrastats_log": None,
    }
    if path is None or not path.is_file():
        return metadata
    data = _read_json(path)
    benchmark = data.get("benchmark")
    if isinstance(benchmark, dict):
        trial_count = benchmark.get("trial_count")
        if isinstance(trial_count, int):
            metadata["trial_count"] = trial_count
        max_tokens = benchmark.get("max_tokens")
        if isinstance(max_tokens, int):
            metadata["max_tokens"] = max_tokens
        temperature = benchmark.get("temperature")
        if isinstance(temperature, (int, float)):
            metadata["temperature"] = float(temperature)
    jetson = data.get("jetson")
    if isinstance(jetson, dict):
        metadata["tegrastats_log"] = _path_or_none(jetson.get("tegrastats_log"))
    return metadata


def _quality_review_report_path(paths: dict[str, Any], *, manifest_path: Path, run_id: str) -> Path | None:
    explicit = _resolve_existing_path(
        _path_or_none(paths.get("quality_review_json")),
        base=manifest_path.parent,
    )
    if explicit is not None:
        return explicit
    fallback = manifest_path.parent / f"{run_id}.quality.json"
    return fallback if fallback.is_file() else None


def _quality_review_summary(report_path: Path | None) -> dict[str, Any]:
    summary = {
        "passed": None,
        "records": None,
        "passed_records": None,
        "failed_case_ids": (),
    }
    if report_path is None or not report_path.is_file():
        return summary
    report = _read_json(report_path)
    passed = report.get("passed")
    summary["passed"] = bool(passed) if isinstance(passed, bool) else None
    records = report.get("records")
    if isinstance(records, int):
        summary["records"] = records
    passed_records = report.get("passed_records")
    if isinstance(passed_records, int):
        summary["passed_records"] = passed_records
    failures = report.get("failures")
    if isinstance(failures, list):
        failed_case_ids: list[str] = []
        seen_case_ids: set[str] = set()
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            case_id = failure.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            failed_case_ids.append(case_id)
        summary["failed_case_ids"] = tuple(failed_case_ids)
    return summary


def _variant_supports_images(variant_plan: dict[str, Any], *, fake_stream_path: Path | None) -> bool:
    explicit = variant_plan.get("supports_images")
    if isinstance(explicit, bool):
        return explicit
    variant = variant_plan.get("variant")
    if isinstance(variant, dict):
        config_path = variant.get("config")
        if isinstance(config_path, str) and config_path.strip():
            try:
                return config_supports_images(load_model_config(config_path))
            except (FileNotFoundError, ValueError):
                pass
    return fake_stream_path is not None


def summarize_sweep_manifest(
    manifest_path: str | Path,
    *,
    min_output_chars: int = 32,
    max_repeat_ratio: float = 0.65,
) -> list[SweepComparisonRow]:
    source = Path(manifest_path)
    manifest = _read_json(source)
    plan = manifest.get("plan")
    result = manifest.get("result")
    if not isinstance(plan, dict) or not isinstance(result, dict):
        raise ValueError(f"{source}: expected sweep manifest with plan and result objects")
    plan_variants = plan.get("variants")
    result_entries = result.get("results")
    if not isinstance(plan_variants, list) or not isinstance(result_entries, list):
        raise ValueError(f"{source}: expected plan.variants and result.results lists")
    fallback_run_prefix = str(plan.get("run_prefix") or source.stem.removesuffix(".manifest"))
    plan_by_run_id = {
        str(variant.get("run_id")): variant
        for variant in plan_variants
        if isinstance(variant, dict) and variant.get("run_id") is not None
    }
    rows: list[SweepComparisonRow] = []
    for index, entry in enumerate(result_entries):
        if not isinstance(entry, dict):
            continue
        run_id = str(entry.get("run_id") or "")
        variant_id = str(entry.get("variant_id") or "")
        variant_plan = plan_by_run_id.get(run_id)
        if variant_plan is None and index < len(plan_variants) and isinstance(plan_variants[index], dict):
            variant_plan = plan_variants[index]
        if variant_plan is None:
            variant_plan = {}
        row_model = str(entry.get("model") or "unknown")
        inferred_comparison_group = _comparison_group(variant_plan, row_model)
        selection_context = _selection_context_for_variant(
            plan,
            variant_id=variant_id,
            comparison_group=inferred_comparison_group,
        )
        paths = variant_plan.get("paths") if isinstance(variant_plan, dict) else None
        paths = paths if isinstance(paths, dict) else {}
        benchmark_path = _path_or_none(paths.get("benchmark_jsonl"))
        benchmark_manifest_path = _path_or_none(paths.get("manifest_json"))
        fake_stream_path = _path_or_none(paths.get("fake_stream_jsonl"))
        profile_summary = _profile_summary(entry, variant_plan, manifest_path=source)
        benchmark_metadata = _benchmark_metadata_from_manifest(benchmark_manifest_path)
        quality_review = _quality_review_summary(
            _quality_review_report_path(
                paths,
                manifest_path=source,
                run_id=run_id,
            )
        )
        summary: RunSummary | None = None
        if benchmark_path is not None and benchmark_path.is_file():
            summary = summarize_run(
                benchmark_path,
                fake_stream_path=fake_stream_path if fake_stream_path is not None and fake_stream_path.is_file() else None,
                min_output_chars=min_output_chars,
                max_repeat_ratio=max_repeat_ratio,
            )
        tegrastats_log = benchmark_metadata.get("tegrastats_log")
        tegrastats_summary = (
            summarize_jetson_profile_log(tegrastats_log)
            if tegrastats_log is not None and tegrastats_log.is_file()
            else {
                "max_temp_c": None,
                "avg_power_w": None,
                "avg_gr3d_util_pct": None,
                "avg_emc_util_pct": None,
                "min_lfb_free_blocks": None,
                "bottleneck_labels": (),
            }
        )
        runtime = _runtime_metadata(variant_plan)
        bottleneck_labels = tegrastats_summary.get("bottleneck_labels")
        labels = (
            tuple(str(label) for label in bottleneck_labels)
            if isinstance(bottleneck_labels, list)
            else ()
        )
        rows.append(
            SweepComparisonRow(
                source=str(source),
                run_prefix=_infer_run_prefix(run_id, variant_id, fallback_run_prefix),
                run_id=run_id,
                variant_id=variant_id,
                selection_id=(
                    str(selection_context.get("selection_id"))
                    if isinstance(selection_context.get("selection_id"), str)
                    else ""
                ),
                selection_reason=(
                    str(selection_context.get("selected_reason"))
                    if isinstance(selection_context.get("selected_reason"), str)
                    else ""
                ),
                model=summary.model if summary is not None else str(entry.get("model") or "unknown"),
                comparison_group=_comparison_group(
                    variant_plan,
                    summary.model if summary is not None else row_model,
                ),
                server_image=str(runtime["image"]) if runtime.get("image") else None,
                server_image_id=str(runtime["image_id"]) if runtime.get("image_id") else None,
                llama_cpp_ref=str(runtime["llama_cpp_ref"]) if runtime.get("llama_cpp_ref") else None,
                artifact_phase_status=_artifact_phase_status(profile_summary),
                artifact_phase_duration_s=_artifact_phase_duration_s(profile_summary),
                prepare_context_summary=_prepare_context_summary(plan),
                prepare_max_clocks_enabled=_prepare_context_flag(plan, "max_clocks_enabled"),
                prepare_drop_caches_before_variant=_prepare_context_flag(plan, "drop_caches_before_variant"),
                preflight_before_prepare_lfb=_format_lfb(entry.get("preflight_before_prepare")),
                preflight_lfb=_format_lfb(entry.get("preflight")),
                preflight_required_lfb_blocks=_effective_required_lfb_blocks(
                    entry,
                    plan=plan,
                    variant_id=variant_id,
                ),
                preflight_prepare_lfb_delta=_preflight_prepare_lfb_delta(entry),
                preflight_prepare_mem_available_mb_delta=_preflight_prepare_mem_available_mb_delta(entry),
                preflight_prepare_buddyinfo_max_order_delta=_preflight_prepare_buddyinfo_max_order_delta(entry),
                trials=benchmark_metadata.get("trial_count"),
                benchmark_max_tokens=benchmark_metadata.get("max_tokens"),
                benchmark_temperature=benchmark_metadata.get("temperature"),
                supports_images=_variant_supports_images(
                    variant_plan,
                    fake_stream_path=fake_stream_path,
                ),
                quality_review_passed=quality_review["passed"],
                quality_review_records=quality_review["records"],
                quality_review_passed_records=quality_review["passed_records"],
                quality_review_failed_case_ids=quality_review["failed_case_ids"],
                guard_passed=summary.guard_passed if summary is not None else False,
                successful=summary.successful if summary is not None else 0,
                records=summary.records if summary is not None else 0,
                fake_stream_successful=summary.fake_stream_successful if summary is not None else 0,
                fake_stream_records=summary.fake_stream_records if summary is not None else 0,
                server_startup_seconds=(
                    float(entry["server_startup_seconds"])
                    if isinstance(entry.get("server_startup_seconds"), (int, float))
                    else None
                ),
                text_avg_tokens_per_s=summary.text_avg_tokens_per_s if summary is not None else None,
                image_avg_tokens_per_s=summary.image_avg_tokens_per_s if summary is not None else None,
                text_avg_latency_s=summary.text_avg_latency_s if summary is not None else None,
                image_avg_latency_s=summary.image_avg_latency_s if summary is not None else None,
                fake_stream_avg_latency_s=summary.fake_stream_avg_latency_s if summary is not None else None,
                max_temp_c=(
                    float(tegrastats_summary["max_temp_c"])
                    if isinstance(tegrastats_summary.get("max_temp_c"), (int, float))
                    else None
                ),
                avg_power_w=(
                    float(tegrastats_summary["avg_power_w"])
                    if isinstance(tegrastats_summary.get("avg_power_w"), (int, float))
                    else None
                ),
                avg_gr3d_util_pct=(
                    float(tegrastats_summary["avg_gr3d_util_pct"])
                    if isinstance(tegrastats_summary.get("avg_gr3d_util_pct"), (int, float))
                    else None
                ),
                avg_emc_util_pct=(
                    float(tegrastats_summary["avg_emc_util_pct"])
                    if isinstance(tegrastats_summary.get("avg_emc_util_pct"), (int, float))
                    else None
                ),
                min_lfb_free_blocks=(
                    int(tegrastats_summary["min_lfb_free_blocks"])
                    if isinstance(tegrastats_summary.get("min_lfb_free_blocks"), int)
                    else None
                ),
                bottleneck_labels=labels,
                guard_failures=summary.guard_failures if summary is not None else ("missing_benchmark_jsonl",),
            )
        )
    return rows


def _add_comparison_deltas(rows: list[SweepComparisonRow], baseline_variant_ids: Iterable[str]) -> None:
    requested_baselines = set(baseline_variant_ids)
    baselines: dict[str, SweepComparisonRow] = {}
    for row in rows:
        if row.comparison_group in baselines:
            continue
        if requested_baselines and row.variant_id not in requested_baselines:
            continue
        baselines[row.comparison_group] = row
    if not requested_baselines:
        for row in rows:
            if row.comparison_group not in baselines and row.guard_passed:
                baselines[row.comparison_group] = row
    for row in rows:
        baseline = baselines.get(row.comparison_group)
        if baseline is None:
            continue
        row.delta_text_tokens_per_s_pct = _percent_delta(row.text_avg_tokens_per_s, baseline.text_avg_tokens_per_s)
        row.delta_image_tokens_per_s_pct = _percent_delta(row.image_avg_tokens_per_s, baseline.image_avg_tokens_per_s)
        row.delta_startup_pct = _percent_delta(row.server_startup_seconds, baseline.server_startup_seconds)
        row.delta_fake_stream_latency_pct = _percent_delta(row.fake_stream_avg_latency_s, baseline.fake_stream_avg_latency_s)


def _add_ranking_prechecks(rows: list[SweepComparisonRow], ranking_min_lfb_blocks: int | None) -> None:
    for row in rows:
        row.ranking_precheck_passed, row.ranking_precheck_reason = _ranking_precheck(
            row,
            ranking_min_lfb_blocks=ranking_min_lfb_blocks,
        )


def _promotion_precheck(
    row: SweepComparisonRow,
    *,
    promotion_precheck_stage: str | None,
    ranking_min_lfb_blocks: int | None,
    promotion_require_quality_review: bool,
) -> tuple[bool | None, str]:
    if promotion_precheck_stage is None:
        return None, ""
    stage_requirements = _PROMOTION_PRECHECK_STAGES[promotion_precheck_stage]
    required_lfb_floor = ranking_min_lfb_blocks if ranking_min_lfb_blocks is not None else 150
    failures: list[str] = []
    required_lfb_blocks = row.preflight_required_lfb_blocks
    if required_lfb_blocks is None:
        failures.append("missing_required_lfb")
    elif required_lfb_blocks < required_lfb_floor:
        failures.append(f"required_lfb {required_lfb_blocks} < required {required_lfb_floor}")
    if not row.prepare_max_clocks_enabled:
        failures.append("missing_max_clocks")
    if not row.prepare_drop_caches_before_variant:
        failures.append("missing_drop_caches")
    if row.trials is None:
        failures.append("missing_trial_count")
    elif row.trials < stage_requirements["min_trials"]:
        failures.append(f"trial_count {row.trials} < required {stage_requirements['min_trials']}")
    if row.benchmark_max_tokens is None:
        failures.append("missing_max_tokens")
    elif row.benchmark_max_tokens < 64:
        failures.append(f"max_tokens {row.benchmark_max_tokens} < required 64")
    if row.benchmark_temperature is None:
        failures.append("missing_temperature")
    elif abs(row.benchmark_temperature) > 1e-9:
        failures.append(f"temperature {row.benchmark_temperature:g} != required 0")
    if not row.guard_passed:
        failures.append("guard_failed")
    if row.records > 0 and row.successful < row.records:
        failures.append(f"benchmark_success {row.successful}/{row.records} < {row.records}/{row.records}")
    if row.supports_images:
        if row.fake_stream_records < 1:
            failures.append("missing_fake_stream_records")
        elif row.fake_stream_successful < row.fake_stream_records:
            failures.append(
                f"fake_stream_success {row.fake_stream_successful}/{row.fake_stream_records} < {row.fake_stream_records}/{row.fake_stream_records}"
            )
    if promotion_require_quality_review:
        if row.quality_review_passed is None:
            failures.append("missing_quality_review")
        elif not row.quality_review_passed:
            details: list[str] = []
            if row.quality_review_passed_records is not None and row.quality_review_records is not None:
                details.append(f"{row.quality_review_passed_records}/{row.quality_review_records}")
            if row.quality_review_failed_case_ids:
                details.append(",".join(row.quality_review_failed_case_ids))
            if details:
                failures.append("quality_review_failed " + " ".join(details))
            else:
                failures.append("quality_review_failed")
    if failures:
        return False, "; ".join(failures)
    return True, ""


def _format_promotion_precheck(row: SweepComparisonRow) -> str:
    if row.promotion_precheck_passed is None:
        return ""
    if row.promotion_precheck_passed:
        return "yes"
    if row.promotion_precheck_reason:
        return f"no ({row.promotion_precheck_reason})"
    return "no"


def _format_quality_review(row: SweepComparisonRow, *, show_missing: bool) -> str:
    if row.quality_review_passed is None:
        return "missing" if show_missing else ""
    details: list[str] = []
    if row.quality_review_passed_records is not None and row.quality_review_records is not None:
        details.append(f"{row.quality_review_passed_records}/{row.quality_review_records}")
    if row.quality_review_failed_case_ids:
        details.append(", ".join(row.quality_review_failed_case_ids))
    status = "yes" if row.quality_review_passed else "no"
    if details:
        return f"{status} ({'; '.join(details)})"
    return status


def _add_promotion_prechecks(
    rows: list[SweepComparisonRow],
    *,
    ranking_min_lfb_blocks: int | None,
    promotion_precheck_stage: str | None,
    promotion_require_quality_review: bool,
) -> None:
    for row in rows:
        row.promotion_precheck_passed, row.promotion_precheck_reason = _promotion_precheck(
            row,
            promotion_precheck_stage=promotion_precheck_stage,
            ranking_min_lfb_blocks=ranking_min_lfb_blocks,
            promotion_require_quality_review=promotion_require_quality_review,
        )


def _format_sweep_comparison_report(
    rows: list[SweepComparisonRow],
    *,
    ranking_min_lfb_blocks: int | None,
    promotion_precheck_stage: str | None,
    promotion_require_quality_review: bool,
) -> str:
    ranking_column = " | Ranking precheck" if ranking_min_lfb_blocks is not None else ""
    ranking_separator = " |---" if ranking_min_lfb_blocks is not None else ""
    ranking_note = (
        f" `Ranking precheck` uses `--ranking-min-lfb-blocks {ranking_min_lfb_blocks}` and only checks whether a row's effective required LFB gate is strict enough for ranking."
        if ranking_min_lfb_blocks is not None
        else ""
    )
    promotion_column = " | Promotion precheck" if promotion_precheck_stage is not None else ""
    promotion_separator = " |---" if promotion_precheck_stage is not None else ""
    promotion_note = (
        " `Promotion precheck` uses `--promotion-precheck-stage {stage}` to enforce the mechanical promotion gate: locked clocks, cache drop, strict required-LFB floor, `max_tokens >= 64`, `temperature = 0`, full benchmark success, fake-stream success for image-capable rows, and the stage trial floor.{quality_gate} Raw excerpt review remains manual.".format(
            stage=promotion_precheck_stage,
            quality_gate=(
                " When `--promotion-require-quality-review` is also set, the row must have a passing structured quality review sidecar."
                if promotion_require_quality_review
                else ""
            ),
        )
        if promotion_precheck_stage is not None
        else ""
    )
    quality_review_enabled = any(
        row.quality_review_passed is not None
        or row.quality_review_records is not None
        or row.quality_review_passed_records is not None
        for row in rows
    )
    quality_review_column = " | Quality review" if quality_review_enabled else ""
    quality_review_separator = " |---" if quality_review_enabled else ""
    quality_review_note = (
        " When per-run quality review sidecars exist, compare also adds a `Quality review` column that summarizes structured policy pass/fail plus passed-record counts. This remains separate from the mechanical promotion precheck."
        if quality_review_enabled
        else ""
    )
    artifact_phase_enabled = any(
        row.artifact_phase_status or row.artifact_phase_duration_s is not None
        for row in rows
    )
    artifact_phase_column = " | Artifact phase | Artifact s" if artifact_phase_enabled else ""
    artifact_phase_separator = " |---|---:" if artifact_phase_enabled else ""
    artifact_phase_note = (
        " When profile phase timings include `artifact_check_or_download`, compare also adds `Artifact phase` and `Artifact s` so first-download rows can be separated from cached-start rows."
        if artifact_phase_enabled
        else ""
    )
    lines = [
        "# Jetson Sweep Comparison Report",
        "",
        "Baseline rows use `0.00%` deltas. Positive throughput deltas are faster; positive startup or fake-stream latency deltas are slower."
        + ranking_note,
        promotion_note + quality_review_note + artifact_phase_note,
        "",
        "| Model | Variant | Selection | Run prefix | Runtime"
        + artifact_phase_column
        + " | Prepare ctx | Preflight lfb | Required lfb"
        + ranking_column
        + promotion_column
        + quality_review_column
        + " | Prepare lfb delta | Prepare avail MB delta | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Max temp C | Avg power W | Avg GR3D % | Avg EMC % | Min lfb blocks | Bottlenecks | Text tok/s delta | Image tok/s delta | Startup delta | Fake latency delta |",
        "|---|---|---|---|---"
        + artifact_phase_separator
        + "|---|---:|---:"
        + ranking_separator
        + promotion_separator
        + quality_review_separator
        + "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        failures = ", ".join(row.guard_failures)
        ranking_precheck = ""
        if ranking_min_lfb_blocks is not None:
            ranking_precheck_text = _format_ranking_precheck(row).replace("|", "\\|")
            ranking_precheck = f" | {ranking_precheck_text}"
        promotion_precheck = ""
        if promotion_precheck_stage is not None:
            promotion_precheck_text = _format_promotion_precheck(row).replace("|", "\\|")
            promotion_precheck = f" | {promotion_precheck_text}"
        quality_review = ""
        if quality_review_enabled:
            quality_review_text = _format_quality_review(row, show_missing=True).replace("|", "\\|")
            quality_review = f" | {quality_review_text}"
        artifact_phase = ""
        if artifact_phase_enabled:
            artifact_phase_status = row.artifact_phase_status.replace("|", "\\|")
            artifact_phase = f" | {artifact_phase_status} | {_fmt(row.artifact_phase_duration_s)}"
        lines.append(
            "| {model} | `{variant}` | {selection} | {run_prefix} | {runtime}{artifact_phase} | {prepare_context} | {lfb} | {required_lfb}{ranking_precheck}{promotion_precheck}{quality_review} | {prepare_lfb_delta} | {prepare_avail_delta} | {trials} | {guard} | {success} | {fake_success} | {startup} | {text_tps} | {image_tps} | {text_latency} | {image_latency} | {fake_latency} | {max_temp} | {avg_power} | {avg_gr3d} | {avg_emc} | {min_lfb} | {bottlenecks} | {text_delta} | {image_delta} | {startup_delta} | {fake_delta} |".format(
                model=row.model,
                variant=row.variant_id,
                selection=_format_selection(row).replace("|", "\\|"),
                run_prefix=row.run_prefix,
                runtime=_format_runtime(row),
                artifact_phase=artifact_phase,
                prepare_context=row.prepare_context_summary.replace("|", "\\|"),
                lfb=row.preflight_lfb,
                required_lfb="" if row.preflight_required_lfb_blocks is None else row.preflight_required_lfb_blocks,
                ranking_precheck=ranking_precheck,
                promotion_precheck=promotion_precheck,
                quality_review=quality_review,
                prepare_lfb_delta=_fmt_signed_int(row.preflight_prepare_lfb_delta),
                prepare_avail_delta=_fmt_signed_float(row.preflight_prepare_mem_available_mb_delta),
                trials="" if row.trials is None else row.trials,
                guard="yes" if row.guard_passed else f"no ({failures})",
                success=f"{row.successful}/{row.records}",
                fake_success=(
                    f"{row.fake_stream_successful}/{row.fake_stream_records}"
                    if row.fake_stream_records
                    else ""
                ),
                startup=_fmt(row.server_startup_seconds),
                text_tps=_fmt(row.text_avg_tokens_per_s),
                image_tps=_fmt(row.image_avg_tokens_per_s),
                text_latency=_fmt(row.text_avg_latency_s),
                image_latency=_fmt(row.image_avg_latency_s),
                fake_latency=_fmt(row.fake_stream_avg_latency_s),
                max_temp=_fmt(row.max_temp_c),
                avg_power=_fmt(row.avg_power_w),
                avg_gr3d=_fmt(row.avg_gr3d_util_pct),
                avg_emc=_fmt(row.avg_emc_util_pct),
                min_lfb="" if row.min_lfb_free_blocks is None else row.min_lfb_free_blocks,
                bottlenecks=", ".join(row.bottleneck_labels),
                text_delta=_fmt_pct(row.delta_text_tokens_per_s_pct),
                image_delta=_fmt_pct(row.delta_image_tokens_per_s_pct),
                startup_delta=_fmt_pct(row.delta_startup_pct),
                fake_delta=_fmt_pct(row.delta_fake_stream_latency_pct),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_sweep_comparison_report(
    *,
    manifest_paths: Iterable[str | Path],
    output_path: str | Path,
    baseline_variant_ids: Iterable[str] = (),
    min_output_chars: int = 32,
    max_repeat_ratio: float = 0.65,
    ranking_min_lfb_blocks: int | None = None,
    promotion_precheck_stage: str | None = None,
    promotion_require_quality_review: bool = False,
) -> list[SweepComparisonRow]:
    rows: list[SweepComparisonRow] = []
    for manifest_path in manifest_paths:
        rows.extend(
            summarize_sweep_manifest(
                manifest_path,
                min_output_chars=min_output_chars,
                max_repeat_ratio=max_repeat_ratio,
            )
        )
    _add_comparison_deltas(rows, baseline_variant_ids)
    _add_ranking_prechecks(rows, ranking_min_lfb_blocks)
    _add_promotion_prechecks(
        rows,
        ranking_min_lfb_blocks=ranking_min_lfb_blocks,
        promotion_precheck_stage=promotion_precheck_stage,
        promotion_require_quality_review=promotion_require_quality_review,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _format_sweep_comparison_report(
            rows,
            ranking_min_lfb_blocks=ranking_min_lfb_blocks,
            promotion_precheck_stage=promotion_precheck_stage,
            promotion_require_quality_review=promotion_require_quality_review,
        ),
        encoding="utf-8",
    )
    return rows


def build_optimization_report(
    *,
    input_paths: Iterable[str | Path],
    fake_stream_paths: Iterable[str | Path] | None = None,
    output_path: str | Path,
    min_output_chars: int = 32,
    max_repeat_ratio: float = 0.65,
) -> list[RunSummary]:
    fake_stream_by_stem = {
        Path(path).stem: Path(path)
        for path in (fake_stream_paths or [])
    }
    summaries = _sort_summaries(
        summarize_run(
            path,
            fake_stream_path=fake_stream_by_stem.get(Path(path).stem),
            min_output_chars=min_output_chars,
            max_repeat_ratio=max_repeat_ratio,
        )
        for path in input_paths
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_format_report(summaries), encoding="utf-8")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank benchmark JSONL runs with a lightweight sanity guard.")
    subparsers = parser.add_subparsers(dest="command")
    report_parser = subparsers.add_parser("report", help="Build a Markdown optimization report")
    report_parser.add_argument("--input", action="append", required=True, help="Benchmark JSONL path; repeatable")
    report_parser.add_argument("--fake-stream", action="append", default=[], help="Fake-stream JSONL path; repeatable")
    report_parser.add_argument("--output", required=True, help="Markdown report output path")
    report_parser.add_argument("--min-output-chars", type=int, default=32)
    report_parser.add_argument("--max-repeat-ratio", type=float, default=0.65)
    report_parser.add_argument("--fail-on-guard", action="store_true")
    compare_parser = subparsers.add_parser("compare", help="Build a Markdown sweep comparison report from sweep manifests")
    compare_parser.add_argument("--manifest", action="append", required=True, help="Sweep manifest path; repeatable")
    compare_parser.add_argument("--baseline-variant", action="append", default=[], help="Variant id to use as per-model baseline; repeatable")
    compare_parser.add_argument("--output", required=True, help="Markdown report output path")
    compare_parser.add_argument("--min-output-chars", type=int, default=32)
    compare_parser.add_argument("--max-repeat-ratio", type=float, default=0.65)
    compare_parser.add_argument("--fail-on-guard", action="store_true")
    compare_parser.add_argument(
        "--ranking-min-lfb-blocks",
        type=int,
        help="Strict ranking gate for preflight LFB requirements; rows below this remain in the report but fail ranking precheck",
    )
    compare_parser.add_argument("--fail-on-ranking-precheck", action="store_true")
    compare_parser.add_argument(
        "--promotion-precheck-stage",
        choices=sorted(_PROMOTION_PRECHECK_STAGES),
        help="Mechanical promotion gate profile for comparison rows; raw excerpt review remains manual",
    )
    compare_parser.add_argument(
        "--promotion-require-quality-review",
        action="store_true",
        help="Require a passing structured quality review sidecar when evaluating promotion precheck",
    )
    compare_parser.add_argument("--fail-on-promotion-precheck", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "compare":
        if args.fail_on_ranking_precheck and args.ranking_min_lfb_blocks is None:
            parser.error("--fail-on-ranking-precheck requires --ranking-min-lfb-blocks")
        if args.fail_on_promotion_precheck and args.promotion_precheck_stage is None:
            parser.error("--fail-on-promotion-precheck requires --promotion-precheck-stage")
        if args.promotion_require_quality_review and args.promotion_precheck_stage is None:
            parser.error("--promotion-require-quality-review requires --promotion-precheck-stage")
        rows = build_sweep_comparison_report(
            manifest_paths=args.manifest,
            output_path=args.output,
            baseline_variant_ids=args.baseline_variant,
            min_output_chars=args.min_output_chars,
            max_repeat_ratio=args.max_repeat_ratio,
            ranking_min_lfb_blocks=args.ranking_min_lfb_blocks,
            promotion_precheck_stage=args.promotion_precheck_stage,
            promotion_require_quality_review=args.promotion_require_quality_review,
        )
        print(json.dumps({"runs": len(rows), "output": args.output}, ensure_ascii=False))
        if args.fail_on_guard and any(not row.guard_passed for row in rows):
            return 1
        if args.fail_on_ranking_precheck and any(row.ranking_precheck_passed is False for row in rows):
            return 1
        if args.fail_on_promotion_precheck and any(row.promotion_precheck_passed is False for row in rows):
            return 1
        return 0
    if args.command != "report":
        parser.print_help()
        return 2
    summaries = build_optimization_report(
        input_paths=args.input,
        fake_stream_paths=args.fake_stream,
        output_path=args.output,
        min_output_chars=args.min_output_chars,
        max_repeat_ratio=args.max_repeat_ratio,
    )
    print(json.dumps({"runs": len(summaries), "output": args.output}, ensure_ascii=False))
    if args.fail_on_guard and any(not summary.guard_passed for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
