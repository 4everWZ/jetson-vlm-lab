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
    model: str
    preflight_lfb: str
    trials: int | None
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
    guard_failures: tuple[str, ...]
    delta_text_tokens_per_s_pct: float | None = None
    delta_image_tokens_per_s_pct: float | None = None
    delta_startup_pct: float | None = None
    delta_fake_stream_latency_pct: float | None = None


TEGRASTATS_TEMP_RE = re.compile(r"@([0-9]+(?:\.[0-9]+)?)C")
TEGRASTATS_VDD_IN_RE = re.compile(r"\bVDD_IN\s+(?P<instant_mw>\d+)mW/\d+mW\b")


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


def summarize_tegrastats_log(path: str | Path) -> dict[str, float | int | None]:
    source = Path(path)
    max_temps: list[float] = []
    power_w: list[float] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        temps = [float(value) for value in TEGRASTATS_TEMP_RE.findall(line)]
        if temps:
            max_temps.append(max(temps))
        power_match = TEGRASTATS_VDD_IN_RE.search(line)
        if power_match is not None:
            power_w.append(int(power_match.group("instant_mw")) / 1000.0)
    return {
        "samples": max(len(max_temps), len(power_w)),
        "max_temp_c": max(max_temps) if max_temps else None,
        "avg_power_w": statistics.mean(power_w) if power_w else None,
    }


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


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


def _infer_run_prefix(run_id: str, variant_id: str, fallback: str) -> str:
    suffix = f"-{variant_id}"
    if run_id.endswith(suffix):
        return run_id[: -len(suffix)]
    return fallback


def _trial_count_from_manifest(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    data = _read_json(path)
    benchmark = data.get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    trial_count = benchmark.get("trial_count")
    return int(trial_count) if isinstance(trial_count, int) else None


def _tegrastats_log_from_manifest(path: Path | None) -> Path | None:
    if path is None or not path.is_file():
        return None
    data = _read_json(path)
    jetson = data.get("jetson")
    if not isinstance(jetson, dict):
        return None
    return _path_or_none(jetson.get("tegrastats_log"))


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
        paths = variant_plan.get("paths") if isinstance(variant_plan, dict) else None
        paths = paths if isinstance(paths, dict) else {}
        benchmark_path = _path_or_none(paths.get("benchmark_jsonl"))
        benchmark_manifest_path = _path_or_none(paths.get("manifest_json"))
        fake_stream_path = _path_or_none(paths.get("fake_stream_jsonl"))
        summary: RunSummary | None = None
        if benchmark_path is not None and benchmark_path.is_file():
            summary = summarize_run(
                benchmark_path,
                fake_stream_path=fake_stream_path if fake_stream_path is not None and fake_stream_path.is_file() else None,
                min_output_chars=min_output_chars,
                max_repeat_ratio=max_repeat_ratio,
            )
        tegrastats_log = _tegrastats_log_from_manifest(benchmark_manifest_path)
        tegrastats_summary = (
            summarize_tegrastats_log(tegrastats_log)
            if tegrastats_log is not None and tegrastats_log.is_file()
            else {"max_temp_c": None, "avg_power_w": None}
        )
        rows.append(
            SweepComparisonRow(
                source=str(source),
                run_prefix=_infer_run_prefix(run_id, variant_id, fallback_run_prefix),
                run_id=run_id,
                variant_id=variant_id,
                model=summary.model if summary is not None else str(entry.get("model") or "unknown"),
                preflight_lfb=_format_lfb(entry.get("preflight")),
                trials=_trial_count_from_manifest(benchmark_manifest_path),
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
                guard_failures=summary.guard_failures if summary is not None else ("missing_benchmark_jsonl",),
            )
        )
    return rows


def _add_comparison_deltas(rows: list[SweepComparisonRow], baseline_variant_ids: Iterable[str]) -> None:
    requested_baselines = set(baseline_variant_ids)
    baselines: dict[str, SweepComparisonRow] = {}
    for row in rows:
        if row.model in baselines:
            continue
        if requested_baselines and row.variant_id not in requested_baselines:
            continue
        baselines[row.model] = row
    if not requested_baselines:
        for row in rows:
            if row.model not in baselines and row.guard_passed:
                baselines[row.model] = row
    for row in rows:
        baseline = baselines.get(row.model)
        if baseline is None:
            continue
        row.delta_text_tokens_per_s_pct = _percent_delta(row.text_avg_tokens_per_s, baseline.text_avg_tokens_per_s)
        row.delta_image_tokens_per_s_pct = _percent_delta(row.image_avg_tokens_per_s, baseline.image_avg_tokens_per_s)
        row.delta_startup_pct = _percent_delta(row.server_startup_seconds, baseline.server_startup_seconds)
        row.delta_fake_stream_latency_pct = _percent_delta(row.fake_stream_avg_latency_s, baseline.fake_stream_avg_latency_s)


def _format_sweep_comparison_report(rows: list[SweepComparisonRow]) -> str:
    lines = [
        "# Jetson Sweep Comparison Report",
        "",
        "Baseline rows use `0.00%` deltas. Positive throughput deltas are faster; positive startup or fake-stream latency deltas are slower.",
        "",
        "| Model | Variant | Run prefix | Preflight lfb | Trials | Guard | Success | Fake success | Startup s | Text tok/s | Image tok/s | Text latency s | Image latency s | Fake latency s | Max temp C | Avg power W | Text tok/s delta | Image tok/s delta | Startup delta | Fake latency delta |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        failures = ", ".join(row.guard_failures)
        lines.append(
            "| {model} | `{variant}` | {run_prefix} | {lfb} | {trials} | {guard} | {success} | {fake_success} | {startup} | {text_tps} | {image_tps} | {text_latency} | {image_latency} | {fake_latency} | {max_temp} | {avg_power} | {text_delta} | {image_delta} | {startup_delta} | {fake_delta} |".format(
                model=row.model,
                variant=row.variant_id,
                run_prefix=row.run_prefix,
                lfb=row.preflight_lfb,
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_format_sweep_comparison_report(rows), encoding="utf-8")
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
    args = parser.parse_args(argv)

    if args.command == "compare":
        rows = build_sweep_comparison_report(
            manifest_paths=args.manifest,
            output_path=args.output,
            baseline_variant_ids=args.baseline_variant,
            min_output_chars=args.min_output_chars,
            max_repeat_ratio=args.max_repeat_ratio,
        )
        print(json.dumps({"runs": len(rows), "output": args.output}, ensure_ascii=False))
        if args.fail_on_guard and any(not row.guard_passed for row in rows):
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
