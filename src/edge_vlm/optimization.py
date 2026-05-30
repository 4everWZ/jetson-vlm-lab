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
    args = parser.parse_args(argv)

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
