"""Route-specific quality checks for benchmark JSONL outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _terms(value: Any, *, field: str, case_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(term, str) or not term.strip() for term in value):
        raise ValueError(f"quality policy case {case_id} field {field} must be a list of non-empty strings")
    return list(value)


def _case_policy(policy: dict[str, Any], case_id: str) -> dict[str, Any]:
    cases = policy.get("cases", {})
    if cases is None:
        cases = {}
    if not isinstance(cases, dict):
        raise ValueError("quality policy cases must be an object")
    case_policy = cases.get(case_id, {})
    if case_policy is None:
        case_policy = {}
    if not isinstance(case_policy, dict):
        raise ValueError(f"quality policy case {case_id} must be an object")
    return case_policy


def _default_policy(policy: dict[str, Any]) -> dict[str, Any]:
    default = policy.get("default", {})
    if default is None:
        return {}
    if not isinstance(default, dict):
        raise ValueError("quality policy default must be an object")
    return default


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


def _record_failures(record: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    case_id = str(record.get("prompt_case_id") or "unknown_case")
    default = _default_policy(policy)
    case_policy = _case_policy(policy, case_id)
    output = str(record.get("output_excerpt") or "").strip()
    failures: list[str] = []

    if record.get("success") is not True:
        failures.append("failed")
        return failures
    if not output:
        failures.append("empty_output")
        return failures

    min_output_chars = int(case_policy.get("min_output_chars", default.get("min_output_chars", 0)) or 0)
    if min_output_chars > 0 and len(output) < min_output_chars:
        failures.append(f"short_output:{len(output)}<{min_output_chars}")

    max_repeat_ratio = float(case_policy.get("max_repeat_ratio", default.get("max_repeat_ratio", 1.0)) or 1.0)
    repeat_ratio = max(_word_repeat_ratio(output), _char_repeat_ratio(output))
    if repeat_ratio > max_repeat_ratio:
        failures.append(f"repetitive_output:{repeat_ratio:.3f}>{max_repeat_ratio:.3f}")

    normalized_output = output.casefold()
    for term in _terms(case_policy.get("must_include_all"), field="must_include_all", case_id=case_id):
        if term.casefold() not in normalized_output:
            failures.append(f"missing_all:{term}")

    any_terms = _terms(case_policy.get("must_include_any"), field="must_include_any", case_id=case_id)
    if any_terms and not any(term.casefold() in normalized_output for term in any_terms):
        failures.append("missing_any:" + "|".join(any_terms))

    for term in _terms(case_policy.get("must_not_include_any"), field="must_not_include_any", case_id=case_id):
        if term.casefold() in normalized_output:
            failures.append(f"forbidden:{term}")

    return failures


def _case_summaries(records: Iterable[dict[str, Any]], failures_by_index: dict[int, list[str]]) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        case_id = str(record.get("prompt_case_id") or "unknown_case")
        summary = summaries.setdefault(case_id, {"records": 0, "passed": 0, "failed": 0})
        summary["records"] += 1
        if failures_by_index.get(index):
            summary["failed"] += 1
        else:
            summary["passed"] += 1
    return summaries


def review_benchmark_records(records: list[dict[str, Any]], policy: dict[str, Any], *, inputs: list[str] | None = None) -> dict[str, Any]:
    failures_by_index: dict[int, list[str]] = {}
    failures: list[dict[str, Any]] = []
    latencies = [float(record["latency_s"]) for record in records if isinstance(record.get("latency_s"), (int, float))]
    tokens_per_sec = [
        float(record["tokens_per_sec"]) for record in records if isinstance(record.get("tokens_per_sec"), (int, float))
    ]

    for index, record in enumerate(records):
        record_failures = _record_failures(record, policy)
        if not record_failures:
            continue
        failures_by_index[index] = record_failures
        failures.append(
            {
                "run_id": record.get("run_id"),
                "model": record.get("model"),
                "trial_index": record.get("trial_index"),
                "case_index": record.get("case_index"),
                "case_id": record.get("prompt_case_id"),
                "input_type": record.get("input_type"),
                "failures": record_failures,
                "output_excerpt": record.get("output_excerpt"),
            }
        )

    run_ids = sorted({str(record.get("run_id")) for record in records if record.get("run_id")})
    models = sorted({str(record.get("model")) for record in records if record.get("model")})
    failed_records = len(failures_by_index)
    return {
        "inputs": list(inputs or []),
        "run_ids": run_ids,
        "models": models,
        "records": len(records),
        "passed_records": len(records) - failed_records,
        "failed_records": failed_records,
        "passed": failed_records == 0,
        "avg_latency_s": statistics.mean(latencies) if latencies else None,
        "avg_tokens_per_sec": statistics.mean(tokens_per_sec) if tokens_per_sec else None,
        "case_summaries": _case_summaries(records, failures_by_index),
        "failures": failures,
    }


def review_benchmark_jsonl(path: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    source = Path(path)
    return review_benchmark_records(list(_iter_jsonl(source)), policy, inputs=[str(source)])


def review_benchmark_jsonls(paths: Iterable[str | Path], policy: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    inputs: list[str] = []
    for path in paths:
        source = Path(path)
        inputs.append(str(source))
        records.extend(_iter_jsonl(source))
    return review_benchmark_records(records, policy, inputs=inputs)


def format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Edge VLM Quality Review",
        "",
        f"- Inputs: {', '.join(f'`{path}`' for path in report.get('inputs', []))}",
        f"- Run IDs: {', '.join(f'`{run_id}`' for run_id in report.get('run_ids', []))}",
        f"- Models: {', '.join(f'`{model}`' for model in report.get('models', []))}",
        f"- Records: {report.get('records', 0)}",
        f"- Passed records: {report.get('passed_records', 0)}",
        f"- Failed records: {report.get('failed_records', 0)}",
        f"- Overall: {'pass' if report.get('passed') else 'fail'}",
        "",
        "| Case | Records | Passed | Failed |",
        "|---|---:|---:|---:|",
    ]
    case_summaries = report.get("case_summaries", {})
    if isinstance(case_summaries, dict):
        for case_id, summary in sorted(case_summaries.items()):
            if not isinstance(summary, dict):
                continue
            lines.append(
                "| {case} | {records} | {passed} | {failed} |".format(
                    case=case_id,
                    records=summary.get("records", 0),
                    passed=summary.get("passed", 0),
                    failed=summary.get("failed", 0),
                )
            )
    lines.append("")

    failures = report.get("failures", [])
    if failures:
        lines.extend(["## Failures", "", "| Run | Case | Trial | Failures |", "|---|---|---:|---|"])
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            lines.append(
                "| {run_id} | {case} | {trial} | {failures} |".format(
                    run_id=failure.get("run_id") or "",
                    case=failure.get("case_id") or "",
                    trial=failure.get("trial_index") or "",
                    failures=", ".join(str(item) for item in failure.get("failures", [])),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review benchmark JSONL outputs against route-specific quality policy.")
    parser.add_argument("--input", action="append", required=True, help="Benchmark JSONL input; repeatable")
    parser.add_argument("--policy", required=True, help="Quality review policy JSON")
    parser.add_argument("--output", help="Optional JSON report output")
    parser.add_argument("--markdown-output", help="Optional Markdown report output")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even when quality checks fail")
    args = parser.parse_args(argv)

    policy = _read_json(Path(args.policy))
    report = review_benchmark_jsonls(args.input, policy)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(format_markdown_report(report), encoding="utf-8")
    if not args.output and not args.markdown_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
