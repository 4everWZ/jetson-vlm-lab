"""Generate per-run quality review sidecars for a sweep manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .quality_review import format_markdown_report, review_benchmark_jsonl


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def write_sweep_quality_reviews(
    *,
    manifest_path: str | Path,
    policy_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _read_json_object(manifest_file)
    policy = _read_json_object(Path(policy_path))
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise ValueError(f"{manifest_file}: expected plan object")
    plan_variants = plan.get("variants")
    if not isinstance(plan_variants, list):
        raise ValueError(f"{manifest_file}: expected plan.variants list")

    run_root = Path(output_root) if output_root is not None else manifest_file.parent
    run_root.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    reports: list[dict[str, Any]] = []
    for variant in plan_variants:
        if not isinstance(variant, dict):
            continue
        run_id = str(variant.get("run_id") or "").strip()
        paths = variant.get("paths")
        if not isinstance(paths, dict):
            continue
        benchmark_jsonl = paths.get("benchmark_jsonl")
        if not run_id or not benchmark_jsonl:
            continue
        benchmark_path = Path(str(benchmark_jsonl))
        if not benchmark_path.is_file():
            continue
        report = review_benchmark_jsonl(benchmark_path, policy)
        json_output = run_root / f"{run_id}.quality.json"
        md_output = run_root / f"{run_id}.quality.md"
        json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_output.write_text(format_markdown_report(report), encoding="utf-8")
        paths["quality_review_json"] = str(json_output)
        paths["quality_review_markdown"] = str(md_output)
        written += 1
        if report.get("passed") is not True:
            failed += 1
        reports.append(
            {
                "run_id": run_id,
                "quality_review_json": str(json_output),
                "quality_review_markdown": str(md_output),
                "passed": report.get("passed") is True,
            }
        )

    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_file),
        "output_root": str(run_root),
        "reviews_written": written,
        "failed_reviews": failed,
        "reports": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate quality review sidecars for each benchmark run in a sweep manifest.")
    parser.add_argument("--manifest", required=True, help="Sweep manifest path")
    parser.add_argument("--policy", required=True, help="Quality review policy JSON")
    parser.add_argument("--output-root", help="Directory for per-run quality review outputs; defaults to manifest parent")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even when one or more run reviews fail")
    args = parser.parse_args(argv)

    result = write_sweep_quality_reviews(
        manifest_path=args.manifest,
        policy_path=args.policy,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False))
    if not args.allow_failures and result["failed_reviews"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
