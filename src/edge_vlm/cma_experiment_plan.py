"""Build read-only CMA reservation experiment plans from selector artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
ROUNDING_STEP_MIB = 64
SAFETY_NOTE = (
    "Read-only planning artifact only; it does not edit boot configuration "
    "or guarantee startup."
)


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _memory_assessment(selection: dict[str, Any]) -> dict[str, Any]:
    for key in ("sudo_memory_blocker_assessment", "memory_blocker_assessment"):
        assessment = selection.get(key)
        if isinstance(assessment, dict):
            return assessment
    return {}


def _diagnostics_summaries(selection: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key in ("sudo_memory_diagnostics_summary", "memory_diagnostics_summary"):
        summary = selection.get(key)
        if isinstance(summary, dict):
            summaries.append(summary)
    return summaries


def _observed_lfb_block_mb(selection: dict[str, Any], assessment: dict[str, Any]) -> int | None:
    value = _int_value(assessment.get("observed_lfb_block_mb"))
    if value is not None:
        return value

    for summary in _diagnostics_summaries(selection):
        value = _int_value(summary.get("tegrastats_lfb_block_mb"))
        if value is not None:
            return value

    preflight = _dict_value(selection.get("preflight"))
    tegrastats = _dict_value(preflight.get("tegrastats"))
    lfb = _dict_value(tegrastats.get("lfb"))
    return _int_value(lfb.get("block_mb"))


def _linux_cma_reserved_size_bytes(selection: dict[str, Any], assessment: dict[str, Any]) -> int | None:
    value = _int_value(assessment.get("linux_cma_reserved_size_bytes"))
    if value is not None:
        return value
    for summary in _diagnostics_summaries(selection):
        value = _int_value(summary.get("linux_cma_reserved_size_bytes"))
        if value is not None:
            return value
    return None


def _round_up_to_mib_step(value: int, step_mib: int) -> int:
    step_bytes = step_mib * MIB
    return ((value + step_bytes - 1) // step_bytes) * step_bytes


def _candidate_requirements(
    selection: dict[str, Any],
    *,
    observed_lfb_block_mb: int | None,
    linux_cma_reserved_size_bytes: int | None,
) -> list[dict[str, Any]]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or observed_lfb_block_mb is None:
        return []

    requirements: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        variant_id = candidate.get("variant_id")
        min_lfb_blocks = _int_value(candidate.get("min_lfb_blocks"))
        if not isinstance(variant_id, str) or not variant_id or min_lfb_blocks is None:
            continue

        required_lfb_bytes = min_lfb_blocks * observed_lfb_block_mb * MIB
        if linux_cma_reserved_size_bytes is None:
            deficit_bytes = None
        else:
            deficit_bytes = max(required_lfb_bytes - linux_cma_reserved_size_bytes, 0)
        requirements.append(
            {
                "variant_id": variant_id,
                "min_lfb_blocks": min_lfb_blocks,
                "required_lfb_bytes": required_lfb_bytes,
                "linux_cma_reserved_size_bytes": linux_cma_reserved_size_bytes,
                "linux_cma_reserved_deficit_bytes": deficit_bytes,
                "block_reasons": _string_list(candidate.get("block_reasons")),
            }
        )
    return requirements


def _experiment_candidates(candidate_requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for requirement in candidate_requirements:
        variant_id = requirement["variant_id"]
        experiments.append(
            {
                "name": f"{variant_id}-minimum",
                "source": "variant_required_lfb",
                "variant_id": variant_id,
                "cma_bytes": requirement["required_lfb_bytes"],
            }
        )

    required_values = [
        requirement["required_lfb_bytes"]
        for requirement in candidate_requirements
        if _int_value(requirement.get("required_lfb_bytes")) is not None
    ]
    if required_values:
        rounded_max = _round_up_to_mib_step(max(required_values), ROUNDING_STEP_MIB)
        experiments.append(
            {
                "name": f"max-required-lfb-rounded-{ROUNDING_STEP_MIB}mib",
                "source": f"max_required_lfb_rounded_up_{ROUNDING_STEP_MIB}mib",
                "variant_id": None,
                "cma_bytes": rounded_max,
            }
        )

    return sorted(experiments, key=lambda experiment: (experiment["cma_bytes"], experiment["name"]))


def build_cma_experiment_plan(
    selection: dict[str, Any],
    *,
    source_selector_json: str | None = None,
) -> dict[str, Any]:
    """Derive a read-only preboot CMA experiment plan from selector evidence."""

    assessment = _memory_assessment(selection)
    observed_lfb_block_mb = _observed_lfb_block_mb(selection, assessment)
    linux_cma_reserved_size_bytes = _linux_cma_reserved_size_bytes(selection, assessment)
    candidate_requirements = _candidate_requirements(
        selection,
        observed_lfb_block_mb=observed_lfb_block_mb,
        linux_cma_reserved_size_bytes=linux_cma_reserved_size_bytes,
    )

    required_lfb_bytes_max = _int_value(assessment.get("required_lfb_bytes_max"))
    if required_lfb_bytes_max is None and candidate_requirements:
        required_lfb_bytes_max = max(requirement["required_lfb_bytes"] for requirement in candidate_requirements)

    linux_cma_reserved_deficit_bytes = _int_value(assessment.get("linux_cma_reserved_deficit_bytes"))
    if (
        linux_cma_reserved_deficit_bytes is None
        and linux_cma_reserved_size_bytes is not None
        and required_lfb_bytes_max is not None
    ):
        linux_cma_reserved_deficit_bytes = max(required_lfb_bytes_max - linux_cma_reserved_size_bytes, 0)

    missing_evidence: list[str] = []
    if observed_lfb_block_mb is None:
        missing_evidence.append("observed_lfb_block_mb")
    if linux_cma_reserved_size_bytes is None:
        missing_evidence.append("linux_cma_reserved_size_bytes")
    if not candidate_requirements:
        missing_evidence.append("candidate_min_lfb_blocks")

    return {
        "kind": "cma_experiment_plan",
        "schema_version": 1,
        "status": "insufficient_evidence" if missing_evidence else "ready",
        "missing_evidence": missing_evidence,
        "source_selector_json": source_selector_json,
        "selected_variant_id": selection.get("selected_variant_id"),
        "selected_reason": selection.get("selected_reason"),
        "preboot_capacity_status": assessment.get("preboot_capacity_status", "unknown"),
        "observed_lfb_block_mb": observed_lfb_block_mb,
        "linux_cma_reserved_size_bytes": linux_cma_reserved_size_bytes,
        "required_lfb_bytes_max": required_lfb_bytes_max,
        "linux_cma_reserved_deficit_bytes": linux_cma_reserved_deficit_bytes,
        "candidate_requirements": candidate_requirements,
        "experiment_candidates": _experiment_candidates(candidate_requirements),
        "applies_boot_config": False,
        "requires_manual_boot_config_change": True,
        "safety_note": SAFETY_NOTE,
    }


def write_cma_experiment_plan(
    *,
    selector_json: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    selector_path = Path(selector_json)
    output_path = Path(output)
    selection = _read_json_object(selector_path)
    plan = build_cma_experiment_plan(selection, source_selector_json=str(selector_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-json", required=True, help="Selector JSON artifact to read")
    parser.add_argument("--output", required=True, help="Path for the CMA experiment plan JSON artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = write_cma_experiment_plan(selector_json=args.selector_json, output=args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": args.output,
                "status": plan["status"],
                "experiment_candidate_count": len(plan["experiment_candidates"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
