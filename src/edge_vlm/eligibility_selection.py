"""Helpers for exporting machine-readable selections from compare eligibility artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


_VALID_GATES = ("startup", "ranking", "promotion")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _selected_id(row: dict[str, Any]) -> dict[str, str]:
    return {
        "run_id": str(row.get("run_id") or ""),
        "variant_id": str(row.get("variant_id") or ""),
    }


def _gate_artifact(row: dict[str, Any], *, gate: str, source: Path) -> dict[str, Any]:
    eligibility = row.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError(f"{source}: each row must include an eligibility object")
    gate_artifact = eligibility.get(f"{gate}_precheck")
    if not isinstance(gate_artifact, dict):
        raise ValueError(f"{source}: each row must include eligibility.{gate}_precheck")
    return gate_artifact


def build_eligibility_selection_artifact(
    *,
    input_paths: Iterable[str | Path],
    gate: str,
    output_path: str | Path,
) -> dict[str, Any]:
    if gate not in _VALID_GATES:
        raise ValueError(f"unsupported gate {gate!r}; expected one of {_VALID_GATES}")
    input_path_list = [Path(path) for path in input_paths]
    selected_rows: list[dict[str, Any]] = []
    for input_path in input_path_list:
        artifact = _read_json_object(input_path)
        rows = artifact.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{input_path}: expected rows list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}: each row must be a JSON object")
            gate_artifact = _gate_artifact(row, gate=gate, source=input_path)
            if gate_artifact.get("passed") is True:
                selected_rows.append(row)
    selection_artifact = {
        "gate": gate,
        "input_paths": [str(path) for path in input_path_list],
        "selected_count": len(selected_rows),
        "selected_ids": [_selected_id(row) for row in selected_rows],
        "selected": selected_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection_artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selection_artifact
