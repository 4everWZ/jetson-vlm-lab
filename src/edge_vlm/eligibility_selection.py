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
    require_leq2b_candidate: bool = False,
    candidate_lane: str | None = None,
) -> dict[str, Any]:
    if gate not in _VALID_GATES:
        raise ValueError(f"unsupported gate {gate!r}; expected one of {_VALID_GATES}")
    input_path_list = [Path(path) for path in input_paths]
    normalized_candidate_lane = candidate_lane.strip() if isinstance(candidate_lane, str) else ""
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
                candidate_scope = row.get("candidate_scope")
                if not isinstance(candidate_scope, dict):
                    candidate_scope = {}
                if require_leq2b_candidate and candidate_scope.get("leq2b_candidate") is not True:
                    continue
                row_candidate_lane = candidate_scope.get("lane")
                normalized_row_candidate_lane = row_candidate_lane.strip() if isinstance(row_candidate_lane, str) else ""
                if normalized_candidate_lane and normalized_row_candidate_lane != normalized_candidate_lane:
                    continue
                selected_rows.append(row)
    selection_artifact = {
        "gate": gate,
        "input_paths": [str(path) for path in input_path_list],
        "filters": {
            "require_leq2b_candidate": require_leq2b_candidate,
            "candidate_lane": normalized_candidate_lane,
        },
        "selected_count": len(selected_rows),
        "selected_ids": [_selected_id(row) for row in selected_rows],
        "selected": selected_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection_artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selection_artifact


def _selection_lane(artifact: dict[str, Any]) -> str:
    filters = artifact.get("filters")
    if not isinstance(filters, dict):
        return ""
    lane = filters.get("candidate_lane")
    return lane.strip() if isinstance(lane, str) else ""


def _selected_row_lane(row: dict[str, Any], *, fallback_lane: str) -> str:
    candidate_scope = row.get("candidate_scope")
    if isinstance(candidate_scope, dict):
        lane = candidate_scope.get("lane")
        if isinstance(lane, str) and lane.strip():
            return lane.strip()
    return fallback_lane


def _ensure_gate_entry(bundle: dict[str, Any], gate: str) -> dict[str, Any]:
    gates = bundle.setdefault("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("internal bundle gates must be a dictionary")
    gate_entry = gates.setdefault(gate, {"selected_count": 0, "selected_ids": [], "lanes": {}})
    if not isinstance(gate_entry, dict):
        raise ValueError("internal gate entry must be a dictionary")
    selected = bundle.setdefault("selected", {})
    if not isinstance(selected, dict):
        raise ValueError("internal selected entry must be a dictionary")
    selected.setdefault(gate, [])
    return gate_entry


def _ensure_lane_entry(gate_entry: dict[str, Any], lane: str) -> dict[str, Any]:
    lanes = gate_entry.setdefault("lanes", {})
    if not isinstance(lanes, dict):
        raise ValueError("internal gate lanes must be a dictionary")
    lane_entry = lanes.setdefault(lane, {"selected_count": 0, "selected_ids": []})
    if not isinstance(lane_entry, dict):
        raise ValueError("internal lane entry must be a dictionary")
    return lane_entry


def build_selection_bundle_artifact(
    *,
    input_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    input_path_list = [Path(path) for path in input_paths]
    bundle: dict[str, Any] = {
        "kind": "candidate_selection_bundle",
        "input_paths": [str(path) for path in input_path_list],
        "filters": {
            "require_leq2b_candidate": bool(input_path_list),
            "candidate_lanes": [],
        },
        "source_artifacts": [],
        "gates": {},
        "selected": {},
    }
    candidate_lanes: set[str] = set()
    require_leq2b_values: list[bool] = []
    for input_path in input_path_list:
        artifact = _read_json_object(input_path)
        gate = artifact.get("gate")
        if not isinstance(gate, str) or gate not in _VALID_GATES:
            raise ValueError(f"{input_path}: expected gate to be one of {_VALID_GATES}")
        filters = artifact.get("filters")
        if not isinstance(filters, dict):
            raise ValueError(f"{input_path}: expected filters object")
        require_leq2b_values.append(filters.get("require_leq2b_candidate") is True)
        artifact_lane = _selection_lane(artifact)
        if artifact_lane:
            candidate_lanes.add(artifact_lane)
        selected = artifact.get("selected")
        if not isinstance(selected, list):
            raise ValueError(f"{input_path}: expected selected list")
        gate_entry = _ensure_gate_entry(bundle, gate)
        if artifact_lane:
            _ensure_lane_entry(gate_entry, artifact_lane)
        source_artifacts = bundle["source_artifacts"]
        if not isinstance(source_artifacts, list):
            raise ValueError("internal source_artifacts must be a list")
        source_artifacts.append(
            {
                "path": str(input_path),
                "gate": gate,
                "filters": filters,
                "selected_count": len(selected),
            }
        )
        gate_selected = bundle["selected"][gate]
        if not isinstance(gate_selected, list):
            raise ValueError("internal gate selected entry must be a list")
        for row in selected:
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}: each selected row must be a JSON object")
            lane = _selected_row_lane(row, fallback_lane=artifact_lane)
            if lane:
                candidate_lanes.add(lane)
            row_with_source = dict(row)
            row_with_source["source_selection_path"] = str(input_path)
            row_with_source["source_gate"] = gate
            row_with_source["candidate_lane"] = lane
            selected_id = _selected_id(row)
            gate_entry["selected_ids"].append(selected_id)
            gate_entry["selected_count"] += 1
            lane_entry = _ensure_lane_entry(gate_entry, lane)
            lane_entry["selected_ids"].append(selected_id)
            lane_entry["selected_count"] += 1
            gate_selected.append(row_with_source)
    bundle["filters"] = {
        "require_leq2b_candidate": bool(require_leq2b_values) and all(require_leq2b_values),
        "candidate_lanes": sorted(candidate_lanes),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bundle
