"""Helpers for exporting machine-readable selections from compare eligibility artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


_VALID_GATES = ("startup", "ranking", "promotion")
_ROUTE_SELECTION_POLICY = {
    "name": "q4_first_q8_fallback",
    "group_key": "comparison_group",
    "quantization_source": "model_config.quantization",
    "unknown_quantization": "preserve_input_order",
}


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


def _row_comparison_group(row: dict[str, Any]) -> str:
    group = row.get("comparison_group")
    if isinstance(group, str) and group.strip():
        return group.strip()
    model = row.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return str(row.get("variant_id") or "")


def _row_quantization(row: dict[str, Any]) -> str:
    model_config = row.get("model_config")
    if not isinstance(model_config, dict):
        return ""
    quantization = model_config.get("quantization")
    return quantization.strip() if isinstance(quantization, str) else ""


def _quantization_bucket(row: dict[str, Any]) -> str:
    normalized = _row_quantization(row).upper()
    if normalized.startswith("Q4"):
        return "q4"
    if normalized.startswith("Q8"):
        return "q8"
    return ""


def _route_member(row: dict[str, Any], *, fallback_reason: str | None = None) -> dict[str, Any]:
    member: dict[str, Any] = {
        **_selected_id(row),
        "model": str(row.get("model") or ""),
        "comparison_group": _row_comparison_group(row),
        "quantization": _row_quantization(row),
    }
    selection = row.get("selection")
    if isinstance(selection, dict):
        member["selection"] = selection
    selection_context = row.get("selection_context")
    if isinstance(selection_context, dict) and selection_context:
        member["selection_context"] = selection_context
    if fallback_reason is not None:
        member["fallback_reason"] = fallback_reason
    return member


def _ordered_route_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        group = _row_comparison_group(candidate)
        if group not in grouped:
            group_order.append(group)
            grouped[group] = []
        grouped[group].append(candidate)

    ordered: list[dict[str, Any]] = []
    fallback_groups: list[dict[str, Any]] = []
    for group in group_order:
        group_candidates = grouped[group]
        buckets = [_quantization_bucket(candidate) for candidate in group_candidates]
        has_q4 = "q4" in buckets
        has_q8 = "q8" in buckets
        if has_q4 and has_q8:
            group_candidates = [
                *[candidate for candidate in group_candidates if _quantization_bucket(candidate) == "q4"],
                *[candidate for candidate in group_candidates if _quantization_bucket(candidate) != "q4"],
            ]
            q8_fallbacks = [
                _route_member(candidate, fallback_reason="q4_primary_q8_fallback")
                for candidate in group_candidates
                if _quantization_bucket(candidate) == "q8"
            ]
            fallback_groups.append(
                {
                    "comparison_group": group,
                    "candidate_count": len(group_candidates),
                    "primary": _route_member(group_candidates[0]),
                    "fallbacks": q8_fallbacks,
                }
            )
        ordered.extend(group_candidates)
    return ordered, fallback_groups


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


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def _bundle_filter_lanes(bundle: dict[str, Any]) -> list[str]:
    filters = bundle.get("filters")
    if not isinstance(filters, dict):
        return []
    candidate_lanes = filters.get("candidate_lanes")
    if not isinstance(candidate_lanes, list):
        return []
    return _ordered_unique(str(lane) for lane in candidate_lanes)


def build_candidate_route_export_artifact(
    *,
    input_path: str | Path,
    gate: str,
    output_path: str | Path,
    candidate_lanes: Iterable[str] | None = None,
) -> dict[str, Any]:
    if gate not in _VALID_GATES:
        raise ValueError(f"unsupported gate {gate!r}; expected one of {_VALID_GATES}")
    source = Path(input_path)
    bundle = _read_json_object(source)
    if bundle.get("kind") != "candidate_selection_bundle":
        raise ValueError(f"{source}: expected kind candidate_selection_bundle")
    selected_by_gate = bundle.get("selected")
    if not isinstance(selected_by_gate, dict):
        raise ValueError(f"{source}: expected selected object")
    selected = selected_by_gate.get(gate, [])
    if not isinstance(selected, list):
        raise ValueError(f"{source}: expected selected.{gate} list")

    gates = bundle.get("gates")
    if not isinstance(gates, dict):
        gates = {}
    gate_entry = gates.get(gate)
    gate_lanes = []
    if isinstance(gate_entry, dict):
        lanes = gate_entry.get("lanes")
        if isinstance(lanes, dict):
            gate_lanes = list(lanes)

    requested_lanes = _ordered_unique(candidate_lanes or [])
    lane_order = _ordered_unique(
        [
            *requested_lanes,
            *_bundle_filter_lanes(bundle),
            *gate_lanes,
            *(
                _selected_row_lane(row, fallback_lane="")
                for row in selected
                if isinstance(row, dict)
            ),
        ]
    )
    if requested_lanes:
        requested_lane_set = set(requested_lanes)
        lane_order = [lane for lane in lane_order if lane in requested_lane_set]

    route_candidates: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lane_order}
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: each selected.{gate} row must be a JSON object")
        lane = _selected_row_lane(row, fallback_lane=str(row.get("candidate_lane") or ""))
        if not lane:
            lane = str(row.get("candidate_lane") or "").strip()
        if requested_lanes and lane not in route_candidates:
            continue
        if lane and lane not in route_candidates:
            route_candidates[lane] = []
            lane_order.append(lane)
        if lane:
            route_candidates[lane].append(dict(row))

    routes = {}
    for lane in lane_order:
        candidates, fallback_groups = _ordered_route_candidates(route_candidates.get(lane, []))
        fallbacks: list[dict[str, Any]] = []
        for group in fallback_groups:
            group_fallbacks = group.get("fallbacks")
            if isinstance(group_fallbacks, list):
                fallbacks.extend(fallback for fallback in group_fallbacks if isinstance(fallback, dict))
        routes[lane] = {
            "candidate_count": len(candidates),
            "selected_ids": [_selected_id(row) for row in candidates],
            "primary": candidates[0] if candidates else None,
            "fallbacks": fallbacks,
            "fallback_groups": fallback_groups,
            "selection_policy": dict(_ROUTE_SELECTION_POLICY),
            "candidates": candidates,
        }
    route_export = {
        "kind": "candidate_route_export",
        "source_bundle": str(source),
        "gate": gate,
        "filters": {
            "candidate_lanes": lane_order,
        },
        "routes": routes,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(route_export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return route_export
