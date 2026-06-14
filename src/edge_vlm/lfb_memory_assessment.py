"""Conservative LFB gate evidence summaries for selector diagnostics."""

from __future__ import annotations

from typing import Any


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _lfb_from_selection(selection: dict[str, Any]) -> tuple[int | None, int | None]:
    preflight = selection.get("preflight") if isinstance(selection.get("preflight"), dict) else {}
    tegrastats = preflight.get("tegrastats") if isinstance(preflight.get("tegrastats"), dict) else {}
    lfb = tegrastats.get("lfb") if isinstance(tegrastats.get("lfb"), dict) else {}
    return _int_value(lfb.get("free_blocks")), _int_value(lfb.get("block_mb"))


def _lfb_from_diagnostics(summary: dict[str, Any]) -> tuple[int | None, int | None]:
    return _int_value(summary.get("tegrastats_lfb_free_blocks")), _int_value(summary.get("tegrastats_lfb_block_mb"))


def _candidate_lfb_deficits(
    selection: dict[str, Any],
    *,
    observed_lfb_free_blocks: int | None,
) -> dict[str, int]:
    if observed_lfb_free_blocks is None:
        return {}
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        return {}
    deficits: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        variant_id = candidate.get("variant_id")
        required = _int_value(candidate.get("min_lfb_blocks"))
        if not isinstance(variant_id, str) or not variant_id or required is None:
            continue
        deficit = required - observed_lfb_free_blocks
        if deficit > 0:
            deficits[variant_id] = deficit
    return deficits


def _debugfs_tracked_bytes(summary: dict[str, Any]) -> int | None:
    values = [
        _int_value(summary.get(key))
        for key in (
            "debugfs_dma_buf_total_bytes",
            "debugfs_nvmap_clients_total_bytes",
            "debugfs_nvmap_allocations_total_bytes",
        )
    ]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _debugfs_readable(summary: dict[str, Any]) -> bool:
    statuses = summary.get("debugfs_statuses")
    if not isinstance(statuses, dict):
        return False
    required = (
        statuses.get("dma_buf_bufinfo"),
        statuses.get("nvmap_iovmm_clients"),
        statuses.get("nvmap_iovmm_allocations"),
    )
    return all(status == "readable" for status in required)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _has_linux_cma_reserved_memory(names: list[str]) -> bool:
    return any(name == "linux,cma" or name.startswith("linux,cma@") for name in names)


def assess_selection_memory(
    *,
    selection: dict[str, Any],
    diagnostics_summary: dict[str, Any],
    diagnostics_source: str,
) -> dict[str, Any]:
    """Build an evidence summary without changing selector semantics."""

    observed_lfb_free_blocks, observed_lfb_block_mb = _lfb_from_diagnostics(diagnostics_summary)
    selection_lfb_free_blocks, selection_lfb_block_mb = _lfb_from_selection(selection)
    if observed_lfb_free_blocks is None:
        observed_lfb_free_blocks = selection_lfb_free_blocks
    if observed_lfb_block_mb is None:
        observed_lfb_block_mb = selection_lfb_block_mb

    candidate_deficits = _candidate_lfb_deficits(
        selection,
        observed_lfb_free_blocks=observed_lfb_free_blocks,
    )
    required_lfb_blocks_max = None
    if candidate_deficits and observed_lfb_free_blocks is not None:
        required_lfb_blocks_max = max(deficit + observed_lfb_free_blocks for deficit in candidate_deficits.values())

    observed_lfb_bytes = None
    required_lfb_bytes_max = None
    if observed_lfb_free_blocks is not None and observed_lfb_block_mb is not None:
        observed_lfb_bytes = observed_lfb_free_blocks * observed_lfb_block_mb * 1024 * 1024
        if required_lfb_blocks_max is not None:
            required_lfb_bytes_max = required_lfb_blocks_max * observed_lfb_block_mb * 1024 * 1024

    debugfs_total_bytes = _debugfs_tracked_bytes(diagnostics_summary)
    cma_free_kb = _int_value(diagnostics_summary.get("cma_free_kb"))
    cma_total_kb = _int_value(diagnostics_summary.get("cma_total_kb"))
    boot_cmdline_cma_token = diagnostics_summary.get("boot_cmdline_cma_token")
    if not isinstance(boot_cmdline_cma_token, str):
        boot_cmdline_cma_token = None
    boot_config_readable_paths = _string_list(diagnostics_summary.get("boot_config_readable_paths"))
    boot_config_cma_tokens = _string_list(diagnostics_summary.get("boot_config_cma_tokens"))
    boot_config_has_cma_token = bool(boot_config_cma_tokens)
    boot_config_append_line_count = _int_value(diagnostics_summary.get("boot_config_append_line_count"))
    reserved_memory_names = _string_list(diagnostics_summary.get("reserved_memory_names"))
    reserved_memory_node_count = _int_value(diagnostics_summary.get("reserved_memory_node_count"))
    linux_cma_reserved_memory_present = _has_linux_cma_reserved_memory(reserved_memory_names)
    linux_cma_reserved_size_bytes = _int_value(diagnostics_summary.get("linux_cma_reserved_size_bytes"))
    linux_cma_reserved_deficit_bytes = None
    preboot_capacity_status = "unknown"
    if linux_cma_reserved_size_bytes is not None and required_lfb_bytes_max is not None:
        linux_cma_reserved_deficit_bytes = required_lfb_bytes_max - linux_cma_reserved_size_bytes
        if linux_cma_reserved_deficit_bytes > 0:
            preboot_capacity_status = "linux_cma_reserved_below_required_lfb"
        else:
            linux_cma_reserved_deficit_bytes = 0
            preboot_capacity_status = "linux_cma_reserved_meets_required_lfb"
    signals: list[str] = []
    if candidate_deficits:
        signals.append("lfb_below_required")
    if _debugfs_readable(diagnostics_summary) and debugfs_total_bytes == 0:
        signals.append("debugfs_tracked_allocations_zero")
    if cma_free_kb is not None and required_lfb_bytes_max is not None and cma_free_kb * 1024 < required_lfb_bytes_max:
        signals.append("cma_free_below_required_lfb_bytes")
    if cma_total_kb is not None and required_lfb_bytes_max is not None and cma_total_kb * 1024 < required_lfb_bytes_max:
        signals.append("cma_total_below_required_lfb_bytes")
    if (
        linux_cma_reserved_size_bytes is not None
        and required_lfb_bytes_max is not None
        and linux_cma_reserved_size_bytes < required_lfb_bytes_max
    ):
        signals.append("linux_cma_reserved_below_required_lfb_bytes")

    status = "lfb_gate_not_met" if candidate_deficits else "no_lfb_gate_deficit_detected"
    selected_variant_id = selection.get("selected_variant_id")
    return {
        "status": status,
        "diagnostics_source": str(diagnostics_source),
        "selection_implication": (
            "no_candidate_meets_lfb_gate"
            if selected_variant_id is None and candidate_deficits
            else "selector_semantics_unchanged"
        ),
        "observed_lfb_free_blocks": observed_lfb_free_blocks,
        "observed_lfb_block_mb": observed_lfb_block_mb,
        "observed_lfb_bytes": observed_lfb_bytes,
        "required_lfb_blocks_max": required_lfb_blocks_max,
        "required_lfb_bytes_max": required_lfb_bytes_max,
        "lfb_free_block_deficit_max": max(candidate_deficits.values()) if candidate_deficits else None,
        "candidate_lfb_deficits": candidate_deficits,
        "cma_free_kb": cma_free_kb,
        "cma_total_kb": cma_total_kb,
        "debugfs_total_tracked_bytes": debugfs_total_bytes,
        "debugfs_readable": _debugfs_readable(diagnostics_summary),
        "boot_cmdline_cma_token": boot_cmdline_cma_token,
        "boot_config_readable_paths": boot_config_readable_paths,
        "boot_config_cma_tokens": boot_config_cma_tokens,
        "boot_config_has_cma_token": boot_config_has_cma_token,
        "boot_config_append_line_count": boot_config_append_line_count,
        "reserved_memory_node_count": reserved_memory_node_count,
        "reserved_memory_names": reserved_memory_names,
        "linux_cma_reserved_memory_present": linux_cma_reserved_memory_present,
        "linux_cma_reserved_size_bytes": linux_cma_reserved_size_bytes,
        "linux_cma_reserved_deficit_bytes": linux_cma_reserved_deficit_bytes,
        "preboot_capacity_status": preboot_capacity_status,
        "signals": signals,
        "note": "Evidence summary only; selector gates and ranking semantics are unchanged.",
    }
