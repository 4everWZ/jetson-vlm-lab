"""Capture low-level Jetson memory diagnostics for LFB triage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUDDYINFO_RE = re.compile(r"^Node\s+(?P<node>\d+),\s+zone\s+(?P<zone>\S+)\s+(?P<counts>.+)$")
LFB_RE = re.compile(r"\blfb\s+(?P<free_blocks>\d+)x(?P<block_mb>\d+)MB\b")
DMA_BUF_TOTAL_RE = re.compile(r"^\s*Total\s+(?P<object_count>\d+)\s+objects?,\s+(?P<total_bytes>\d+)\s+bytes\b")
VMSTAT_KEYS = {
    "compact_success",
    "compact_fail",
    "compact_stall",
    "compact_daemon_wake",
    "compact_daemon_migrate_scanned",
    "compact_daemon_free_scanned",
    "pgscan_kswapd",
    "pgscan_direct",
    "pgsteal_kswapd",
    "pgsteal_direct",
}
DEBUGFS_PATHS = {
    "nvmap_iovmm_clients": "kernel/debug/nvmap/iovmm/clients",
    "nvmap_iovmm_allocations": "kernel/debug/nvmap/iovmm/allocations",
    "dma_buf_bufinfo": "kernel/debug/dma_buf/bufinfo",
    "cma_debug": "kernel/debug/cma",
}
DEVICE_TREE_STRING_PROPERTIES = {
    "compatible",
    "status",
    "name",
}
SIZE_TOKEN_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "KIB": 1024,
    "M": 1024 * 1024,
    "MB": 1024 * 1024,
    "MIB": 1024 * 1024,
    "G": 1024 * 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
    "GIB": 1024 * 1024 * 1024,
}


def _read_text(path: Path, *, max_chars: int | None = None) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text if max_chars is None else text[:max_chars]


def _read_int_file(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _read_meminfo(proc_root: Path) -> dict[str, int]:
    meminfo: dict[str, int] = {}
    text = _read_text(proc_root / "meminfo")
    if text is None:
        return meminfo
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if not parts:
            continue
        try:
            meminfo[key] = int(parts[0])
        except ValueError:
            continue
    return meminfo


def parse_buddyinfo_line(line: str) -> dict[str, Any] | None:
    match = BUDDYINFO_RE.match(line.strip())
    if match is None:
        return None
    try:
        counts = [int(part) for part in match.group("counts").split()]
    except ValueError:
        return None
    max_order = None
    for order, count in enumerate(counts):
        if count > 0:
            max_order = order
    return {
        "node": int(match.group("node")),
        "zone": match.group("zone"),
        "free_blocks_by_order": counts,
        "max_order_with_free_block": max_order,
    }


def _read_buddyinfo(proc_root: Path) -> dict[str, Any]:
    text = _read_text(proc_root / "buddyinfo")
    if text is None:
        return {"available": False, "zones": [], "max_order_with_free_block": None}
    zones = []
    max_order = None
    for line in text.splitlines():
        parsed = parse_buddyinfo_line(line)
        if parsed is None:
            continue
        zones.append(parsed)
        zone_max_order = parsed.get("max_order_with_free_block")
        if isinstance(zone_max_order, int) and (max_order is None or zone_max_order > max_order):
            max_order = zone_max_order
    return {
        "available": bool(zones),
        "zones": zones,
        "max_order_with_free_block": max_order,
    }


def _parse_swaps(text: str | None) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    if text is None:
        return {"available": False, "devices": [], "total_kb": 0, "used_kb": 0}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            size_kb = int(parts[2])
            used_kb = int(parts[3])
            priority = int(parts[4])
        except ValueError:
            continue
        devices.append(
            {
                "filename": parts[0],
                "type": parts[1],
                "size_kb": size_kb,
                "used_kb": used_kb,
                "priority": priority,
            }
        )
    return {
        "available": True,
        "devices": devices,
        "total_kb": sum(int(device["size_kb"]) for device in devices),
        "used_kb": sum(int(device["used_kb"]) for device in devices),
    }


def _read_vmstat(proc_root: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    text = _read_text(proc_root / "vmstat")
    if text is None:
        return values
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in VMSTAT_KEYS:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _parse_pressure_line(line: str) -> dict[str, Any] | None:
    parts = line.split()
    if not parts:
        return None
    parsed: dict[str, Any] = {"kind": parts[0]}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        try:
            parsed[key] = float(raw_value) if "." in raw_value else int(raw_value)
        except ValueError:
            parsed[key] = raw_value
    return parsed


def _read_pressure(proc_root: Path) -> dict[str, Any]:
    pressure: dict[str, Any] = {}
    for name in ("memory", "cpu", "io"):
        text = _read_text(proc_root / "pressure" / name)
        if text is None:
            pressure[name] = []
            continue
        pressure[name] = [parsed for line in text.splitlines() if (parsed := _parse_pressure_line(line))]
    return pressure


def _read_boot_cmdline(proc_root: Path) -> dict[str, Any]:
    path = proc_root / "cmdline"
    text = _read_text(path, max_chars=8192)
    if text is None:
        return {"path": str(path), "available": False, "raw": None, "cma_token": None}
    raw = " ".join(text.replace("\0", " ").split())
    cma_token = next((token for token in raw.split() if token.startswith("cma=")), None)
    return {"path": str(path), "available": True, "raw": raw, "cma_token": cma_token}


def _looks_like_device_tree_string_list(raw: bytes) -> bool:
    if not raw:
        return False
    values = [part for part in raw.rstrip(b"\0").split(b"\0") if part]
    if not values:
        return False
    return all(all(32 <= byte < 127 for byte in value) for value in values)


def _decode_device_tree_string_list(raw: bytes) -> list[str] | None:
    if not _looks_like_device_tree_string_list(raw):
        return None
    return [part.decode("ascii", errors="strict") for part in raw.rstrip(b"\0").split(b"\0") if part]


def _decode_device_tree_u32(raw: bytes) -> int | None:
    if len(raw) != 4:
        return None
    return int.from_bytes(raw, byteorder="big", signed=False)


def _read_device_tree_u32(path: Path) -> int | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return _decode_device_tree_u32(raw)


def _decode_device_tree_cells(raw: bytes, cell_count: int | None) -> int | None:
    if cell_count is None or cell_count <= 0:
        return None
    if len(raw) != cell_count * 4:
        return None
    value = 0
    for offset in range(0, len(raw), 4):
        value = (value << 32) | int.from_bytes(raw[offset : offset + 4], byteorder="big", signed=False)
    return value


def _decode_reg_regions(raw: bytes, *, address_cells: int | None, size_cells: int | None) -> list[dict[str, int]] | None:
    if address_cells is None or size_cells is None or address_cells <= 0 or size_cells <= 0:
        return None
    region_cell_count = address_cells + size_cells
    region_bytes = region_cell_count * 4
    if region_bytes <= 0 or len(raw) == 0 or len(raw) % region_bytes != 0:
        return None
    regions: list[dict[str, int]] = []
    for offset in range(0, len(raw), region_bytes):
        address_raw = raw[offset : offset + address_cells * 4]
        size_raw = raw[offset + address_cells * 4 : offset + region_bytes]
        address_bytes = _decode_device_tree_cells(address_raw, address_cells)
        size_bytes = _decode_device_tree_cells(size_raw, size_cells)
        if address_bytes is None or size_bytes is None:
            return None
        regions.append({"address_bytes": address_bytes, "size_bytes": size_bytes})
    return regions


def _read_reserved_memory_property(
    path: Path,
    *,
    address_cells: int | None,
    size_cells: int | None,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {f"{path.name}_error": str(exc)}
    if raw == b"":
        return {path.name: True}
    if path.name in DEVICE_TREE_STRING_PROPERTIES:
        decoded = _decode_device_tree_string_list(raw)
        if decoded is not None:
            return {path.name: decoded}
    properties: dict[str, Any] = {f"{path.name}_hex": raw.hex()}
    if path.name == "size":
        size_bytes = _decode_device_tree_cells(raw, size_cells)
        if size_bytes is not None:
            properties["size_bytes"] = size_bytes
    if path.name == "reg":
        regions = _decode_reg_regions(raw, address_cells=address_cells, size_cells=size_cells)
        if regions is not None:
            properties["reg_regions"] = regions
    return properties


def _read_reserved_memory_node(
    path: Path,
    *,
    address_cells: int | None,
    size_cells: int | None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    try:
        entries = sorted(path.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        return {"name": path.name, "status": "unreadable", "error": str(exc), "properties": properties}
    for entry in entries:
        if entry.is_dir():
            continue
        properties.update(_read_reserved_memory_property(entry, address_cells=address_cells, size_cells=size_cells))
    return {"name": path.name, "status": "readable", "properties": properties}


def _read_reserved_memory(proc_root: Path, sys_root: Path) -> dict[str, Any]:
    candidates = [
        sys_root / "firmware" / "devicetree" / "base" / "reserved-memory",
        proc_root / "device-tree" / "reserved-memory",
    ]
    selected_path = candidates[0]
    for candidate in candidates:
        try:
            if candidate.exists():
                selected_path = candidate
                break
        except OSError:
            selected_path = candidate
            break
    try:
        exists = selected_path.exists()
    except OSError as exc:
        return {
            "path": str(selected_path),
            "available": False,
            "status": "unreadable",
            "error": str(exc),
            "nodes": [],
            "node_count": 0,
        }
    if not exists:
        return {
            "path": str(selected_path),
            "available": False,
            "status": "missing",
            "nodes": [],
            "node_count": 0,
        }
    try:
        entries = sorted(selected_path.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        return {
            "path": str(selected_path),
            "available": False,
            "status": "unreadable",
            "error": str(exc),
            "nodes": [],
            "node_count": 0,
        }
    address_cells = _read_device_tree_u32(selected_path / "#address-cells")
    size_cells = _read_device_tree_u32(selected_path / "#size-cells")
    nodes = [
        _read_reserved_memory_node(entry, address_cells=address_cells, size_cells=size_cells)
        for entry in entries
        if entry.is_dir()
    ]
    return {
        "path": str(selected_path),
        "available": True,
        "status": "readable",
        "address_cells": address_cells,
        "size_cells": size_cells,
        "nodes": nodes,
        "node_count": len(nodes),
    }


def _read_boot_memory(proc_root: Path, sys_root: Path) -> dict[str, Any]:
    return {
        "cmdline": _read_boot_cmdline(proc_root),
        "reserved_memory": _read_reserved_memory(proc_root, sys_root),
    }


def _linux_cma_reserved_size_bytes(nodes: list[Any]) -> int | None:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name")
        if not isinstance(name, str) or (name != "linux,cma" and not name.startswith("linux,cma@")):
            continue
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        size_bytes = properties.get("size_bytes")
        if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
            return size_bytes
        regions = properties.get("reg_regions")
        if isinstance(regions, list) and len(regions) == 1 and isinstance(regions[0], dict):
            region_size = regions[0].get("size_bytes")
            if isinstance(region_size, int) and not isinstance(region_size, bool):
                return region_size
    return None


def _read_cmdline(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def _read_processes(proc_root: Path, *, limit: int) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for entry in proc_root.iterdir() if proc_root.is_dir() else []:
        if not entry.name.isdigit():
            continue
        status_text = _read_text(entry / "status")
        if status_text is None:
            continue
        fields: dict[str, str] = {}
        for line in status_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key] = value.strip()
        rss_text = fields.get("VmRSS", "")
        rss_parts = rss_text.split()
        if not rss_parts:
            continue
        try:
            rss_kb = int(rss_parts[0])
        except ValueError:
            continue
        uid_text = fields.get("Uid", "").split()
        uid = int(uid_text[0]) if uid_text and uid_text[0].isdigit() else None
        command = _read_cmdline(entry / "cmdline") or fields.get("Name", "")
        processes.append(
            {
                "pid": int(entry.name),
                "name": fields.get("Name", ""),
                "state": fields.get("State", ""),
                "uid": uid,
                "rss_kb": rss_kb,
                "command": command,
            }
        )
    processes.sort(key=lambda process: int(process["rss_kb"]), reverse=True)
    return processes[:limit]


def _read_zram(sys_root: Path) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    for entry in sorted((sys_root / "block").glob("zram*")):
        if not entry.is_dir():
            continue
        device: dict[str, Any] = {
            "name": entry.name,
            "disksize_bytes": _read_int_file(entry / "disksize"),
        }
        mm_stat = _read_text(entry / "mm_stat")
        if mm_stat:
            fields = mm_stat.split()
            if len(fields) >= 3:
                for key, index in (
                    ("orig_data_size_bytes", 0),
                    ("compr_data_size_bytes", 1),
                    ("mem_used_total_bytes", 2),
                ):
                    try:
                        device[key] = int(fields[index])
                    except ValueError:
                        pass
        devices.append(device)
    return {"devices": devices}


def _debugfs_status(sys_root: Path) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for key, relative_path in DEBUGFS_PATHS.items():
        path = sys_root / relative_path
        try:
            exists = path.exists()
        except OSError as exc:
            statuses[key] = {"path": str(path), "status": "unreadable", "error": str(exc)}
            continue
        if not exists:
            statuses[key] = {"path": str(path), "status": "missing"}
            continue
        try:
            is_directory = path.is_dir()
        except OSError as exc:
            statuses[key] = {"path": str(path), "status": "unreadable", "error": str(exc)}
            continue
        if is_directory:
            statuses[key] = {"path": str(path), "status": "directory"}
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                preview = handle.read(4096)
        except OSError as exc:
            statuses[key] = {"path": str(path), "status": "unreadable", "error": str(exc)}
            continue
        statuses[key] = {"path": str(path), "status": "readable", "preview": preview}
    return statuses


def _parse_size_token_to_bytes(token: str) -> int | None:
    match = re.fullmatch(r"(?P<value>\d+)(?P<unit>[A-Za-z]*)", token.strip())
    if match is None:
        return None
    multiplier = SIZE_TOKEN_MULTIPLIERS.get(match.group("unit").upper())
    if multiplier is None:
        return None
    return int(match.group("value")) * multiplier


def _parse_nvmap_total_bytes(preview: str) -> int | None:
    for line in preview.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "total":
            return _parse_size_token_to_bytes(parts[-1])
    return None


def _parse_dma_buf_totals(preview: str) -> dict[str, int | None]:
    for line in preview.splitlines():
        match = DMA_BUF_TOTAL_RE.match(line)
        if match is None:
            continue
        return {
            "object_count": int(match.group("object_count")),
            "total_bytes": int(match.group("total_bytes")),
        }
    return {"object_count": None, "total_bytes": None}


def _summarize_debugfs(debugfs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "statuses": {
            key: value.get("status") if isinstance(value, dict) else None for key, value in debugfs.items()
        }
    }
    for key in ("nvmap_iovmm_clients", "nvmap_iovmm_allocations"):
        value = debugfs.get(key)
        preview = value.get("preview", "") if isinstance(value, dict) else ""
        summary[key] = {"total_bytes": _parse_nvmap_total_bytes(preview)}
    dma_buf = debugfs.get("dma_buf_bufinfo")
    dma_buf_preview = dma_buf.get("preview", "") if isinstance(dma_buf, dict) else ""
    summary["dma_buf_bufinfo"] = _parse_dma_buf_totals(dma_buf_preview)
    return summary


def parse_tegrastats_lfb(line: str) -> dict[str, int] | None:
    match = LFB_RE.search(line)
    if match is None:
        return None
    return {
        "free_blocks": int(match.group("free_blocks")),
        "block_mb": int(match.group("block_mb")),
    }


def _sample_tegrastats(interval_ms: int = 1000, timeout_s: float = 2.5) -> dict[str, Any]:
    if shutil.which("tegrastats") is None:
        return {"available": False, "raw": None, "lfb": None}
    process = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=2.0)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    raw = lines[-1] if lines else None
    return {
        "available": True,
        "raw": raw,
        "lfb": parse_tegrastats_lfb(raw or ""),
    }


def _summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    meminfo = diagnostics.get("meminfo_kb") if isinstance(diagnostics.get("meminfo_kb"), dict) else {}
    swap = diagnostics.get("swap") if isinstance(diagnostics.get("swap"), dict) else {}
    buddyinfo = diagnostics.get("buddyinfo") if isinstance(diagnostics.get("buddyinfo"), dict) else {}
    tegrastats = diagnostics.get("tegrastats") if isinstance(diagnostics.get("tegrastats"), dict) else {}
    lfb = tegrastats.get("lfb") if isinstance(tegrastats.get("lfb"), dict) else {}
    debugfs_summary = (
        diagnostics.get("debugfs_summary") if isinstance(diagnostics.get("debugfs_summary"), dict) else {}
    )
    debugfs_statuses = (
        debugfs_summary.get("statuses") if isinstance(debugfs_summary.get("statuses"), dict) else {}
    )
    dma_buf_summary = (
        debugfs_summary.get("dma_buf_bufinfo") if isinstance(debugfs_summary.get("dma_buf_bufinfo"), dict) else {}
    )
    nvmap_clients_summary = (
        debugfs_summary.get("nvmap_iovmm_clients")
        if isinstance(debugfs_summary.get("nvmap_iovmm_clients"), dict)
        else {}
    )
    nvmap_allocations_summary = (
        debugfs_summary.get("nvmap_iovmm_allocations")
        if isinstance(debugfs_summary.get("nvmap_iovmm_allocations"), dict)
        else {}
    )
    boot_memory = diagnostics.get("boot_memory") if isinstance(diagnostics.get("boot_memory"), dict) else {}
    cmdline = boot_memory.get("cmdline") if isinstance(boot_memory.get("cmdline"), dict) else {}
    reserved_memory = (
        boot_memory.get("reserved_memory") if isinstance(boot_memory.get("reserved_memory"), dict) else {}
    )
    reserved_nodes = reserved_memory.get("nodes") if isinstance(reserved_memory.get("nodes"), list) else []
    return {
        "mem_available_kb": meminfo.get("MemAvailable"),
        "mem_free_kb": meminfo.get("MemFree"),
        "cma_total_kb": meminfo.get("CmaTotal"),
        "cma_free_kb": meminfo.get("CmaFree"),
        "swap_total_kb": swap.get("total_kb"),
        "swap_used_kb": swap.get("used_kb"),
        "buddyinfo_max_order_with_free_block": buddyinfo.get("max_order_with_free_block"),
        "tegrastats_lfb_free_blocks": lfb.get("free_blocks"),
        "tegrastats_lfb_block_mb": lfb.get("block_mb"),
        "zram_device_count": len(diagnostics.get("zram", {}).get("devices", [])),
        "top_rss_processes": diagnostics.get("top_rss_processes", [])[:5],
        "debugfs_statuses": debugfs_statuses,
        "debugfs_dma_buf_object_count": dma_buf_summary.get("object_count"),
        "debugfs_dma_buf_total_bytes": dma_buf_summary.get("total_bytes"),
        "debugfs_nvmap_clients_total_bytes": nvmap_clients_summary.get("total_bytes"),
        "debugfs_nvmap_allocations_total_bytes": nvmap_allocations_summary.get("total_bytes"),
        "boot_cmdline_cma_token": cmdline.get("cma_token"),
        "reserved_memory_node_count": reserved_memory.get("node_count", 0),
        "reserved_memory_names": [node.get("name") for node in reserved_nodes if isinstance(node, dict)],
        "linux_cma_reserved_size_bytes": _linux_cma_reserved_size_bytes(reserved_nodes),
    }


def capture_memory_diagnostics(
    output: str | Path,
    *,
    proc_root: str | Path = "/proc",
    sys_root: str | Path = "/sys",
    top_process_limit: int = 10,
    sample_tegrastats: bool = True,
) -> dict[str, Any]:
    proc = Path(proc_root)
    sys = Path(sys_root)
    debugfs = _debugfs_status(sys)
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "meminfo_kb": _read_meminfo(proc),
        "buddyinfo": _read_buddyinfo(proc),
        "swap": _parse_swaps(_read_text(proc / "swaps")),
        "zram": _read_zram(sys),
        "vmstat": _read_vmstat(proc),
        "pressure": _read_pressure(proc),
        "boot_memory": _read_boot_memory(proc, sys),
        "top_rss_processes": _read_processes(proc, limit=top_process_limit),
        "debugfs": debugfs,
        "debugfs_summary": _summarize_debugfs(debugfs),
        "tegrastats": _sample_tegrastats() if sample_tegrastats else {"available": False, "raw": None, "lfb": None},
    }
    diagnostics["summary"] = _summary(diagnostics)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--top-process-limit", type=int, default=10)
    parser.add_argument("--skip-tegrastats", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    diagnostics = capture_memory_diagnostics(
        args.output,
        top_process_limit=args.top_process_limit,
        sample_tegrastats=not args.skip_tegrastats,
    )
    print(json.dumps({"output": args.output, "summary": diagnostics["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
