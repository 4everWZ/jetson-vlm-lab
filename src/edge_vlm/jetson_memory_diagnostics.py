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
        "top_rss_processes": _read_processes(proc, limit=top_process_limit),
        "debugfs": _debugfs_status(sys),
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
