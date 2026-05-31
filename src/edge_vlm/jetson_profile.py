"""Jetson profiling helpers for tegrastats logs."""

from __future__ import annotations

import re
from typing import Any

RAM_RE = re.compile(
    r"\bRAM\s+(?P<used>\d+)/(?P<total>\d+)MB"
    r"(?:\s+\(lfb\s+(?P<lfb_blocks>\d+)x(?P<lfb_mb>\d+)MB\))?"
)
SWAP_RE = re.compile(
    r"\bSWAP\s+(?P<used>\d+)/(?P<total>\d+)MB"
    r"(?:\s+\(cached\s+(?P<cached>\d+)MB\))?"
)
CPU_RE = re.compile(r"\bCPU\s+\[(?P<cores>[^\]]*)\]")
CPU_CORE_RE = re.compile(r"(?P<util>\d+)%@(?P<freq>\d+)")
ENGINE_RE = re.compile(
    r"\b(?P<name>GR3D_FREQ|EMC_FREQ)\s+"
    r"(?P<util>\d+)%(?:@\[?(?P<freq>\d+)\]?)?"
)
TEMP_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_]+)@(?P<temp>[0-9]+(?:\.[0-9]+)?)C\b")
POWER_RE = re.compile(r"\b(?P<rail>VDD_[A-Z0-9_]+)\s+(?P<instant>\d+)mW/(?P<average>\d+)mW\b")


def _parse_cpu_cores(raw_cores: str) -> list[dict[str, int | str | None]]:
    cores: list[dict[str, int | str | None]] = []
    for raw_core in raw_cores.split(","):
        core = raw_core.strip()
        if core == "off":
            cores.append({"state": "off", "util_pct": None, "freq_mhz": None})
            continue
        match = CPU_CORE_RE.fullmatch(core)
        if match is None:
            cores.append({"state": "unknown", "util_pct": None, "freq_mhz": None})
            continue
        cores.append(
            {
                "state": "online",
                "util_pct": int(match.group("util")),
                "freq_mhz": int(match.group("freq")),
            }
        )
    return cores


def parse_tegrastats_line(line: str) -> dict[str, Any]:
    sample: dict[str, Any] = {"raw": line}

    ram_match = RAM_RE.search(line)
    if ram_match is not None:
        sample["ram"] = {
            "used_mb": int(ram_match.group("used")),
            "total_mb": int(ram_match.group("total")),
        }
        if ram_match.group("lfb_blocks") is not None and ram_match.group("lfb_mb") is not None:
            sample["lfb"] = {
                "free_blocks": int(ram_match.group("lfb_blocks")),
                "block_mb": int(ram_match.group("lfb_mb")),
            }

    swap_match = SWAP_RE.search(line)
    if swap_match is not None:
        swap = {
            "used_mb": int(swap_match.group("used")),
            "total_mb": int(swap_match.group("total")),
            "cached_mb": int(swap_match.group("cached") or 0),
        }
        sample["swap"] = swap

    cpu_match = CPU_RE.search(line)
    if cpu_match is not None:
        sample["cpu"] = {"cores": _parse_cpu_cores(cpu_match.group("cores"))}

    for engine_match in ENGINE_RE.finditer(line):
        key = "gr3d" if engine_match.group("name") == "GR3D_FREQ" else "emc"
        freq = engine_match.group("freq")
        sample[key] = {
            "util_pct": int(engine_match.group("util")),
            "freq_mhz": int(freq) if freq is not None else None,
        }

    temps = {
        match.group("name"): float(match.group("temp"))
        for match in TEMP_RE.finditer(line)
    }
    if temps:
        sample["temps_c"] = temps

    power = {
        match.group("rail"): {
            "instant": int(match.group("instant")),
            "average": int(match.group("average")),
        }
        for match in POWER_RE.finditer(line)
    }
    if power:
        sample["power_mw"] = power

    return sample
