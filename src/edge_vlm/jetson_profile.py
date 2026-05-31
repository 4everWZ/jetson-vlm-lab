"""Jetson profiling helpers for tegrastats logs."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator

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


def iter_tegrastats_samples(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield parse_tegrastats_line(stripped)


def _mean_or_none(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return statistics.mean(items)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def summarize_tegrastats_samples(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    parsed_samples = list(samples)
    temps: list[float] = []
    vdd_in_w: list[float] = []
    gr3d_utils: list[float] = []
    emc_utils: list[float] = []
    lfb_blocks: list[int] = []
    ram_used: list[int] = []
    cpu_utils: list[float] = []

    for sample in parsed_samples:
        sample_temps = sample.get("temps_c")
        if isinstance(sample_temps, dict):
            temps.extend(float(value) for value in sample_temps.values() if isinstance(value, (int, float)))

        power = sample.get("power_mw")
        if isinstance(power, dict):
            vdd_in = power.get("VDD_IN")
            if isinstance(vdd_in, dict) and isinstance(vdd_in.get("instant"), int):
                vdd_in_w.append(vdd_in["instant"] / 1000.0)

        gr3d = sample.get("gr3d")
        if isinstance(gr3d, dict) and isinstance(gr3d.get("util_pct"), int):
            gr3d_utils.append(float(gr3d["util_pct"]))

        emc = sample.get("emc")
        if isinstance(emc, dict) and isinstance(emc.get("util_pct"), int):
            emc_utils.append(float(emc["util_pct"]))

        lfb = sample.get("lfb")
        if isinstance(lfb, dict) and isinstance(lfb.get("free_blocks"), int):
            lfb_blocks.append(int(lfb["free_blocks"]))

        ram = sample.get("ram")
        if isinstance(ram, dict) and isinstance(ram.get("used_mb"), int):
            ram_used.append(int(ram["used_mb"]))

        cpu = sample.get("cpu")
        cores = cpu.get("cores") if isinstance(cpu, dict) else None
        if isinstance(cores, list):
            cpu_utils.extend(
                float(core["util_pct"])
                for core in cores
                if isinstance(core, dict) and isinstance(core.get("util_pct"), int)
            )

    avg_gr3d = _mean_or_none(gr3d_utils)
    avg_emc = _mean_or_none(emc_utils)
    max_temp = max(temps) if temps else None
    bottlenecks: list[str] = []
    if avg_gr3d is not None and avg_gr3d >= 85.0:
        bottlenecks.append("gpu_compute")
    if avg_emc is not None and avg_emc >= 80.0:
        bottlenecks.append("emc_memory_bandwidth")
    if max_temp is not None and max_temp >= 80.0:
        bottlenecks.append("power_or_thermal")
    if not bottlenecks:
        bottlenecks.append("not_identified")

    return {
        "available": True,
        "samples": len(parsed_samples),
        "max_temp_c": max_temp,
        "avg_power_w": _round_or_none(_mean_or_none(vdd_in_w)),
        "avg_gr3d_util_pct": _round_or_none(avg_gr3d),
        "max_gr3d_util_pct": max(gr3d_utils) if gr3d_utils else None,
        "avg_emc_util_pct": _round_or_none(avg_emc),
        "max_emc_util_pct": max(emc_utils) if emc_utils else None,
        "min_lfb_free_blocks": min(lfb_blocks) if lfb_blocks else None,
        "max_ram_used_mb": max(ram_used) if ram_used else None,
        "avg_cpu_util_pct": _round_or_none(_mean_or_none(cpu_utils)),
        "bottleneck_labels": bottlenecks,
    }


def summarize_tegrastats_log(path: str | Path) -> dict[str, Any]:
    return summarize_tegrastats_samples(iter_tegrastats_samples(path))


def write_profile_summary(log_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary = summarize_tegrastats_log(log_path)
    summary["source"] = str(log_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
