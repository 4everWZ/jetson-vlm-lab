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
REQUIRED_PHASES = (
    "artifact_check_or_download",
    "server_startup",
    "warmup",
    "formal_text",
    "formal_image",
    "fake_stream",
    "shutdown",
)
INPUT_TIMING_COMPONENT_KEYS = (
    "mime_detect_s",
    "image_read_s",
    "base64_encode_s",
    "data_url_build_s",
    "payload_build_s",
    "json_serialize_s",
    "http_request_s",
    "response_parse_s",
)
INPUT_PAYLOAD_FALLBACK_KEYS = (
    "mime_detect_s",
    "image_read_s",
    "base64_encode_s",
    "data_url_build_s",
)


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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sum_numeric_values(values: Iterable[float | None]) -> float | None:
    total = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _input_payload_overhead_s(timing: dict[str, Any]) -> float | None:
    payload_build = _float_or_none(timing.get("payload_build_s"))
    if payload_build is None:
        payload_build = _sum_numeric_values(
            _float_or_none(timing.get(key)) for key in INPUT_PAYLOAD_FALLBACK_KEYS
        )
    json_serialize = _float_or_none(timing.get("json_serialize_s"))
    return _sum_numeric_values((payload_build, json_serialize))


def _normalize_input_timing_summary(input_timing_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(input_timing_summary, dict):
        return {
            "available": False,
            "reason": "not_recorded",
            "records": 0,
            "records_with_latency": 0,
            "sources": {},
        }
    normalized = dict(input_timing_summary)
    normalized.setdefault("available", bool(normalized.get("records")))
    normalized.setdefault("reason", None if normalized.get("available") else "not_recorded")
    normalized.setdefault("records", 0)
    normalized.setdefault("records_with_latency", 0)
    sources = normalized.get("sources")
    normalized["sources"] = dict(sources) if isinstance(sources, dict) else {}
    return normalized


def summarize_input_timing_records(
    records: Iterable[dict[str, Any]],
    *,
    sources: dict[str, int] | None = None,
) -> dict[str, Any]:
    timing_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        timing = record.get("input_timing") if isinstance(record, dict) else None
        if isinstance(timing, dict):
            timing_records.append((record, timing))

    if not timing_records:
        return {
            "available": False,
            "reason": "not_recorded",
            "records": 0,
            "records_with_latency": 0,
            "sources": dict(sources or {}),
        }

    payload_overheads: list[float] = []
    runtime_waits: list[float] = []
    e2e_latencies: list[float] = []
    payload_ratio_payloads: list[float] = []
    payload_ratio_e2es: list[float] = []
    runtime_ratio_waits: list[float] = []
    runtime_ratio_e2es: list[float] = []
    request_body_bytes: list[float] = []
    image_bytes: list[float] = []
    component_values: dict[str, list[float]] = {key: [] for key in INPUT_TIMING_COMPONENT_KEYS}

    for record, timing in timing_records:
        payload_overhead = _input_payload_overhead_s(timing)
        if payload_overhead is not None:
            payload_overheads.append(payload_overhead)

        latency = _float_or_none(record.get("latency_s"))
        if latency is not None:
            e2e_latency = latency + (payload_overhead or 0.0)
            e2e_latencies.append(e2e_latency)
            if payload_overhead is not None:
                payload_ratio_payloads.append(payload_overhead)
                payload_ratio_e2es.append(e2e_latency)

        runtime_wait = _float_or_none(timing.get("http_request_s"))
        if runtime_wait is not None:
            runtime_waits.append(runtime_wait)
            if latency is not None:
                runtime_ratio_waits.append(runtime_wait)
                runtime_ratio_e2es.append(latency + (payload_overhead or 0.0))

        request_body = _float_or_none(timing.get("request_body_bytes"))
        if request_body is not None:
            request_body_bytes.append(request_body)

        image_size = _float_or_none(timing.get("image_bytes"))
        if image_size is not None:
            image_bytes.append(image_size)

        for key in INPUT_TIMING_COMPONENT_KEYS:
            value = _float_or_none(timing.get(key))
            if value is not None:
                component_values[key].append(value)

    total_payload_ratio_e2e = sum(payload_ratio_e2es)
    total_runtime_ratio_e2e = sum(runtime_ratio_e2es)
    payload_ratio = (
        sum(payload_ratio_payloads) / total_payload_ratio_e2e
        if total_payload_ratio_e2e > 0 and payload_ratio_payloads
        else None
    )
    runtime_wait_ratio = (
        sum(runtime_ratio_waits) / total_runtime_ratio_e2e
        if total_runtime_ratio_e2e > 0 and runtime_ratio_waits
        else None
    )
    component_averages = {
        key: _round_or_none(_mean_or_none(values))
        for key, values in component_values.items()
        if values
    }

    return {
        "available": True,
        "reason": None,
        "records": len(timing_records),
        "records_with_latency": len(e2e_latencies),
        "sources": dict(sources or {}),
        "avg_payload_overhead_s": _round_or_none(_mean_or_none(payload_overheads)),
        "avg_payload_overhead_ratio": _round_or_none(payload_ratio),
        "avg_e2e_latency_s": _round_or_none(_mean_or_none(e2e_latencies)),
        "avg_runtime_wait_s": _round_or_none(_mean_or_none(runtime_waits)),
        "avg_runtime_wait_ratio": _round_or_none(runtime_wait_ratio),
        "avg_request_body_bytes": _round_or_none(_mean_or_none(request_body_bytes)),
        "max_request_body_bytes": int(max(request_body_bytes)) if request_body_bytes else None,
        "avg_image_bytes": _round_or_none(_mean_or_none(image_bytes)),
        "max_image_bytes": int(max(image_bytes)) if image_bytes else None,
        "components_avg_s": component_averages,
    }


def _normalize_phase_timings(phase_timings: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    normalized = {
        phase: {
            "available": False,
            "duration_s": None,
            "reason": "not_recorded",
        }
        for phase in REQUIRED_PHASES
    }
    if not isinstance(phase_timings, dict):
        return normalized
    for phase, raw_value in phase_timings.items():
        if phase not in normalized or not isinstance(raw_value, dict):
            continue
        available = raw_value.get("available")
        duration = raw_value.get("duration_s")
        entry: dict[str, Any] = {
            "available": bool(available) if available is not None else isinstance(duration, (int, float)),
            "duration_s": float(duration) if isinstance(duration, (int, float)) else None,
            "reason": str(raw_value.get("reason") or "") or None,
        }
        source = raw_value.get("source")
        if isinstance(source, str) and source:
            entry["source"] = source
        details = raw_value.get("details")
        if isinstance(details, dict):
            entry["details"] = dict(details)
        normalized[phase] = entry
    return normalized


def _derive_bottleneck_labels(
    *,
    avg_gr3d: float | None,
    avg_emc: float | None,
    max_temp: float | None,
    avg_cpu: float | None,
    phase_timings: dict[str, dict[str, Any]],
    input_timing_summary: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if avg_gr3d is not None and avg_gr3d >= 85.0:
        labels.append("gpu_compute")
    if avg_emc is not None and avg_emc >= 80.0:
        labels.append("emc_memory_bandwidth")
    if max_temp is not None and max_temp >= 80.0:
        labels.append("power_or_thermal")
    if avg_cpu is not None and avg_cpu >= 80.0 and (avg_gr3d is None or avg_gr3d < 50.0):
        labels.append("cpu_prepost")
    startup = phase_timings.get("server_startup", {})
    artifact = phase_timings.get("artifact_check_or_download", {})
    startup_duration = startup.get("duration_s")
    artifact_duration = artifact.get("duration_s")
    if (
        isinstance(startup_duration, (int, float)) and startup_duration >= 30.0
    ) or (
        isinstance(artifact_duration, (int, float)) and artifact_duration >= 30.0
    ):
        labels.append("startup_or_download")
    input_payload_labeled = False
    if input_timing_summary.get("available") is True:
        payload_overhead = _float_or_none(input_timing_summary.get("avg_payload_overhead_s"))
        payload_ratio = _float_or_none(input_timing_summary.get("avg_payload_overhead_ratio"))
        if (
            payload_overhead is not None
            and payload_overhead >= 0.25
            and payload_ratio is not None
            and payload_ratio >= 0.15
        ):
            labels.append("input_payload")
            input_payload_labeled = True
        runtime_wait = _float_or_none(input_timing_summary.get("avg_runtime_wait_s"))
        runtime_ratio = _float_or_none(input_timing_summary.get("avg_runtime_wait_ratio"))
        if (
            not input_payload_labeled
            and runtime_wait is not None
            and runtime_wait >= 1.0
            and runtime_ratio is not None
            and runtime_ratio >= 0.75
            and (avg_gr3d is None or avg_gr3d < 50.0)
            and (avg_emc is None or avg_emc < 60.0)
            and (avg_cpu is None or avg_cpu < 80.0)
        ):
            labels.append("runtime_overhead")
    if not labels:
        labels.append("not_identified")
    return labels


def summarize_tegrastats_samples(
    samples: Iterable[dict[str, Any]],
    *,
    phase_timings: dict[str, Any] | None = None,
    input_timing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_samples = list(samples)
    normalized_phases = _normalize_phase_timings(phase_timings)
    normalized_input_summary = _normalize_input_timing_summary(input_timing_summary)
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
    avg_cpu = _mean_or_none(cpu_utils)
    max_temp = max(temps) if temps else None
    bottlenecks = _derive_bottleneck_labels(
        avg_gr3d=avg_gr3d,
        avg_emc=avg_emc,
        max_temp=max_temp,
        avg_cpu=avg_cpu,
        phase_timings=normalized_phases,
        input_timing_summary=normalized_input_summary,
    )

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
        "avg_cpu_util_pct": _round_or_none(avg_cpu),
        "phase_timings": normalized_phases,
        "input_timing_summary": normalized_input_summary,
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


def write_profile_artifacts(
    *,
    tegrastats_log: str | Path,
    profile_jsonl_path: str | Path,
    summary_path: str | Path,
    phase_timings: dict[str, Any] | None = None,
    input_timing_summary: dict[str, Any] | None = None,
    profile_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    samples = list(iter_tegrastats_samples(tegrastats_log))
    profile_jsonl = Path(profile_jsonl_path)
    profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with profile_jsonl.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(
                json.dumps({"sample_index": index, "sample": sample}, ensure_ascii=False)
                + "\n"
            )

    summary = summarize_tegrastats_samples(
        samples,
        phase_timings=phase_timings,
        input_timing_summary=input_timing_summary,
    )
    summary["source"] = str(tegrastats_log)
    summary["profile_jsonl"] = str(profile_jsonl)
    summary["profile_files"] = dict(profile_files or {})
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
