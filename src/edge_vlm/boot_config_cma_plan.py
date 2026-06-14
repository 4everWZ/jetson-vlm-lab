"""Build read-only boot config CMA patch plans from CMA experiment plans."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
CMA_TOKEN_RE = re.compile(r"(?<!\S)cma=[^\s]+")
SAFETY_NOTE = (
    "Read-only planning artifact only; it does not edit boot configuration, "
    "create backups, reboot, or guarantee startup."
)


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _resolve_boot_path(boot_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return boot_root / path.relative_to("/")
    return boot_root / path


def _candidate_boot_config_path(cma_plan: dict[str, Any], boot_root: Path) -> tuple[str | None, Path | None]:
    for path_text in _string_list(cma_plan.get("boot_config_readable_paths")):
        resolved_path = _resolve_boot_path(boot_root, path_text)
        try:
            if resolved_path.exists():
                return path_text, resolved_path
        except OSError:
            return path_text, resolved_path
    return None, None


def _append_lines(lines: list[str]) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if parts and parts[0].upper() == "APPEND":
            entries.append((line_number, line))
    return entries


def _cma_token(cma_bytes: int) -> str | None:
    if cma_bytes <= 0 or cma_bytes % MIB != 0:
        return None
    return f"cma={cma_bytes // MIB}M"


def _replace_or_append_cma_token(line: str, cma_token: str) -> tuple[str, str]:
    if CMA_TOKEN_RE.search(line):
        return CMA_TOKEN_RE.sub(cma_token, line), "replace_cma_token"
    separator = "" if line.endswith(" ") else " "
    return f"{line}{separator}{cma_token}", "append_cma_token"


def _unified_diff(
    *,
    path_text: str,
    lines: list[str],
    line_number: int,
    proposed_line: str,
    candidate_name: str,
) -> str:
    modified = list(lines)
    modified[line_number - 1] = proposed_line
    return "\n".join(
        difflib.unified_diff(
            lines,
            modified,
            fromfile=path_text,
            tofile=f"{path_text} ({candidate_name})",
            lineterm="",
        )
    )


def _patch_candidates(
    cma_plan: dict[str, Any],
    *,
    path_text: str,
    lines: list[str],
    line_number: int,
    current_append_line: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    raw_candidates = cma_plan.get("experiment_candidates")
    if not isinstance(raw_candidates, list):
        return candidates
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        cma_bytes = _int_value(raw_candidate.get("cma_bytes"))
        name = raw_candidate.get("name")
        if cma_bytes is None or not isinstance(name, str) or not name:
            continue
        token = _cma_token(cma_bytes)
        if token is None:
            continue
        proposed_line, action = _replace_or_append_cma_token(current_append_line, token)
        candidates.append(
            {
                "name": name,
                "source": raw_candidate.get("source"),
                "variant_id": raw_candidate.get("variant_id"),
                "cma_bytes": cma_bytes,
                "cma_token": token,
                "action": action,
                "line_number": line_number,
                "proposed_append_line": proposed_line,
                "unified_diff": _unified_diff(
                    path_text=path_text,
                    lines=lines,
                    line_number=line_number,
                    proposed_line=proposed_line,
                    candidate_name=name,
                ),
            }
        )
    return candidates


def build_boot_config_cma_plan(
    cma_plan: dict[str, Any],
    *,
    boot_root: str | Path = "/",
    source_cma_plan: str | None = None,
) -> dict[str, Any]:
    """Derive manual-review extlinux CMA patch candidates without writing files."""

    root = Path(boot_root)
    missing_evidence: list[str] = []
    path_text, resolved_path = _candidate_boot_config_path(cma_plan, root)
    if path_text is None or resolved_path is None:
        missing_evidence.append("boot_config_readable_paths")
        lines: list[str] = []
        append_entries: list[tuple[int, str]] = []
        target_line_number = None
        current_append_line = None
        current_cma_tokens: list[str] = []
        patch_candidates: list[dict[str, Any]] = []
    else:
        try:
            text = resolved_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
        except OSError:
            missing_evidence.append("boot_config_readable_path_content")
            lines = []
        append_entries = _append_lines(lines)
        if len(append_entries) != 1:
            missing_evidence.append("single_boot_config_append_line")
            target_line_number = None
            current_append_line = None
            current_cma_tokens = []
            patch_candidates = []
        else:
            target_line_number, current_append_line = append_entries[0]
            current_cma_tokens = CMA_TOKEN_RE.findall(current_append_line)
            patch_candidates = _patch_candidates(
                cma_plan,
                path_text=path_text,
                lines=lines,
                line_number=target_line_number,
                current_append_line=current_append_line,
            )
            if not patch_candidates:
                missing_evidence.append("mib_aligned_cma_experiment_candidates")

    status = "ready" if not missing_evidence else "review_required"
    return {
        "kind": "boot_config_cma_patch_plan",
        "schema_version": 1,
        "status": status,
        "missing_evidence": missing_evidence,
        "source_cma_plan": source_cma_plan,
        "source_selector_json": cma_plan.get("source_selector_json"),
        "target_boot_config_path": path_text,
        "resolved_boot_config_path": str(resolved_path) if resolved_path is not None else None,
        "target_append_line_number": target_line_number,
        "current_append_line": current_append_line,
        "current_cma_tokens": current_cma_tokens,
        "patch_candidates": patch_candidates,
        "applies_boot_config": False,
        "requires_manual_approval": True,
        "required_manual_steps": [
            "review the selected patch candidate",
            "create an out-of-band backup of the boot config",
            "apply the change manually on the Jetson",
            "reboot with serial or physical recovery available",
            "rerun memory diagnostics and the Qwen selector after reboot",
        ],
        "safety_note": SAFETY_NOTE,
    }


def write_boot_config_cma_plan(
    *,
    cma_plan: str | Path,
    output: str | Path,
    boot_root: str | Path = "/",
) -> dict[str, Any]:
    cma_plan_path = Path(cma_plan)
    output_path = Path(output)
    plan = build_boot_config_cma_plan(
        _read_json_object(cma_plan_path),
        boot_root=boot_root,
        source_cma_plan=str(cma_plan_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cma-plan", required=True, help="CMA experiment plan JSON artifact to read")
    parser.add_argument("--output", required=True, help="Path for the boot config CMA patch plan JSON artifact")
    parser.add_argument("--boot-root", default="/", help="Root directory for read-only boot config path inspection")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = write_boot_config_cma_plan(
            cma_plan=args.cma_plan,
            output=args.output,
            boot_root=args.boot_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": args.output,
                "status": plan["status"],
                "patch_candidate_count": len(plan["patch_candidates"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
