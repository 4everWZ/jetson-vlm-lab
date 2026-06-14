"""Enrich selector context JSON with follow-up diagnostics artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def enrich_with_sudo_memory_diagnostics(
    *,
    selector_output: str | Path,
    sudo_memory_diagnostics_output: str | Path,
) -> dict[str, Any]:
    selector_path = Path(selector_output)
    diagnostics_path = Path(sudo_memory_diagnostics_output)
    selector = _read_json_object(selector_path)
    diagnostics = _read_json_object(diagnostics_path)
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}

    selector["sudo_memory_diagnostics_path"] = str(diagnostics_path)
    selector["sudo_memory_diagnostics_summary"] = dict(summary)
    selector_path.write_text(json.dumps(selector, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-output", required=True, help="Selector JSON file to update in place")
    parser.add_argument(
        "--sudo-memory-diagnostics-output",
        required=True,
        help="Sudo memory diagnostics JSON sidecar to summarize into the selector JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        enriched = enrich_with_sudo_memory_diagnostics(
            selector_output=args.selector_output,
            sudo_memory_diagnostics_output=args.sudo_memory_diagnostics_output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "selector_output": args.selector_output,
                "sudo_memory_diagnostics_path": enriched.get("sudo_memory_diagnostics_path"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
