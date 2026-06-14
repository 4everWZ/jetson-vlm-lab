"""Preflight checks for host-side GGUF artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


GGUF_MAGIC = b"GGUF"


def inspect_gguf_artifact(role: str, path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    exists = artifact_path.exists()
    is_file = artifact_path.is_file()
    record: dict[str, Any] = {
        "role": str(role),
        "path": str(artifact_path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": None,
        "gguf_magic": False,
        "status": "missing",
    }
    if not exists:
        return record
    if not is_file:
        record["status"] = "not_file"
        return record
    try:
        stat = artifact_path.stat()
        with artifact_path.open("rb") as handle:
            magic = handle.read(len(GGUF_MAGIC))
    except OSError as exc:
        record["status"] = "read_error"
        record["error"] = str(exc)
        return record
    record["size_bytes"] = stat.st_size
    record["gguf_magic"] = magic == GGUF_MAGIC
    record["status"] = "ok" if record["gguf_magic"] else "invalid_magic"
    return record


def inspect_gguf_artifacts(artifacts: Iterable[tuple[str, str | Path]]) -> dict[str, Any]:
    artifact_records = [inspect_gguf_artifact(role, path) for role, path in artifacts]
    failed_records = [record for record in artifact_records if record["status"] != "ok"]
    return {
        "schema_version": 1,
        "status": "ok" if not failed_records else "failed",
        "artifact_count": len(artifact_records),
        "failed_count": len(failed_records),
        "artifacts": artifact_records,
    }


def _write_manifest(manifest: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    check_parser = subparsers.add_parser("check", help="Write a machine-readable GGUF artifact preflight manifest")
    check_parser.add_argument(
        "--artifact",
        nargs=2,
        action="append",
        metavar=("ROLE", "PATH"),
        required=True,
        help="Artifact role and host path. Repeat for model, mmproj, or other explicit GGUF files.",
    )
    check_parser.add_argument("--output", type=Path, help="Manifest JSON output path. Defaults to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "check":
        parser.print_help(sys.stderr)
        return 2
    manifest = inspect_gguf_artifacts([(role, Path(path)) for role, path in args.artifact])
    _write_manifest(manifest, args.output)
    print(
        f"gguf_artifact_check={manifest['status']} "
        f"artifacts={manifest['artifact_count']} failed={manifest['failed_count']}",
        file=sys.stderr,
    )
    return 0 if manifest["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
