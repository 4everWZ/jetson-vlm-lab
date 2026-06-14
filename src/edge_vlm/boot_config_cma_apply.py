"""Apply a reviewed boot config CMA patch plan with stale-plan guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


class BootConfigCmaApplyError(ValueError):
    """Raised when a boot config patch plan is not safe to apply."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_boot_path(boot_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return boot_root / path.relative_to("/")
    return boot_root / path


def _require(condition: bool, *, reason: str, message: str) -> None:
    if not condition:
        raise BootConfigCmaApplyError(reason, message)


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _candidate_by_id(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = plan.get("patch_candidates")
    _require(isinstance(candidates, list), reason="invalid_patch_candidates", message="patch_candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("candidate_id") == candidate_id or candidate.get("name") == candidate_id:
            return candidate
    raise BootConfigCmaApplyError("candidate_not_found", f"candidate not found: {candidate_id}")


def _line_body_and_ending(line: str) -> tuple[str, str]:
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _build_proposed_bytes(
    *,
    current_bytes: bytes,
    line_number: int,
    expected_line: str,
    proposed_line: str,
) -> bytes:
    current_text = current_bytes.decode("utf-8", errors="replace")
    lines = current_text.splitlines(keepends=True)
    _require(line_number > 0, reason="invalid_line_number", message="line_number must be positive")
    _require(line_number <= len(lines), reason="line_number_out_of_range", message="line_number exceeds file length")

    body, ending = _line_body_and_ending(lines[line_number - 1])
    _require(
        body == expected_line,
        reason="current_append_line_mismatch",
        message="current APPEND line no longer matches patch plan",
    )
    lines[line_number - 1] = proposed_line + ending
    return "".join(lines).encode("utf-8")


def _validate_plan_and_candidate(
    *,
    plan: dict[str, Any],
    candidate: dict[str, Any],
    current_bytes: bytes,
) -> tuple[str, str, int, str, str]:
    _require(
        plan.get("kind") == "boot_config_cma_patch_plan",
        reason="invalid_plan_kind",
        message="plan kind must be boot_config_cma_patch_plan",
    )
    _require(plan.get("status") == "ready", reason="plan_not_ready", message="patch plan status must be ready")

    target_path_text = _string_value(plan.get("target_boot_config_path"))
    target_hash = _string_value(plan.get("target_file_sha256"))
    expected_hash = _string_value(candidate.get("expected_current_sha256"))
    proposed_hash = _string_value(candidate.get("proposed_file_sha256"))
    current_line = _string_value(plan.get("current_append_line"))
    proposed_line = _string_value(candidate.get("proposed_append_line"))
    line_number = _int_value(candidate.get("line_number"))
    if line_number is None:
        line_number = _int_value(plan.get("target_append_line_number"))

    _require(target_path_text is not None, reason="missing_target_path", message="missing target_boot_config_path")
    _require(target_hash is not None, reason="missing_target_hash", message="missing target_file_sha256")
    _require(expected_hash is not None, reason="missing_expected_hash", message="missing expected_current_sha256")
    _require(proposed_hash is not None, reason="missing_proposed_hash", message="missing proposed_file_sha256")
    _require(current_line is not None, reason="missing_current_append_line", message="missing current_append_line")
    _require(proposed_line is not None, reason="missing_proposed_append_line", message="missing proposed_append_line")
    _require(line_number is not None, reason="missing_line_number", message="missing line_number")

    current_hash = _sha256_bytes(current_bytes)
    _require(
        current_hash == target_hash == expected_hash,
        reason="current_hash_mismatch",
        message="current boot config SHA256 does not match patch plan",
    )

    size_bytes = _int_value(plan.get("target_file_size_bytes"))
    if size_bytes is not None:
        _require(
            len(current_bytes) == size_bytes,
            reason="current_size_mismatch",
            message="current boot config size does not match patch plan",
        )

    return target_path_text, current_line, line_number, proposed_line, proposed_hash


def _backup_path_for(*, target_path: Path, backup_dir: Path | None, current_hash: str) -> Path:
    root = backup_dir if backup_dir is not None else target_path.parent
    return root / f"{target_path.name}.pre-cma-{current_hash[:12]}.bak"


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    existing_stat = path.stat()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, stat.S_IMODE(existing_stat.st_mode))
        try:
            os.chown(tmp_path, existing_stat.st_uid, existing_stat.st_gid)
        except PermissionError:
            pass
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def apply_boot_config_cma_candidate(
    plan: dict[str, Any],
    *,
    candidate_id: str,
    boot_root: str | Path = "/",
    dry_run: bool = True,
    confirm_manual_approval: bool = False,
    backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and optionally apply one CMA boot config patch candidate."""

    if not dry_run:
        _require(
            confirm_manual_approval,
            reason="manual_confirmation_required",
            message="--apply requires --confirm-manual-approval",
        )

    candidate = _candidate_by_id(plan, candidate_id)
    target_path_text = _string_value(plan.get("target_boot_config_path"))
    _require(target_path_text is not None, reason="missing_target_path", message="missing target_boot_config_path")
    target_path = _resolve_boot_path(Path(boot_root), target_path_text)
    current_bytes = target_path.read_bytes()
    current_hash = _sha256_bytes(current_bytes)
    _, current_line, line_number, proposed_line, proposed_hash = _validate_plan_and_candidate(
        plan=plan,
        candidate=candidate,
        current_bytes=current_bytes,
    )
    proposed_bytes = _build_proposed_bytes(
        current_bytes=current_bytes,
        line_number=line_number,
        expected_line=current_line,
        proposed_line=proposed_line,
    )
    computed_proposed_hash = _sha256_bytes(proposed_bytes)
    _require(
        computed_proposed_hash == proposed_hash,
        reason="proposed_hash_mismatch",
        message="computed proposed boot config SHA256 does not match patch plan",
    )

    result = {
        "kind": "boot_config_cma_apply_result",
        "schema_version": 1,
        "status": "ready" if dry_run else "applied",
        "dry_run": dry_run,
        "candidate_id": candidate.get("candidate_id") or candidate_id,
        "cma_token": candidate.get("cma_token"),
        "target_boot_config_path": target_path_text,
        "resolved_boot_config_path": str(target_path),
        "current_file_sha256": current_hash,
        "expected_current_sha256": candidate.get("expected_current_sha256"),
        "proposed_file_sha256": proposed_hash,
        "final_file_sha256": current_hash if dry_run else computed_proposed_hash,
        "backup_path": None,
        "applies_boot_config": not dry_run,
    }

    if dry_run:
        return result

    backup_root = Path(backup_dir) if backup_dir is not None else None
    backup_path = _backup_path_for(target_path=target_path, backup_dir=backup_root, current_hash=current_hash)
    _require(not backup_path.exists(), reason="backup_already_exists", message=f"backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, backup_path)
    _atomic_write_bytes(target_path, proposed_bytes)

    final_hash = _sha256_bytes(target_path.read_bytes())
    _require(
        final_hash == computed_proposed_hash,
        reason="final_hash_mismatch",
        message="final boot config SHA256 does not match applied patch",
    )
    result["backup_path"] = str(backup_path)
    result["final_file_sha256"] = final_hash
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Boot config CMA patch plan JSON artifact")
    parser.add_argument("--candidate-id", required=True, help="Patch candidate_id to validate or apply")
    parser.add_argument("--boot-root", default="/", help="Root directory for target boot config resolution")
    parser.add_argument("--backup-dir", help="Directory for apply-mode backup files")
    parser.add_argument("--output", help="Optional path for the apply result JSON artifact")
    parser.add_argument("--apply", action="store_true", help="Write the selected candidate after all guards pass")
    parser.add_argument(
        "--confirm-manual-approval",
        action="store_true",
        help="Required with --apply to acknowledge out-of-band approval and recovery readiness",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = apply_boot_config_cma_candidate(
            _read_json_object(Path(args.plan)),
            candidate_id=args.candidate_id,
            boot_root=args.boot_root,
            dry_run=not args.apply,
            confirm_manual_approval=args.confirm_manual_approval,
            backup_dir=args.backup_dir,
        )
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0
    except (BootConfigCmaApplyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
