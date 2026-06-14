"""Guarded boot config CMA patch apply contract tests."""

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _patch_plan_fixture(*, original_text: str) -> dict:
    current_append_line = "APPEND root=/dev/mmcblk0p1 rw quiet"
    proposed_append_line = "APPEND root=/dev/mmcblk0p1 rw quiet cma=600M"
    proposed_text = original_text.replace(current_append_line, proposed_append_line, 1)
    current_hash = _sha256(original_text)
    return {
        "kind": "boot_config_cma_patch_plan",
        "schema_version": 1,
        "status": "ready",
        "missing_evidence": [],
        "target_boot_config_path": "/boot/extlinux/extlinux.conf",
        "target_file_sha256": current_hash,
        "target_file_size_bytes": len(original_text.encode("utf-8")),
        "target_append_line_number": 1,
        "current_append_line": current_append_line,
        "current_cma_tokens": [],
        "patch_candidates": [
            {
                "candidate_id": "qwen3-vl-2b-instruct-q4-smoke-minimum",
                "name": "qwen3-vl-2b-instruct-q4-smoke-minimum",
                "cma_token": "cma=600M",
                "action": "append_cma_token",
                "line_number": 1,
                "expected_current_sha256": current_hash,
                "proposed_file_sha256": _sha256(proposed_text),
                "proposed_append_line": proposed_append_line,
            }
        ],
        "applies_boot_config": False,
        "requires_manual_approval": True,
    }


def _write_extlinux(root: Path, text: str) -> Path:
    extlinux = root / "boot" / "extlinux" / "extlinux.conf"
    extlinux.parent.mkdir(parents=True)
    extlinux.write_text(text, encoding="utf-8")
    return extlinux


class BootConfigCmaApplyContractsTest(unittest.TestCase):
    def test_dry_run_validates_candidate_without_writing_or_backup(self):
        from edge_vlm.boot_config_cma_apply import apply_boot_config_cma_candidate

        original_text = "APPEND root=/dev/mmcblk0p1 rw quiet\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "backups"
            extlinux = _write_extlinux(root, original_text)
            plan = _patch_plan_fixture(original_text=original_text)

            result = apply_boot_config_cma_candidate(
                plan,
                candidate_id="qwen3-vl-2b-instruct-q4-smoke-minimum",
                boot_root=root,
                dry_run=True,
                backup_dir=backup_dir,
            )

            self.assertEqual(extlinux.read_text(encoding="utf-8"), original_text)
            self.assertFalse(backup_dir.exists())

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["applies_boot_config"])
        self.assertEqual(result["candidate_id"], "qwen3-vl-2b-instruct-q4-smoke-minimum")
        self.assertEqual(result["cma_token"], "cma=600M")
        self.assertEqual(result["backup_path"], None)

    def test_apply_rejects_stale_current_hash_without_writing(self):
        from edge_vlm.boot_config_cma_apply import BootConfigCmaApplyError, apply_boot_config_cma_candidate

        original_text = "APPEND root=/dev/mmcblk0p1 rw quiet\n"
        stale_text = "APPEND root=/dev/mmcblk0p1 rw quiet debug\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extlinux = _write_extlinux(root, original_text)
            plan = _patch_plan_fixture(original_text=original_text)
            extlinux.write_text(stale_text, encoding="utf-8")

            with self.assertRaises(BootConfigCmaApplyError) as caught:
                apply_boot_config_cma_candidate(
                    plan,
                    candidate_id="qwen3-vl-2b-instruct-q4-smoke-minimum",
                    boot_root=root,
                    dry_run=False,
                    confirm_manual_approval=True,
                    backup_dir=root / "backups",
                )

            self.assertEqual(extlinux.read_text(encoding="utf-8"), stale_text)
            self.assertFalse((root / "backups").exists())

        self.assertEqual(caught.exception.reason, "current_hash_mismatch")

    def test_apply_requires_manual_confirmation_for_write(self):
        from edge_vlm.boot_config_cma_apply import BootConfigCmaApplyError, apply_boot_config_cma_candidate

        original_text = "APPEND root=/dev/mmcblk0p1 rw quiet\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extlinux = _write_extlinux(root, original_text)
            plan = _patch_plan_fixture(original_text=original_text)

            with self.assertRaises(BootConfigCmaApplyError) as caught:
                apply_boot_config_cma_candidate(
                    plan,
                    candidate_id="qwen3-vl-2b-instruct-q4-smoke-minimum",
                    boot_root=root,
                    dry_run=False,
                    backup_dir=root / "backups",
                )

            self.assertEqual(extlinux.read_text(encoding="utf-8"), original_text)
            self.assertFalse((root / "backups").exists())

        self.assertEqual(caught.exception.reason, "manual_confirmation_required")

    def test_apply_writes_backup_before_replacing_boot_config(self):
        from edge_vlm.boot_config_cma_apply import apply_boot_config_cma_candidate

        original_text = "APPEND root=/dev/mmcblk0p1 rw quiet\n"
        proposed_text = "APPEND root=/dev/mmcblk0p1 rw quiet cma=600M\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "backups"
            extlinux = _write_extlinux(root, original_text)
            plan = _patch_plan_fixture(original_text=original_text)

            result = apply_boot_config_cma_candidate(
                plan,
                candidate_id="qwen3-vl-2b-instruct-q4-smoke-minimum",
                boot_root=root,
                dry_run=False,
                confirm_manual_approval=True,
                backup_dir=backup_dir,
            )
            backup_path = Path(result["backup_path"])

            self.assertEqual(extlinux.read_text(encoding="utf-8"), proposed_text)
            self.assertEqual(backup_path.read_text(encoding="utf-8"), original_text)

        self.assertEqual(result["status"], "applied")
        self.assertFalse(result["dry_run"])
        self.assertTrue(result["applies_boot_config"])
        self.assertEqual(result["final_file_sha256"], _sha256(proposed_text))

    def test_cli_defaults_to_dry_run_and_writes_output_json(self):
        from edge_vlm.boot_config_cma_apply import main

        original_text = "APPEND root=/dev/mmcblk0p1 rw quiet\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extlinux = _write_extlinux(root, original_text)
            plan_path = root / "plan.json"
            output_path = root / "apply-result.json"
            plan_path.write_text(json.dumps(_patch_plan_fixture(original_text=original_text)) + "\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--plan",
                        str(plan_path),
                        "--candidate-id",
                        "qwen3-vl-2b-instruct-q4-smoke-minimum",
                        "--boot-root",
                        str(root),
                        "--output",
                        str(output_path),
                    ]
                )
            written = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(extlinux.read_text(encoding="utf-8"), original_text)

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")
        self.assertTrue(written["dry_run"])
        self.assertFalse(written["applies_boot_config"])


if __name__ == "__main__":
    unittest.main()
