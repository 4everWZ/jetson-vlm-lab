"""Boot config CMA patch planning contract tests."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


MiB = 1024 * 1024


def _cma_plan_fixture() -> dict:
    return {
        "kind": "cma_experiment_plan",
        "schema_version": 1,
        "status": "ready",
        "source_selector_json": "outputs/unit/qwen-selector.json",
        "boot_config_readable_paths": ["/boot/extlinux/extlinux.conf"],
        "boot_config_cma_tokens": [],
        "boot_config_has_cma_token": False,
        "applies_boot_config": False,
        "requires_manual_boot_config_change": True,
        "experiment_candidates": [
            {
                "name": "qwen3-vl-2b-instruct-q8-smoke-minimum",
                "source": "variant_required_lfb",
                "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                "cma_bytes": 400 * MiB,
            },
            {
                "name": "qwen3-vl-2b-instruct-q4-smoke-minimum",
                "source": "variant_required_lfb",
                "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                "cma_bytes": 600 * MiB,
            },
            {
                "name": "max-required-lfb-rounded-64mib",
                "source": "max_required_lfb_rounded_up_64mib",
                "variant_id": None,
                "cma_bytes": 640 * MiB,
            },
        ],
    }


class BootConfigCmaPlanContractsTest(unittest.TestCase):
    def test_plan_builds_read_only_extlinux_patch_candidates(self):
        from edge_vlm.boot_config_cma_plan import build_boot_config_cma_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            boot_root = tmp_path
            extlinux = boot_root / "boot" / "extlinux" / "extlinux.conf"
            extlinux.parent.mkdir(parents=True)
            extlinux.write_text(
                "\n".join(
                    [
                        "TIMEOUT 30",
                        "DEFAULT primary",
                        "LABEL primary",
                        "      APPEND root=/dev/mmcblk0p1 rw quiet",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            plan = build_boot_config_cma_plan(_cma_plan_fixture(), boot_root=boot_root)
            after_text = extlinux.read_text(encoding="utf-8")

        self.assertEqual(plan["kind"], "boot_config_cma_patch_plan")
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["status"], "ready")
        self.assertFalse(plan["applies_boot_config"])
        self.assertTrue(plan["requires_manual_approval"])
        self.assertIn("does not edit boot configuration", plan["safety_note"])
        self.assertEqual(plan["source_cma_plan"], None)
        self.assertEqual(plan["target_boot_config_path"], "/boot/extlinux/extlinux.conf")
        self.assertEqual(plan["resolved_boot_config_path"], str(extlinux))
        self.assertEqual(plan["target_append_line_number"], 4)
        self.assertEqual(plan["current_cma_tokens"], [])
        self.assertEqual(plan["current_append_line"], "      APPEND root=/dev/mmcblk0p1 rw quiet")
        self.assertEqual(after_text, "TIMEOUT 30\nDEFAULT primary\nLABEL primary\n      APPEND root=/dev/mmcblk0p1 rw quiet\n")
        self.assertEqual(
            [
                (
                    candidate["name"],
                    candidate["cma_token"],
                    candidate["action"],
                    candidate["proposed_append_line"],
                )
                for candidate in plan["patch_candidates"]
            ],
            [
                (
                    "qwen3-vl-2b-instruct-q8-smoke-minimum",
                    "cma=400M",
                    "append_cma_token",
                    "      APPEND root=/dev/mmcblk0p1 rw quiet cma=400M",
                ),
                (
                    "qwen3-vl-2b-instruct-q4-smoke-minimum",
                    "cma=600M",
                    "append_cma_token",
                    "      APPEND root=/dev/mmcblk0p1 rw quiet cma=600M",
                ),
                (
                    "max-required-lfb-rounded-64mib",
                    "cma=640M",
                    "append_cma_token",
                    "      APPEND root=/dev/mmcblk0p1 rw quiet cma=640M",
                ),
            ],
        )
        self.assertIn("-      APPEND root=/dev/mmcblk0p1 rw quiet", plan["patch_candidates"][2]["unified_diff"])
        self.assertIn("+      APPEND root=/dev/mmcblk0p1 rw quiet cma=640M", plan["patch_candidates"][2]["unified_diff"])

    def test_plan_replaces_existing_cma_token_for_review(self):
        from edge_vlm.boot_config_cma_plan import build_boot_config_cma_plan

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            extlinux = tmp_path / "boot" / "extlinux" / "extlinux.conf"
            extlinux.parent.mkdir(parents=True)
            extlinux.write_text("APPEND root=/dev/mmcblk0p1 rw cma=256M quiet\n", encoding="utf-8")

            plan = build_boot_config_cma_plan(_cma_plan_fixture(), boot_root=tmp_path)

        self.assertEqual(plan["current_cma_tokens"], ["cma=256M"])
        self.assertEqual(plan["patch_candidates"][0]["action"], "replace_cma_token")
        self.assertEqual(
            plan["patch_candidates"][0]["proposed_append_line"],
            "APPEND root=/dev/mmcblk0p1 rw cma=400M quiet",
        )

    def test_cli_writes_patch_plan_json(self):
        from edge_vlm.boot_config_cma_plan import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            extlinux = tmp_path / "boot" / "extlinux" / "extlinux.conf"
            extlinux.parent.mkdir(parents=True)
            extlinux.write_text("APPEND root=/dev/mmcblk0p1 rw quiet\n", encoding="utf-8")
            cma_plan_path = tmp_path / "selector.cma-experiment-plan.json"
            output_path = tmp_path / "selector.boot-config-cma-plan.json"
            cma_plan_path.write_text(json.dumps(_cma_plan_fixture()) + "\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--cma-plan",
                        str(cma_plan_path),
                        "--boot-root",
                        str(tmp_path),
                        "--output",
                        str(output_path),
                    ]
                )
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")
        self.assertEqual(written["source_cma_plan"], str(cma_plan_path))
        self.assertEqual(written["patch_candidates"][2]["cma_token"], "cma=640M")


if __name__ == "__main__":
    unittest.main()
