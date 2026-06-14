"""CMA experiment planning artifact contract tests."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


MiB = 1024 * 1024


def _selector_fixture() -> dict:
    return {
        "selection_id": "qwen3-vl-2b-instruct-auto",
        "selected_variant_id": None,
        "selected_reason": "no_usable_variant",
        "preflight": {"tegrastats": {"lfb": {"free_blocks": 69, "block_mb": 4}}},
        "memory_diagnostics_summary": {
            "tegrastats_lfb_free_blocks": 69,
            "tegrastats_lfb_block_mb": 4,
            "reserved_memory_node_count": 7,
            "reserved_memory_names": ["camdbg_carveout", "linux,cma", "vpr-carveout"],
            "boot_config_readable_paths": ["/boot/extlinux/extlinux.conf"],
            "boot_config_cma_tokens": ["cma=256M"],
            "boot_config_has_cma_token": True,
            "linux_cma_reserved_size_bytes": 256 * MiB,
        },
        "memory_blocker_assessment": {
            "status": "lfb_gate_not_met",
            "preboot_capacity_status": "linux_cma_reserved_below_required_lfb",
            "observed_lfb_block_mb": 4,
            "required_lfb_bytes_max": 150 * 4 * MiB,
            "boot_config_readable_paths": ["/boot/extlinux/extlinux.conf"],
            "boot_config_cma_tokens": ["cma=256M"],
            "boot_config_has_cma_token": True,
            "linux_cma_reserved_size_bytes": 256 * MiB,
            "linux_cma_reserved_deficit_bytes": (150 - 64) * 4 * MiB,
        },
        "candidates": [
            {
                "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                "min_lfb_blocks": 150,
                "block_reasons": [
                    "lfb_free_blocks 69 < required 150",
                    "preboot_linux_cma_reserved_bytes 268435456 < required_lfb_bytes 629145600",
                ],
                "usable": False,
            },
            {
                "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                "min_lfb_blocks": 100,
                "block_reasons": [
                    "lfb_free_blocks 69 < required 100",
                    "preboot_linux_cma_reserved_bytes 268435456 < required_lfb_bytes 419430400",
                ],
                "usable": False,
            },
        ],
    }


class CmaExperimentPlanContractsTest(unittest.TestCase):
    def test_plan_derives_read_only_cma_experiment_candidates_from_selector(self):
        from edge_vlm.cma_experiment_plan import build_cma_experiment_plan

        plan = build_cma_experiment_plan(
            _selector_fixture(),
            source_selector_json="outputs/jetson_inspect/qwen-selector.json",
        )

        self.assertEqual(plan["kind"], "cma_experiment_plan")
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["status"], "ready")
        self.assertFalse(plan["applies_boot_config"])
        self.assertTrue(plan["requires_manual_boot_config_change"])
        self.assertIn("does not edit boot configuration", plan["safety_note"])
        self.assertEqual(plan["source_selector_json"], "outputs/jetson_inspect/qwen-selector.json")
        self.assertIsNone(plan["selected_variant_id"])
        self.assertEqual(plan["selected_reason"], "no_usable_variant")
        self.assertEqual(plan["preboot_capacity_status"], "linux_cma_reserved_below_required_lfb")
        self.assertEqual(plan["boot_config_readable_paths"], ["/boot/extlinux/extlinux.conf"])
        self.assertEqual(plan["boot_config_cma_tokens"], ["cma=256M"])
        self.assertTrue(plan["boot_config_has_cma_token"])
        self.assertEqual(plan["linux_cma_reserved_size_bytes"], 256 * MiB)
        self.assertEqual(plan["required_lfb_bytes_max"], 150 * 4 * MiB)
        self.assertEqual(plan["linux_cma_reserved_deficit_bytes"], (150 - 64) * 4 * MiB)

        self.assertEqual(
            plan["candidate_requirements"],
            [
                {
                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                    "min_lfb_blocks": 150,
                    "required_lfb_bytes": 150 * 4 * MiB,
                    "linux_cma_reserved_size_bytes": 256 * MiB,
                    "linux_cma_reserved_deficit_bytes": (150 - 64) * 4 * MiB,
                    "block_reasons": [
                        "lfb_free_blocks 69 < required 150",
                        "preboot_linux_cma_reserved_bytes 268435456 < required_lfb_bytes 629145600",
                    ],
                },
                {
                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                    "min_lfb_blocks": 100,
                    "required_lfb_bytes": 100 * 4 * MiB,
                    "linux_cma_reserved_size_bytes": 256 * MiB,
                    "linux_cma_reserved_deficit_bytes": (100 - 64) * 4 * MiB,
                    "block_reasons": [
                        "lfb_free_blocks 69 < required 100",
                        "preboot_linux_cma_reserved_bytes 268435456 < required_lfb_bytes 419430400",
                    ],
                },
            ],
        )
        self.assertEqual(
            plan["experiment_candidates"],
            [
                {
                    "name": "qwen3-vl-2b-instruct-q8-smoke-minimum",
                    "source": "variant_required_lfb",
                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                    "cma_bytes": 100 * 4 * MiB,
                },
                {
                    "name": "qwen3-vl-2b-instruct-q4-smoke-minimum",
                    "source": "variant_required_lfb",
                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                    "cma_bytes": 150 * 4 * MiB,
                },
                {
                    "name": "max-required-lfb-rounded-64mib",
                    "source": "max_required_lfb_rounded_up_64mib",
                    "variant_id": None,
                    "cma_bytes": 640 * MiB,
                },
            ],
        )

    def test_plan_marks_insufficient_evidence_when_linux_cma_size_is_missing(self):
        from edge_vlm.cma_experiment_plan import build_cma_experiment_plan

        selector = _selector_fixture()
        selector["memory_diagnostics_summary"].pop("linux_cma_reserved_size_bytes")
        selector["memory_blocker_assessment"].pop("linux_cma_reserved_size_bytes")
        selector["memory_blocker_assessment"].pop("linux_cma_reserved_deficit_bytes")

        plan = build_cma_experiment_plan(selector)

        self.assertEqual(plan["status"], "insufficient_evidence")
        self.assertIn("linux_cma_reserved_size_bytes", plan["missing_evidence"])
        self.assertIsNone(plan["linux_cma_reserved_size_bytes"])
        self.assertIsNone(plan["candidate_requirements"][0]["linux_cma_reserved_deficit_bytes"])
        self.assertEqual(plan["candidate_requirements"][0]["required_lfb_bytes"], 150 * 4 * MiB)

    def test_cli_writes_plan_json(self):
        from edge_vlm.cma_experiment_plan import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selector_path = tmp_path / "selector.json"
            output_path = tmp_path / "selector.cma-experiment-plan.json"
            selector_path.write_text(json.dumps(_selector_fixture()) + "\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["--selector-json", str(selector_path), "--output", str(output_path)])
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")
        self.assertEqual(written["source_selector_json"], str(selector_path))
        self.assertEqual(written["experiment_candidates"][2]["cma_bytes"], 640 * MiB)


if __name__ == "__main__":
    unittest.main()
