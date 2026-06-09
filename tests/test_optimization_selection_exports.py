"""Comparison eligibility selection artifact contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OptimizationSelectionExportContractsTest(unittest.TestCase):
    def test_select_eligible_cli_writes_filtered_selection_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eligibility_input = tmp_path / "comparison.eligibility.json"
            selection_output = tmp_path / "ranking.selection.json"
            eligibility_input.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "run_id": "run-a",
                                "run_prefix": "sweep-a",
                                "variant_id": "variant-a",
                                "model": "model-a",
                                "comparison_group": "group-a",
                                "selection": {"id": "auto-a", "reason": "primary_usable"},
                                "eligibility": {
                                    "startup_precheck": {"passed": True, "reason": ""},
                                    "ranking_precheck": {"passed": True, "reason": ""},
                                    "promotion_precheck": {"passed": False, "reason": "quality_review_failed 10/20 text_case"},
                                },
                            },
                            {
                                "run_id": "run-b",
                                "run_prefix": "sweep-b",
                                "variant_id": "variant-b",
                                "model": "model-b",
                                "comparison_group": "group-b",
                                "selection": {"id": "", "reason": ""},
                                "eligibility": {
                                    "startup_precheck": {"passed": False, "reason": "artifact_phase downloaded_or_checked != cached"},
                                    "ranking_precheck": {"passed": False, "reason": "startup_precheck_failed"},
                                    "promotion_precheck": {"passed": False, "reason": "startup_precheck_failed"},
                                },
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.optimization",
                    "select-eligible",
                    "--input",
                    str(eligibility_input),
                    "--gate",
                    "ranking",
                    "--output",
                    str(selection_output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            artifact = json.loads(selection_output.read_text(encoding="utf-8"))

        self.assertEqual(stdout, {"gate": "ranking", "selected": 1, "output": str(selection_output)})
        self.assertEqual(artifact["gate"], "ranking")
        self.assertEqual(artifact["input_paths"], [str(eligibility_input)])
        self.assertEqual(artifact["selected_count"], 1)
        self.assertEqual(artifact["selected_ids"], [{"run_id": "run-a", "variant_id": "variant-a"}])
        self.assertEqual(len(artifact["selected"]), 1)
        self.assertEqual(artifact["selected"][0]["run_id"], "run-a")
        self.assertEqual(artifact["selected"][0]["eligibility"]["ranking_precheck"], {"passed": True, "reason": ""})

