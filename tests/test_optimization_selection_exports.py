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

    def test_select_eligible_cli_can_filter_by_candidate_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eligibility_input = tmp_path / "comparison.eligibility.json"
            selection_output = tmp_path / "ranking.leq2b-vlm.selection.json"
            eligibility_input.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "run_id": "run-vlm-2b",
                                "run_prefix": "sweep-vlm-2b",
                                "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                "model": "qwen3-vl-2b-instruct-q4",
                                "comparison_group": "qwen3-vl-2b-instruct",
                                "selection": {"id": "qwen3-vl-2b-instruct-auto", "reason": "primary_usable"},
                                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
                                "eligibility": {
                                    "startup_precheck": {"passed": True, "reason": ""},
                                    "ranking_precheck": {"passed": True, "reason": ""},
                                    "promotion_precheck": {"passed": False, "reason": "missing_quality_review"},
                                },
                            },
                            {
                                "run_id": "run-vlm-4b",
                                "run_prefix": "sweep-vlm-4b",
                                "variant_id": "youtu-vl-4b-q4-thirdparty-smoke",
                                "model": "youtu-vl-4b-q4-thirdparty",
                                "comparison_group": "youtu-vl-4b-q4-thirdparty",
                                "selection": {"id": "", "reason": ""},
                                "candidate_scope": {"leq2b_candidate": False, "lane": "vlm"},
                                "eligibility": {
                                    "startup_precheck": {"passed": True, "reason": ""},
                                    "ranking_precheck": {"passed": True, "reason": ""},
                                    "promotion_precheck": {"passed": True, "reason": ""},
                                },
                            },
                            {
                                "run_id": "run-text-2b",
                                "run_prefix": "sweep-text-2b",
                                "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                                "model": "tencent-youtu-llm-2b-q8",
                                "comparison_group": "tencent-youtu-llm-2b-q8",
                                "selection": {"id": "", "reason": ""},
                                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
                                "eligibility": {
                                    "startup_precheck": {"passed": True, "reason": ""},
                                    "ranking_precheck": {"passed": True, "reason": ""},
                                    "promotion_precheck": {"passed": True, "reason": ""},
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
                    "--require-leq2b-candidate",
                    "--candidate-lane",
                    "vlm",
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
        self.assertEqual(
            artifact["filters"],
            {"require_leq2b_candidate": True, "candidate_lane": "vlm"},
        )
        self.assertEqual(artifact["selected_count"], 1)
        self.assertEqual(artifact["selected_ids"], [{"run_id": "run-vlm-2b", "variant_id": "qwen3-vl-2b-instruct-q4-smoke"}])
        self.assertEqual(artifact["selected"][0]["candidate_scope"], {"leq2b_candidate": True, "lane": "vlm"})
