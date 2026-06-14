"""Candidate bundle route export contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OptimizationRouteExportContractsTest(unittest.TestCase):
    def test_export_routes_cli_writes_lane_grouped_route_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_input = tmp_path / "leq2b.candidate_bundle.json"
            route_output = tmp_path / "leq2b.routes.json"
            bundle_input.write_text(
                json.dumps(
                    {
                        "kind": "candidate_selection_bundle",
                        "input_paths": ["vlm/ranking.selection.json", "text/promotion.selection.json"],
                        "filters": {
                            "require_leq2b_candidate": True,
                            "candidate_lanes": ["text", "vlm"],
                        },
                        "source_artifacts": [],
                        "gates": {
                            "promotion": {
                                "selected_count": 2,
                                "selected_ids": [
                                    {"run_id": "run-text-primary", "variant_id": "tencent-youtu-llm-2b-q8-text-smoke"},
                                    {"run_id": "run-text-backup", "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke"},
                                ],
                                "lanes": {
                                    "text": {
                                        "selected_count": 2,
                                        "selected_ids": [
                                            {
                                                "run_id": "run-text-primary",
                                                "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                                            },
                                            {
                                                "run_id": "run-text-backup",
                                                "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke",
                                            },
                                        ],
                                    },
                                    "vlm": {
                                        "selected_count": 0,
                                        "selected_ids": [],
                                    },
                                },
                            }
                        },
                        "selected": {
                            "promotion": [
                                {
                                    "run_id": "run-text-primary",
                                    "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                                    "model": "tencent-youtu-llm-2b-q8",
                                    "candidate_lane": "text",
                                    "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
                                    "source_selection_path": "text/promotion.selection.json",
                                    "source_gate": "promotion",
                                },
                                {
                                    "run_id": "run-text-backup",
                                    "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke",
                                    "model": "tencent-hy-mt2-1p8b-q4",
                                    "candidate_lane": "text",
                                    "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
                                    "source_selection_path": "text/promotion.selection.json",
                                    "source_gate": "promotion",
                                },
                            ]
                        },
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
                    "export-routes",
                    "--input",
                    str(bundle_input),
                    "--gate",
                    "promotion",
                    "--output",
                    str(route_output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            artifact = json.loads(route_output.read_text(encoding="utf-8"))

        self.assertEqual(stdout, {"gate": "promotion", "lanes": 2, "output": str(route_output)})
        self.assertEqual(artifact["kind"], "candidate_route_export")
        self.assertEqual(artifact["source_bundle"], str(bundle_input))
        self.assertEqual(artifact["gate"], "promotion")
        self.assertEqual(artifact["filters"], {"candidate_lanes": ["text", "vlm"]})
        self.assertEqual(artifact["routes"]["text"]["candidate_count"], 2)
        self.assertEqual(
            artifact["routes"]["text"]["selected_ids"],
            [
                {"run_id": "run-text-primary", "variant_id": "tencent-youtu-llm-2b-q8-text-smoke"},
                {"run_id": "run-text-backup", "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke"},
            ],
        )
        self.assertEqual(
            artifact["routes"]["text"]["primary"]["variant_id"],
            "tencent-youtu-llm-2b-q8-text-smoke",
        )
        self.assertEqual(
            artifact["routes"]["text"]["candidates"][1]["variant_id"],
            "tencent-hy-mt2-1p8b-q4-text-smoke",
        )
        self.assertEqual(artifact["routes"]["vlm"]["candidate_count"], 0)
        self.assertEqual(artifact["routes"]["vlm"]["selected_ids"], [])
        self.assertIsNone(artifact["routes"]["vlm"]["primary"])
        self.assertEqual(artifact["routes"]["vlm"]["candidates"], [])
