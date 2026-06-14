"""Candidate bundle route export contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OptimizationRouteExportContractsTest(unittest.TestCase):
    def test_export_routes_promotes_q4_primary_and_marks_q8_fallback_within_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_input = tmp_path / "leq2b.candidate_bundle.json"
            route_output = tmp_path / "leq2b.routes.json"
            bundle_input.write_text(
                json.dumps(
                    {
                        "kind": "candidate_selection_bundle",
                        "input_paths": ["vlm/ranking.selection.json"],
                        "filters": {
                            "require_leq2b_candidate": True,
                            "candidate_lanes": ["vlm"],
                        },
                        "source_artifacts": [],
                        "gates": {
                            "ranking": {
                                "selected_count": 2,
                                "selected_ids": [
                                    {
                                        "run_id": "qwen-q8",
                                        "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                                    },
                                    {
                                        "run_id": "qwen-q4",
                                        "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                    },
                                ],
                                "lanes": {
                                    "vlm": {
                                        "selected_count": 2,
                                        "selected_ids": [
                                            {
                                                "run_id": "qwen-q8",
                                                "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                                            },
                                            {
                                                "run_id": "qwen-q4",
                                                "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                            },
                                        ],
                                    },
                                },
                            }
                        },
                        "selected": {
                            "ranking": [
                                {
                                    "run_id": "qwen-q8",
                                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                                    "model": "qwen3-vl-2b-instruct-q8",
                                    "comparison_group": "qwen3-vl-2b-instruct",
                                    "model_config": {"quantization": "Q8_0"},
                                    "candidate_lane": "vlm",
                                    "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
                                    "source_selection_path": "vlm/ranking.selection.json",
                                    "source_gate": "ranking",
                                },
                                {
                                    "run_id": "qwen-q4",
                                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                    "model": "qwen3-vl-2b-instruct-q4",
                                    "comparison_group": "qwen3-vl-2b-instruct",
                                    "model_config": {"quantization": "Q4_K_M"},
                                    "candidate_lane": "vlm",
                                    "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
                                    "source_selection_path": "vlm/ranking.selection.json",
                                    "source_gate": "ranking",
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
                    "ranking",
                    "--output",
                    str(route_output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads(route_output.read_text(encoding="utf-8"))

        route = artifact["routes"]["vlm"]
        self.assertEqual(route["selection_policy"]["name"], "q4_first_q8_fallback")
        self.assertEqual(
            [candidate["variant_id"] for candidate in route["candidates"]],
            ["qwen3-vl-2b-instruct-q4-smoke", "qwen3-vl-2b-instruct-q8-smoke"],
        )
        self.assertEqual(route["primary"]["variant_id"], "qwen3-vl-2b-instruct-q4-smoke")
        self.assertEqual(
            route["fallbacks"],
            [
                {
                    "run_id": "qwen-q8",
                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                    "model": "qwen3-vl-2b-instruct-q8",
                    "comparison_group": "qwen3-vl-2b-instruct",
                    "quantization": "Q8_0",
                    "fallback_reason": "q4_primary_q8_fallback",
                }
            ],
        )
        self.assertEqual(
            route["fallback_groups"],
            [
                {
                    "comparison_group": "qwen3-vl-2b-instruct",
                    "candidate_count": 2,
                    "primary": {
                        "run_id": "qwen-q4",
                        "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                        "model": "qwen3-vl-2b-instruct-q4",
                        "comparison_group": "qwen3-vl-2b-instruct",
                        "quantization": "Q4_K_M",
                    },
                    "fallbacks": [
                        {
                            "run_id": "qwen-q8",
                            "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                            "model": "qwen3-vl-2b-instruct-q8",
                            "comparison_group": "qwen3-vl-2b-instruct",
                            "quantization": "Q8_0",
                            "fallback_reason": "q4_primary_q8_fallback",
                        }
                    ],
                }
            ],
        )

    def test_export_routes_preserves_cross_group_and_non_q4_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle_input = tmp_path / "leq2b.candidate_bundle.json"
            route_output = tmp_path / "leq2b.routes.json"

            def row(run_id: str, variant_id: str, *, model: str, group: str, quantization: str) -> dict[str, object]:
                return {
                    "run_id": run_id,
                    "variant_id": variant_id,
                    "model": model,
                    "comparison_group": group,
                    "model_config": {"quantization": quantization},
                    "candidate_lane": "text",
                    "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
                    "source_selection_path": "text/ranking.selection.json",
                    "source_gate": "ranking",
                }

            selected = [
                row(
                    "youtu-q8",
                    "tencent-youtu-llm-2b-q8-text-smoke",
                    model="tencent-youtu-llm-2b-q8",
                    group="tencent-youtu-llm-2b",
                    quantization="Q8_0",
                ),
                row(
                    "hy-q8",
                    "tencent-hy-mt2-1p8b-q8-text-smoke",
                    model="tencent-hy-mt2-1p8b-q8",
                    group="tencent-hy-mt2-1p8b",
                    quantization="Q8_0",
                ),
                row(
                    "hy-q6",
                    "tencent-hy-mt2-1p8b-q6-text-smoke",
                    model="tencent-hy-mt2-1p8b-q6",
                    group="tencent-hy-mt2-1p8b",
                    quantization="Q6_K",
                ),
                row(
                    "hy-q4",
                    "tencent-hy-mt2-1p8b-q4-text-smoke",
                    model="tencent-hy-mt2-1p8b-q4",
                    group="tencent-hy-mt2-1p8b",
                    quantization="Q4_K_M",
                ),
            ]
            bundle_input.write_text(
                json.dumps(
                    {
                        "kind": "candidate_selection_bundle",
                        "input_paths": ["text/ranking.selection.json"],
                        "filters": {
                            "require_leq2b_candidate": True,
                            "candidate_lanes": ["text"],
                        },
                        "source_artifacts": [],
                        "gates": {
                            "ranking": {
                                "selected_count": len(selected),
                                "selected_ids": [
                                    {"run_id": str(row["run_id"]), "variant_id": str(row["variant_id"])}
                                    for row in selected
                                ],
                                "lanes": {
                                    "text": {
                                        "selected_count": len(selected),
                                        "selected_ids": [
                                            {"run_id": str(row["run_id"]), "variant_id": str(row["variant_id"])}
                                            for row in selected
                                        ],
                                    }
                                },
                            }
                        },
                        "selected": {"ranking": selected},
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
                    "ranking",
                    "--output",
                    str(route_output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads(route_output.read_text(encoding="utf-8"))

        route = artifact["routes"]["text"]
        self.assertEqual(
            [candidate["variant_id"] for candidate in route["candidates"]],
            [
                "tencent-youtu-llm-2b-q8-text-smoke",
                "tencent-hy-mt2-1p8b-q4-text-smoke",
                "tencent-hy-mt2-1p8b-q8-text-smoke",
                "tencent-hy-mt2-1p8b-q6-text-smoke",
            ],
        )
        self.assertEqual(route["primary"]["variant_id"], "tencent-youtu-llm-2b-q8-text-smoke")
        self.assertEqual(
            route["fallback_groups"],
            [
                {
                    "comparison_group": "tencent-hy-mt2-1p8b",
                    "candidate_count": 3,
                    "primary": {
                        "run_id": "hy-q4",
                        "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke",
                        "model": "tencent-hy-mt2-1p8b-q4",
                        "comparison_group": "tencent-hy-mt2-1p8b",
                        "quantization": "Q4_K_M",
                    },
                    "fallbacks": [
                        {
                            "run_id": "hy-q8",
                            "variant_id": "tencent-hy-mt2-1p8b-q8-text-smoke",
                            "model": "tencent-hy-mt2-1p8b-q8",
                            "comparison_group": "tencent-hy-mt2-1p8b",
                            "quantization": "Q8_0",
                            "fallback_reason": "q4_primary_q8_fallback",
                        }
                    ],
                }
            ],
        )

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
