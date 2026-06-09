"""Contracts for sweep-level quality review sidecars."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class SweepQualityReviewContractsTest(unittest.TestCase):
    def test_write_sweep_quality_reviews_writes_sidecars_and_updates_manifest(self):
        from edge_vlm.sweep_quality_review import write_sweep_quality_reviews

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_dir = tmp_path / "benchmarks"
            benchmark_dir.mkdir()
            manifest_path = tmp_path / "suite.manifest.json"
            benchmark_jsonl = benchmark_dir / "run-a.jsonl"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "mini",
                        "run_id": "run-a",
                        "prompt_case_id": "text_translation_zh_to_en_short",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 1.0,
                        "tokens": 64,
                        "tokens_per_sec": 64.0,
                        "output_excerpt": "Jetson Orin runs an edge vision language model with memory, bandwidth, power, and latency constraints.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "suite",
                            "variants": [
                                {
                                    "run_id": "run-a",
                                    "variant": {"id": "mini-smoke"},
                                    "paths": {"benchmark_jsonl": str(benchmark_jsonl)},
                                }
                            ],
                        },
                        "result": {"results": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = write_sweep_quality_reviews(
                manifest_path=manifest_path,
                policy_path="configs/benchmark/quality_review_policy.json",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result["reviews_written"], 1)
            self.assertEqual(result["failed_reviews"], 0)
            paths = manifest["plan"]["variants"][0]["paths"]
            self.assertTrue(Path(paths["quality_review_json"]).is_file())
            self.assertTrue(Path(paths["quality_review_markdown"]).is_file())
            report = json.loads(Path(paths["quality_review_json"]).read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])

    def test_sweep_quality_review_cli_can_fail_when_any_review_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_dir = tmp_path / "benchmarks"
            benchmark_dir.mkdir()
            manifest_path = tmp_path / "suite.manifest.json"
            benchmark_jsonl = benchmark_dir / "run-a.jsonl"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "mini",
                        "run_id": "run-a",
                        "prompt_case_id": "text_code_short",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 1.0,
                        "tokens": 64,
                        "tokens_per_sec": 64.0,
                        "output_excerpt": "This answer avoids the required code detail.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "suite",
                            "variants": [
                                {
                                    "run_id": "run-a",
                                    "variant": {"id": "mini-smoke"},
                                    "paths": {"benchmark_jsonl": str(benchmark_jsonl)},
                                }
                            ],
                        },
                        "result": {"results": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.sweep_quality_review",
                    "--manifest",
                    str(manifest_path),
                    "--policy",
                    "configs/benchmark/quality_review_policy.json",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)


if __name__ == "__main__":
    unittest.main()
