"""Optimization report compare CLI contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OptimizationReportCliContractsTest(unittest.TestCase):
    def test_compare_cli_can_fail_on_promotion_precheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "promotion-cli"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "promotion-cli-qwen3-vl-2b-instruct-q4-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                "quality_terms_any": ["memory", "bandwidth"],
                            }
                        ),
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                "quality_terms_any": ["square", "background"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps(
                    {
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 2.1,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 151x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 5, "max_tokens": 64, "temperature": 0.0},
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                        "jetson": {"tegrastats_log": str(tegrastats_log)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "promotion-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "promotion-cli",
                            "variants": [
                                {
                                    "variant": {"id": "qwen3-vl-2b-instruct-q4-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "fake_stream_jsonl": str(fake_jsonl),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 151, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 5.0,
                                    "benchmark_returncode": 0,
                                    "fake_stream_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--promotion-precheck-stage",
                    "formal-repeat",
                    "--fail-on-promotion-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_fail_on_startup_precheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "startup-cli"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "startup-cli-tencent-youtu-llm-2b-q8-text-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            profile_summary_json = benchmark_dir / f"{run_id}.profile.json"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "tencent-youtu-llm-2b-q8",
                        "run_id": run_id,
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 64.0 / 24.621,
                        "tokens": 64,
                        "tokens_per_sec": 24.621,
                        "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 5, "max_tokens": 64, "temperature": 0.0},
                        "cases_written": 1,
                        "successful": 1,
                        "failed": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            profile_summary_json.write_text(
                json.dumps(
                    {
                        "phase_timings": {
                            "artifact_check_or_download": {
                                "available": True,
                                "duration_s": 351.027,
                                "source": "launcher",
                                "details": {"status": "downloaded_or_checked"},
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "startup-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "startup-cli",
                            "variants": [
                                {
                                    "variant": {"id": "tencent-youtu-llm-2b-q8-text-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "profile_summary_json": str(profile_summary_json),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 198, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 354.568,
                                    "benchmark_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--startup-require-cached-artifacts",
                    "--fail-on-startup-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_fail_on_promotion_precheck_when_startup_precheck_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "promotion-startup-cli"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "promotion-startup-cli-qwen3-vl-2b-instruct-q4-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            profile_summary_json = output_root / "profiles" / f"{run_id}.summary.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            profile_summary_json.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                "quality_terms_any": ["memory", "bandwidth"],
                            }
                        ),
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                "quality_terms_any": ["square", "background"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps(
                    {
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 2.1,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 151x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 5, "max_tokens": 64, "temperature": 0.0},
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                        "jetson": {"tegrastats_log": str(tegrastats_log)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            profile_summary_json.write_text(
                json.dumps(
                    {
                        "phase_timings": {
                            "artifact_check_or_download": {
                                "available": True,
                                "duration_s": 351.027,
                                "source": "launcher",
                                "details": {"status": "downloaded_or_checked"},
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "promotion-startup-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "promotion-startup-cli",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [
                                {
                                    "variant": {"id": "qwen3-vl-2b-instruct-q4-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "fake_stream_jsonl": str(fake_jsonl),
                                        "profile_summary_json": str(profile_summary_json),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 151, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 354.568,
                                    "benchmark_returncode": 0,
                                    "fake_stream_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--startup-require-cached-artifacts",
                    "--promotion-precheck-stage",
                    "formal-repeat",
                    "--promotion-require-startup-precheck",
                    "--fail-on-promotion-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_fail_on_promotion_precheck_when_quality_review_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "promotion-quality-cli"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "promotion-quality-cli-qwen3-vl-2b-instruct-q4-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                "quality_terms_any": ["memory", "bandwidth"],
                            }
                        ),
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                "quality_terms_any": ["square", "background"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps(
                    {
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 2.1,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 151x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 5, "max_tokens": 64, "temperature": 0.0},
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                        "jetson": {"tegrastats_log": str(tegrastats_log)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "promotion-quality-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "promotion-quality-cli",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [
                                {
                                    "variant": {"id": "qwen3-vl-2b-instruct-q4-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "fake_stream_jsonl": str(fake_jsonl),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 151, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 5.0,
                                    "benchmark_returncode": 0,
                                    "fake_stream_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--promotion-precheck-stage",
                    "formal-repeat",
                    "--promotion-require-quality-review",
                    "--fail-on-promotion-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_fail_on_ranking_precheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "ranking-cli"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "ranking-cli-qwen3-vl-2b-instruct-q8-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q8",
                                "run_id": run_id,
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                "quality_terms_any": ["memory", "bandwidth"],
                            }
                        ),
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q8",
                                "run_id": run_id,
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                "quality_terms_any": ["square", "background"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps(
                    {
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 2.1,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 121x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 1},
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                        "jetson": {"tegrastats_log": str(tegrastats_log)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "ranking-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "ranking-cli",
                            "variants": [
                                {
                                    "variant": {"id": "qwen3-vl-2b-instruct-q8-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "fake_stream_jsonl": str(fake_jsonl),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                                    "preflight_required_lfb_blocks": 100,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 121, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 5.0,
                                    "benchmark_returncode": 0,
                                    "fake_stream_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--ranking-min-lfb-blocks",
                    "150",
                    "--fail-on-ranking-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_fail_on_ranking_precheck_when_startup_precheck_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "ranking-startup-cli"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "ranking-startup-cli-qwen3-vl-2b-instruct-q8-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q8",
                                "run_id": run_id,
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                "quality_terms_any": ["memory", "bandwidth"],
                            }
                        ),
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q8",
                                "run_id": run_id,
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image shows two contrasting square shapes on a background.",
                                "quality_terms_any": ["square", "background"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps(
                    {
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 2.1,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 121x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 1},
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                        "jetson": {"tegrastats_log": str(tegrastats_log)},
                        "profile_summary": {
                            "phase_timings": {
                                "artifact_check_or_download": {
                                    "status": "downloaded_or_checked",
                                    "duration_s": 351.027,
                                }
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "ranking-startup-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "ranking-startup-cli",
                            "variants": [
                                {
                                    "variant": {"id": "qwen3-vl-2b-instruct-q8-smoke"},
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "fake_stream_jsonl": str(fake_jsonl),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 121, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 354.568,
                                    "benchmark_returncode": 0,
                                    "fake_stream_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--startup-require-cached-artifacts",
                    "--ranking-min-lfb-blocks",
                    "150",
                    "--ranking-require-startup-precheck",
                    "--fail-on-ranking-precheck",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_compare_cli_can_write_eligibility_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "eligibility-cli"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            eligibility_output = tmp_path / "comparison.eligibility.json"
            run_id = "eligibility-cli-tencent-youtu-llm-2b-q8-text-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            profile_summary_json = benchmark_dir / f"{run_id}.profile.json"
            quality_review_json = benchmark_dir / f"{run_id}.quality.json"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "tencent-youtu-llm-2b-q8",
                        "run_id": run_id,
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 64.0 / 24.621,
                        "tokens": 64,
                        "tokens_per_sec": 24.621,
                        "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "benchmark": {"trial_count": 5, "max_tokens": 64, "temperature": 0.0},
                        "cases_written": 1,
                        "successful": 1,
                        "failed": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            profile_summary_json.write_text(
                json.dumps(
                    {
                        "phase_timings": {
                            "artifact_check_or_download": {
                                "available": True,
                                "duration_s": 0.002,
                                "source": "launcher",
                                "details": {"status": "cached"},
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            quality_review_json.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "records": 1,
                        "passed_records": 1,
                        "failures": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "eligibility-cli.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "eligibility-cli",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [
                                {
                                    "variant": {
                                        "id": "tencent-youtu-llm-2b-q8-text-smoke",
                                        "config": "configs/models/tencent_youtu_llm_2b_q8.yaml",
                                    },
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "profile_summary_json": str(profile_summary_json),
                                        "quality_review_json": str(quality_review_json),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 198, "block_mb": 4},
                                        }
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 5.015,
                                    "benchmark_returncode": 0,
                                }
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
                    "compare",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                    "--eligibility-output",
                    str(eligibility_output),
                    "--startup-require-cached-artifacts",
                    "--ranking-min-lfb-blocks",
                    "150",
                    "--ranking-require-startup-precheck",
                    "--promotion-precheck-stage",
                    "formal-repeat",
                    "--promotion-require-startup-precheck",
                    "--promotion-require-quality-review",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src"},
            )
            stdout = json.loads(result.stdout)
            artifact = json.loads(eligibility_output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(stdout["output"], str(report))
        self.assertEqual(stdout["eligibility_output"], str(eligibility_output))
        self.assertEqual(artifact["eligible"]["startup"], [{"run_id": run_id, "variant_id": "tencent-youtu-llm-2b-q8-text-smoke"}])
        self.assertEqual(artifact["eligible"]["ranking"], artifact["eligible"]["startup"])
        self.assertEqual(artifact["eligible"]["promotion"], artifact["eligible"]["startup"])
        self.assertEqual(artifact["rows"][0]["candidate_scope"], {"leq2b_candidate": True, "lane": "text"})
        self.assertEqual(artifact["rows"][0]["eligibility"]["promotion_precheck"], {"passed": True, "reason": ""})


if __name__ == "__main__":
    unittest.main()
