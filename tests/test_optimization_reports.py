"""Optimization report and sweep comparison contract tests."""

import json
import tempfile
import unittest
from pathlib import Path


class OptimizationReportContractsTest(unittest.TestCase):

    def test_optimization_report_ranks_only_sanity_passing_runs(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fast_but_bad = tmp_path / "fast_bad.jsonl"
            steady_good = tmp_path / "steady_good.jsonl"
            report = tmp_path / "report.md"
            fast_but_bad.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-fast",
                                "run_id": "fast-bad",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 0.5,
                                "tokens": 64,
                                "tokens_per_sec": 128.0,
                                "output_excerpt": "ok ok ok ok ok ok ok ok ok ok ok ok",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-fast",
                                "run_id": "fast-bad",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 0.5,
                                "tokens": 64,
                                "tokens_per_sec": 128.0,
                                "output_excerpt": "",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            steady_good.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-steady",
                                "run_id": "steady-good",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "A concise answer that mentions memory, bandwidth, and thermal limits.",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-steady",
                                "run_id": "steady-good",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.5,
                                "tokens": 64,
                                "tokens_per_sec": 25.6,
                                "output_excerpt": "The image contains two high-contrast squares on a simple background.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[fast_but_bad, steady_good],
                output_path=report,
                min_output_chars=24,
                max_repeat_ratio=0.6,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([summary.run_id for summary in summaries], ["steady-good", "fast-bad"])
        self.assertTrue(summaries[0].guard_passed)
        self.assertFalse(summaries[1].guard_passed)
        self.assertIn("| 1 | steady-good | local-steady | yes |", report_text)
        self.assertIn("| - | fast-bad | local-fast | no |", report_text)
        self.assertIn("repetitive_output", report_text)
        self.assertIn("empty_output", report_text)

    def test_optimization_report_rejects_quality_term_miss(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fast_off_topic = tmp_path / "fast_off_topic.jsonl"
            steady_relevant = tmp_path / "steady_relevant.jsonl"
            report = tmp_path / "report.md"
            fast_off_topic.write_text(
                json.dumps(
                    {
                        "model": "local-fast",
                        "run_id": "fast-off-topic",
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 0.5,
                        "tokens": 64,
                        "tokens_per_sec": 128.0,
                        "output_excerpt": "A fluent answer that is long enough but avoids the required subject.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            steady_relevant.write_text(
                json.dumps(
                    {
                        "model": "local-steady",
                        "run_id": "steady-relevant",
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 2.0,
                        "tokens": 64,
                        "tokens_per_sec": 32.0,
                        "output_excerpt": "A useful answer that names memory pressure and bandwidth limits.",
                        "quality_terms_any": ["memory", "bandwidth"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[fast_off_topic, steady_relevant],
                output_path=report,
                min_output_chars=24,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([summary.run_id for summary in summaries], ["steady-relevant", "fast-off-topic"])
        self.assertTrue(summaries[0].guard_passed)
        self.assertFalse(summaries[1].guard_passed)
        self.assertIn("quality_terms_miss", report_text)

    def test_optimization_report_includes_fake_stream_guard_and_latency(self):
        from edge_vlm.optimization import build_optimization_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bench = tmp_path / "benchmarks" / "run-a.jsonl"
            fake_stream = tmp_path / "fake_stream" / "run-a.jsonl"
            report = tmp_path / "report.md"
            bench.parent.mkdir()
            fake_stream.parent.mkdir()
            bench.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "local-model",
                                "run_id": "run-a",
                                "prompt_case_id": "text_case",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 1.0,
                                "tokens": 64,
                                "tokens_per_sec": 64.0,
                                "output_excerpt": "A useful answer with enough detail to pass the sanity guard.",
                            }
                        ),
                        json.dumps(
                            {
                                "model": "local-model",
                                "run_id": "run-a",
                                "prompt_case_id": "image_case",
                                "input_type": "image",
                                "success": True,
                                "latency_s": 2.0,
                                "tokens": 64,
                                "tokens_per_sec": 32.0,
                                "output_excerpt": "The image shows two contrasting square shapes.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_stream.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "frame_id": "frame_001.png",
                        "success": True,
                        "latency_s": 3.25,
                        "output_excerpt": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summaries = build_optimization_report(
                input_paths=[bench],
                fake_stream_paths=[fake_stream],
                output_path=report,
                min_output_chars=24,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(len(summaries), 1)
        self.assertFalse(summaries[0].guard_passed)
        self.assertEqual(summaries[0].fake_stream_records, 1)
        self.assertEqual(summaries[0].fake_stream_successful, 1)
        self.assertEqual(summaries[0].fake_stream_avg_latency_s, 3.25)
        self.assertIn("Fake latency s", report_text)
        self.assertIn("fake_stream:frame_001.png:empty_output", report_text)

    def test_optimization_comparison_report_adds_manifest_context_and_deltas(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "gemma-compare"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
                lfb_blocks,
                power_w,
                required_lfb_blocks=150,
            ):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                profile_summary_json = benchmark_dir / f"{run_id}.summary.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": "gemma4-e2b-it-q4",
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": "gemma4-e2b-it-q4",
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "latency_s": fake_latency,
                            "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tegrastats_log.write_text(
                    "\n".join(
                        [
                            (
                                "05-31-2026 RAM 2000/7620MB (lfb 200x4MB) "
                                "GR3D_FREQ 80%@[1020] EMC_FREQ 82%@3199 cpu@50.0C gpu@51.0C tj@51.0C "
                                f"VDD_IN {int(power_w * 1000)}mW/{int(power_w * 1000)}mW"
                            ),
                            (
                                "05-31-2026 RAM 2100/7620MB (lfb 180x4MB) "
                                "GR3D_FREQ 90%@[1020] EMC_FREQ 84%@3199 cpu@52.0C gpu@54.5C tj@54.5C "
                                f"VDD_IN {int((power_w + 1) * 1000)}mW/{int((power_w + 1) * 1000)}mW"
                            ),
                        ]
                    )
                    + "\n",
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
                return {
                    "plan": {
                        "server_runtime": {
                            "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                            "image_id": "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
                            "llama_cpp_ref": "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774",
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        }
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": required_lfb_blocks,
                        "preflight_before_prepare": {
                            "meminfo_kb": {"MemAvailable": 5900000},
                            "buddyinfo": {"max_order_with_free_block": 10},
                            "tegrastats": {
                                "lfb": {"free_blocks": lfb_blocks - 40, "block_mb": 4},
                            },
                        },
                        "preflight_delta": {
                            "lfb_free_blocks_delta": 40,
                            "mem_available_kb_delta": 500000,
                            "buddyinfo_max_order_delta": 2,
                        },
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": lfb_blocks, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "gemma-baseline",
                "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                text_tps=10.0,
                image_tps=8.0,
                fake_latency=9.0,
                startup_s=5.0,
                lfb_blocks=180,
                power_w=10.0,
            )
            candidate = write_run(
                "gemma-directio",
                "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio",
                text_tps=12.0,
                image_tps=7.0,
                fake_latency=8.0,
                startup_s=6.0,
                lfb_blocks=190,
                power_w=12.0,
            )
            manifest = output_root / "gemma-compare.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "gemma-compare",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [baseline["plan"], candidate["plan"]],
                        },
                        "result": {"results": [baseline["result"], candidate["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["gemma-q4-baseline-gpu12-b512-u512-kvq8"],
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual([row.variant_id for row in rows], [
            "gemma-q4-baseline-gpu12-b512-u512-kvq8",
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio",
        ])
        self.assertEqual(rows[0].server_image, "ghcr.io/4everwz/jetson-llama-cpp:test")
        self.assertEqual(rows[0].server_image_id, "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844")
        self.assertEqual(rows[0].llama_cpp_ref, "b4c0549a49be9e6dc59ac9d0a5bc21dbda910774")
        self.assertEqual(rows[0].preflight_before_prepare_lfb, "140x4MB")
        self.assertEqual(rows[0].preflight_required_lfb_blocks, 150)
        self.assertEqual(rows[0].preflight_prepare_lfb_delta, 40)
        self.assertEqual(rows[0].preflight_prepare_mem_available_mb_delta, 500000 / 1024.0)
        self.assertEqual(rows[0].preflight_prepare_buddyinfo_max_order_delta, 2)
        self.assertEqual(rows[1].delta_text_tokens_per_s_pct, 20.0)
        self.assertEqual(rows[1].delta_image_tokens_per_s_pct, -12.5)
        self.assertAlmostEqual(rows[1].delta_fake_stream_latency_pct, -11.111111, places=5)
        self.assertEqual(rows[0].prepare_context_summary, "max_clocks, drop_caches")
        self.assertIn(
            "| Model | Variant | Selection | Run prefix | Runtime | Prepare ctx | Preflight lfb | Required lfb | Prepare lfb delta | Prepare avail MB delta |",
            report_text,
        )
        self.assertIn("Avg GR3D %", report_text)
        self.assertIn("Avg EMC %", report_text)
        self.assertIn("Bottlenecks", report_text)
        self.assertIn("gpu_compute", report_text)
        self.assertEqual(rows[0].avg_gr3d_util_pct, 85.0)
        self.assertEqual(rows[0].avg_emc_util_pct, 83.0)
        self.assertEqual(rows[0].min_lfb_free_blocks, 180)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:test / 52a8ad644e41 / b4c0549a49be", report_text)
        self.assertIn("| gemma4-e2b-it-q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` |  | gemma-directio | ghcr.io/4everwz/jetson-llama-cpp:test / 52a8ad644e41 / b4c0549a49be | max_clocks, drop_caches | 190x4MB | 150 | +40 | +488.281 | 1 | yes | 2/2 | 1/1 | 6.000 | 12.000 | 7.000 | 5.333 | 9.143 | 8.000 | 54.500 | 12.500 | 85.000 | 83.000 | 180 | gpu_compute, emc_memory_bandwidth | +20.00% | -12.50% | +20.00% | -11.11% |", report_text)
        self.assertIn("Baseline rows use `0.00%` deltas", report_text)

    def test_optimization_comparison_report_can_surface_quality_review_sidecars(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "quality-compare"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "quality-compare-qwen3-vl-2b-instruct-q4-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            quality_review_json = output_root / f"{run_id}.quality.json"
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "qwen3-vl-2b-instruct-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_code_short",
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
                        "latency_s": 1.9,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
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
                        "cases_written": 2,
                        "successful": 2,
                        "failed": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            quality_review_json.write_text(
                json.dumps(
                    {
                        "run_ids": [run_id],
                        "models": ["qwen3-vl-2b-instruct-q4"],
                        "records": 2,
                        "passed_records": 1,
                        "failed_records": 1,
                        "passed": False,
                        "failures": [
                            {
                                "run_id": run_id,
                                "case_id": "text_code_short",
                                "failures": ["missing_all:def "],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "quality-compare.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "quality-compare",
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
                                            "lfb": {"free_blocks": 160, "block_mb": 4},
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

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertFalse(rows[0].quality_review_passed)
        self.assertEqual(rows[0].quality_review_records, 2)
        self.assertEqual(rows[0].quality_review_passed_records, 1)
        self.assertEqual(rows[0].quality_review_failed_case_ids, ("text_code_short",))
        self.assertIn("Quality review", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` |  | quality-compare |  |  | 160x4MB | 150 | no (1/2; text_code_short) |", report_text)

    def test_optimization_comparison_report_can_surface_artifact_phase_timings(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "artifact-compare"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "artifact-compare-tencent-youtu-llm-2b-q8-text-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            profile_summary_json = output_root / "profiles" / f"{run_id}.summary.json"
            profile_summary_json.parent.mkdir(parents=True)
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "tencent-youtu-llm-2b-q8",
                        "run_id": run_id,
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 1.861,
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
            manifest = output_root / "artifact-compare.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "artifact-compare",
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

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(rows[0].artifact_phase_status, "cached")
        self.assertEqual(rows[0].artifact_phase_duration_s, 0.002)
        self.assertIn("Artifact phase", report_text)
        self.assertIn("Artifact s", report_text)
        self.assertIn("| tencent-youtu-llm-2b-q8 | `tencent-youtu-llm-2b-q8-text-smoke` |  | artifact-compare |  | cached | 0.002 |  | 198x4MB | 150 |  |  | 5 | yes | 1/1 |  | 5.015 | 24.621 |  | 1.861 |  |  |  |  |  |  |  |  | +0.00% |  | +0.00% |  |", report_text)

    def test_optimization_comparison_report_can_mark_cached_startup_precheck(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "startup-precheck"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(run_prefix, variant_id, *, model, artifact_status, artifact_s, startup_s):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                profile_summary_json = benchmark_dir / f"{run_id}.profile.json"
                benchmark_jsonl.write_text(
                    json.dumps(
                        {
                            "model": model,
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
                                    "duration_s": artifact_s,
                                    "source": "launcher",
                                    "details": {"status": artifact_status},
                                }
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {"id": variant_id},
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "profile_summary_json": str(profile_summary_json),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": 150,
                        "preflight": {"tegrastats": {"lfb": {"free_blocks": 198, "block_mb": 4}}},
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                    },
                }

            cached = write_run(
                "startup-cached",
                "tencent-youtu-llm-2b-q8-text-smoke",
                model="tencent-youtu-llm-2b-q8",
                artifact_status="cached",
                artifact_s=0.002,
                startup_s=5.015,
            )
            first_download = write_run(
                "startup-first",
                "tencent-youtu-llm-2b-q8-text-smoke-first",
                model="tencent-youtu-llm-2b-q8",
                artifact_status="downloaded_or_checked",
                artifact_s=351.027,
                startup_s=354.568,
            )
            manifest = output_root / "startup-precheck.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "startup-precheck",
                            "variants": [cached["plan"], first_download["plan"]],
                        },
                        "result": {"results": [cached["result"], first_download["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                startup_require_cached_artifacts=True,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].startup_precheck_passed)
        self.assertEqual(rows[0].startup_precheck_reason, "")
        self.assertFalse(rows[1].startup_precheck_passed)
        self.assertEqual(rows[1].startup_precheck_reason, "artifact_phase downloaded_or_checked != cached")
        self.assertIn("Startup precheck", report_text)
        self.assertIn("| tencent-youtu-llm-2b-q8 | `tencent-youtu-llm-2b-q8-text-smoke` |  | startup-cached |  | cached | 0.002 |  | 198x4MB | 150 | yes |", report_text)
        self.assertIn("no (artifact_phase downloaded_or_checked != cached)", report_text)

    def test_optimization_comparison_report_can_use_shared_comparison_group(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-compare"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                comparison_group,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
            ):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                profile_summary_json = benchmark_dir / f"{run_id}.summary.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "latency_s": fake_latency,
                            "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tegrastats_log.write_text(
                    "\n".join(
                        [
                            "05-31-2026 RAM 2000/7620MB (lfb 121x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW",
                            "05-31-2026 RAM 2100/7620MB (lfb 118x4MB) GR3D_FREQ 96%@[1020] cpu@52.0C gpu@54.5C tj@54.5C VDD_IN 22000mW/22000mW",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                benchmark_manifest.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "benchmark": {"trial_count": 3},
                            "cases_written": 2,
                            "successful": 2,
                            "failed": 0,
                            "jetson": {"tegrastats_log": str(tegrastats_log)},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "comparison_group": comparison_group,
                        },
                        "server_runtime": {
                            "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": 121, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=34.865,
                image_tps=31.958,
                fake_latency=1.827,
                startup_s=8.023,
            )
            candidate = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=31.346,
                image_tps=29.393,
                fake_latency=2.144,
                startup_s=5.017,
            )
            manifest = output_root / "qwen-compare.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {"run_prefix": "qwen-compare", "variants": [baseline["plan"], candidate["plan"]]},
                        "result": {"results": [baseline["result"], candidate["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
            )

        baseline_row, candidate_row = rows
        self.assertEqual(baseline_row.variant_id, "qwen3-vl-2b-instruct-q4-smoke")
        self.assertEqual(candidate_row.variant_id, "qwen3-vl-2b-instruct-q8-smoke")
        self.assertAlmostEqual(
            candidate_row.delta_text_tokens_per_s_pct,
            ((31.346 - 34.865) / 34.865) * 100.0,
            places=5,
        )
        self.assertAlmostEqual(
            candidate_row.delta_image_tokens_per_s_pct,
            ((29.393 - 31.958) / 31.958) * 100.0,
            places=5,
        )
        self.assertAlmostEqual(
            candidate_row.delta_startup_pct,
            ((5.017 - 8.023) / 8.023) * 100.0,
            places=5,
        )
        self.assertAlmostEqual(
            candidate_row.delta_fake_stream_latency_pct,
            ((2.144 - 1.827) / 1.827) * 100.0,
            places=5,
        )

    def test_optimization_comparison_report_surfaces_auto_selected_lane_context(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-auto"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_prefix = "qwen-auto"
            variant_id = "qwen3-vl-2b-instruct-q4-smoke"
            run_id = f"{run_prefix}-{variant_id}"
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
                                "latency_s": 64.0 / 34.865,
                                "tokens": 64,
                                "tokens_per_sec": 34.865,
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
                                "latency_s": 64.0 / 31.958,
                                "tokens": 64,
                                "tokens_per_sec": 31.958,
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
                        "latency_s": 1.827,
                        "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "\n".join(
                    [
                        "05-31-2026 RAM 2000/7620MB (lfb 121x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW",
                        "05-31-2026 RAM 2100/7620MB (lfb 118x4MB) GR3D_FREQ 96%@[1020] cpu@52.0C gpu@54.5C tj@54.5C VDD_IN 22000mW/22000mW",
                    ]
                )
                + "\n",
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
            manifest = output_root / "qwen-auto.manifest.json"
            eligibility_output = tmp_path / "qwen-auto.eligibility.json"
            selection_context = {
                "selection_id": "qwen3-vl-2b-instruct-auto",
                "comparison_group": "qwen3-vl-2b-instruct",
                "selected_variant_id": variant_id,
                "selected_reason": "primary_usable",
                "primary_variant_id": variant_id,
                "fallback_variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                "memory_diagnostics_summary": {
                    "debugfs_statuses": {"dma_buf_bufinfo": "unreadable"},
                    "debugfs_dma_buf_total_bytes": None,
                    "debugfs_nvmap_clients_total_bytes": None,
                    "tegrastats_lfb_free_blocks": 69,
                    "cma_free_kb": 221296,
                },
                "sudo_memory_diagnostics_summary": {
                    "debugfs_statuses": {
                        "nvmap_iovmm_clients": "readable",
                        "nvmap_iovmm_allocations": "readable",
                        "dma_buf_bufinfo": "readable",
                    },
                    "debugfs_dma_buf_total_bytes": 0,
                    "debugfs_nvmap_clients_total_bytes": 0,
                    "debugfs_nvmap_allocations_total_bytes": 0,
                    "tegrastats_lfb_free_blocks": 69,
                    "cma_free_kb": 221296,
                    "boot_cmdline_cma_token": None,
                    "boot_config_readable_paths": ["/boot/extlinux/extlinux.conf"],
                    "boot_config_cma_tokens": ["cma=256M"],
                    "boot_config_has_cma_token": True,
                    "boot_config_append_line_count": 1,
                    "reserved_memory_node_count": 7,
                    "reserved_memory_names": [
                        "camdbg_carveout",
                        "linux,cma",
                        "vpr-carveout",
                    ],
                    "linux_cma_reserved_size_bytes": 256 * 1024 * 1024,
                },
                "sudo_memory_blocker_assessment": {
                    "status": "lfb_gate_not_met",
                    "diagnostics_source": "sudo",
                    "lfb_free_block_deficit_max": 81,
                    "boot_cmdline_cma_token": None,
                    "boot_config_readable_paths": ["/boot/extlinux/extlinux.conf"],
                    "boot_config_cma_tokens": ["cma=256M"],
                    "boot_config_has_cma_token": True,
                    "boot_config_append_line_count": 1,
                    "reserved_memory_node_count": 7,
                    "reserved_memory_names": [
                        "camdbg_carveout",
                        "linux,cma",
                        "vpr-carveout",
                    ],
                    "linux_cma_reserved_memory_present": True,
                    "linux_cma_reserved_size_bytes": 256 * 1024 * 1024,
                    "preboot_capacity_status": "linux_cma_reserved_below_required_lfb",
                    "linux_cma_reserved_deficit_bytes": (150 - 64) * 4 * 1024 * 1024,
                    "signals": [
                        "lfb_below_required",
                        "debugfs_tracked_allocations_zero",
                        "cma_free_below_required_lfb_bytes",
                        "linux_cma_reserved_below_required_lfb_bytes",
                    ],
                    "selection_implication": "no_candidate_meets_lfb_gate",
                },
                "candidates": [
                    {
                        "variant_id": variant_id,
                        "usable": True,
                        "chosen": True,
                        "block_reasons": [],
                        "artifact_manifest": {"status": "ok", "failed_count": 0},
                    },
                    {
                        "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                        "usable": False,
                        "chosen": False,
                        "block_reasons": ["primary_usable_not_needed"],
                    },
                ],
            }
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": run_prefix,
                            "selection_contexts": [selection_context],
                            "variants": [
                                {
                                    "variant": {
                                        "id": variant_id,
                                        "comparison_group": "qwen3-vl-2b-instruct",
                                    },
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
                                    "variant_id": variant_id,
                                    "preflight": {
                                        "tegrastats": {
                                            "lfb": {"free_blocks": 121, "block_mb": 4},
                                        }
                                    },
                                    "preflight_required_lfb_blocks": 100,
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

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=[variant_id],
                eligibility_output_path=eligibility_output,
            )
            report_text = report.read_text(encoding="utf-8")
            eligibility = json.loads(eligibility_output.read_text(encoding="utf-8"))

        self.assertEqual(rows[0].variant_id, variant_id)
        self.assertEqual(rows[0].selection_id, "qwen3-vl-2b-instruct-auto")
        self.assertEqual(rows[0].selection_reason, "primary_usable")
        self.assertEqual(rows[0].selection_context["candidates"], selection_context["candidates"])
        self.assertEqual(eligibility["rows"][0]["selection_context"]["candidates"], selection_context["candidates"])
        self.assertEqual(rows[0].selection_memory_diagnostics["source"], "sudo")
        self.assertEqual(rows[0].selection_memory_diagnostics["debugfs_dma_buf_total_bytes"], 0)
        self.assertEqual(rows[0].selection_memory_diagnostics["debugfs_nvmap_clients_total_bytes"], 0)
        self.assertEqual(rows[0].selection_memory_diagnostics["tegrastats_lfb_free_blocks"], 69)
        self.assertEqual(
            rows[0].selection_memory_diagnostics["debugfs_statuses"]["dma_buf_bufinfo"],
            "readable",
        )
        self.assertEqual(rows[0].selection_memory_assessment["status"], "lfb_gate_not_met")
        self.assertEqual(rows[0].selection_memory_assessment["lfb_free_block_deficit_max"], 81)
        self.assertEqual(rows[0].selection_memory_diagnostics["boot_config_cma_tokens"], ["cma=256M"])
        self.assertTrue(rows[0].selection_memory_assessment["boot_config_has_cma_token"])
        self.assertEqual(rows[0].selection_memory_diagnostics["reserved_memory_node_count"], 7)
        self.assertIn("linux,cma", rows[0].selection_memory_diagnostics["reserved_memory_names"])
        self.assertTrue(rows[0].selection_memory_assessment["linux_cma_reserved_memory_present"])
        self.assertEqual(
            rows[0].selection_memory_assessment["linux_cma_reserved_size_bytes"],
            256 * 1024 * 1024,
        )
        self.assertEqual(
            rows[0].selection_memory_assessment["preboot_capacity_status"],
            "linux_cma_reserved_below_required_lfb",
        )
        self.assertEqual(
            rows[0].selection_memory_assessment["linux_cma_reserved_deficit_bytes"],
            (150 - 64) * 4 * 1024 * 1024,
        )
        self.assertEqual(
            eligibility["rows"][0]["selection_memory_diagnostics"]["debugfs_nvmap_clients_total_bytes"],
            0,
        )
        self.assertEqual(
            eligibility["rows"][0]["selection_memory_assessment"]["signals"][0],
            "lfb_below_required",
        )
        self.assertEqual(rows[0].preflight_required_lfb_blocks, 100)
        self.assertIn("Selection", report_text)
        self.assertIn("Selector memory", report_text)
        self.assertIn(
            "sudo lfb=69 cma=216.1MiB debugfs=dma_buf:readable,nvmap:readable dma=0B nvmap=0B cmdline_cma=none boot_config_cma=cma=256M reserved_memory=7:linux,cma linux_cma=256.0MiB linux_cma_deficit=344.0MiB assessment=lfb_gate_not_met deficit=81 blocks",
            report_text,
        )
        self.assertIn("Required lfb", report_text)
        self.assertIn("qwen3-vl-2b-instruct-auto (primary_usable)", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` | qwen3-vl-2b-instruct-auto (primary_usable) | qwen-auto |  | sudo lfb=69", report_text)

    def test_optimization_comparison_report_attaches_blocked_selector_context_to_candidate_rows(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-blocked"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            eligibility_output = tmp_path / "qwen-blocked.eligibility.json"
            run_prefix = "qwen-blocked"
            variant_id = "qwen3-vl-2b-instruct-q4-smoke"
            run_id = f"{run_prefix}-{variant_id}"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            fake_jsonl = fake_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "model": "qwen3-vl-2b-instruct-q4",
                        "run_id": run_id,
                        "prompt_case_id": "text_case",
                        "input_type": "text",
                        "success": True,
                        "latency_s": 2.0,
                        "tokens": 32,
                        "tokens_per_sec": 16.0,
                        "output_excerpt": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fake_jsonl.write_text(
                json.dumps({"frame_id": "frame_001.png", "success": True, "latency_s": 1.0})
                + "\n",
                encoding="utf-8",
            )
            benchmark_manifest.write_text(
                json.dumps({"run_id": run_id, "benchmark": {"trial_count": 1}, "cases_written": 1, "successful": 1, "failed": 0})
                + "\n",
                encoding="utf-8",
            )
            selection_context = {
                "selection_id": "qwen3-vl-2b-instruct-auto",
                "comparison_group": "qwen3-vl-2b-instruct",
                "selected_variant_id": None,
                "selected_reason": "no_usable_variant",
                "primary_variant_id": variant_id,
                "fallback_variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                "sudo_memory_diagnostics_summary": {
                    "debugfs_statuses": {"dma_buf_bufinfo": "readable"},
                    "debugfs_dma_buf_total_bytes": 0,
                    "debugfs_nvmap_clients_total_bytes": 0,
                    "tegrastats_lfb_free_blocks": 69,
                    "cma_free_kb": 221296,
                },
                "sudo_memory_blocker_assessment": {
                    "status": "lfb_gate_not_met",
                    "diagnostics_source": "sudo",
                    "lfb_free_block_deficit_max": 81,
                    "signals": ["lfb_below_required"],
                },
                "candidates": [
                    {
                        "variant_id": variant_id,
                        "usable": False,
                        "chosen": False,
                        "block_reasons": ["lfb_free_blocks 69 < required 150"],
                    },
                    {
                        "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                        "usable": False,
                        "chosen": False,
                        "block_reasons": ["lfb_free_blocks 69 < required 100"],
                    },
                ],
            }
            manifest = output_root / "qwen-blocked.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": run_prefix,
                            "selection_contexts": [selection_context],
                            "variants": [
                                {
                                    "variant": {
                                        "id": variant_id,
                                        "comparison_group": "qwen3-vl-2b-instruct",
                                    },
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
                                    "variant_id": variant_id,
                                    "preflight": {"tegrastats": {"lfb": {"free_blocks": 69, "block_mb": 4}}},
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight_passed": False,
                                }
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=[variant_id],
                eligibility_output_path=eligibility_output,
            )
            report_text = report.read_text(encoding="utf-8")
            eligibility = json.loads(eligibility_output.read_text(encoding="utf-8"))

        self.assertEqual(rows[0].selection_id, "qwen3-vl-2b-instruct-auto")
        self.assertEqual(rows[0].selection_reason, "no_usable_variant")
        self.assertEqual(rows[0].selection_context["candidates"], selection_context["candidates"])
        self.assertEqual(rows[0].selection_memory_diagnostics["source"], "sudo")
        self.assertEqual(rows[0].selection_memory_assessment["status"], "lfb_gate_not_met")
        self.assertIn("qwen3-vl-2b-instruct-auto (no_usable_variant)", report_text)
        self.assertIn("sudo lfb=69 cma=216.1MiB", report_text)
        self.assertIn("assessment=lfb_gate_not_met deficit=81 blocks", report_text)
        self.assertEqual(
            eligibility["rows"][0]["selection_memory_diagnostics"]["debugfs_dma_buf_total_bytes"],
            0,
        )
        self.assertEqual(
            eligibility["rows"][0]["selection_memory_assessment"]["lfb_free_block_deficit_max"],
            81,
        )

    def test_optimization_comparison_report_can_mark_ranking_precheck_failures(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-ranking"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                comparison_group,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
                required_lfb_blocks,
            ):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                profile_summary_json = benchmark_dir / f"{run_id}.summary.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "latency_s": fake_latency,
                            "output_excerpt": "The frame shows a simple scene with contrasting square shapes.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tegrastats_log.write_text(
                    "\n".join(
                        [
                            "05-31-2026 RAM 2000/7620MB (lfb 121x4MB) GR3D_FREQ 95%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 21000mW/21000mW",
                            "05-31-2026 RAM 2100/7620MB (lfb 118x4MB) GR3D_FREQ 96%@[1020] cpu@52.0C gpu@54.5C tj@54.5C VDD_IN 22000mW/22000mW",
                        ]
                    )
                    + "\n",
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
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "comparison_group": comparison_group,
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": required_lfb_blocks,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": 121, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=34.865,
                image_tps=31.958,
                fake_latency=1.827,
                startup_s=8.023,
                required_lfb_blocks=150,
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=31.346,
                image_tps=29.393,
                fake_latency=2.144,
                startup_s=5.017,
                required_lfb_blocks=100,
            )
            manifest = output_root / "qwen-ranking.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {"run_prefix": "qwen-ranking", "variants": [baseline["plan"], fallback["plan"]]},
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                ranking_min_lfb_blocks=150,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].ranking_precheck_passed)
        self.assertEqual(rows[0].ranking_precheck_reason, "")
        self.assertFalse(rows[1].ranking_precheck_passed)
        self.assertEqual(rows[1].ranking_precheck_reason, "required_lfb 100 < ranking 150")
        self.assertIn("Ranking precheck", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` |  | qwen-q4 |  |  | 121x4MB | 150 | yes |", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q8 | `qwen3-vl-2b-instruct-q8-smoke` |  | qwen-q8 |  |  | 121x4MB | 100 | no (required_lfb 100 < ranking 150) |", report_text)

    def test_optimization_comparison_report_can_require_cached_startup_for_ranking(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-ranking-startup"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                comparison_group,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
                required_lfb_blocks,
                artifact_phase_status,
            ):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                profile_summary_json = benchmark_dir / f"{run_id}.summary.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "latency_s": fake_latency,
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
                profile_summary_json.write_text(
                    json.dumps(
                        {
                            "phase_timings": {
                                "artifact_check_or_download": {
                                    "available": True,
                                    "duration_s": 0.002 if artifact_phase_status == "cached" else 351.027,
                                    "source": "launcher",
                                    "details": {"status": artifact_phase_status},
                                }
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "comparison_group": comparison_group,
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                            "profile_summary_json": str(profile_summary_json),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": required_lfb_blocks,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": 121, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=34.865,
                image_tps=31.958,
                fake_latency=1.827,
                startup_s=8.023,
                required_lfb_blocks=150,
                artifact_phase_status="cached",
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=31.346,
                image_tps=29.393,
                fake_latency=2.144,
                startup_s=354.568,
                required_lfb_blocks=150,
                artifact_phase_status="downloaded_or_checked",
            )
            manifest = output_root / "qwen-ranking-startup.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {"run_prefix": "qwen-ranking-startup", "variants": [baseline["plan"], fallback["plan"]]},
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                startup_require_cached_artifacts=True,
                ranking_min_lfb_blocks=150,
                ranking_require_startup_precheck=True,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].startup_precheck_passed)
        self.assertTrue(rows[0].ranking_precheck_passed)
        self.assertFalse(rows[1].startup_precheck_passed)
        self.assertFalse(rows[1].ranking_precheck_passed)
        self.assertEqual(rows[1].ranking_precheck_reason, "startup_precheck_failed")
        self.assertIn("Startup precheck", report_text)
        self.assertIn("Ranking precheck", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` |  | qwen-q4 |  | cached | 0.002 |  | 121x4MB | 150 | yes | yes |", report_text)
        self.assertIn(
            "| qwen3-vl-2b-instruct-q8 | `qwen3-vl-2b-instruct-q8-smoke` |  | qwen-q8 |  | downloaded_or_checked | 351.027 |  | 121x4MB | 150 | no (artifact_phase downloaded_or_checked != cached) | no (startup_precheck_failed) |",
            report_text,
        )

    def test_optimization_comparison_report_can_backfill_required_lfb_from_plan(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-plan-required"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                comparison_group,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
            ):
                run_id = f"{run_prefix}-{variant_id}"
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
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "latency_s": fake_latency,
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
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "comparison_group": comparison_group,
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": 121, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=34.865,
                image_tps=31.958,
                fake_latency=1.827,
                startup_s=8.023,
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=31.346,
                image_tps=29.393,
                fake_latency=2.144,
                startup_s=5.017,
            )
            manifest = output_root / "qwen-plan-required.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "qwen-plan-required",
                            "min_lfb_blocks": 150,
                            "variant_min_lfb_blocks": {
                                "qwen3-vl-2b-instruct-q8-smoke": 100,
                            },
                            "variants": [baseline["plan"], fallback["plan"]],
                        },
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                ranking_min_lfb_blocks=150,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(rows[0].preflight_required_lfb_blocks, 150)
        self.assertEqual(rows[1].preflight_required_lfb_blocks, 100)
        self.assertTrue(rows[0].ranking_precheck_passed)
        self.assertFalse(rows[1].ranking_precheck_passed)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` |  | qwen-q4 |  |  | 121x4MB | 150 | yes |", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q8 | `qwen3-vl-2b-instruct-q8-smoke` |  | qwen-q8 |  |  | 121x4MB | 100 | no (required_lfb 100 < ranking 150) |", report_text)

    def test_optimization_comparison_report_can_mark_promotion_precheck_failures(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-promotion"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                comparison_group,
                text_tps,
                image_tps,
                fake_latency,
                startup_s,
                required_lfb_blocks,
                trial_count,
                max_tokens,
                temperature,
                fake_stream_success,
            ):
                run_id = f"{run_prefix}-{variant_id}"
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
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / text_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": text_tps,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / image_tps,
                                    "tokens": 64,
                                    "tokens_per_sec": image_tps,
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
                            "success": fake_stream_success,
                            "latency_s": fake_latency,
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
                            "benchmark": {
                                "trial_count": trial_count,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            },
                            "cases_written": 2,
                            "successful": 2,
                            "failed": 0,
                            "jetson": {"tegrastats_log": str(tegrastats_log)},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "comparison_group": comparison_group,
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": required_lfb_blocks,
                        "preflight": {
                            "tegrastats": {
                                "lfb": {"free_blocks": 151, "block_mb": 4},
                            }
                        },
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=34.865,
                image_tps=31.958,
                fake_latency=1.827,
                startup_s=8.023,
                required_lfb_blocks=150,
                trial_count=5,
                max_tokens=64,
                temperature=0.0,
                fake_stream_success=True,
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                comparison_group="qwen3-vl-2b-instruct",
                text_tps=31.346,
                image_tps=29.393,
                fake_latency=2.144,
                startup_s=5.017,
                required_lfb_blocks=150,
                trial_count=4,
                max_tokens=32,
                temperature=0.2,
                fake_stream_success=False,
            )
            manifest = output_root / "qwen-promotion.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "qwen-promotion",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [baseline["plan"], fallback["plan"]],
                        },
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                ranking_min_lfb_blocks=150,
                promotion_precheck_stage="formal-repeat",
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].promotion_precheck_passed)
        self.assertEqual(rows[0].promotion_precheck_reason, "")
        self.assertFalse(rows[1].promotion_precheck_passed)
        self.assertIn("trial_count 4 < required 5", rows[1].promotion_precheck_reason)
        self.assertIn("max_tokens 32 < required 64", rows[1].promotion_precheck_reason)
        self.assertIn("temperature 0.2 != required 0", rows[1].promotion_precheck_reason)
        self.assertIn("fake_stream_success 0/1 < 1/1", rows[1].promotion_precheck_reason)
        self.assertIn("Promotion precheck", report_text)
        self.assertIn("max_clocks, drop_caches", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` |  | qwen-q4 |  | max_clocks, drop_caches | 151x4MB | 150 | yes | yes |", report_text)
        self.assertIn("no (trial_count 4 < required 5; max_tokens 32 < required 64; temperature 0.2 != required 0; guard_failed; fake_stream_success 0/1 < 1/1)", report_text)

    def test_optimization_comparison_report_can_require_quality_review_for_promotion(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-promotion-quality"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(
                run_prefix,
                variant_id,
                *,
                model,
                quality_review_passed,
                quality_review_passed_records,
                quality_review_records,
                quality_review_failed_case_ids,
            ):
                run_id = f"{run_prefix}-{variant_id}"
                benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
                fake_jsonl = fake_dir / f"{run_id}.jsonl"
                benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
                quality_review_json = output_root / f"{run_id}.quality.json"
                tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
                tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
                benchmark_jsonl.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / 34.0,
                                    "tokens": 64,
                                    "tokens_per_sec": 34.0,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / 30.0,
                                    "tokens": 64,
                                    "tokens_per_sec": 30.0,
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
                            "latency_s": 2.0,
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
                quality_review_json.write_text(
                    json.dumps(
                        {
                            "run_ids": [run_id],
                            "models": [model],
                            "records": quality_review_records,
                            "passed_records": quality_review_passed_records,
                            "failed_records": quality_review_records - quality_review_passed_records,
                            "passed": quality_review_passed,
                            "failures": [
                                {
                                    "run_id": run_id,
                                    "case_id": case_id,
                                    "failures": ["missing_all:placeholder"],
                                }
                                for case_id in quality_review_failed_case_ids
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {"id": variant_id, "comparison_group": "qwen3-vl-2b-instruct"},
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                            "quality_review_json": str(quality_review_json),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": 150,
                        "preflight": {"tegrastats": {"lfb": {"free_blocks": 151, "block_mb": 4}}},
                        "preflight_passed": True,
                        "server_startup_seconds": 5.0,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                quality_review_passed=True,
                quality_review_passed_records=30,
                quality_review_records=30,
                quality_review_failed_case_ids=(),
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                quality_review_passed=False,
                quality_review_passed_records=20,
                quality_review_records=30,
                quality_review_failed_case_ids=("text_cn_short", "text_code_short"),
            )
            manifest = output_root / "qwen-promotion-quality.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "qwen-promotion-quality",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [baseline["plan"], fallback["plan"]],
                        },
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                ranking_min_lfb_blocks=150,
                promotion_precheck_stage="formal-repeat",
                promotion_require_quality_review=True,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].promotion_precheck_passed)
        self.assertFalse(rows[1].promotion_precheck_passed)
        self.assertIn("quality_review_failed 20/30", rows[1].promotion_precheck_reason)
        self.assertIn("text_cn_short,text_code_short", rows[1].promotion_precheck_reason)
        self.assertIn("Promotion precheck", report_text)
        self.assertIn("Quality review", report_text)
        self.assertIn("quality_review_failed 20/30", report_text)

    def test_optimization_comparison_report_can_require_cached_startup_for_promotion(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "qwen-promotion-startup"
            benchmark_dir = output_root / "benchmarks"
            fake_dir = output_root / "fake_stream"
            benchmark_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"

            def write_run(run_prefix, variant_id, *, model, artifact_status, artifact_s, startup_s):
                run_id = f"{run_prefix}-{variant_id}"
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
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "text_case",
                                    "input_type": "text",
                                    "success": True,
                                    "latency_s": 64.0 / 34.0,
                                    "tokens": 64,
                                    "tokens_per_sec": 34.0,
                                    "output_excerpt": "A useful answer that mentions memory and bandwidth limits.",
                                    "quality_terms_any": ["memory", "bandwidth"],
                                }
                            ),
                            json.dumps(
                                {
                                    "model": model,
                                    "run_id": run_id,
                                    "prompt_case_id": "image_case",
                                    "input_type": "image",
                                    "success": True,
                                    "latency_s": 64.0 / 30.0,
                                    "tokens": 64,
                                    "tokens_per_sec": 30.0,
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
                            "latency_s": 2.0,
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
                                    "duration_s": artifact_s,
                                    "source": "launcher",
                                    "details": {"status": artifact_status},
                                }
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {
                    "plan": {
                        "variant": {"id": variant_id, "comparison_group": "qwen3-vl-2b-instruct"},
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "fake_stream_jsonl": str(fake_jsonl),
                            "profile_summary_json": str(profile_summary_json),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": 150,
                        "preflight": {"tegrastats": {"lfb": {"free_blocks": 151, "block_mb": 4}}},
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                        "fake_stream_returncode": 0,
                    },
                }

            baseline = write_run(
                "qwen-q4",
                "qwen3-vl-2b-instruct-q4-smoke",
                model="qwen3-vl-2b-instruct-q4",
                artifact_status="cached",
                artifact_s=0.002,
                startup_s=5.015,
            )
            fallback = write_run(
                "qwen-q8",
                "qwen3-vl-2b-instruct-q8-smoke",
                model="qwen3-vl-2b-instruct-q8",
                artifact_status="downloaded_or_checked",
                artifact_s=351.027,
                startup_s=354.568,
            )
            manifest = output_root / "qwen-promotion-startup.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "qwen-promotion-startup",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [baseline["plan"], fallback["plan"]],
                        },
                        "result": {"results": [baseline["result"], fallback["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                baseline_variant_ids=["qwen3-vl-2b-instruct-q4-smoke"],
                startup_require_cached_artifacts=True,
                promotion_precheck_stage="formal-repeat",
                promotion_require_startup_precheck=True,
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertTrue(rows[0].promotion_precheck_passed)
        self.assertEqual(rows[0].promotion_precheck_reason, "")
        self.assertFalse(rows[1].promotion_precheck_passed)
        self.assertIn("startup_precheck_failed", rows[1].promotion_precheck_reason)
        self.assertIn("Promotion precheck", report_text)
        self.assertIn("Startup precheck", report_text)
        self.assertIn("no (startup_precheck_failed)", report_text)

    def test_optimization_comparison_report_skips_fake_stream_requirement_for_text_only_rows(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "tencent-text-promotion"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            run_id = "tencent-text-promotion-tencent-hy-mt2-1p8b-q4-text-smoke"
            benchmark_jsonl = benchmark_dir / f"{run_id}.jsonl"
            benchmark_manifest = benchmark_dir / f"{run_id}.manifest.json"
            quality_review_json = output_root / f"{run_id}.quality.json"
            tegrastats_log = tmp_path / "outputs" / "tegrastats" / f"{run_id}.log"
            tegrastats_log.parent.mkdir(parents=True, exist_ok=True)
            benchmark_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "tencent-hy-mt2-1p8b-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_case_a",
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
                                "model": "tencent-hy-mt2-1p8b-q4",
                                "run_id": run_id,
                                "prompt_case_id": "text_case_b",
                                "input_type": "text",
                                "success": True,
                                "latency_s": 2.2,
                                "tokens": 64,
                                "tokens_per_sec": 29.0,
                                "output_excerpt": "Another useful answer that stays on topic and mentions thermal limits.",
                                "quality_terms_any": ["thermal", "limits"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tegrastats_log.write_text(
                "05-31-2026 RAM 2000/7620MB (lfb 151x4MB) GR3D_FREQ 45%@[1020] cpu@50.0C gpu@51.0C tj@51.0C VDD_IN 18000mW/18000mW\n",
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
            quality_review_json.write_text(
                json.dumps(
                    {
                        "run_ids": [run_id],
                        "models": ["tencent-hy-mt2-1p8b-q4"],
                        "records": 20,
                        "passed_records": 20,
                        "failed_records": 0,
                        "passed": True,
                        "failures": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = output_root / "tencent-text-promotion.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "tencent-text-promotion",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [
                                {
                                    "run_id": run_id,
                                    "supports_images": False,
                                    "variant": {
                                        "id": "tencent-hy-mt2-1p8b-q4-text-smoke",
                                        "comparison_group": "tencent-hy-mt2-1p8b",
                                    },
                                    "paths": {
                                        "benchmark_jsonl": str(benchmark_jsonl),
                                        "manifest_json": str(benchmark_manifest),
                                        "quality_review_json": str(quality_review_json),
                                    },
                                }
                            ],
                        },
                        "result": {
                            "results": [
                                {
                                    "run_id": run_id,
                                    "variant_id": "tencent-hy-mt2-1p8b-q4-text-smoke",
                                    "model": "tencent-hy-mt2-1p8b-q4",
                                    "preflight_required_lfb_blocks": 150,
                                    "preflight": {
                                        "tegrastats": {"lfb": {"free_blocks": 151, "block_mb": 4}}
                                    },
                                    "preflight_passed": True,
                                    "server_startup_seconds": 4.0,
                                    "benchmark_returncode": 0,
                                }
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                ranking_min_lfb_blocks=150,
                promotion_precheck_stage="formal-repeat",
                promotion_require_quality_review=True,
            )

        self.assertTrue(rows[0].promotion_precheck_passed)
        self.assertEqual(rows[0].promotion_precheck_reason, "")

    def test_optimization_comparison_report_can_write_eligibility_artifact(self):
        from edge_vlm.optimization import build_sweep_comparison_report

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "outputs" / "optimization_sweeps" / "eligibility-artifact"
            benchmark_dir = output_root / "benchmarks"
            benchmark_dir.mkdir(parents=True)
            report = tmp_path / "comparison.md"
            eligibility_output = tmp_path / "comparison.eligibility.json"

            def write_run(run_prefix, variant_id, *, artifact_status, artifact_s, startup_s):
                run_id = f"{run_prefix}-{variant_id}"
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
                                    "duration_s": artifact_s,
                                    "source": "launcher",
                                    "details": {"status": artifact_status},
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
                return {
                    "plan": {
                        "variant": {
                            "id": variant_id,
                            "config": "configs/models/tencent_youtu_llm_2b_q8.yaml",
                        },
                        "paths": {
                            "benchmark_jsonl": str(benchmark_jsonl),
                            "manifest_json": str(benchmark_manifest),
                            "profile_summary_json": str(profile_summary_json),
                            "quality_review_json": str(quality_review_json),
                        },
                    },
                    "result": {
                        "run_id": run_id,
                        "variant_id": variant_id,
                        "preflight_required_lfb_blocks": 150,
                        "preflight": {"tegrastats": {"lfb": {"free_blocks": 198, "block_mb": 4}}},
                        "preflight_passed": True,
                        "server_startup_seconds": startup_s,
                        "benchmark_returncode": 0,
                    },
                }

            cached = write_run(
                "eligibility-cached",
                "tencent-youtu-llm-2b-q8-text-smoke",
                artifact_status="cached",
                artifact_s=0.002,
                startup_s=5.015,
            )
            first_download = write_run(
                "eligibility-first",
                "tencent-youtu-llm-2b-q8-text-smoke-first",
                artifact_status="downloaded_or_checked",
                artifact_s=351.027,
                startup_s=354.568,
            )
            manifest = output_root / "eligibility-artifact.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": "eligibility-artifact",
                            "prepare_context": {
                                "max_clocks_enabled": True,
                                "drop_caches_before_variant": True,
                            },
                            "variants": [cached["plan"], first_download["plan"]],
                        },
                        "result": {"results": [cached["result"], first_download["result"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_sweep_comparison_report(
                manifest_paths=[manifest],
                output_path=report,
                eligibility_output_path=eligibility_output,
                baseline_variant_ids=["tencent-youtu-llm-2b-q8-text-smoke"],
                startup_require_cached_artifacts=True,
                ranking_min_lfb_blocks=150,
                ranking_require_startup_precheck=True,
                promotion_precheck_stage="formal-repeat",
                promotion_require_quality_review=True,
                promotion_require_startup_precheck=True,
            )
            artifact = json.loads(eligibility_output.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(artifact["runs"], 2)
        self.assertEqual(artifact["markdown_output"], str(report))
        self.assertEqual(artifact["eligibility_output"], str(eligibility_output))
        self.assertTrue(artifact["startup_require_cached_artifacts"])
        self.assertEqual(artifact["ranking_min_lfb_blocks"], 150)
        self.assertEqual(artifact["promotion_precheck_stage"], "formal-repeat")
        self.assertEqual(
            artifact["eligible"]["startup"],
            [
                {
                    "run_id": "eligibility-cached-tencent-youtu-llm-2b-q8-text-smoke",
                    "variant_id": "tencent-youtu-llm-2b-q8-text-smoke",
                }
            ],
        )
        self.assertEqual(artifact["eligible"]["ranking"], artifact["eligible"]["startup"])
        self.assertEqual(artifact["eligible"]["promotion"], artifact["eligible"]["startup"])
        self.assertEqual(artifact["rows"][0]["artifact_phase"]["status"], "cached")
        self.assertEqual(artifact["rows"][0]["model_config"]["family"], "Tencent Youtu-LLM")
        self.assertEqual(artifact["rows"][0]["model_config"]["quantization"], "Q8_0")
        self.assertEqual(artifact["rows"][0]["candidate_scope"], {"leq2b_candidate": True, "lane": "text"})
        self.assertEqual(artifact["rows"][0]["eligibility"]["startup_precheck"], {"passed": True, "reason": ""})
        self.assertEqual(artifact["rows"][0]["eligibility"]["ranking_precheck"], {"passed": True, "reason": ""})
        self.assertEqual(artifact["rows"][0]["eligibility"]["promotion_precheck"], {"passed": True, "reason": ""})
        self.assertEqual(artifact["rows"][1]["artifact_phase"]["status"], "downloaded_or_checked")
        self.assertEqual(
            artifact["rows"][1]["eligibility"]["startup_precheck"],
            {"passed": False, "reason": "artifact_phase downloaded_or_checked != cached"},
        )
        self.assertEqual(
            artifact["rows"][1]["eligibility"]["ranking_precheck"],
            {"passed": False, "reason": "startup_precheck_failed"},
        )
        self.assertEqual(
            artifact["rows"][1]["eligibility"]["promotion_precheck"],
            {"passed": False, "reason": "startup_precheck_failed"},
        )

if __name__ == "__main__":
    unittest.main()
