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
                        "plan": {"run_prefix": "gemma-compare", "variants": [baseline["plan"], candidate["plan"]]},
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
        self.assertIn(
            "| Model | Variant | Selection | Run prefix | Runtime | Preflight lfb | Required lfb | Prepare lfb delta | Prepare avail MB delta |",
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
        self.assertIn("| gemma4-e2b-it-q4 | `gemma-q4-baseline-gpu12-b512-u512-kvq8-directio` |  | gemma-directio | ghcr.io/4everwz/jetson-llama-cpp:test / 52a8ad644e41 / b4c0549a49be | 190x4MB | 150 | +40 | +488.281 | 1 | yes | 2/2 | 1/1 | 6.000 | 12.000 | 7.000 | 5.333 | 9.143 | 8.000 | 54.500 | 12.500 | 85.000 | 83.000 | 180 | gpu_compute, emc_memory_bandwidth | +20.00% | -12.50% | +20.00% | -11.11% |", report_text)
        self.assertIn("Baseline rows use `0.00%` deltas", report_text)

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
            manifest.write_text(
                json.dumps(
                    {
                        "plan": {
                            "run_prefix": run_prefix,
                            "selection_contexts": [
                                {
                                    "selection_id": "qwen3-vl-2b-instruct-auto",
                                    "comparison_group": "qwen3-vl-2b-instruct",
                                    "selected_variant_id": variant_id,
                                    "selected_reason": "primary_usable",
                                }
                            ],
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
            )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(rows[0].variant_id, variant_id)
        self.assertEqual(rows[0].selection_id, "qwen3-vl-2b-instruct-auto")
        self.assertEqual(rows[0].selection_reason, "primary_usable")
        self.assertEqual(rows[0].preflight_required_lfb_blocks, 100)
        self.assertIn("Selection", report_text)
        self.assertIn("Required lfb", report_text)
        self.assertIn("qwen3-vl-2b-instruct-auto (primary_usable)", report_text)
        self.assertIn("| qwen3-vl-2b-instruct-q4 | `qwen3-vl-2b-instruct-q4-smoke` | qwen3-vl-2b-instruct-auto (primary_usable) | qwen-auto |  | 121x4MB | 100 |", report_text)


if __name__ == "__main__":
    unittest.main()
