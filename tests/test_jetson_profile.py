"""Jetson tegrastats parsing, profile summary, and report artifact tests."""

import json
import tempfile
import unittest
from pathlib import Path


class JetsonProfileContractsTest(unittest.TestCase):

    def test_jetson_profile_parses_core_tegrastats_fields(self):
        from edge_vlm.jetson_profile import parse_tegrastats_line

        sample = parse_tegrastats_line(
            "05-31-2026 RAM 2100/7620MB (lfb 180x4MB) "
            "SWAP 12/3810MB (cached 4MB) CPU [10%@1728,off,35%@1728] "
            "GR3D_FREQ 89%@[1020] EMC_FREQ 76%@3199 "
            "cpu@52.0C gpu@54.5C tj@55.0C "
            "VDD_IN 17400mW/16800mW VDD_CPU_GPU_CV 8900mW/8200mW"
        )

        self.assertEqual(sample["ram"], {"used_mb": 2100, "total_mb": 7620})
        self.assertEqual(sample["swap"], {"used_mb": 12, "total_mb": 3810, "cached_mb": 4})
        self.assertEqual(sample["lfb"], {"free_blocks": 180, "block_mb": 4})
        self.assertEqual(sample["cpu"]["cores"][0], {"state": "online", "util_pct": 10, "freq_mhz": 1728})
        self.assertEqual(sample["cpu"]["cores"][1], {"state": "off", "util_pct": None, "freq_mhz": None})
        self.assertEqual(sample["gr3d"], {"util_pct": 89, "freq_mhz": 1020})
        self.assertEqual(sample["emc"], {"util_pct": 76, "freq_mhz": 3199})
        self.assertEqual(sample["temps_c"]["gpu"], 54.5)
        self.assertEqual(sample["power_mw"]["VDD_IN"], {"instant": 17400, "average": 16800})
        self.assertEqual(sample["power_mw"]["VDD_CPU_GPU_CV"], {"instant": 8900, "average": 8200})

    def test_jetson_profile_summarizes_log_and_labels_bottlenecks(self):
        from edge_vlm.jetson_profile import summarize_tegrastats_log

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "tegrastats.log"
            log.write_text(
                "\n".join(
                    [
                        "RAM 2000/7620MB (lfb 200x4MB) CPU [40%@1728] GR3D_FREQ 92%@[1020] EMC_FREQ 81%@3199 gpu@54.0C VDD_IN 18000mW/17000mW",
                        "RAM 2200/7620MB (lfb 160x4MB) CPU [45%@1728] GR3D_FREQ 88%@[1020] EMC_FREQ 86%@3199 gpu@58.0C VDD_IN 19000mW/18000mW",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_tegrastats_log(log)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["min_lfb_free_blocks"], 160)
        self.assertEqual(summary["max_temp_c"], 58.0)
        self.assertEqual(summary["avg_power_w"], 18.5)
        self.assertEqual(summary["avg_gr3d_util_pct"], 90.0)
        self.assertEqual(summary["avg_emc_util_pct"], 83.5)
        self.assertIn("gpu_compute", summary["bottleneck_labels"])
        self.assertIn("emc_memory_bandwidth", summary["bottleneck_labels"])

    def test_jetson_profile_labels_input_payload_and_runtime_overhead_from_input_timing(self):
        from edge_vlm.jetson_profile import summarize_input_timing_records, summarize_tegrastats_samples

        payload_input_summary = summarize_input_timing_records(
            [
                {
                    "latency_s": 1.0,
                    "input_timing": {
                        "image_bytes": 1024,
                        "payload_build_s": 0.30,
                        "json_serialize_s": 0.10,
                        "http_request_s": 0.80,
                        "response_parse_s": 0.02,
                        "request_body_bytes": 2048,
                    },
                },
                {
                    "latency_s": 1.4,
                    "input_timing": {
                        "image_bytes": 512,
                        "payload_build_s": 0.20,
                        "json_serialize_s": 0.10,
                        "http_request_s": 1.10,
                        "response_parse_s": 0.02,
                        "request_body_bytes": 1024,
                    },
                },
            ],
            sources={"benchmark_jsonl": 2},
        )
        payload_summary = summarize_tegrastats_samples(
            [
                {
                    "cpu": {"cores": [{"util_pct": 20}]},
                    "gr3d": {"util_pct": 10},
                    "emc": {"util_pct": 25},
                }
            ],
            input_timing_summary=payload_input_summary,
        )

        runtime_input_summary = summarize_input_timing_records(
            [
                {
                    "latency_s": 2.5,
                    "input_timing": {
                        "payload_build_s": 0.02,
                        "json_serialize_s": 0.01,
                        "http_request_s": 2.40,
                        "response_parse_s": 0.01,
                        "request_body_bytes": 640,
                    },
                }
            ],
            sources={"benchmark_jsonl": 1},
        )
        runtime_summary = summarize_tegrastats_samples(
            [
                {
                    "cpu": {"cores": [{"util_pct": 25}]},
                    "gr3d": {"util_pct": 12},
                    "emc": {"util_pct": 30},
                }
            ],
            input_timing_summary=runtime_input_summary,
        )

        self.assertTrue(payload_input_summary["available"])
        self.assertEqual(payload_input_summary["records"], 2)
        self.assertEqual(payload_input_summary["avg_payload_overhead_s"], 0.35)
        self.assertEqual(payload_input_summary["avg_payload_overhead_ratio"], 0.226)
        self.assertIn("input_payload", payload_summary["bottleneck_labels"])
        self.assertNotIn("runtime_overhead", payload_summary["bottleneck_labels"])
        self.assertIn("runtime_overhead", runtime_summary["bottleneck_labels"])
        self.assertNotIn("input_payload", runtime_summary["bottleneck_labels"])

    def test_jetson_profile_writes_profile_jsonl_and_phase_summary(self):
        from edge_vlm.jetson_profile import write_profile_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log = tmp_path / "tegrastats.log"
            profile_jsonl = tmp_path / "profile.jsonl"
            summary_json = tmp_path / "summary.json"
            log.write_text(
                "\n".join(
                    [
                        "2026-05-31T13:45:54.000000Z RAM 2000/7620MB (lfb 200x4MB) CPU [40%@1728] GR3D_FREQ 12%@[1020] EMC_FREQ 31%@3199 gpu@54.0C VDD_IN 8000mW/7000mW",
                        "2026-05-31T13:45:55.500000Z RAM 2200/7620MB (lfb 160x4MB) CPU [45%@1728] GR3D_FREQ 18%@[1020] EMC_FREQ 36%@3199 gpu@58.0C VDD_IN 9000mW/8000mW",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_profile_artifacts(
                tegrastats_log=log,
                profile_jsonl_path=profile_jsonl,
                summary_path=summary_json,
                phase_timings={"server_startup": {"available": True, "duration_s": 35.0}},
                profile_files={"jetson_clocks": "outputs/benchmarks/unit.profile/jetson-clocks.txt"},
            )
            profile_records = [json.loads(line) for line in profile_jsonl.read_text(encoding="utf-8").splitlines()]
            summary_record = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(len(profile_records), 2)
        self.assertEqual(profile_records[0]["sample_index"], 0)
        self.assertEqual(profile_records[0]["captured_at"], "2026-05-31T13:45:54.000000Z")
        self.assertEqual(profile_records[0]["elapsed_s"], 0.0)
        self.assertEqual(profile_records[1]["captured_at"], "2026-05-31T13:45:55.500000Z")
        self.assertEqual(profile_records[1]["elapsed_s"], 1.5)
        self.assertEqual(profile_records[0]["sample"]["ram"]["used_mb"], 2000)
        self.assertEqual(summary["first_captured_at"], "2026-05-31T13:45:54.000000Z")
        self.assertEqual(summary["last_captured_at"], "2026-05-31T13:45:55.500000Z")
        self.assertEqual(summary["captured_duration_s"], 1.5)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["phase_timings"]["server_startup"]["duration_s"], 35.0)
        self.assertFalse(summary["phase_timings"]["artifact_check_or_download"]["available"])
        self.assertIn("startup_or_download", summary["bottleneck_labels"])
        self.assertEqual(
            summary["profile_files"]["jetson_clocks"],
            "outputs/benchmarks/unit.profile/jetson-clocks.txt",
        )
        self.assertEqual(summary_record, summary)
        for doc_path in (
            "docs/benchmark_protocol.md",
            "docs/specs/jetson_optimization_loop.md",
            "docs/specs/next_phase_infra_and_model_strategy.md",
            "docs/matrix_edge_vlm_workflow.md",
        ):
            doc = Path(doc_path).read_text(encoding="utf-8")
            with self.subTest(doc_path=doc_path):
                self.assertIn("captured_at", doc)
                self.assertIn("elapsed_s", doc)


if __name__ == "__main__":
    unittest.main()
