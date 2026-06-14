"""Model catalog, documentation, and quality review contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path



class ModelCatalogDocsQualityContractsTest(unittest.TestCase):

    def test_quality_review_applies_route_specific_policy_to_benchmark_jsonl(self):
        from edge_vlm.quality_review import format_markdown_report, review_benchmark_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_jsonl = tmp_path / "benchmark.jsonl"
            benchmark_jsonl.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 1,
                            "prompt_case_id": "text_code_short",
                            "input_type": "text",
                            "success": True,
                            "output_excerpt": "def tokens_per_second(latency_s, token_count):\n    return token_count / latency_s",
                        },
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 2,
                            "prompt_case_id": "text_code_short",
                            "input_type": "text",
                            "success": True,
                            "output_excerpt": "return latency * token_count",
                        },
                        {
                            "run_id": "quality-unit",
                            "model": "candidate",
                            "trial_index": 1,
                            "case_index": 3,
                            "prompt_case_id": "image_safety_scene_single",
                            "input_type": "image",
                            "success": True,
                            "output_excerpt": "No visible hazards are present in the simple square scene.",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            policy = {
                "default": {"min_output_chars": 12, "max_repeat_ratio": 0.8},
                "cases": {
                    "text_code_short": {
                        "must_include_all": ["def ", "tokens_per_second"],
                        "must_not_include_any": ["latency * token_count"],
                    },
                    "image_safety_scene_single": {
                        "must_include_any": ["no visible", "none"],
                        "must_not_include_any": ["fire", "knife"],
                    },
                },
            }

            report = review_benchmark_jsonl(benchmark_jsonl, policy)
            markdown = format_markdown_report(report)

        self.assertFalse(report["passed"])
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["failed_records"], 1)
        self.assertEqual(report["case_summaries"]["text_code_short"]["failed"], 1)
        self.assertIn("text_code_short", markdown)
        self.assertIn("missing_all:def ", json.dumps(report, ensure_ascii=False))
        self.assertIn("forbidden:latency * token_count", json.dumps(report, ensure_ascii=False))

    def test_quality_review_cli_and_docs_are_wired(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark_jsonl = tmp_path / "benchmark.jsonl"
            report_json = tmp_path / "quality.json"
            report_md = tmp_path / "quality.md"
            benchmark_jsonl.write_text(
                json.dumps(
                    {
                        "run_id": "quality-cli-unit",
                        "model": "candidate",
                        "trial_index": 1,
                        "case_index": 1,
                        "prompt_case_id": "text_translation_zh_to_en_short",
                        "input_type": "text",
                        "success": True,
                        "output_excerpt": "Jetson Orin runs a vision language model on the edge with memory bandwidth, power, and latency constraints.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.quality_review",
                    "--input",
                    str(benchmark_jsonl),
                    "--policy",
                    "configs/benchmark/quality_review_policy.json",
                    "--output",
                    str(report_json),
                    "--markdown-output",
                    str(report_md),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONPATH": "src", "PYTHONPYCACHEPREFIX": "/tmp/edge-vlm-pycache"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            markdown = report_md.read_text(encoding="utf-8")

        self.assertTrue(report["passed"])
        self.assertIn("quality-cli-unit", report["run_ids"])
        self.assertIn("text_translation_zh_to_en_short", markdown)

        protocol = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")
        for text in (protocol, strategy, matrix):
            self.assertIn("edge_vlm.quality_review", text)
            self.assertIn("configs/benchmark/quality_review_policy.json", text)

    def test_next_phase_spec_orders_infra_before_model_expansion_and_lists_tencent_youtu_vl(self):
        spec = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")

        self.assertIn("Phase 1: Formal Benchmark Infra", spec)
        self.assertIn("Phase 2: Lightweight Model Expansion", spec)
        self.assertLess(
            spec.index("Phase 1: Formal Benchmark Infra"),
            spec.index("Phase 2: Lightweight Model Expansion"),
        )
        self.assertIn("tencent/Youtu-VL-4B-Instruct-GGUF", spec)
        self.assertIn("ggml-org/HunyuanOCR-GGUF", spec)
        self.assertIn("tencent/HY-MT1.5-1.8B-GGUF", spec)
        self.assertIn("tencent/Youtu-LLM-2B-GGUF", spec)
        self.assertIn("Hy-MT1.5 1.8B Safetensors", spec)
        self.assertIn("Hy-MT1.5", spec)
        self.assertIn("SmolVLM2", spec)
        self.assertIn("Qwen3-VL-2B", spec)

    def test_workflow_matrix_tracks_next_phase_infra_and_model_specs(self):
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        self.assertIn("next_phase_infra_and_model_strategy.md", matrix)
        self.assertIn("jetson_optimization_loop.md", matrix)
        self.assertIn("src/edge_vlm/jetson_profile.py", matrix)
        self.assertIn("scripts/jetson/run_optimization_sweep.sh", matrix)
        self.assertIn("scripts/jetson/run_remote_current_defaults_suite.sh", matrix)
        self.assertIn("run_hf_gguf_vlm_llama_docker.sh", matrix)
        self.assertIn("run_remote_tencent_text_suite.sh", matrix)
        self.assertIn("SmolVLM2", matrix)
        self.assertIn("Qwen3-VL", matrix)
        self.assertIn("HunyuanOCR", matrix)
        self.assertIn("Hy-MT1.5", matrix)
        self.assertIn("Youtu-VL", matrix)

    def test_lightweight_hf_gguf_vlm_configs_and_variants_exist(self):
        from edge_vlm.config import config_supports_images, load_model_config

        expected = {
            "smolvlm2-256m-q8": {
                "config": "configs/models/smolvlm2_256m_q8.yaml",
                "model_ref": "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0",
                "model_file": "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "mmproj_file": "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "ctx_size": 512,
                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
            },
            "qwen3-vl-2b-thinking-q4": {
                "config": "configs/models/qwen3_vl_2b_thinking_q4.yaml",
                "model_ref": "Qwen/Qwen3-VL-2B-Thinking-GGUF:Q4_K_M",
                "model_file": "Qwen3VL-2B-Thinking-Q4_K_M.gguf",
                "mmproj_file": "mmproj-Qwen3VL-2B-Thinking-Q8_0.gguf",
                "ctx_size": 1024,
                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
            },
            "qwen3-vl-2b-instruct-q4": {
                "config": "configs/models/qwen3_vl_2b_instruct_q4.yaml",
                "model_ref": "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M",
                "model_file": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                "mmproj_file": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "comparison_group": "qwen3-vl-2b-instruct",
                "ctx_size": 1024,
                "batch_size": 256,
                "ubatch_size": 64,
                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
            },
            "qwen3-vl-2b-instruct-q8": {
                "config": "configs/models/qwen3_vl_2b_instruct_q8.yaml",
                "model_ref": "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q8_0",
                "model_file": "Qwen3VL-2B-Instruct-Q8_0.gguf",
                "mmproj_file": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "comparison_group": "qwen3-vl-2b-instruct",
                "ctx_size": 1024,
                "batch_size": 128,
                "ubatch_size": 32,
                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
            },
            "hunyuanocr-q8": {
                "config": "configs/models/hunyuanocr_q8.yaml",
                "model_ref": "ggml-org/HunyuanOCR-GGUF:Q8_0",
                "model_file": "HunyuanOCR-Q8_0.gguf",
                "mmproj_file": "mmproj-HunyuanOCR-Q8_0.gguf",
                "ctx_size": 1024,
                "batch_size": 128,
                "ubatch_size": 32,
                "candidate_scope": {"leq2b_candidate": True, "lane": "vlm"},
            },
            "youtu-vl-4b-q8": {
                "config": "configs/models/youtu_vl_4b_q8.yaml",
                "model_ref": "tencent/Youtu-VL-4B-Instruct-GGUF:Q8_0",
                "model_file": "Youtu-VL-4B-Instruct-Q8_0.gguf",
                "mmproj_file": "mmproj-Youtu-VL-4b-Instruct-BF16.gguf",
                "ctx_size": 2048,
            },
            "youtu-vl-4b-q4-thirdparty": {
                "config": "configs/models/youtu_vl_4b_q4_thirdparty.yaml",
                "model_ref": "mradermacher/Youtu-VL-4B-Instruct-GGUF:Q4_K_M",
                "model_file": "Youtu-VL-4B-Instruct.Q4_K_M.gguf",
                "mmproj_file": "Youtu-VL-4B-Instruct.mmproj-Q8_0.gguf",
                "ctx_size": 1024,
                "n_gpu_layers": 8,
                "batch_size": 256,
                "ubatch_size": 256,
                "required_arg": "--no-mmproj-offload",
            },
        }
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}

        def assert_arg_value(args: list[str], flag: str, expected_value: object) -> None:
            self.assertIn(flag, args)
            flag_index = args.index(flag)
            self.assertLess(flag_index + 1, len(args), f"{flag} has no value")
            self.assertEqual(args[flag_index + 1], str(expected_value))

        for model_name, expected_values in expected.items():
            with self.subTest(model_name=model_name):
                config = load_model_config(expected_values["config"])
                self.assertEqual(config["model"]["name"], model_name)
                self.assertEqual(config["model"]["model_ref"], expected_values["model_ref"])
                self.assertTrue(config_supports_images(config))
                self.assertEqual(config["runtime"]["model_file"], expected_values["model_file"])
                self.assertEqual(config["runtime"]["mmproj_file"], expected_values["mmproj_file"])
                self.assertEqual(
                    config["runtime"]["jetson_script"],
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                )
                if "candidate_scope" in expected_values:
                    self.assertEqual(config["candidate_scope"], expected_values["candidate_scope"])

                variant_id = f"{model_name}-smoke"
                variant = by_id[variant_id]
                self.assertEqual(variant["model"], model_name)
                self.assertEqual(variant["config"], expected_values["config"])
                self.assertEqual(variant["launcher"], "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh")
                self.assertEqual(variant["env"]["MODEL_REF"], expected_values["model_ref"])
                self.assertEqual(variant["env"]["MODEL_FILE"], expected_values["model_file"])
                self.assertEqual(variant["env"]["MMPROJ_FILE"], expected_values["mmproj_file"])
                self.assertEqual(variant["env"]["CTX_SIZE"], expected_values["ctx_size"])
                self.assertEqual(variant["env"]["MODEL_ALIAS"], model_name)
                if "comparison_group" in expected_values:
                    self.assertEqual(variant["comparison_group"], expected_values["comparison_group"])
                if "n_gpu_layers" in expected_values:
                    self.assertEqual(variant["env"]["N_GPU_LAYERS"], expected_values["n_gpu_layers"])
                if "batch_size" in expected_values:
                    self.assertEqual(variant["env"]["LLAMA_BATCH_SIZE"], expected_values["batch_size"])
                    assert_arg_value(variant["args"], "--batch-size", expected_values["batch_size"])
                if "ubatch_size" in expected_values:
                    self.assertEqual(variant["env"]["LLAMA_UBATCH_SIZE"], expected_values["ubatch_size"])
                    assert_arg_value(variant["args"], "--ubatch-size", expected_values["ubatch_size"])
                if "required_arg" in expected_values:
                    self.assertIn(expected_values["required_arg"], variant["args"])
                self.assertIn("--parallel", variant["args"])
                self.assertIn("--no-warmup", variant["args"])
                if model_name.endswith("-thirdparty"):
                    status = config["notes"]["status"]
                    self.assertIn("third-party", status)
                    self.assertIn("not an official Tencent GGUF artifact", status)

    def test_tencent_text_gguf_configs_and_variants_exist(self):
        from edge_vlm.config import config_supports_images, load_model_config

        expected = {
            "tencent-hy-mt1p5-1p8b-1p25bit": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_1p25bit.yaml",
                "model_ref": "tencent/Hy-MT1.5-1.8B-1.25bit-GGUF:1.25bit",
                "model_file": "Hy-MT1.5-1.8B-1.25bit.gguf",
                "quantization": "1.25bit",
            },
            "tencent-hy-mt1p5-1p8b-2bit": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_2bit.yaml",
                "model_ref": "tencent/Hy-MT1.5-1.8B-2bit-GGUF:2bit",
                "model_file": "Hy-MT1.5-1.8B-2bit.gguf",
                "quantization": "2bit",
            },
            "tencent-hy-mt1p5-1p8b-q4": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_q4.yaml",
                "model_ref": "tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M",
                "model_file": "HY-MT1.5-1.8B-Q4_K_M.gguf",
                "quantization": "Q4_K_M",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-hy-mt1p5-1p8b-q6": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_q6.yaml",
                "model_ref": "tencent/HY-MT1.5-1.8B-GGUF:Q6_K",
                "model_file": "HY-MT1.5-1.8B-Q6_K.gguf",
                "quantization": "Q6_K",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-hy-mt1p5-1p8b-q8": {
                "config": "configs/models/tencent_hy_mt1p5_1p8b_q8.yaml",
                "model_ref": "tencent/HY-MT1.5-1.8B-GGUF:Q8_0",
                "model_file": "HY-MT1.5-1.8B-Q8_0.gguf",
                "quantization": "Q8_0",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-hy-mt2-1p8b-1p25bit": {
                "config": "configs/models/tencent_hy_mt2_1p8b_1p25bit.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit",
                "model_file": "Hy-MT2-1.8B-1.25Bit.gguf",
                "quantization": "1.25Bit",
            },
            "tencent-hy-mt2-1p8b-2bit": {
                "config": "configs/models/tencent_hy_mt2_1p8b_2bit.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-2Bit-GGUF:2Bit",
                "model_file": "Hy-MT2-1.8B-2Bit.gguf",
                "quantization": "2Bit",
            },
            "tencent-hy-mt2-1p8b-q4": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q4.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
                "model_file": "Hy-MT2-1.8B-Q4_K_M.gguf",
                "quantization": "Q4_K_M",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-hy-mt2-1p8b-q6": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q6.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q6_K",
                "model_file": "Hy-MT2-1.8B-Q6_K.gguf",
                "quantization": "Q6_K",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-hy-mt2-1p8b-q8": {
                "config": "configs/models/tencent_hy_mt2_1p8b_q8.yaml",
                "model_ref": "tencent/Hy-MT2-1.8B-GGUF:Q8_0",
                "model_file": "Hy-MT2-1.8B-Q8_0.gguf",
                "quantization": "Q8_0",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
            "tencent-youtu-llm-2b-q8": {
                "config": "configs/models/tencent_youtu_llm_2b_q8.yaml",
                "model_ref": "tencent/Youtu-LLM-2B-GGUF:Q8_0",
                "model_file": "Youtu-LLM-2B-Q8_0.gguf",
                "quantization": "Q8_0",
                "candidate_scope": {"leq2b_candidate": True, "lane": "text"},
            },
        }
        variants = [
            json.loads(line)
            for line in Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {variant["id"]: variant for variant in variants}
        text_cases = [
            json.loads(line)
            for line in Path("configs/benchmark/text_prompt_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertGreaterEqual(len(text_cases), 4)
        self.assertTrue(all(case.get("input_type") == "text" for case in text_cases))
        self.assertIn("text_translation_zh_to_en_short", {case["id"] for case in text_cases})
        for model_name, expected_values in expected.items():
            with self.subTest(model_name=model_name):
                config = load_model_config(expected_values["config"])
                self.assertEqual(config["model"]["name"], model_name)
                self.assertEqual(config["model"]["model_ref"], expected_values["model_ref"])
                self.assertEqual(config["model"]["quantization"], expected_values["quantization"])
                self.assertFalse(config_supports_images(config))
                self.assertEqual(config["runtime"]["model_file"], expected_values["model_file"])
                self.assertEqual(
                    config["runtime"]["jetson_script"],
                    "scripts/jetson/run_hf_gguf_llama_docker.sh",
                )
                if "candidate_scope" in expected_values:
                    self.assertEqual(config["candidate_scope"], expected_values["candidate_scope"])

                variant = by_id[f"{model_name}-text-smoke"]
                self.assertEqual(variant["model"], model_name)
                self.assertEqual(variant["config"], expected_values["config"])
                self.assertEqual(variant["launcher"], "scripts/jetson/run_hf_gguf_llama_docker.sh")
                self.assertEqual(variant["env"]["MODEL_REF"], expected_values["model_ref"])
                self.assertEqual(variant["env"]["MODEL_FILE"], expected_values["model_file"])
                self.assertEqual(variant["env"]["MODEL_ALIAS"], model_name)
                self.assertEqual(variant["env"]["EDGE_VLM_CASES"], "configs/benchmark/text_prompt_cases.jsonl")
                self.assertIn("--parallel", variant["args"])
                self.assertIn("--no-warmup", variant["args"])

    def test_tencent_text_suite_docs_match_all_variant_default_policy(self):
        checked_paths = (
            "docs/specs/next_phase_infra_and_model_strategy.md",
            "docs/specs/next_phase_benchmark_and_models.md",
            "docs/benchmarks/jetson_lightweight_models_20260531.md",
            "docs/benchmark_protocol.md",
            "docs/matrix_edge_vlm_workflow.md",
            "configs/benchmark/jetson_optimization_variants.jsonl",
            "configs/models/tencent_hy_mt1p5_1p8b_1p25bit.yaml",
            "configs/models/tencent_hy_mt1p5_1p8b_2bit.yaml",
            "configs/models/tencent_hy_mt1p5_1p8b_q4.yaml",
            "configs/models/tencent_hy_mt1p5_1p8b_q6.yaml",
            "configs/models/tencent_hy_mt1p5_1p8b_q8.yaml",
            "configs/models/tencent_hy_mt2_1p8b_1p25bit.yaml",
            "configs/models/tencent_hy_mt2_1p8b_2bit.yaml",
            "configs/models/tencent_youtu_llm_2b_q8.yaml",
        )
        stale_phrases = (
            "defaults to Hy-MT2 Q4/Q6/Q8",
            "default-suite rows",
            "current-runtime-compatible Hy-MT2 Q4_K_M/Q6_K/Q8_0",
            "keep out of default text repeats",
            "stay out of default repeats",
            "not in the default text suite",
            "must be passed through `JETSON_TENCENT_TEXT_VARIANTS`",
            "Hy-MT2 Q4/Q6/Q8 rows only",
            "Youtu-LLM Q8 rows still need Jetson evidence",
            "No Jetson evidence yet",
        )

        for path in checked_paths:
            text = Path(path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                for phrase in stale_phrases:
                    self.assertNotIn(phrase, text)

    def test_tencent_hy_mt2_q4_smoke_evidence_is_documented_as_text_only(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc):
            self.assertIn("tencent-hy-mt2-q4-smoke64-cached-20260531b", text)
            self.assertIn("19.759", text)
        self.assertIn("text/router", benchmark_doc)
        self.assertIn("not a VLM ranking row", model_doc)
        self.assertIn("guard-passing cached text smoke", matrix)

    def test_tencent_text_suite_full_smoke_evidence_is_documented_with_invalid_q8(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc, protocol_doc):
            self.assertIn("tencent-text-smoke64-20260531T120054Z", text)
            self.assertIn("33.541", text)
            self.assertIn("26.287", text)
            self.assertIn("tencent-hy-mt2-q6-smoke64-cached-20260531a", text)
            self.assertIn("26.298", text)
            self.assertIn("Q8 full-suite row is invalidated", text)
            self.assertIn("tencent-hy-mt2-q8-smoke64-cached-20260531T132724Z", text)
            self.assertIn("30.756", text)
            self.assertIn("terminate_group", text)
            self.assertIn("tencent-text-repeat5-20260531a", text)
            self.assertIn("34.528", text)
            self.assertIn("31.395", text)
            self.assertIn("26.778", text)
            self.assertIn("20/20", text)
        self.assertIn("tencent-text-repeat5-20260531a", matrix)
        self.assertIn("invalid ggml type 42", benchmark_doc)
        self.assertIn("offset 203248672", benchmark_doc)
        self.assertIn("server_port_still_open_before_start", protocol_doc)

    def test_tencent_text_prepared_repeat_evidence_is_documented(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(
            encoding="utf-8"
        )

        for text in (benchmark_doc, strategy_doc, protocol_doc):
            self.assertIn("tencent-text-repeat5-prepctx-20260609T131412Z", text)
            self.assertIn("max_clocks, drop_caches", text)
            self.assertIn("Required lfb", text)
            self.assertIn("Youtu-LLM 2B Q8", text)
        self.assertIn("24.577", benchmark_doc)
        self.assertIn("354.568", benchmark_doc)
        self.assertIn("20/20", benchmark_doc)
        self.assertIn("quality_review_failed 15/20", benchmark_doc)
        self.assertIn("quality_review_failed 10/20", benchmark_doc)
        self.assertIn("Promotion precheck = yes", benchmark_doc)
        for text in (benchmark_doc, strategy_doc, protocol_doc, model_doc):
            self.assertIn("youtu-llm-q8-cached-20260609T133339Z", text)
            self.assertIn("cached", text)
        self.assertIn("artifact_check_or_download", benchmark_doc)
        self.assertIn("0.002", benchmark_doc)
        self.assertIn("24.621", benchmark_doc)
        self.assertIn("5.015", benchmark_doc)
        self.assertIn("first artifact download", benchmark_doc)

    def test_lightweight_formal_repeat_evidence_is_documented(self):
        benchmark_doc = Path("docs/benchmarks/jetson_lightweight_models_20260531.md").read_text(
            encoding="utf-8"
        )
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        model_doc = Path("docs/specs/next_phase_benchmark_and_models.md").read_text(encoding="utf-8")
        matrix = Path("docs/matrix_edge_vlm_workflow.md").read_text(encoding="utf-8")

        for text in (benchmark_doc, strategy_doc, model_doc, matrix):
            self.assertIn("lightweight-repeat5-20260531T134554Z", text)
        for metric in ("199.847", "48.598", "34.761", "7.502"):
            self.assertIn(metric, benchmark_doc)
        self.assertIn("latency_floor", strategy_doc)
        self.assertIn("balanced_candidate", strategy_doc)
        self.assertIn("runtime_overhead", strategy_doc)
        self.assertIn("official Youtu-VL Q8", model_doc)
        self.assertIn("Structured quality review", benchmark_doc)
        self.assertIn("MiniCPM-V 4.6 Q4 | 30/30", benchmark_doc)
        self.assertIn("SmolVLM2 256M Q8 | 20/30", benchmark_doc)
        self.assertIn("Qwen3-VL 2B Thinking Q4 | 20/30", benchmark_doc)
        self.assertIn("Youtu-VL 4B Q4 third-party | 30/30", benchmark_doc)
        self.assertIn("Qwen remains image/fake-stream balanced_candidate", strategy_doc)

    def test_qwen3_instruct_diagnostic_smoke_status_is_documented(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )
        config_doc = Path("configs/models/qwen3_vl_2b_instruct_q4.yaml").read_text(encoding="utf-8")
        variant_catalog = Path("configs/benchmark/jetson_optimization_variants.jsonl").read_text(
            encoding="utf-8"
        )
        q8_config_doc = Path("configs/models/qwen3_vl_2b_instruct_q8.yaml").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc):
            self.assertIn("qwen3-instruct-q4-smoke-lfb32-20260609T075208Z", text)
            self.assertIn("34.349", text)
            self.assertIn("26.957", text)
            self.assertIn("1.827", text)
            self.assertIn("qwen3-instruct-q4-smoke-compact-20260609T074600Z", text)
            self.assertIn("46x4MB", text)
            self.assertIn("quality review", text)
            self.assertIn("qwen3-instruct-q4q8-lfb100-20260609T091700Z", text)
            self.assertIn("34.865", text)
            self.assertIn("31.958", text)
            self.assertIn("31.346", text)
            self.assertIn("29.393", text)
            self.assertIn("125x4MB", text)
            self.assertIn("963.598", text)
        for text in (config_doc, variant_catalog):
            self.assertIn("Jetson smoke", text)
            self.assertIn("min-lfb-blocks 32", text)
        self.assertIn("qwen3-instruct-q4q8-lfb100-20260609T091700Z", config_doc)
        self.assertIn("1.816", config_doc)
        for text in (protocol_doc, variant_catalog):
            self.assertIn("qwen3-vl-2b-instruct-q8-smoke", text)
        for text in (protocol_doc, strategy_doc, q8_config_doc):
            self.assertIn("qwen3-instruct-q8-smoke-lfb100-20260609T082321Z", text)
            self.assertIn("31.076", text)
            self.assertIn("25.997", text)
            self.assertIn("2.142", text)
            self.assertIn("323.05", text)
            self.assertIn("106x4MB", text)
            self.assertIn("qwen3-instruct-q4q8-lfb100-20260609T091700Z", text)
            self.assertIn("31.346", text)
            self.assertIn("29.393", text)
            self.assertIn("963.598", text)
        for text in (protocol_doc, strategy_doc, variant_catalog):
            self.assertIn("150-LFB", text)
        self.assertIn("relaxed 100-LFB smoke passed with guard", variant_catalog)
        self.assertIn("default lightweight ranking row", variant_catalog)

    def test_readmes_summarize_q4_first_q8_fallback_state(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (readme, readme_zh):
            self.assertIn("Q4", text)
            self.assertIn("Q8", text)
            self.assertIn("Qwen3-VL 2B Instruct", text)
            self.assertIn("qwen3-instruct-q4q8-lfb100-20260609T091700Z", text)
            self.assertIn("34.865", text)
            self.assertIn("31.346", text)

    def test_preflight_docs_include_buddyinfo_fragmentation_context(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("`meminfo_kb`, `/proc/buddyinfo`, and structured `tegrastats` `lfb`", protocol_doc)
        self.assertIn("/proc/buddyinfo", strategy_doc)
        self.assertIn("fragmentation", strategy_doc)

    def test_jetson_runtime_docs_explain_mmproj_probe_and_skip_reason(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        migration_doc = Path("docs/migration_wsl_to_jetson.md").read_text(encoding="utf-8")

        for text in (readme, readme_zh, protocol_doc, migration_doc):
            self.assertIn("--mmproj", text)
            self.assertIn("runtime_missing_mmproj_support", text)
            self.assertIn("edge_vlm.llama_cpp_runtime", text)
            self.assertIn("probe-image", text)
            self.assertIn("multimodal_ready", text)
            self.assertIn("LLAMA_CPP_RUNTIME_PROBE_OUTPUT", text)

        for text in (readme, readme_zh, protocol_doc, migration_doc):
            self.assertIn("invalid argument: --mmproj", text)
            self.assertIn("dustynv/llama_cpp:b5283-r36.4-cu128-24.04", text)

    def test_qwen3_selector_docs_are_wired(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        migration_doc = Path("docs/migration_wsl_to_jetson.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")

        for text in (readme, readme_zh, protocol_doc, migration_doc, strategy_doc):
            self.assertIn("select_qwen3_instruct_variant.sh", text)
        for text in (readme, readme_zh, protocol_doc):
            self.assertIn("fallback-min-lfb-blocks", text)
        for text in (readme, readme_zh, protocol_doc, migration_doc):
            self.assertIn("JETSON_LIGHTWEIGHT_QWEN3_SELECTOR", text)
            self.assertIn("JETSON_LIGHTWEIGHT_QWEN3_FALLBACK_MIN_LFB_BLOCKS", text)
        for text in (readme, readme_zh, protocol_doc, strategy_doc):
            self.assertIn("Selection", text)
            self.assertIn("ranking-min-lfb-blocks", text)
            self.assertIn("Ranking precheck", text)
        for text in (protocol_doc,):
            self.assertIn("selection_contexts", text)
            self.assertIn("--selection-context-json", text)
            self.assertIn("variant_min_lfb_blocks", text)
            self.assertIn("Required lfb", text)
        for text in (readme, readme_zh, protocol_doc, strategy_doc):
            self.assertIn("--variant-min-lfb-blocks", text)

    def test_compare_docs_explain_required_lfb_backfill_from_plan(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc):
            self.assertIn("plan.min_lfb_blocks", text)
            self.assertIn("variant_min_lfb_blocks", text)
            self.assertIn("preflight_required_lfb_blocks", text)

    def test_compare_docs_explain_prepare_context_signals(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc):
            self.assertIn("prepare_context", text)
            self.assertIn("Prepare ctx", text)
            self.assertIn("max_clocks", text)
            self.assertIn("drop_caches", text)

    def test_compare_docs_explain_artifact_phase_columns(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc):
            self.assertIn("artifact_check_or_download", text)
            self.assertIn("Artifact phase", text)
            self.assertIn("Artifact s", text)

    def test_docs_explain_gguf_artifact_magic_validation(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc, readme, readme_zh):
            self.assertIn("GGUF magic", text)
            self.assertIn("llama-server", text)
        for text in (protocol_doc, strategy_doc):
            self.assertIn("model and mmproj", text)

    def test_docs_explain_gguf_artifact_preflight_manifest(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc, readme, readme_zh):
            self.assertIn("check_gguf_artifacts.sh", text)
            self.assertIn("gguf-artifacts.json", text)
            self.assertIn("JETSON_REMOTE_GGUF_PREFLIGHT", text)
            self.assertIn("JETSON_REMOTE_GGUF_PREFLIGHT_FAIL", text)
        for text in (protocol_doc, strategy_doc):
            self.assertIn("edge_vlm.gguf_artifacts", text)

    def test_compare_docs_explain_startup_precheck_cached_artifact_gate(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc, readme, readme_zh):
            self.assertIn("startup-require-cached-artifacts", text)
            self.assertIn("Startup precheck", text)
            self.assertIn("ranking-require-startup-precheck", text)
        for text in (protocol_doc, strategy_doc):
            self.assertIn("artifact_check_or_download", text)
            self.assertIn("cached", text)
        for text in (protocol_doc, readme, readme_zh):
            self.assertIn("JETSON_CURRENT_DEFAULTS_FAIL_ON_STARTUP_PRECHECK", text)
            self.assertIn("JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK", text)
            self.assertIn("JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK", text)
            self.assertIn("JETSON_CURRENT_DEFAULTS_FAIL_ON_RANKING_PRECHECK", text)
            self.assertIn("JETSON_LIGHTWEIGHT_FAIL_ON_RANKING_PRECHECK", text)
            self.assertIn("JETSON_TENCENT_TEXT_FAIL_ON_RANKING_PRECHECK", text)

    def test_compare_docs_explain_machine_readable_eligibility_output(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc, readme, readme_zh):
            self.assertIn("eligibility-output", text)
            self.assertIn("comparison.eligibility.json", text)
        for text in (protocol_doc, strategy_doc):
            self.assertIn("Ranking precheck", text)
            self.assertIn("Promotion precheck", text)

    def test_compare_docs_explain_selection_exports_from_eligibility_artifacts(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, strategy_doc, readme, readme_zh):
            self.assertIn("select-eligible", text)
            self.assertIn("ranking.selection.json", text)
            self.assertIn("promotion.selection.json", text)
            self.assertIn("require-leq2b-candidate", text)
            self.assertIn("candidate-lane", text)
            self.assertIn("ranking.leq2b-vlm.selection.json", text)
            self.assertIn("promotion.leq2b-vlm.selection.json", text)
            self.assertIn("ranking.leq2b-text.selection.json", text)
            self.assertIn("promotion.leq2b-text.selection.json", text)
            self.assertIn("bundle-selections", text)
            self.assertIn("leq2b.candidate_bundle.json", text)
            self.assertIn("export-routes", text)
            self.assertIn("leq2b.routes.json", text)
            self.assertIn("q4_first_q8_fallback", text)
            self.assertIn("JETSON_LEQ2B_BUILD_ROUTES", text)

    def test_compare_docs_explain_promotion_precheck_stage(self):
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        loop_doc = Path("docs/specs/jetson_optimization_loop.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")

        for text in (protocol_doc, loop_doc, strategy_doc):
            self.assertIn("promotion-precheck-stage", text)
            self.assertIn("Promotion precheck", text)
            self.assertIn("formal-repeat", text)
            self.assertIn("promotion-reference", text)
            self.assertIn("raw excerpt review", text)
        for text in (protocol_doc, loop_doc, strategy_doc, readme, readme_zh):
            self.assertIn("promotion-require-quality-review", text)
            self.assertIn("Quality review", text)
        for text in (protocol_doc, readme, readme_zh):
            self.assertIn("promotion-require-startup-precheck", text)
            self.assertIn("Startup precheck", text)
        for text in (protocol_doc, readme, readme_zh):
            self.assertIn("JETSON_CURRENT_DEFAULTS_FAIL_ON_PROMOTION_PRECHECK", text)
            self.assertIn("JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK", text)
            self.assertIn("JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK", text)

    def test_sweep_quality_review_sidecars_and_compare_docs_are_wired(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        readme_zh = Path("README.zh-CN.md").read_text(encoding="utf-8")
        protocol_doc = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        loop_doc = Path("docs/specs/jetson_optimization_loop.md").read_text(encoding="utf-8")
        strategy_doc = Path("docs/specs/next_phase_infra_and_model_strategy.md").read_text(
            encoding="utf-8"
        )

        for text in (readme, readme_zh, protocol_doc, loop_doc, strategy_doc):
            self.assertIn("edge_vlm.sweep_quality_review", text)
            self.assertIn("Quality review", text)
        for text in (protocol_doc, loop_doc, strategy_doc):
            self.assertIn("quality_review_json", text)
            self.assertIn("quality_review_markdown", text)
            self.assertIn("configs/benchmark/quality_review_policy.json", text)

    def test_shared_prompt_case_assets_exist_for_out_of_box_dry_runs(self):
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        cases = [
            json.loads(line)
            for line in Path("configs/benchmark/prompt_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for case in cases:
            input_type = case.get("input_type")
            if input_type == "image":
                image_path = Path(case["image_path"])
                self.assertTrue(image_path.is_file(), f"missing sample image: {image_path}")
                self.assertIn(image_path.suffix.lower(), image_suffixes)
                self.assertTrue(
                    image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                    or image_path.read_bytes().startswith(b"\xff\xd8\xff"),
                    f"sample image is not a PNG or JPEG: {image_path}",
                )
            elif input_type == "fake_stream":
                image_dir = Path(case["image_dir"])
                self.assertTrue(image_dir.is_dir(), f"missing fake-stream directory: {image_dir}")
                frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in image_suffixes)
                self.assertGreaterEqual(
                    len(frames),
                    3,
                    f"fake-stream sample should include at least three frames: {image_dir}",
                )


if __name__ == "__main__":
    unittest.main()
