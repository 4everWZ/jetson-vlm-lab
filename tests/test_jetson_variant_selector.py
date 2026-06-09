"""Q4-first / Q8 fallback selector contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _write_variant_catalog(path: Path, model_root: Path) -> Path:
    variants = [
        {
            "id": "qwen3-vl-2b-instruct-q4-smoke",
            "model": "qwen3-vl-2b-instruct-q4",
            "config": "configs/models/qwen3_vl_2b_instruct_q4.yaml",
            "launcher": "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
            "env": {
                "MODEL_DIR": str(model_root),
                "MODEL_REF": "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M",
                "MODEL_FILE": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                "MMPROJ_FILE": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "MODEL_ALIAS": "qwen3-vl-2b-instruct-q4",
                "CTX_SIZE": 1024,
                "N_GPU_LAYERS": 99,
                "LLAMA_BATCH_SIZE": 256,
                "LLAMA_UBATCH_SIZE": 64,
            },
            "args": ["--parallel", "1", "--batch-size", "256", "--ubatch-size", "64", "--no-warmup"],
        },
        {
            "id": "qwen3-vl-2b-instruct-q8-smoke",
            "model": "qwen3-vl-2b-instruct-q8",
            "config": "configs/models/qwen3_vl_2b_instruct_q8.yaml",
            "launcher": "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
            "env": {
                "MODEL_DIR": str(model_root),
                "MODEL_REF": "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q8_0",
                "MODEL_FILE": "Qwen3VL-2B-Instruct-Q8_0.gguf",
                "MMPROJ_FILE": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "MODEL_ALIAS": "qwen3-vl-2b-instruct-q8",
                "CTX_SIZE": 1024,
                "N_GPU_LAYERS": 99,
                "LLAMA_BATCH_SIZE": 128,
                "LLAMA_UBATCH_SIZE": 32,
            },
            "args": ["--parallel", "1", "--batch-size", "128", "--ubatch-size", "32", "--no-warmup"],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(variant) for variant in variants) + "\n",
        encoding="utf-8",
    )
    return path


def _preflight(free_blocks: int) -> dict[str, object]:
    return {
        "captured_at": "2026-06-09T00:00:00+00:00",
        "meminfo_kb": {"MemAvailable": 6911728},
        "buddyinfo": {"available": True, "zones": [], "max_order_with_free_block": 12},
        "tegrastats": {
            "available": True,
            "raw": f"RAM 716/7620MB (lfb {free_blocks}x4MB)",
            "lfb": {"free_blocks": free_blocks, "block_mb": 4},
        },
    }


class JetsonVariantSelectorContractsTest(unittest.TestCase):
    def test_selector_prefers_q4_when_primary_is_usable(self):
        from edge_vlm.jetson_variant_selector import select_preferred_variant

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = _write_variant_catalog(tmp_path / "variants.jsonl", tmp_path / "models")
            repo_dir = tmp_path / "models" / "Qwen" / "Qwen3-VL-2B-Instruct-GGUF"
            repo_dir.mkdir(parents=True)
            for name in (
                "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "Qwen3VL-2B-Instruct-Q8_0.gguf",
            ):
                (repo_dir / name).write_bytes(b"GGUF")

            with patch(
                "edge_vlm.jetson_variant_selector.capture_preflight_sample",
                return_value=_preflight(180),
            ):
                with patch(
                    "edge_vlm.jetson_variant_selector._runtime_metadata",
                    return_value={
                        "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                        "llama_server_probe_ok": True,
                        "llama_server_found": True,
                        "llama_server_help_ok": True,
                        "llama_server_supports_mmproj": True,
                        "llama_server_multimodal_markers": ["--mmproj", "mmproj"],
                    },
                ):
                    selection = select_preferred_variant(
                        variants_path=catalog,
                        primary_variant_id="qwen3-vl-2b-instruct-q4-smoke",
                        fallback_variant_id="qwen3-vl-2b-instruct-q8-smoke",
                        min_lfb_blocks=150,
                        base_env={
                            "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:test",
                            "DOCKER_GPU_ARGS": "--runtime nvidia",
                        },
                    )

        self.assertEqual(selection["selected_variant_id"], "qwen3-vl-2b-instruct-q4-smoke")
        self.assertEqual(selection["selected_reason"], "primary_usable")
        self.assertTrue(selection["candidates"][0]["usable"])
        self.assertFalse(selection["candidates"][1]["chosen"])
        self.assertEqual(
            selection["candidates"][0]["launcher_env"]["LLAMA_CPP_DOCKER_IMAGE"],
            "ghcr.io/4everwz/jetson-llama-cpp:test",
        )

    def test_selector_falls_back_to_q8_when_q4_artifact_is_missing(self):
        from edge_vlm.jetson_variant_selector import select_preferred_variant

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = _write_variant_catalog(tmp_path / "variants.jsonl", tmp_path / "models")
            repo_dir = tmp_path / "models" / "Qwen" / "Qwen3-VL-2B-Instruct-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf").write_bytes(b"GGUF")
            (repo_dir / "Qwen3VL-2B-Instruct-Q8_0.gguf").write_bytes(b"GGUF")

            with patch(
                "edge_vlm.jetson_variant_selector.capture_preflight_sample",
                return_value=_preflight(180),
            ):
                with patch(
                    "edge_vlm.jetson_variant_selector._runtime_metadata",
                    return_value={
                        "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                        "llama_server_probe_ok": True,
                        "llama_server_found": True,
                        "llama_server_help_ok": True,
                        "llama_server_supports_mmproj": True,
                        "llama_server_multimodal_markers": ["--mmproj", "mmproj"],
                    },
                ):
                    selection = select_preferred_variant(
                        variants_path=catalog,
                        primary_variant_id="qwen3-vl-2b-instruct-q4-smoke",
                        fallback_variant_id="qwen3-vl-2b-instruct-q8-smoke",
                        min_lfb_blocks=150,
                    )

        self.assertEqual(selection["selected_variant_id"], "qwen3-vl-2b-instruct-q8-smoke")
        self.assertEqual(selection["selected_reason"], "primary_blocked_selected_fallback")
        self.assertIn("missing_model_artifact", selection["candidates"][0]["block_reasons"])
        self.assertTrue(selection["candidates"][1]["usable"])
        self.assertTrue(selection["candidates"][1]["chosen"])

    def test_selector_returns_no_selection_when_both_lanes_fail_strict_gate(self):
        from edge_vlm.jetson_variant_selector import select_preferred_variant

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = _write_variant_catalog(tmp_path / "variants.jsonl", tmp_path / "models")
            repo_dir = tmp_path / "models" / "Qwen" / "Qwen3-VL-2B-Instruct-GGUF"
            repo_dir.mkdir(parents=True)
            for name in (
                "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "Qwen3VL-2B-Instruct-Q8_0.gguf",
            ):
                (repo_dir / name).write_bytes(b"GGUF")

            with patch(
                "edge_vlm.jetson_variant_selector.capture_preflight_sample",
                return_value=_preflight(125),
            ):
                with patch(
                    "edge_vlm.jetson_variant_selector._runtime_metadata",
                    return_value={
                        "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                        "llama_server_probe_ok": True,
                        "llama_server_found": True,
                        "llama_server_help_ok": True,
                        "llama_server_supports_mmproj": True,
                        "llama_server_multimodal_markers": ["--mmproj", "mmproj"],
                    },
                ):
                    selection = select_preferred_variant(
                        variants_path=catalog,
                        primary_variant_id="qwen3-vl-2b-instruct-q4-smoke",
                        fallback_variant_id="qwen3-vl-2b-instruct-q8-smoke",
                        min_lfb_blocks=150,
                    )

        self.assertIsNone(selection["selected_variant_id"])
        self.assertEqual(selection["selected_reason"], "no_usable_variant")
        self.assertIn("lfb_free_blocks 125 < required 150", selection["candidates"][0]["block_reasons"])
        self.assertIn("lfb_free_blocks 125 < required 150", selection["candidates"][1]["block_reasons"])

    def test_selector_can_use_relaxed_gate_for_fallback_lane(self):
        from edge_vlm.jetson_variant_selector import select_preferred_variant

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = _write_variant_catalog(tmp_path / "variants.jsonl", tmp_path / "models")
            repo_dir = tmp_path / "models" / "Qwen" / "Qwen3-VL-2B-Instruct-GGUF"
            repo_dir.mkdir(parents=True)
            for name in (
                "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                "Qwen3VL-2B-Instruct-Q8_0.gguf",
            ):
                (repo_dir / name).write_bytes(b"GGUF")

            with patch(
                "edge_vlm.jetson_variant_selector.capture_preflight_sample",
                return_value=_preflight(125),
            ):
                with patch(
                    "edge_vlm.jetson_variant_selector._runtime_metadata",
                    return_value={
                        "image": "ghcr.io/4everwz/jetson-llama-cpp:test",
                        "llama_server_probe_ok": True,
                        "llama_server_found": True,
                        "llama_server_help_ok": True,
                        "llama_server_supports_mmproj": True,
                        "llama_server_multimodal_markers": ["--mmproj", "mmproj"],
                    },
                ):
                    selection = select_preferred_variant(
                        variants_path=catalog,
                        primary_variant_id="qwen3-vl-2b-instruct-q4-smoke",
                        fallback_variant_id="qwen3-vl-2b-instruct-q8-smoke",
                        min_lfb_blocks=150,
                        fallback_min_lfb_blocks=100,
                    )

        self.assertEqual(selection["selected_variant_id"], "qwen3-vl-2b-instruct-q8-smoke")
        self.assertEqual(selection["selected_reason"], "primary_blocked_selected_fallback")
        self.assertIn("lfb_free_blocks 125 < required 150", selection["candidates"][0]["block_reasons"])
        self.assertEqual(selection["candidates"][1]["min_lfb_blocks"], 100)
        self.assertTrue(selection["candidates"][1]["usable"])


if __name__ == "__main__":
    unittest.main()
