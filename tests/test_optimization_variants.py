"""Jetson optimization variant catalog contract tests."""

import json
import unittest
from pathlib import Path


VARIANT_CATALOG = Path("configs/benchmark/jetson_optimization_variants.jsonl")


def load_variant_catalog_by_id():
    variants = [
        json.loads(line)
        for line in VARIANT_CATALOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {variant["id"]: variant for variant in variants}


class JetsonOptimizationVariantsContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants_by_id = load_variant_catalog_by_id()

    def test_jetson_optimization_variants_include_gemma_mid_batch_candidate(self):
        by_id = self.variants_by_id

        candidate = by_id["gemma-q4-gpu12-b384-u384-kvq8"]

        self.assertEqual(candidate["model"], "gemma4-e2b-it-q4")
        self.assertEqual(candidate["env"]["N_GPU_LAYERS"], 12)
        self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], 384)
        self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], 384)
        self.assertIn("--batch-size", candidate["args"])
        self.assertIn("384", candidate["args"])
        self.assertIn("--cache-type-k", candidate["args"])
        self.assertIn("q8_0", candidate["args"])

    def test_jetson_optimization_variants_include_flash_attention_candidates(self):
        by_id = self.variants_by_id

        minicpm = by_id["minicpm-q4-baseline-b128-u32-kvq8-faon"]
        gemma = by_id["gemma-q4-baseline-gpu12-b512-u512-kvq8-faon"]

        for candidate in (minicpm, gemma):
            self.assertIn("--flash-attn", candidate["args"])
            flag_index = candidate["args"].index("--flash-attn")
            self.assertEqual(candidate["args"][flag_index + 1], "on")
            self.assertIn("--cache-type-k", candidate["args"])
            self.assertIn("--cache-type-v", candidate["args"])

        self.assertEqual(minicpm["env"]["LLAMA_BATCH_SIZE"], 128)
        self.assertEqual(minicpm["env"]["LLAMA_UBATCH_SIZE"], 32)
        self.assertEqual(gemma["env"]["LLAMA_BATCH_SIZE"], 512)
        self.assertEqual(gemma["env"]["LLAMA_UBATCH_SIZE"], 512)

    def test_jetson_optimization_variants_include_memory_mapping_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-mlock": ("minicpmv46-q4", 128, 32, "--mlock"),
            "minicpm-q4-baseline-b128-u32-kvq8-nommap": ("minicpmv46-q4", 128, 32, "--no-mmap"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock": ("gemma4-e2b-it-q4", 512, 512, "--mlock"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nommap": ("gemma4-e2b-it-q4", 512, 512, "--no-mmap"),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, candidate["args"])
                self.assertIn("--cache-type-k", candidate["args"])
                self.assertIn("--cache-type-v", candidate["args"])

    def test_jetson_optimization_variants_include_mlock_ulimit_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-mlock-ulimit": ("minicpmv46-q4", 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-mlock-ulimit": ("gemma4-e2b-it-q4", 512, 512),
        }

        for variant_id, (model, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertEqual(candidate["env"]["DOCKER_GPU_ARGS"], "--runtime nvidia --ulimit memlock=-1:-1")
                self.assertIn("--mlock", candidate["args"])

    def test_jetson_optimization_variants_include_cache_precision_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kq4-vq8": ("minicpmv46-q4", 128, 32, "q4_0", "q8_0"),
            "minicpm-q4-baseline-b128-u32-kvq4": ("minicpmv46-q4", 128, 32, "q4_0", "q4_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kq4-vq8": ("gemma4-e2b-it-q4", 512, 512, "q4_0", "q8_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq4": ("gemma4-e2b-it-q4", 512, 512, "q4_0", "q4_0"),
        }

        for variant_id, (model, batch_size, ubatch_size, cache_k, cache_v) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertEqual(args[args.index("--cache-type-k") + 1], cache_k)
                self.assertEqual(args[args.index("--cache-type-v") + 1], cache_v)

    def test_jetson_optimization_variants_include_gemma_flash_attention_cache_precision_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "gemma-q4-baseline-gpu12-b512-u512-kq4-vq8-faon": ("q4_0", "q8_0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq4-faon": ("q4_0", "q4_0"),
        }

        for variant_id, (cache_k, cache_v) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], "gemma4-e2b-it-q4")
                self.assertEqual(candidate["env"]["N_GPU_LAYERS"], 12)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], 512)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], 512)
                self.assertEqual(args[args.index("--cache-type-k") + 1], cache_k)
                self.assertEqual(args[args.index("--cache-type-v") + 1], cache_v)
                self.assertEqual(args[args.index("--flash-attn") + 1], "on")

    def test_jetson_optimization_variants_include_no_cont_batching_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-nocb": ("minicpmv46-q4", 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nocb": ("gemma4-e2b-it-q4", 512, 512),
        }

        for variant_id, (model, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn("--no-cont-batching", candidate["args"])
                self.assertIn("--cache-type-k", candidate["args"])
                self.assertIn("--cache-type-v", candidate["args"])

    def test_jetson_optimization_variants_include_prompt_cache_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-cache-ram0": ("minicpmv46-q4", 128, 32, "--cache-ram", "0"),
            "minicpm-q4-baseline-b128-u32-kvq8-nocacheprompt": ("minicpmv46-q4", 128, 32, "--no-cache-prompt", None),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-cache-ram0": ("gemma4-e2b-it-q4", 512, 512, "--cache-ram", "0"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nocacheprompt": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--no-cache-prompt",
                None,
            ),
        }

        for variant_id, (model, batch_size, ubatch_size, flag, value) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                if value is not None:
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_host_repack_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-nohost": ("minicpmv46-q4", 128, 32, "--no-host"),
            "minicpm-q4-baseline-b128-u32-kvq8-norepack": ("minicpmv46-q4", 128, 32, "--no-repack"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nohost": ("gemma4-e2b-it-q4", 512, 512, "--no-host"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-norepack": ("gemma4-e2b-it-q4", 512, 512, "--no-repack"),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_direct_io_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-directio": ("minicpmv46-q4", 128, 32, "--direct-io"),
            "minicpm-q4-baseline-b128-u32-kvq8-nodirectio": ("minicpmv46-q4", 128, 32, "--no-direct-io"),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-directio": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--direct-io",
            ),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-nodirectio": (
                "gemma4-e2b-it-q4",
                512,
                512,
                "--no-direct-io",
            ),
        }

        for variant_id, (model, batch_size, ubatch_size, flag) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn(flag, args)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)

    def test_jetson_optimization_variants_include_warmup_candidates(self):
        by_id = self.variants_by_id

        expected = {
            "minicpm-q4-baseline-b128-u32-kvq8-warmup": ("minicpmv46-q4", 32, 128, 32),
            "gemma-q4-baseline-gpu12-b512-u512-kvq8-warmup": ("gemma4-e2b-it-q4", 12, 512, 512),
        }

        for variant_id, (model, n_gpu_layers, batch_size, ubatch_size) in expected.items():
            with self.subTest(variant_id=variant_id):
                candidate = by_id[variant_id]
                args = candidate["args"]
                self.assertEqual(candidate["model"], model)
                self.assertEqual(candidate["env"]["N_GPU_LAYERS"], n_gpu_layers)
                self.assertEqual(candidate["env"]["LLAMA_BATCH_SIZE"], batch_size)
                self.assertEqual(candidate["env"]["LLAMA_UBATCH_SIZE"], ubatch_size)
                self.assertIn("--cache-type-k", args)
                self.assertIn("--cache-type-v", args)
                self.assertNotIn("--no-warmup", args)


if __name__ == "__main__":
    unittest.main()
