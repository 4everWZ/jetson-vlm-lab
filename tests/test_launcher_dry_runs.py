"""Jetson launcher dry-run command contract tests."""

import tempfile
import unittest
from pathlib import Path

from tests._launcher_helpers import launcher_env, run_launcher


class LauncherDryRunCommandContractsTest(unittest.TestCase):

    def test_jetson_hf_gguf_vlm_launcher_can_dry_run_model_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = launcher_env(
                JETSON_DRY_RUN="1",
                DOCKER_TTY="0",
                MODEL_DIR=Path(tmp) / "models",
                MODEL_REF="ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0",
                MODEL_FILE="SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                MMPROJ_FILE="mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                MODEL_ALIAS="smolvlm2-256m-q8",
                CTX_SIZE="512",
                N_GPU_LAYERS="99",
                VLM_SERVER_PORT="19101",
            )
            result = run_launcher(
                "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                "--parallel",
                "1",
                "--batch-size",
                "128",
                "--ubatch-size",
                "32",
                "--no-warmup",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("-m /models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/SmolVLM2-256M-Video-Instruct-Q8_0.gguf", result.stdout)
        self.assertIn("--mmproj /models/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf", result.stdout)
        self.assertNotIn("-hf ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0", result.stdout)
        self.assertIn("--alias smolvlm2-256m-q8", result.stdout)
        self.assertIn("-p 19101:8080", result.stdout)
        self.assertIn("-c 512", result.stdout)
        self.assertIn("--n-gpu-layers 99", result.stdout)
        self.assertIn("--batch-size 128", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_jetson_hf_gguf_text_launcher_can_dry_run_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = launcher_env(
                JETSON_DRY_RUN="1",
                DOCKER_TTY="0",
                MODEL_DIR=Path(tmp) / "models",
                MODEL_REF="tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit",
                MODEL_FILE="Hy-MT2-1.8B-1.25Bit.gguf",
                MODEL_ALIAS="tencent-hy-mt2-1p8b-1p25bit",
                CTX_SIZE="1024",
                N_GPU_LAYERS="99",
                VLM_SERVER_PORT="19102",
            )
            result = run_launcher(
                "scripts/jetson/run_hf_gguf_llama_docker.sh",
                "--parallel",
                "1",
                "--batch-size",
                "128",
                "--ubatch-size",
                "32",
                "--no-warmup",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("-m /models/tencent/Hy-MT2-1.8B-1.25Bit-GGUF/Hy-MT2-1.8B-1.25Bit.gguf", result.stdout)
        self.assertNotIn("--mmproj", result.stdout)
        self.assertNotIn("-hf tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit", result.stdout)
        self.assertIn("--alias tencent-hy-mt2-1p8b-1p25bit", result.stdout)
        self.assertIn("-p 19102:8080", result.stdout)
        self.assertIn("-c 1024", result.stdout)
        self.assertIn("--n-gpu-layers 99", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_jetson_hf_gguf_text_launcher_defaults_to_verified_hy_mt2_q4(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_launcher(
                "scripts/jetson/run_hf_gguf_llama_docker.sh",
                env=launcher_env(
                    JETSON_DRY_RUN="1",
                    DOCKER_TTY="0",
                    MODEL_DIR=Path(tmp) / "models",
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tencent/Hy-MT2-1.8B-GGUF", result.stdout)
        self.assertIn("Hy-MT2-1.8B-Q4_K_M.gguf", result.stdout)
        self.assertIn("--alias tencent-hy-mt2-1p8b-q4", result.stdout)

    def test_jetson_gemma_launcher_can_dry_run_without_docker_or_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_launcher(
                "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
                env=launcher_env(
                    JETSON_DRY_RUN="1",
                    MODEL_DIR=Path(tmp) / "models",
                    VLM_SERVER_PORT="19090",
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("ggml-org/gemma-4-E2B-it-GGUF:Q8_0", result.stdout)
        self.assertIn("-p 19090:8080", result.stdout)

    def test_jetson_launcher_allows_explicit_llama_cpp_image_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_launcher(
                "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
                env=launcher_env(
                    JETSON_DRY_RUN="1",
                    MODEL_DIR=Path(tmp) / "models",
                    LLAMA_CPP_DOCKER_IMAGE="dustynv/llama_cpp:b5283-r36.4-cu128-24.04",
                    LLAMA_SERVER_CMD="/usr/local/bin/llama-server",
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dustynv/llama_cpp:b5283-r36.4-cu128-24.04", result.stdout)
        self.assertIn("/usr/local/bin/llama-server", result.stdout)

    def test_jetson_gemma_launcher_dry_run_allows_explicit_missing_model_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            env = launcher_env(
                JETSON_DRY_RUN="1",
                MODEL_DIR=model_dir,
                LLAMA_CPP_DOCKER_IMAGE="ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87",
                MODEL_PATH=model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.Q4_K_M.gguf",
                MMPROJ_PATH=model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.mmproj-Q8_0.gguf",
                MODEL_ALIAS="gemma4-e2b-it-q4",
                CTX_SIZE="512",
                N_GPU_LAYERS="12",
            )
            result = run_launcher(
                "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
                "-fit",
                "off",
                "--parallel",
                "1",
                "--batch-size",
                "512",
                "--ubatch-size",
                "512",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--no-warmup",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("-m /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf", result.stdout)
        self.assertIn("--batch-size 512", result.stdout)

    def test_jetson_minicpm_launcher_can_dry_run_without_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_launcher(
                "scripts/jetson/run_minicpmv46_llama_docker.sh",
                env=launcher_env(
                    JETSON_DRY_RUN="1",
                    MODEL_DIR=Path(tmp) / "models",
                    VLM_SERVER_PORT="19091",
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("-m /models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf", result.stdout)

    def test_jetson_launchers_can_disable_docker_tty_for_automation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_launcher(
                "scripts/jetson/run_minicpmv46_llama_docker.sh",
                env=launcher_env(
                    JETSON_DRY_RUN="1",
                    DOCKER_TTY="0",
                    MODEL_DIR=Path(tmp) / "models",
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_launcher_dry_run_preserves_explicit_empty_docker_gpu_args(self):
        launcher_paths = [
            "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
            "scripts/jetson/run_minicpmv46_llama_docker.sh",
            "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
            "scripts/jetson/run_hf_gguf_llama_docker.sh",
        ]
        for launcher_path in launcher_paths:
            with self.subTest(launcher_path=launcher_path):
                with tempfile.TemporaryDirectory() as tmp:
                    result = run_launcher(
                        launcher_path,
                        "--parallel",
                        "1",
                        env=launcher_env(
                            JETSON_DRY_RUN="1",
                            DOCKER_GPU_ARGS="",
                            DOCKER_TTY="0",
                            MODEL_DIR=Path(tmp) / "models",
                        ),
                    )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("docker run", result.stdout)
                self.assertNotIn("--runtime nvidia", result.stdout)


if __name__ == "__main__":
    unittest.main()
