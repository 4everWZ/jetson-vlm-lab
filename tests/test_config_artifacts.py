"""Configuration, WSL wrapper, and llama.cpp artifact contract tests."""

import unittest
from pathlib import Path



class ConfigArtifactContractsTest(unittest.TestCase):

    def test_gemma_q8_config_is_available_for_low_memory_wsl_path(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/gemma4_e2b_q8.yaml")

        self.assertEqual(config["model"]["name"], "gemma4-e2b-it-q8")
        self.assertEqual(config["model"]["quantization"], "Q8_0")
        self.assertTrue(config["capabilities"]["image"])

    def test_gemma_q8_wsl_defaults_stay_within_observed_low_memory_smoke_path(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/gemma4_e2b_q8.yaml")
        launch_script = Path("scripts/wsl/run_gemma4_e2b_llama.sh").read_text(encoding="utf-8")

        self.assertEqual(config["runtime"]["ctx_size"], 512)
        self.assertEqual(config["runtime"]["n_gpu_layers"], 0)
        self.assertIn('ctx_size="${CTX_SIZE:-512}"', launch_script)
        self.assertIn('llama_threads="${LLAMA_THREADS:-2}"', launch_script)
        self.assertIn('llama_parallel="${LLAMA_PARALLEL:-1}"', launch_script)
        self.assertIn('llama_batch_size="${LLAMA_BATCH_SIZE:-128}"', launch_script)
        self.assertIn('llama_ubatch_size="${LLAMA_UBATCH_SIZE:-32}"', launch_script)

    def test_gemma_q4_prepare_downloads_prequantized_artifacts(self):
        prepare_script = Path("scripts/wsl/prepare_gemma4_e2b_q4.sh").read_text(encoding="utf-8")
        config_text = Path("configs/models/gemma4_e2b_q4.yaml").read_text(encoding="utf-8")

        self.assertIn("mradermacher/gemma-4-E2B-it-GGUF", prepare_script)
        self.assertIn("gemma-4-E2B-it.Q4_K_M.gguf", prepare_script)
        self.assertIn("gemma-4-E2B-it.mmproj-Q8_0.gguf", prepare_script)
        self.assertIn("hf_hub_download", prepare_script)
        self.assertNotIn("llama-quantize", prepare_script)
        self.assertNotIn("ALLOW_HIGH_MEMORY_QUANTIZE", prepare_script)
        self.assertNotIn("ALLOW_Q8_REQUANTIZE", prepare_script)
        self.assertIn("prequantized_gguf", config_text)

    def test_wsl_cuda_wrappers_use_separate_cuda_build_and_gpu_layers(self):
        cuda_build_script = Path("scripts/wsl/build_llama_cpp_cuda.sh").read_text(encoding="utf-8")
        gemma_cuda_script = Path("scripts/wsl/run_gemma4_e2b_llama_cuda.sh").read_text(encoding="utf-8")
        minicpm_cuda_script = Path("scripts/wsl/run_minicpmv46_llama_cuda.sh").read_text(encoding="utf-8")

        self.assertIn('ENABLE_CUDA="${ENABLE_CUDA:-1}"', cuda_build_script)
        self.assertIn('LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_dir}/build-cuda}"', cuda_build_script)
        self.assertIn('BUILD_JOBS="${BUILD_JOBS:-8}"', cuda_build_script)
        self.assertIn('CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"', cuda_build_script)
        self.assertIn("nvcc", cuda_build_script)
        self.assertIn('LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build-cuda/bin/llama-server}"', gemma_cuda_script)
        self.assertIn('N_GPU_LAYERS="${N_GPU_LAYERS:-32}"', gemma_cuda_script)
        self.assertIn('LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-512}"', gemma_cuda_script)
        self.assertIn('LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}"', gemma_cuda_script)
        self.assertIn('scripts/wsl/run_gemma4_e2b_llama.sh', gemma_cuda_script)
        self.assertIn('LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${llama_cpp_dir}/build-cuda/bin/llama-server}"', minicpm_cuda_script)
        self.assertIn('N_GPU_LAYERS="${N_GPU_LAYERS:-32}"', minicpm_cuda_script)
        self.assertIn('LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-128}"', minicpm_cuda_script)
        self.assertIn('LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-32}"', minicpm_cuda_script)
        self.assertIn('scripts/wsl/run_minicpmv46_llama.sh', minicpm_cuda_script)

    def test_llama_cpp_artifact_builder_prefers_direct_docker_when_available(self):
        artifact_builder = Path("scripts/build_llama_cpp_artifacts.sh").read_text(encoding="utf-8")

        self.assertIn('BUILD_JOBS="${BUILD_JOBS:-6}"', artifact_builder)
        self.assertIn('BUILD_LOG_DIR="${BUILD_LOG_DIR:-$PWD/outputs/build/llama-cpp}"', artifact_builder)
        self.assertIn('DOCKER_BIN="${DOCKER_BIN:-}"', artifact_builder)
        self.assertIn("docker ps >/dev/null 2>&1", artifact_builder)
        self.assertIn("DOCKER_CMD=(docker)", artifact_builder)
        self.assertIn("DOCKER_CMD=(sudo docker)", artifact_builder)
        self.assertIn('-v "$BUILD_LOG_DIR:/build-logs"', artifact_builder)
        self.assertIn("llama-server.help.txt", artifact_builder)
        self.assertIn("copied-files.txt", artifact_builder)
        self.assertIn('"${DOCKER_CMD[@]}" run --rm', artifact_builder)

    def test_llama_cpp_image_builder_records_ref_in_tag_and_labels(self):
        image_builder = Path("scripts/build_llama_cpp_image.sh").read_text(encoding="utf-8")

        self.assertIn('LLAMA_CPP_REF="${LLAMA_CPP_REF:-', image_builder)
        self.assertIn('BUILD_LOG_DIR="${BUILD_LOG_DIR:-$PWD/outputs/build/llama-cpp}"', image_builder)
        self.assertIn('IMAGE_TAG="${IMAGE_TAG:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-${LLAMA_CPP_REF:0:7}}"', image_builder)
        self.assertIn("--build-arg", image_builder)
        self.assertIn("LLAMA_CPP_REF=${LLAMA_CPP_REF}", image_builder)
        self.assertIn("image-manifest", image_builder)
        self.assertIn("artifact-files", image_builder)
        self.assertIn("-f docker/llama-cpp/Dockerfile", image_builder)

    def test_llama_cpp_image_builder_rejects_unexpected_artifact_files(self):
        image_builder = Path("scripts/build_llama_cpp_image.sh").read_text(encoding="utf-8")

        self.assertIn("allowed_artifact_files=(", image_builder)
        self.assertIn("LLAMA_CPP_REF", image_builder)
        self.assertIn("bin/llama-server", image_builder)
        self.assertIn("bin/llama-mtmd-cli", image_builder)
        self.assertIn("Unexpected file in artifacts/llama.cpp-install", image_builder)

    def test_llama_cpp_docker_context_excludes_local_state_and_keeps_artifacts(self):
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

        self.assertIn("**", dockerignore)
        self.assertIn("!.dockerignore", dockerignore)
        self.assertIn("!docker/llama-cpp/Dockerfile", dockerignore)
        self.assertIn("!artifacts/llama.cpp-install/**", dockerignore)
        self.assertNotIn("!.env", dockerignore)
        self.assertNotIn("!models/", dockerignore)
        self.assertNotIn("!outputs/", dockerignore)

    def test_llama_cpp_build_artifacts_are_not_git_tracked(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("artifacts/", gitignore)

    def test_minicpm_prepare_downloads_official_prebuilt_artifacts(self):
        prepare_script = Path("scripts/wsl/prepare_minicpmv46_q4.sh").read_text(encoding="utf-8")
        config_text = Path("configs/models/minicpmv46_q4.yaml").read_text(encoding="utf-8")

        self.assertIn("openbmb/MiniCPM-V-4.6-gguf", prepare_script)
        self.assertIn("MiniCPM-V-4_6-Q4_K_M.gguf", prepare_script)
        self.assertIn("mmproj-model-f16.gguf", prepare_script)
        self.assertIn("hf_hub_download", prepare_script)
        self.assertNotIn("ALLOW_MINICPM_FULL_PREPARE", prepare_script)
        self.assertNotIn("convert_hf_to_gguf", prepare_script)
        self.assertNotIn("llama-quantize", prepare_script)
        self.assertIn("prequantized_gguf", config_text)

    def test_minicpm_inspection_script_does_not_download_model_files(self):
        inspect_script = Path("scripts/wsl/inspect_minicpmv46_hf.sh").read_text(encoding="utf-8")

        self.assertIn("model_info", inspect_script)
        self.assertIn("files_metadata=True", inspect_script)
        self.assertIn("openbmb/MiniCPM-V-4.6-gguf", inspect_script)
        self.assertIn("scripts/wsl/prepare_minicpmv46_q4.sh", inspect_script)
        self.assertNotIn("snapshot_download", inspect_script)
        self.assertNotIn("hf_hub_download", inspect_script)
        self.assertNotIn("convert_hf_to_gguf", inspect_script)

    def test_minicpm_config_uses_prebuilt_wsl_defaults(self):
        from edge_vlm.config import load_model_config

        config = load_model_config("configs/models/minicpmv46_q4.yaml")
        launch_script = Path("scripts/wsl/run_minicpmv46_llama.sh").read_text(encoding="utf-8")

        self.assertEqual(config["runtime"]["ctx_size"], 512)
        self.assertEqual(config["runtime"]["n_gpu_layers"], 0)
        self.assertIn('ctx_size="${CTX_SIZE:-512}"', launch_script)
        self.assertIn('llama_threads="${LLAMA_THREADS:-2}"', launch_script)
        self.assertIn('llama_parallel="${LLAMA_PARALLEL:-1}"', launch_script)
        self.assertIn('llama_batch_size="${LLAMA_BATCH_SIZE:-128}"', launch_script)
        self.assertIn('llama_ubatch_size="${LLAMA_UBATCH_SIZE:-32}"', launch_script)
        self.assertIn("pre-built", config["notes"]["status"].lower())


if __name__ == "__main__":
    unittest.main()
