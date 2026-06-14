"""Jetson launcher and Hugging Face GGUF artifact contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path



class LauncherArtifactContractsTest(unittest.TestCase):

    def test_jetson_hf_gguf_vlm_launcher_can_dry_run_model_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "MODEL_REF": "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0",
                "MODEL_FILE": "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "MMPROJ_FILE": "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                "MODEL_ALIAS": "smolvlm2-256m-q8",
                "CTX_SIZE": "512",
                "N_GPU_LAYERS": "99",
                "VLM_SERVER_PORT": "19101",
            }
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                    "--parallel",
                    "1",
                    "--batch-size",
                    "128",
                    "--ubatch-size",
                    "32",
                    "--no-warmup",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
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

    def test_hf_gguf_launcher_accepts_completed_partial_after_resume_416(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            fake_curl = bin_dir / "curl"
            fake_docker = bin_dir / "docker"
            fake_curl.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'curl: (22) The requested URL returned error: 416\\n' >&2",
                        "exit 22",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_docker.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "if [[ \"${1:-}\" == \"image\" && \"${2:-}\" == \"inspect\" ]]; then",
                        "  cat <<'JSON'",
                        json.dumps(
                            [
                                {
                                    "Id": "sha256:unit-test",
                                    "Created": "2026-06-14T00:00:00+00:00",
                                    "RepoDigests": [],
                                    "Config": {
                                        "Labels": {
                                            "org.opencontainers.image.version": "unit-ref",
                                            "org.opencontainers.image.source": "https://github.com/ggml-org/llama.cpp",
                                        }
                                    },
                                }
                            ]
                        ),
                        "JSON",
                        "  exit 0",
                        "fi",
                        "if [[ \"${1:-}\" == \"run\" && \"$*\" == *\"--entrypoint /bin/bash\"* ]]; then",
                        "  cat <<'PROBE'",
                        "llama_server_found=1",
                        "llama_server_path=/usr/local/bin/llama-server",
                        "llama_server_help_ok=1",
                        "llama_server_supports_mmproj=1",
                        "llama_server_multimodal_markers=--mmproj,mmproj",
                        "PROBE",
                        "  exit 0",
                        "fi",
                        "printf '%s\\n' \"$*\" > \"${FAKE_DOCKER_LOG:?}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_curl, 0o755)
            os.chmod(fake_docker, 0o755)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "ggml-org" / "HunyuanOCR-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "HunyuanOCR-Q8_0.gguf").write_bytes(b"GGUFmodel")
            partial = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf.partial"
            partial.write_bytes(b"GGUFmmproj")

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                    "--parallel",
                    "1",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_DOCKER_LOG": str(docker_log),
                    "DOCKER_TTY": "0",
                    "DOCKER_GPU_ARGS": "",
                    "LLAMA_CPP_DOCKER_IMAGE": "unit/llama-cpp:test",
                    "MODEL_DIR": str(model_dir),
                    "MODEL_REF": "ggml-org/HunyuanOCR-GGUF:Q8_0",
                    "MODEL_FILE": "HunyuanOCR-Q8_0.gguf",
                    "MMPROJ_FILE": "mmproj-HunyuanOCR-Q8_0.gguf",
                    "MODEL_ALIAS": "hunyuanocr-q8",
                },
            )

            completed = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf"
            docker_command = docker_log.read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(completed.is_file())
            self.assertFalse(partial.exists())
            self.assertIn("HTTP 416", result.stderr)
            self.assertIn("--mmproj /models/ggml-org/HunyuanOCR-GGUF/mmproj-HunyuanOCR-Q8_0.gguf", docker_command)

    def test_jetson_hf_gguf_text_launcher_can_dry_run_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "MODEL_REF": "tencent/Hy-MT2-1.8B-1.25Bit-GGUF:1.25Bit",
                "MODEL_FILE": "Hy-MT2-1.8B-1.25Bit.gguf",
                "MODEL_ALIAS": "tencent-hy-mt2-1p8b-1p25bit",
                "CTX_SIZE": "1024",
                "N_GPU_LAYERS": "99",
                "VLM_SERVER_PORT": "19102",
            }
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_hf_gguf_llama_docker.sh",
                    "--parallel",
                    "1",
                    "--batch-size",
                    "128",
                    "--ubatch-size",
                    "32",
                    "--no-warmup",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
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
            result = subprocess.run(
                ["bash", "scripts/jetson/run_hf_gguf_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_DRY_RUN": "1",
                    "DOCKER_TTY": "0",
                    "MODEL_DIR": str(Path(tmp) / "models"),
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tencent/Hy-MT2-1.8B-GGUF", result.stdout)
        self.assertIn("Hy-MT2-1.8B-Q4_K_M.gguf", result.stdout)
        self.assertIn("--alias tencent-hy-mt2-1p8b-q4", result.stdout)

    def test_jetson_gemma_launcher_can_dry_run_without_docker_or_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "VLM_SERVER_PORT": "19090",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("ggml-org/gemma-4-E2B-it-GGUF:Q8_0", result.stdout)
        self.assertIn("-p 19090:8080", result.stdout)

    def test_jetson_gemma_launcher_dry_run_does_not_create_model_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocked_parent = Path(tmp) / "blocked"
            blocked_parent.mkdir()
            blocked_parent.chmod(0o500)
            try:
                env = {
                    **os.environ,
                    "JETSON_DRY_RUN": "1",
                    "MODEL_DIR": str(blocked_parent / "models"),
                }
                result = subprocess.run(
                    ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh"],
                    check=False,
                    capture_output=True,
                    encoding="utf-8",
                    env=env,
                )
            finally:
                blocked_parent.chmod(0o700)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertFalse((blocked_parent / "models").exists())

    def test_jetson_launcher_allows_explicit_llama_cpp_image_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(Path(tmp) / "models"),
                "LLAMA_CPP_DOCKER_IMAGE": "dustynv/llama_cpp:b5283-r36.4-cu128-24.04",
                "LLAMA_SERVER_CMD": "/usr/local/bin/llama-server",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dustynv/llama_cpp:b5283-r36.4-cu128-24.04", result.stdout)
        self.assertIn("/usr/local/bin/llama-server", result.stdout)

    def test_jetson_image_resolver_defaults_to_verified_multimodal_image_even_with_autotag(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            autotag = fake_bin / "autotag"
            autotag.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == \"llama_cpp\" ]]; then\n"
                "  printf '%s\\n' 'dustynv/llama_cpp:r36.4.0'\n"
                "fi\n",
                encoding="utf-8",
            )
            autotag.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            }
            env.pop("LLAMA_CPP_DOCKER_IMAGE", None)
            env.pop("LLAMA_CPP_DOCKER_IMAGE_FALLBACK", None)
            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "source scripts/jetson/resolve_llama_cpp_image.sh; resolve_llama_cpp_image",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87",
        )

    def test_jetson_gemma_launcher_dry_run_allows_explicit_missing_model_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(model_dir),
                "LLAMA_CPP_DOCKER_IMAGE": "ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87",
                "MODEL_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.Q4_K_M.gguf"),
                "MMPROJ_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "gemma-4-E2B-it.mmproj-Q8_0.gguf"),
                "MODEL_ALIAS": "gemma4-e2b-it-q4",
                "CTX_SIZE": "512",
                "N_GPU_LAYERS": "12",
            }
            result = subprocess.run(
                [
                    "bash",
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
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("-m /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf", result.stdout)
        self.assertIn("--batch-size 512", result.stdout)

    def test_jetson_minicpm_launcher_can_dry_run_without_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "MODEL_DIR": str(model_dir),
                "VLM_SERVER_PORT": "19091",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_minicpmv46_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertIn("ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87", result.stdout)
        self.assertIn("/bin/bash -lc", result.stdout)
        self.assertIn("-m /models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf", result.stdout)
        self.assertIn("--mmproj /models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf", result.stdout)

    def test_jetson_launchers_can_disable_docker_tty_for_automation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "JETSON_DRY_RUN": "1",
                "DOCKER_TTY": "0",
                "MODEL_DIR": str(Path(tmp) / "models"),
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_minicpmv46_llama_docker.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertNotIn("-it", result.stdout)

    def test_jetson_launchers_record_optional_artifact_phase_logs(self):
        phase_helper = Path("scripts/jetson/phase_logging.sh").read_text(encoding="utf-8")
        self.assertIn("EDGE_VLM_LAUNCH_PHASE_LOG", phase_helper)
        self.assertIn("write_launch_phase", phase_helper)
        launcher_paths = [
            Path("scripts/jetson/run_minicpmv46_llama_docker.sh"),
            Path("scripts/jetson/run_gemma4_e2b_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_vlm_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_llama_docker.sh"),
        ]

        for launcher_path in launcher_paths:
            launcher = launcher_path.read_text(encoding="utf-8")
            self.assertIn("phase_logging.sh", launcher, str(launcher_path))
            self.assertIn("artifact_check_or_download", launcher, str(launcher_path))
            self.assertIn("write_launch_phase", launcher, str(launcher_path))

    def test_hf_gguf_launchers_share_artifact_download_helper(self):
        helper_path = Path("scripts/jetson/hf_artifacts.sh")
        self.assertTrue(helper_path.is_file())
        helper = helper_path.read_text(encoding="utf-8")
        self.assertIn("download_hf_file", helper)
        self.assertIn("accepting existing GGUF partial as complete", helper)

        launcher_paths = [
            Path("scripts/jetson/run_hf_gguf_vlm_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_llama_docker.sh"),
        ]
        for launcher_path in launcher_paths:
            launcher = launcher_path.read_text(encoding="utf-8")
            self.assertIn("hf_artifacts.sh", launcher, str(launcher_path))
            self.assertNotIn("download_hf_file() {", launcher, str(launcher_path))

    def test_hf_gguf_launcher_dry_runs_do_not_create_model_directories(self):
        launcher_paths = [
            "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
            "scripts/jetson/run_hf_gguf_llama_docker.sh",
        ]
        for launcher_path in launcher_paths:
            with self.subTest(launcher_path=launcher_path):
                with tempfile.TemporaryDirectory() as tmp:
                    blocked_parent = Path(tmp) / "blocked"
                    blocked_parent.mkdir()
                    blocked_parent.chmod(0o500)
                    try:
                        env = {
                            **os.environ,
                            "JETSON_DRY_RUN": "1",
                            "MODEL_DIR": str(blocked_parent / "models"),
                            "DOCKER_TTY": "0",
                        }
                        result = subprocess.run(
                            ["bash", launcher_path],
                            check=False,
                            capture_output=True,
                            encoding="utf-8",
                            env=env,
                        )
                    finally:
                        blocked_parent.chmod(0o700)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("docker run", result.stdout)
                self.assertFalse((blocked_parent / "models").exists())


if __name__ == "__main__":
    unittest.main()
