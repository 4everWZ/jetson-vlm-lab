"""Jetson launcher artifact and download contract tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests._launcher_helpers import launcher_env, run_launcher, write_executable


class LauncherArtifactContractsTest(unittest.TestCase):

    def _write_mmproj_ready_docker(self, docker_path, docker_log):
        write_executable(
            docker_path,
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
                f"printf '%s\\n' \"$*\" > {str(docker_log)!r}",
            ],
        )

    def test_hf_gguf_launcher_accepts_completed_partial_after_resume_416(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            write_executable(
                bin_dir / "curl",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'curl: (22) The requested URL returned error: 416\\n' >&2",
                    "exit 22",
                ],
            )
            self._write_mmproj_ready_docker(bin_dir / "docker", docker_log)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "ggml-org" / "HunyuanOCR-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "HunyuanOCR-Q8_0.gguf").write_bytes(b"GGUFmodel")
            partial = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf.partial"
            partial.write_bytes(b"GGUFmmproj")

            result = run_launcher(
                "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                "--parallel",
                "1",
                env=launcher_env(
                    PATH=f"{bin_dir}:{os.environ['PATH']}",
                    DOCKER_TTY="0",
                    DOCKER_GPU_ARGS="",
                    LLAMA_CPP_DOCKER_IMAGE="unit/llama-cpp:test",
                    MODEL_DIR=model_dir,
                    MODEL_REF="ggml-org/HunyuanOCR-GGUF:Q8_0",
                    MODEL_FILE="HunyuanOCR-Q8_0.gguf",
                    MMPROJ_FILE="mmproj-HunyuanOCR-Q8_0.gguf",
                    MODEL_ALIAS="hunyuanocr-q8",
                ),
            )

            completed = repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf"
            docker_command = docker_log.read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(completed.is_file())
            self.assertFalse(partial.exists())
            self.assertIn("HTTP 416", result.stderr)
            self.assertIn("--mmproj /models/ggml-org/HunyuanOCR-GGUF/mmproj-HunyuanOCR-Q8_0.gguf", docker_command)

    def test_hf_gguf_text_launcher_rejects_cached_model_without_gguf_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            self._write_mmproj_ready_docker(bin_dir / "docker", docker_log)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "tencent" / "Hy-MT2-1.8B-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "Hy-MT2-1.8B-Q4_K_M.gguf").write_bytes(b"not-a-gguf")

            result = run_launcher(
                "scripts/jetson/run_hf_gguf_llama_docker.sh",
                env=launcher_env(
                    PATH=f"{bin_dir}:{os.environ['PATH']}",
                    DOCKER_GPU_ARGS="",
                    DOCKER_TTY="0",
                    LLAMA_CPP_DOCKER_IMAGE="unit/llama-cpp:test",
                    LLAMA_CPP_RUNTIME_PROBE_OUTPUT=tmp_path / "runtime-probe.json",
                    MODEL_DIR=model_dir,
                    MODEL_REF="tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
                    MODEL_FILE="Hy-MT2-1.8B-Q4_K_M.gguf",
                    MODEL_ALIAS="tencent-hy-mt2-1p8b-q4",
                ),
            )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("model GGUF failed magic check", result.stderr)
        self.assertIn("Hy-MT2-1.8B-Q4_K_M.gguf", result.stderr)

    def test_hf_gguf_vlm_launcher_rejects_cached_mmproj_without_gguf_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            self._write_mmproj_ready_docker(bin_dir / "docker", docker_log)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "ggml-org" / "HunyuanOCR-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "HunyuanOCR-Q8_0.gguf").write_bytes(b"GGUFmodel")
            (repo_dir / "mmproj-HunyuanOCR-Q8_0.gguf").write_bytes(b"not-a-gguf")

            result = run_launcher(
                "scripts/jetson/run_hf_gguf_vlm_llama_docker.sh",
                "--parallel",
                "1",
                env=launcher_env(
                    PATH=f"{bin_dir}:{os.environ['PATH']}",
                    DOCKER_TTY="0",
                    DOCKER_GPU_ARGS="",
                    LLAMA_CPP_DOCKER_IMAGE="unit/llama-cpp:test",
                    MODEL_DIR=model_dir,
                    MODEL_REF="ggml-org/HunyuanOCR-GGUF:Q8_0",
                    MODEL_FILE="HunyuanOCR-Q8_0.gguf",
                    MMPROJ_FILE="mmproj-HunyuanOCR-Q8_0.gguf",
                    MODEL_ALIAS="hunyuanocr-q8",
                ),
            )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("mmproj GGUF failed magic check", result.stderr)
        self.assertFalse(docker_log.exists())

    def test_minicpm_launcher_rejects_cached_model_without_gguf_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            docker_log = tmp_path / "docker.log"
            phase_log = tmp_path / "phase.jsonl"
            self._write_mmproj_ready_docker(bin_dir / "docker", docker_log)
            model_dir = tmp_path / "models"
            repo_dir = model_dir / "MiniCPM-V-4.6-gguf"
            repo_dir.mkdir(parents=True)
            (repo_dir / "MiniCPM-V-4_6-Q4_K_M.gguf").write_bytes(b"not-a-gguf")
            (repo_dir / "mmproj-model-f16.gguf").write_bytes(b"GGUFmmproj")

            result = run_launcher(
                "scripts/jetson/run_minicpmv46_llama_docker.sh",
                "--parallel",
                "1",
                env=launcher_env(
                    PATH=f"{bin_dir}:{os.environ['PATH']}",
                    DOCKER_TTY="0",
                    DOCKER_GPU_ARGS="",
                    LLAMA_CPP_DOCKER_IMAGE="unit/llama-cpp:test",
                    MODEL_DIR=model_dir,
                    EDGE_VLM_LAUNCH_PHASE_LOG=phase_log,
                ),
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("model GGUF failed magic check", result.stderr)
            self.assertFalse(docker_log.exists())
            phase_events = [json.loads(line) for line in phase_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(phase_events[-1]["phase"], "artifact_check_or_download")
            self.assertEqual(phase_events[-1]["details"]["status"], "invalid_model")

    def test_jetson_gemma_launcher_dry_run_does_not_create_model_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocked_parent = Path(tmp) / "blocked"
            blocked_parent.mkdir()
            blocked_parent.chmod(0o500)
            try:
                result = run_launcher(
                    "scripts/jetson/run_gemma4_e2b_llama_docker.sh",
                    env=launcher_env(
                        JETSON_DRY_RUN="1",
                        MODEL_DIR=blocked_parent / "models",
                    ),
                )
            finally:
                blocked_parent.chmod(0o700)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker run", result.stdout)
        self.assertFalse((blocked_parent / "models").exists())

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
        gguf_helper_path = Path("scripts/jetson/gguf_artifacts.sh")
        self.assertTrue(helper_path.is_file())
        self.assertTrue(gguf_helper_path.is_file())
        helper = helper_path.read_text(encoding="utf-8")
        gguf_helper = gguf_helper_path.read_text(encoding="utf-8")
        self.assertIn("download_hf_file", helper)
        self.assertIn("download_hf_gguf_file", helper)
        self.assertIn("require_gguf_artifact", gguf_helper)
        self.assertIn("accepting existing GGUF partial as complete", helper)

        launcher_paths = [
            Path("scripts/jetson/run_hf_gguf_vlm_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_llama_docker.sh"),
        ]
        for launcher_path in launcher_paths:
            launcher = launcher_path.read_text(encoding="utf-8")
            self.assertIn("hf_artifacts.sh", launcher, str(launcher_path))
            self.assertIn("download_hf_gguf_file", launcher, str(launcher_path))
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
                        result = run_launcher(
                            launcher_path,
                            env=launcher_env(
                                JETSON_DRY_RUN="1",
                                MODEL_DIR=blocked_parent / "models",
                                DOCKER_TTY="0",
                            ),
                        )
                    finally:
                        blocked_parent.chmod(0o700)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("docker run", result.stdout)
                self.assertFalse((blocked_parent / "models").exists())


if __name__ == "__main__":
    unittest.main()
