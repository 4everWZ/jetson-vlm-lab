"""Direct Jetson VLM launcher runtime-gate contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class VlmLauncherRuntimeGateContractsTest(unittest.TestCase):
    def _write_fake_docker(self, fake_bin: Path, *, supports_mmproj: bool) -> None:
        fake_docker = fake_bin / "docker"
        markers = "--mmproj,mmproj" if supports_mmproj else "mmproj"
        fake_docker.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf '%s\\n' \"$*\" >> \"${DOCKER_LOG:?}\"",
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
                    "if [[ \"${1:-}\" == \"run\" ]]; then",
                    "  cat <<'PROBE'",
                    "llama_server_found=1",
                    "llama_server_path=/usr/local/bin/llama-server",
                    "llama_server_help_ok=1",
                    f"llama_server_supports_mmproj={1 if supports_mmproj else 0}",
                    f"llama_server_multimodal_markers={markers}",
                    "PROBE",
                    "  exit 0",
                    "fi",
                    "exit 91",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(fake_docker, 0o755)

    def _gemma_env(self, tmp_path: Path, fake_bin: Path, probe_output: Path) -> dict[str, str]:
        model_dir = tmp_path / "models"
        return {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHONPATH": "src",
            "DOCKER_LOG": str(tmp_path / "docker.log"),
            "DOCKER_GPU_ARGS": "",
            "DOCKER_TTY": "0",
            "HF_HOME": str(tmp_path / "hf-cache"),
            "LLAMA_CPP_DOCKER_IMAGE": "dustynv/llama_cpp:b5283-r36.4-cu128-24.04",
            "LLAMA_CPP_RUNTIME_PROBE_OUTPUT": str(probe_output),
            "LLAMA_SERVER_CMD": "/usr/local/bin/llama-server",
            "MODEL_DIR": str(model_dir),
            "MODEL_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "missing-model.gguf"),
            "MMPROJ_PATH": str(model_dir / "gemma-4-E2B-it-GGUF" / "missing-mmproj.gguf"),
        }

    def test_vlm_launcher_rejects_runtime_without_exact_mmproj_before_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            probe_output = tmp_path / "runtime-probe.json"
            self._write_fake_docker(fake_bin, supports_mmproj=False)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh", "--parallel", "1"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=self._gemma_env(tmp_path, fake_bin, probe_output),
            )

            self.assertTrue(probe_output.is_file(), result.stderr)
            artifact = json.loads(probe_output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2)
        self.assertIn("runtime_missing_mmproj_support", result.stderr)
        self.assertIn("dustynv/llama_cpp:b5283-r36.4-cu128-24.04", result.stderr)
        self.assertNotIn("MODEL_PATH not found", result.stderr)
        self.assertFalse(artifact["multimodal_ready"])
        self.assertFalse(artifact["llama_server_supports_mmproj"])
        self.assertEqual(artifact["llama_server_multimodal_markers"], ["mmproj"])

    def test_vlm_launcher_allows_multimodal_ready_runtime_to_reach_artifact_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            probe_output = tmp_path / "runtime-probe.json"
            self._write_fake_docker(fake_bin, supports_mmproj=True)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_gemma4_e2b_llama_docker.sh", "--parallel", "1"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=self._gemma_env(tmp_path, fake_bin, probe_output),
            )

            self.assertTrue(probe_output.is_file(), result.stderr)
            artifact = json.loads(probe_output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2)
        self.assertNotIn("runtime_missing_mmproj_support", result.stderr)
        self.assertIn("MODEL_PATH not found", result.stderr)
        self.assertTrue(artifact["multimodal_ready"])
        self.assertTrue(artifact["llama_server_supports_mmproj"])
        self.assertEqual(artifact["llama_server_multimodal_markers"], ["--mmproj", "mmproj"])

    def test_only_vlm_launchers_source_multimodal_runtime_gate(self):
        vlm_launchers = [
            Path("scripts/jetson/run_gemma4_e2b_llama_docker.sh"),
            Path("scripts/jetson/run_minicpmv46_llama_docker.sh"),
            Path("scripts/jetson/run_hf_gguf_vlm_llama_docker.sh"),
        ]
        for launcher_path in vlm_launchers:
            with self.subTest(launcher=str(launcher_path)):
                launcher = launcher_path.read_text(encoding="utf-8")
                self.assertIn("llama_cpp_runtime_gate.sh", launcher)
                self.assertIn("require_llama_cpp_multimodal_runtime", launcher)

        text_launcher = Path("scripts/jetson/run_hf_gguf_llama_docker.sh").read_text(encoding="utf-8")
        self.assertNotIn("llama_cpp_runtime_gate.sh", text_launcher)
        self.assertNotIn("require_llama_cpp_multimodal_runtime", text_launcher)


if __name__ == "__main__":
    unittest.main()
