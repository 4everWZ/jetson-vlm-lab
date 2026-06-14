"""Text-only Jetson launcher runtime-gate contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TextLauncherRuntimeGateContractsTest(unittest.TestCase):
    def _write_fake_runtime_docker(
        self,
        fake_bin: Path,
        *,
        server_found: bool,
        help_ok: bool = True,
        supports_mmproj: bool = False,
    ) -> None:
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
                                "Id": "sha256:unit-text-runtime",
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
                    f"llama_server_found={1 if server_found else 0}",
                    "llama_server_path=/usr/local/bin/llama-server" if server_found else "llama_server_path=",
                    f"llama_server_help_ok={1 if help_ok else 0}" if server_found else "llama_server_help_ok=",
                    f"llama_server_supports_mmproj={1 if supports_mmproj else 0}" if server_found and help_ok else "llama_server_supports_mmproj=",
                    f"llama_server_multimodal_markers={markers}" if server_found and help_ok else "llama_server_multimodal_markers=",
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

    def _launcher_env(self, tmp_path: Path, fake_bin: Path, probe_output: Path) -> dict[str, str]:
        return {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHONPATH": "src",
            "DOCKER_LOG": str(tmp_path / "docker.log"),
            "DOCKER_GPU_ARGS": "",
            "DOCKER_TTY": "0",
            "HF_HOME": str(tmp_path / "hf-cache"),
            "LLAMA_CPP_DOCKER_IMAGE": "unit/llama-cpp-text:test",
            "LLAMA_CPP_RUNTIME_PROBE_OUTPUT": str(probe_output),
            "LLAMA_SERVER_CMD": "/usr/local/bin/llama-server",
            "MODEL_DIR": str(tmp_path / "models"),
            "MODEL_REF": "tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
            "MODEL_FILE": "Hy-MT2-1.8B-Q4_K_M.gguf",
            "MODEL_ALIAS": "tencent-hy-mt2-1p8b-q4",
        }

    def test_text_launcher_rejects_runtime_without_llama_server_before_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            probe_output = tmp_path / "text-runtime-probe.json"
            download_log = tmp_path / "download.log"
            self._write_fake_runtime_docker(fake_bin, server_found=False)
            curl = fake_bin / "curl"
            curl.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        f"printf 'curl\\n' >> {str(download_log)!r}",
                        "exit 91",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(curl, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_hf_gguf_llama_docker.sh", "--parallel", "1"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=self._launcher_env(tmp_path, fake_bin, probe_output),
            )

            self.assertTrue(probe_output.is_file(), result.stderr)
            artifact = json.loads(probe_output.read_text(encoding="utf-8"))
            model_dir_exists = (tmp_path / "models").exists()

        self.assertEqual(result.returncode, 2)
        self.assertIn("runtime_missing_llama_server", result.stderr)
        self.assertFalse(artifact["llama_server_found"])
        self.assertFalse(download_log.exists())
        self.assertFalse(model_dir_exists)

    def test_text_launcher_allows_server_runtime_without_mmproj_to_reach_artifact_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            probe_output = tmp_path / "text-runtime-probe.json"
            self._write_fake_runtime_docker(fake_bin, server_found=True, supports_mmproj=False)
            model_path = tmp_path / "models" / "tencent" / "Hy-MT2-1.8B-GGUF" / "Hy-MT2-1.8B-Q4_K_M.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"not-a-gguf")

            result = subprocess.run(
                ["bash", "scripts/jetson/run_hf_gguf_llama_docker.sh", "--parallel", "1"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=self._launcher_env(tmp_path, fake_bin, probe_output),
            )

            self.assertTrue(probe_output.is_file(), result.stderr)
            artifact = json.loads(probe_output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2)
        self.assertNotIn("runtime_missing_mmproj_support", result.stderr)
        self.assertNotIn("runtime_missing_llama_server", result.stderr)
        self.assertIn("model GGUF failed magic check", result.stderr)
        self.assertTrue(artifact["llama_server_found"])
        self.assertTrue(artifact["llama_server_help_ok"])
        self.assertFalse(artifact["llama_server_supports_mmproj"])

    def test_text_launcher_sources_server_runtime_gate_only(self):
        text_launcher = Path("scripts/jetson/run_hf_gguf_llama_docker.sh").read_text(encoding="utf-8")

        self.assertIn("llama_cpp_runtime_gate.sh", text_launcher)
        self.assertIn("require_llama_cpp_server_runtime", text_launcher)
        self.assertNotIn("require_llama_cpp_multimodal_runtime", text_launcher)


if __name__ == "__main__":
    unittest.main()
