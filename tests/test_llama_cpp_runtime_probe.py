"""Standalone llama.cpp runtime probe contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class LlamaCppRuntimeProbeContractsTest(unittest.TestCase):
    def _write_fake_docker(self, fake_bin: Path, *, run_stdout: str) -> Path:
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf '%s\\n' \"$*\" >> \"${DOCKER_LOG}\"",
                    "if [[ \"${1:-}\" == \"image\" && \"${2:-}\" == \"inspect\" ]]; then",
                    "  cat <<'JSON'",
                    json.dumps(
                        [
                            {
                                "Id": "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
                                "Created": "2026-05-27T13:47:04.31937282+09:30",
                                "RepoDigests": [
                                    "ghcr.io/4everwz/jetson-llama-cpp@sha256:c39cdc50c4564f29490c69b30f601da23f4d086f99d5c4b562426dbf65fec263"
                                ],
                                "Config": {
                                    "Labels": {
                                        "org.opencontainers.image.version": "d749821db3bd8c5c52360f55ca1a1b76ae46d8e7",
                                        "org.opencontainers.image.revision": "735d6e569bf8",
                                        "org.opencontainers.image.source": "https://github.com/ggml-org/llama.cpp",
                                        "org.opencontainers.image.base.name": "dustynv/cuda-python:r36.4.0-cu128-24.04",
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
                    run_stdout,
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
        return fake_docker

    def test_probe_image_cli_writes_multimodal_ready_runtime_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            docker_log = tmp_path / "docker.log"
            output = tmp_path / "runtime.json"
            image = "ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87"
            self._write_fake_docker(
                fake_bin,
                run_stdout="\n".join(
                    [
                        "llama_server_found=1",
                        "llama_server_path=/usr/local/bin/llama-server",
                        "llama_server_help_ok=1",
                        "llama_server_supports_mmproj=1",
                        "llama_server_multimodal_markers=--mmproj,mmproj",
                    ]
                ),
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.llama_cpp_runtime",
                    "probe-image",
                    "--image",
                    image,
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "DOCKER_LOG": str(docker_log),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "PYTHONPATH": "src",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            docker_commands = docker_log.read_text(encoding="utf-8")

        self.assertEqual(stdout, {"image": image, "multimodal_ready": True, "output": str(output)})
        self.assertEqual(artifact["kind"], "llama_cpp_runtime_probe")
        self.assertEqual(artifact["image"], image)
        self.assertTrue(artifact["inspect_ok"])
        self.assertEqual(
            artifact["image_id"],
            "sha256:52a8ad644e416b014466be5a35be1c8f92cf58ecd7fc9cffe8133a8955cb7844",
        )
        self.assertEqual(artifact["llama_cpp_ref"], "d749821db3bd8c5c52360f55ca1a1b76ae46d8e7")
        self.assertEqual(artifact["source_revision"], "735d6e569bf8")
        self.assertEqual(artifact["base_image"], "dustynv/cuda-python:r36.4.0-cu128-24.04")
        self.assertTrue(artifact["llama_server_probe_ok"])
        self.assertEqual(artifact["llama_server_path"], "/usr/local/bin/llama-server")
        self.assertTrue(artifact["llama_server_supports_mmproj"])
        self.assertEqual(artifact["llama_server_multimodal_markers"], ["--mmproj", "mmproj"])
        self.assertTrue(artifact["multimodal_ready"])
        self.assertIn(f"image inspect {image}", docker_commands)
        self.assertIn(f"run --rm --runtime nvidia --entrypoint /bin/bash {image}", docker_commands)

    def test_probe_image_cli_rejects_generic_mmproj_marker_without_exact_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            output = tmp_path / "runtime.json"
            image = "dustynv/llama_cpp:b5283-r36.4-cu128-24.04"
            self._write_fake_docker(
                fake_bin,
                run_stdout="\n".join(
                    [
                        "llama_server_found=1",
                        "llama_server_path=/usr/local/bin/llama-server",
                        "llama_server_help_ok=1",
                        "llama_server_supports_mmproj=0",
                        "llama_server_multimodal_markers=mmproj",
                    ]
                ),
            )

            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "edge_vlm.llama_cpp_runtime",
                    "probe-image",
                    "--image",
                    image,
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "DOCKER_LOG": str(tmp_path / "docker.log"),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "PYTHONPATH": "src",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(stdout, {"image": image, "multimodal_ready": False, "output": str(output)})
        self.assertTrue(artifact["llama_server_probe_ok"])
        self.assertTrue(artifact["llama_server_help_ok"])
        self.assertFalse(artifact["llama_server_supports_mmproj"])
        self.assertEqual(artifact["llama_server_multimodal_markers"], ["mmproj"])
        self.assertFalse(artifact["multimodal_ready"])
