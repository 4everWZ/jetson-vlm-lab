"""Jetson llama.cpp image resolver contract tests."""

import os
import tempfile
import unittest
from pathlib import Path

from tests._launcher_helpers import launcher_env, run_bash, write_executable


class LauncherImageResolverContractsTest(unittest.TestCase):

    def test_jetson_image_resolver_defaults_to_verified_multimodal_image_even_with_autotag(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            write_executable(
                fake_bin / "autotag",
                [
                    "#!/usr/bin/env bash",
                    "if [[ \"$1\" == \"llama_cpp\" ]]; then",
                    "  printf '%s\\n' 'dustynv/llama_cpp:r36.4.0'",
                    "fi",
                ],
            )
            result = run_bash(
                [
                    "-lc",
                    "source scripts/jetson/resolve_llama_cpp_image.sh; resolve_llama_cpp_image",
                ],
                env=launcher_env(
                    PATH=f"{fake_bin}:{os.environ['PATH']}",
                    LLAMA_CPP_DOCKER_IMAGE=None,
                    LLAMA_CPP_DOCKER_IMAGE_FALLBACK=None,
                ),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87",
        )


if __name__ == "__main__":
    unittest.main()
