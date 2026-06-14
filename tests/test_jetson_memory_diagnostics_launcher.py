"""Jetson memory diagnostics launcher contract tests."""

import unittest
from pathlib import Path


class JetsonMemoryDiagnosticsLauncherContractsTest(unittest.TestCase):
    def test_memory_diagnostics_launcher_runs_python_module_with_default_output(self):
        script = Path("scripts/jetson/capture_memory_diagnostics.sh").read_text(encoding="utf-8")

        self.assertIn('env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"', script)
        self.assertIn('jetson_load_env_file "${env_file}"', script)
        self.assertIn(
            'output="${JETSON_MEMORY_DIAGNOSTICS_OUTPUT:-${repo_root}/outputs/jetson_inspect/memory-diagnostics-',
            script,
        )
        self.assertIn('exec "${python_bin}" -m edge_vlm.jetson_memory_diagnostics --output "${output}" "$@"', script)


if __name__ == "__main__":
    unittest.main()
