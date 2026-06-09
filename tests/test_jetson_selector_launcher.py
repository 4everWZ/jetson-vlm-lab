"""Jetson selector launcher contract tests."""

import unittest
from pathlib import Path


class JetsonSelectorLauncherContractsTest(unittest.TestCase):
    def test_qwen3_selector_launcher_sources_env_and_runs_python_module(self):
        script = Path("scripts/jetson/select_qwen3_instruct_variant.sh").read_text(encoding="utf-8")

        self.assertIn('env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"', script)
        self.assertIn('jetson_load_env_file "${env_file}"', script)
        self.assertIn('cd "${repo_root}"', script)
        self.assertIn('export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"', script)
        self.assertIn('exec "${python_bin}" -m edge_vlm.jetson_variant_selector "$@"', script)


if __name__ == "__main__":
    unittest.main()
