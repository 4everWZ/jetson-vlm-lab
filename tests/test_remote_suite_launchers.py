"""Remote suite launcher contract tests."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import isolated_remote_env, write_fake_suite_scripts


class RemoteSuiteLauncherContractsTest(unittest.TestCase):
    def test_remote_current_defaults_suite_runs_locked_sweep_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep, fake_remote = write_fake_suite_scripts(tmp_path)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_current_defaults_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_CURRENT_DEFAULTS_RUN_PREFIX="defaults-unit",
                    JETSON_CURRENT_DEFAULTS_TRIAL_COUNT="7",
                    JETSON_CURRENT_DEFAULTS_MAX_TOKENS="33",
                    JETSON_CURRENT_DEFAULTS_FAKE_STREAM_MAX_FRAMES="2",
                    JETSON_CURRENT_DEFAULTS_MIN_LFB_BLOCKS="199",
                    JETSON_CURRENT_DEFAULTS_WAIT_TIMEOUT_S="123",
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    FAKE_SUITE_LOG=str(log_file),
                ),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=defaults-unit\n", log_text)
        self.assertIn("SWEEP_ARG=--variant\nSWEEP_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("SWEEP_ARG=--variant\nSWEEP_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=7\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=33\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=2\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=199\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=123\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/defaults-unit/defaults-unit.manifest.json\n",
            log_text,
        )
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/defaults-unit/comparison.md\n",
            log_text,
        )

    def test_remote_lightweight_model_suite_runs_baselines_and_candidates_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep, fake_remote = write_fake_suite_scripts(tmp_path)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_lightweight_model_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_LIGHTWEIGHT_RUN_PREFIX="light-unit",
                    JETSON_LIGHTWEIGHT_TRIAL_COUNT="6",
                    JETSON_LIGHTWEIGHT_MAX_TOKENS="44",
                    JETSON_LIGHTWEIGHT_FAKE_STREAM_MAX_FRAMES="4",
                    JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS="177",
                    JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S="321",
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    FAKE_SUITE_LOG=str(log_file),
                ),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("ENV_QWEN3_SELECTOR=1\n", log_text)
        self.assertIn("ENV_QWEN3_FALLBACK_MIN_LFB=100\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=light-unit\n", log_text)
        for variant_id in (
            "minicpm-q4-baseline-b128-u32-kvq8",
            "gemma-q4-baseline-gpu12-b512-u512-kvq8",
            "smolvlm2-256m-q8-smoke",
            "qwen3-vl-2b-thinking-q4-smoke",
            "youtu-vl-4b-q4-thirdparty-smoke",
        ):
            self.assertIn(f"SWEEP_ARG=--variant\nSWEEP_ARG={variant_id}\n", log_text)
        self.assertNotIn("SWEEP_ARG=--variant\nSWEEP_ARG=qwen3-vl-2b-instruct-q4-smoke\n", log_text)
        self.assertNotIn("SWEEP_ARG=--variant\nSWEEP_ARG=youtu-vl-4b-q8-smoke\n", log_text)
        self.assertNotIn("SWEEP_ARG=--variant\nSWEEP_ARG=hunyuanocr-q8-smoke\n", log_text)
        protocol_text = Path("docs/benchmark_protocol.md").read_text(encoding="utf-8")
        self.assertIn("excludes HunyuanOCR and official Youtu-VL Q8", protocol_text)
        self.assertIn("JETSON_LIGHTWEIGHT_EXTRA_VARIANTS=youtu-vl-4b-q8-smoke", protocol_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=6\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=44\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=4\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=177\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=321\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/light-unit/light-unit.manifest.json\n",
            log_text,
        )
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("REMOTE_ARG=--baseline-variant\nREMOTE_ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/light-unit/comparison.md\n",
            log_text,
        )

    def test_remote_tencent_text_suite_runs_text_candidates_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep, fake_remote = write_fake_suite_scripts(tmp_path)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_tencent_text_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_TENCENT_TEXT_RUN_PREFIX="tencent-text-unit",
                    JETSON_TENCENT_TEXT_TRIAL_COUNT="4",
                    JETSON_TENCENT_TEXT_MAX_TOKENS="55",
                    JETSON_TENCENT_TEXT_MIN_LFB_BLOCKS="188",
                    JETSON_TENCENT_TEXT_WAIT_TIMEOUT_S="222",
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    FAKE_SUITE_LOG=str(log_file),
                ),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=tencent-text-unit\n", log_text)
        for variant_id in (
            "tencent-hy-mt1p5-1p8b-1p25bit-text-smoke",
            "tencent-hy-mt1p5-1p8b-2bit-text-smoke",
            "tencent-hy-mt1p5-1p8b-q4-text-smoke",
            "tencent-hy-mt1p5-1p8b-q6-text-smoke",
            "tencent-hy-mt1p5-1p8b-q8-text-smoke",
            "tencent-hy-mt2-1p8b-1p25bit-text-smoke",
            "tencent-hy-mt2-1p8b-2bit-text-smoke",
            "tencent-hy-mt2-1p8b-q4-text-smoke",
            "tencent-hy-mt2-1p8b-q6-text-smoke",
            "tencent-hy-mt2-1p8b-q8-text-smoke",
            "tencent-youtu-llm-2b-q8-text-smoke",
        ):
            self.assertIn(f"SWEEP_ARG=--variant\nSWEEP_ARG={variant_id}\n", log_text)
        self.assertIn("SWEEP_ARG=--trial-count\nSWEEP_ARG=4\n", log_text)
        self.assertIn("SWEEP_ARG=--max-tokens\nSWEEP_ARG=55\n", log_text)
        self.assertIn("SWEEP_ARG=--fake-stream-max-frames\nSWEEP_ARG=0\n", log_text)
        self.assertIn("SWEEP_ARG=--min-lfb-blocks\nSWEEP_ARG=188\n", log_text)
        self.assertIn("SWEEP_ARG=--wait-timeout-s\nSWEEP_ARG=222\n", log_text)
        self.assertIn("REMOTE\n", log_text)
        self.assertIn("REMOTE_ARG=PYTHONPATH=src\nREMOTE_ARG=python3\nREMOTE_ARG=-m\nREMOTE_ARG=edge_vlm.optimization\nREMOTE_ARG=compare\n", log_text)
        self.assertIn(
            "REMOTE_ARG=--manifest\nREMOTE_ARG=outputs/optimization_sweeps/tencent-text-unit/tencent-text-unit.manifest.json\n",
            log_text,
        )
        self.assertIn(
            "REMOTE_ARG=--output\nREMOTE_ARG=outputs/optimization_sweeps/tencent-text-unit/comparison.md\n",
            log_text,
        )


if __name__ == "__main__":
    unittest.main()
