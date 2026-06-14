"""Remote optimization sweep CMA sidecar contract tests."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import isolated_remote_env, write_executable


class RemoteOptimizationSweepCmaSidecarContractsTest(unittest.TestCase):
    def test_no_variant_selector_writes_boot_config_apply_dry_run_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_executable(
                fake_remote,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                    "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    "if printf '%s\\n' \"$@\" | grep -q 'select_qwen3_instruct_variant.sh'; then",
                    "  printf '%s\\n' '{\"selected_variant_id\":\"\",\"selected_reason\":\"no_usable_variant\"}'",
                    "fi",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-selector-cma-sidecar",
                    "--variant",
                    "smolvlm2-256m-q8-smoke",
                    "--min-lfb-blocks",
                    "150",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR="1",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARG=edge_vlm.boot_config_cma_apply\n", log_text)
        self.assertIn(
            "ARG=--plan\n"
            "ARG=outputs/optimization_sweeps/unit-selector-cma-sidecar/unit-selector-cma-sidecar.qwen3-selector.boot-config-cma-plan.json\n"
            "ARG=--candidate-id\n"
            "ARG=max-required-lfb-rounded-64mib\n"
            "ARG=--output\n"
            "ARG=outputs/optimization_sweeps/unit-selector-cma-sidecar/unit-selector-cma-sidecar.qwen3-selector.boot-config-cma-apply-dry-run.json\n",
            log_text,
        )
        self.assertIn(
            "Qwen3 selector boot config CMA apply dry-run output: "
            "outputs/optimization_sweeps/unit-selector-cma-sidecar/"
            "unit-selector-cma-sidecar.qwen3-selector.boot-config-cma-apply-dry-run.json",
            result.stderr,
        )

    def test_cma_apply_dry_run_sidecar_can_be_skipped_without_skipping_patch_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_executable(
                fake_remote,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                    "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    "if printf '%s\\n' \"$@\" | grep -q 'select_qwen3_instruct_variant.sh'; then",
                    "  printf '%s\\n' '{\"selected_variant_id\":\"\",\"selected_reason\":\"no_usable_variant\"}'",
                    "fi",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-selector-cma-sidecar-skip",
                    "--variant",
                    "smolvlm2-256m-q8-smoke",
                    "--min-lfb-blocks",
                    "150",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR="1",
                    JETSON_REMOTE_QWEN3_INSTRUCT_CMA_APPLY_DRY_RUN="0",
                    FAKE_REMOTE_LOG=str(log_file),
                    PATH=os.environ["PATH"],
                ),
            )
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARG=edge_vlm.cma_experiment_plan\n", log_text)
        self.assertIn("ARG=edge_vlm.boot_config_cma_plan\n", log_text)
        self.assertNotIn("edge_vlm.boot_config_cma_apply", log_text)
        self.assertNotIn("boot config CMA apply dry-run output", result.stderr)


if __name__ == "__main__":
    unittest.main()
