"""run_remote_optimization_sweep.sh contract tests."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import (
    isolated_remote_env,
    write_executable,
    write_remote_arg_logger,
)


class RemoteOptimizationSweepContractsTest(unittest.TestCase):
    def test_remote_optimization_sweep_syncs_branch_and_forwards_pinned_sweep_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-remote",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                    "--min-lfb-blocks",
                    "150",
                    "--pre-variant-command",
                    "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertIn("ARG=git\nARG=fetch\nARG=origin\nARG=main\n", log_text)
        self.assertIn("ARG=git\nARG=checkout\nARG=--detach\nARG=FETCH_HEAD\n", log_text)
        self.assertNotIn("ARG=pull\n", log_text)
        self.assertIn("ARG=env\n", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertIn("ARG=--run-prefix\nARG=unit-remote\n", log_text)
        self.assertIn("ARG=--variant\nARG=minicpm-q4-baseline-b128-u32-kvq8\n", log_text)
        self.assertIn("ARG=--min-lfb-blocks\nARG=150\n", log_text)
        self.assertIn("ARG=--pre-variant-command\nARG=sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n", log_text)

    def test_remote_optimization_sweep_can_sync_explicit_remote_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "qwen3-vl-2b-instruct-q4-smoke",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_BRANCH="bench/qwen3-vl-instruct-q4",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertIn("ARG=git\nARG=fetch\nARG=origin\nARG=bench/qwen3-vl-instruct-q4\n", log_text)
        self.assertIn("ARG=git\nARG=checkout\nARG=--detach\nARG=FETCH_HEAD\n", log_text)
        self.assertNotIn("ARG=pull\n", log_text)
        self.assertIn("ARG=--variant\nARG=qwen3-vl-2b-instruct-q4-smoke\n", log_text)

    def test_remote_optimization_sweep_can_select_qwen3_fallback_variant_before_sweep(self):
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
                    "  printf '%s\\n' '{\"selected_variant_id\":\"qwen3-vl-2b-instruct-q8-smoke\",\"selected_reason\":\"primary_blocked_selected_fallback\"}'",
                    "fi",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-selector",
                    "--variant",
                    "smolvlm2-256m-q8-smoke",
                    "--min-lfb-blocks",
                    "150",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR="1",
                    JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_MIN_LFB_BLOCKS="100",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=bash\nARG=scripts/jetson/select_qwen3_instruct_variant.sh\n", log_text)
        self.assertIn("ARG=--run-prefix\nARG=unit-selector\n", log_text)
        self.assertIn("ARG=--min-lfb-blocks\nARG=150\n", log_text)
        self.assertIn("ARG=--fallback-min-lfb-blocks\nARG=100\n", log_text)
        self.assertIn(
            "ARG=--selection-context-json\nARG=outputs/optimization_sweeps/unit-selector/unit-selector.qwen3-selector.json\n",
            log_text,
        )
        self.assertIn(
            "ARG=--variant-min-lfb-blocks\nARG=qwen3-vl-2b-instruct-q8-smoke=100\n",
            log_text,
        )
        self.assertIn("ARG=--variant\nARG=smolvlm2-256m-q8-smoke\n", log_text)
        self.assertIn("ARG=--variant\nARG=qwen3-vl-2b-instruct-q8-smoke\n", log_text)

    def test_remote_optimization_sweep_can_skip_git_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 1)
        self.assertNotIn("ARG=pull\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertIn("ARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)

    def test_remote_optimization_sweep_can_prepare_max_clocks_without_logging_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote, stdin_mode="sudo")

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_PREPARE_MAX_CLOCKS="1",
                    JETSON_SSH_PASSWORD="secret-password",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=sudo\nARG=-S\nARG=sh\nARG=-c\n", log_text)
        self.assertIn("jetson_clocks && jetson_clocks --show > outputs/jetson_inspect/jetson-clocks-max-", log_text)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertIn("ARG=EDGE_VLM_PREPARE_MAX_CLOCKS_ENABLED=1\n", log_text)
        self.assertIn("ARG=EDGE_VLM_PREPARE_MAX_CLOCKS_CAPTURE=outputs/jetson_inspect/jetson-clocks-max-", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_sources_ignored_env_for_sudo_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            env_file = tmp_path / ".env.jetson"
            fake_remote = tmp_path / "remote_exec.sh"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_remote_arg_logger(fake_remote, stdin_mode="sudo")
            env = isolated_remote_env(
                JETSON_ENV_FILE=str(env_file),
                JETSON_REMOTE_EXEC=str(fake_remote),
                JETSON_REMOTE_SYNC="0",
                JETSON_REMOTE_PREPARE_MAX_CLOCKS="1",
                FAKE_REMOTE_LOG=str(log_file),
            )
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=sudo\nARG=-S\nARG=sh\nARG=-c\n", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_prepare_max_clocks_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            write_executable(fake_remote, ["#!/usr/bin/env bash", "exit 0"])

            env = isolated_remote_env(
                JETSON_REMOTE_EXEC=str(fake_remote),
                JETSON_REMOTE_SYNC="0",
                JETSON_REMOTE_PREPARE_MAX_CLOCKS="1",
            )
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_REMOTE_PREPARE_MAX_CLOCKS requires", result.stderr)

    def test_remote_optimization_sweep_can_drop_caches_before_variants_without_logging_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote, stdin_mode="bash_lc")

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
                    JETSON_SSH_PASSWORD="secret-password",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 1)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=bash\nARG=-lc\n", log_text)
        self.assertIn("mkfifo", log_text)
        self.assertIn("trap '' PIPE", log_text)
        self.assertIn('> "${pw_fifo}" 2>/dev/null', log_text)
        self.assertIn("--pre-variant-command", log_text)
        self.assertIn("sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches", log_text)
        self.assertIn("echo 1 > /proc/sys/vm/compact_memory", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=1\n", log_text)
        self.assertIn("ARG=EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=remote_wrapper_drop_caches\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertNotIn("sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertNotIn("|| exit 0", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_drop_caches_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            write_executable(fake_remote, ["#!/usr/bin/env bash", "exit 0"])

            env = isolated_remote_env(
                JETSON_REMOTE_EXEC=str(fake_remote),
                JETSON_REMOTE_SYNC="0",
                JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
            )
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT requires", result.stderr)

    def test_remote_optimization_sweep_drop_caches_rejects_existing_pre_variant_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            write_executable(fake_remote, ["#!/usr/bin/env bash", "exit 0"])

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--dry-run",
                    "--variant",
                    "minicpm-q4-baseline-b128-u32-kvq8",
                    "--pre-variant-command",
                    "true",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
                    JETSON_SSH_PASSWORD="secret-password",
                ),
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined with --pre-variant-command", result.stderr)


if __name__ == "__main__":
    unittest.main()
