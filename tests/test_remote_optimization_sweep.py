"""run_remote_optimization_sweep.sh contract tests."""

import os
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
    def test_remote_optimization_sweep_can_fail_before_ssh_when_access_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            nc_log = tmp_path / "nc.log"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote)
            write_executable(
                fake_bin / "nc",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ARGS=%s\\n' \"$*\" > \"${FAKE_NC_LOG:?}\"",
                    "printf 'nc: connect to 100.95.31.18 port 22 timed out\\n' >&2",
                    "exit 1",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-access-preflight",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_ACCESS_PREFLIGHT="1",
                    JETSON_REMOTE_ACCESS_TCP_PROBE="nc",
                    JETSON_SSH_HOST="100.95.31.18",
                    JETSON_SSH_USER="weizheng",
                    PATH=f"{fake_bin}:{os.environ['PATH']}",
                    FAKE_NC_LOG=str(nc_log),
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            nc_log_exists = nc_log.exists()
            nc_log_text = nc_log.read_text(encoding="utf-8") if nc_log_exists else ""
            remote_log_exists = log_file.exists()

        self.assertEqual(result.returncode, 1)
        self.assertTrue(nc_log_exists, "access preflight did not run nc")
        self.assertIn("tcp_probe=tcp_connect_failed", result.stderr)
        self.assertIn("ARGS=-vz -w 5 100.95.31.18 22", nc_log_text)
        self.assertFalse(remote_log_exists)

    def test_remote_optimization_sweep_runs_after_successful_access_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            nc_log = tmp_path / "nc.log"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_remote = tmp_path / "remote_exec.sh"
            write_remote_arg_logger(fake_remote)
            write_executable(
                fake_bin / "nc",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ARGS=%s\\n' \"$*\" > \"${FAKE_NC_LOG:?}\"",
                ],
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
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_ACCESS_PREFLIGHT="1",
                    JETSON_SSH_HOST="100.95.31.18",
                    JETSON_SSH_USER="weizheng",
                    PATH=f"{fake_bin}:{os.environ['PATH']}",
                    FAKE_NC_LOG=str(nc_log),
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            nc_log_exists = nc_log.exists()
            nc_log_text = nc_log.read_text(encoding="utf-8") if nc_log_exists else ""
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(nc_log_exists, "access preflight did not run nc")
        self.assertIn("ARGS=-vz -w 5 100.95.31.18 22", nc_log_text)
        self.assertEqual(log_text.count("CALL\n"), 1)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)

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
            "ARG=--memory-diagnostics-output\n"
            "ARG=outputs/optimization_sweeps/unit-selector/unit-selector.qwen3-selector.memory-diagnostics.json\n",
            log_text,
        )
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

    def test_remote_optimization_sweep_can_capture_sudo_memory_diagnostics_after_qwen3_selector(self):
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
                    "if [[ \"${1:-}\" == \"sudo\" ]]; then",
                    "  IFS= read -r password_from_stdin || true",
                    "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                    "  printf '%s\\n' '{\"output\":\"sudo-sidecar.json\",\"summary\":{\"debugfs_statuses\":{\"dma_buf_bufinfo\":\"readable\"},\"debugfs_dma_buf_total_bytes\":0}}'",
                    "fi",
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
                    "unit-selector-sudo",
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
                    JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO="1",
                    JETSON_REMOTE_SUDO_PASSWORD="pw",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 4)
        self.assertIn("STDIN_BYTES=2\n", log_text)
        self.assertIn(
            "ARG=sudo\n"
            "ARG=-S\n"
            "ARG=-p\n"
            "ARG=\n"
            "ARG=env\n"
            "ARG=PYTHONPATH=src\n"
            "ARG=python3\n"
            "ARG=-m\n"
            "ARG=edge_vlm.jetson_memory_diagnostics\n"
            "ARG=--output\n"
            "ARG=outputs/optimization_sweeps/unit-selector-sudo/unit-selector-sudo.qwen3-selector.memory-diagnostics.sudo.json\n",
            log_text,
        )
        self.assertIn(
            "Qwen3 selector sudo memory diagnostics output: "
            "outputs/optimization_sweeps/unit-selector-sudo/unit-selector-sudo.qwen3-selector.memory-diagnostics.sudo.json",
            result.stderr,
        )
        self.assertNotIn("sudo-sidecar.json", result.stdout)
        self.assertIn("ARG=edge_vlm.selection_context_diagnostics\n", log_text)
        self.assertIn(
            "ARG=--selector-output\n"
            "ARG=outputs/optimization_sweeps/unit-selector-sudo/unit-selector-sudo.qwen3-selector.json\n"
            "ARG=--sudo-memory-diagnostics-output\n"
            "ARG=outputs/optimization_sweeps/unit-selector-sudo/unit-selector-sudo.qwen3-selector.memory-diagnostics.sudo.json\n",
            log_text,
        )

    def test_remote_optimization_sweep_requires_password_for_sudo_memory_diagnostics(self):
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
                    "unit-selector-sudo",
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
                    JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO="1",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO requires "
            "JETSON_REMOTE_SUDO_PASSWORD or JETSON_SSH_PASSWORD.",
            result.stderr,
        )
        self.assertFalse(log_file.exists())

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

    def test_remote_optimization_sweep_can_write_advisory_gguf_preflight_manifest(self):
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
                    "if printf '%s\\n' \"$@\" | grep -q 'check-plan'; then",
                    "  exit 2",
                    "fi",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-preflight",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_GGUF_PREFLIGHT="1",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertIn("ARG=edge_vlm.jetson_sweep\n", log_text)
        self.assertIn("ARG=--dry-run\nARG=--plan-output\nARG=outputs/optimization_sweeps/unit-preflight/unit-preflight.preflight-plan.json\n", log_text)
        self.assertIn("ARG=edge_vlm.gguf_artifacts\nARG=check-plan\n", log_text)
        self.assertIn("ARG=--plan\nARG=outputs/optimization_sweeps/unit-preflight/unit-preflight.preflight-plan.json\n", log_text)
        self.assertIn("ARG=--output\nARG=outputs/optimization_sweeps/unit-preflight/unit-preflight.gguf-artifacts.json\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertIn("GGUF artifact preflight failed; continuing because JETSON_REMOTE_GGUF_PREFLIGHT_FAIL=0", result.stderr)

    def test_remote_optimization_sweep_can_fail_on_gguf_preflight_when_enabled(self):
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
                    "if printf '%s\\n' \"$@\" | grep -q 'check-plan'; then",
                    "  exit 2",
                    "fi",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-preflight-fail",
                    "--variant",
                    "gemma-q4-baseline-gpu12-b512-u512-kvq8",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_GGUF_PREFLIGHT="1",
                    JETSON_REMOTE_GGUF_PREFLIGHT_FAIL="1",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 2)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=edge_vlm.gguf_artifacts\nARG=check-plan\n", log_text)
        self.assertNotIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertIn("GGUF artifact preflight failed", result.stderr)

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

if __name__ == "__main__":
    unittest.main()
