"""Remote memory preparation contract tests."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import (
    isolated_remote_env,
    write_executable,
    write_remote_arg_logger,
)


class RemoteMemoryPrepareContractsTest(unittest.TestCase):
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
        self.assertIn("prepare_attempt=1; while", log_text)
        self.assertIn('while [ \\"\\${prepare_attempt}\\" -le %s ]', log_text)
        self.assertIn("prepare_attempt=\\$((prepare_attempt + 1))", log_text)
        self.assertIn("echo 1 > /proc/sys/vm/compact_memory", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=1\n", log_text)
        self.assertIn("ARG=EDGE_VLM_MEMORY_PREPARE_ATTEMPTS=1\n", log_text)
        self.assertIn("ARG=EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=remote_wrapper_drop_caches\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertNotIn("sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertNotIn("|| exit 0", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_can_repeat_memory_prepare_attempts_without_logging_password(self):
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
                    "if [[ \"${1:-}\" == \"sudo\" || (\"${1:-}\" == \"bash\" && \"${2:-}\" == \"-lc\") ]]; then",
                    "  IFS= read -r password_from_stdin || true",
                    "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                    "fi",
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
                    "unit-selector-prepare",
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
                    JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
                    JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS="3",
                    JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR="1",
                    JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_MIN_LFB_BLOCKS="100",
                    JETSON_SSH_PASSWORD="secret-password",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertEqual(log_text.count("STDIN_BYTES=15\n"), 2)
        self.assertIn("ARG=sudo\nARG=-S\nARG=-p\nARG=\nARG=sh\nARG=-c\n", log_text)
        self.assertIn('while [ "${prepare_attempt}" -le 3 ]', log_text)
        self.assertIn("prepare_attempt=$((prepare_attempt + 1))", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/select_qwen3_instruct_variant.sh\n", log_text)
        self.assertIn("ARG=bash\nARG=-lc\n", log_text)
        self.assertIn("ARG=EDGE_VLM_MEMORY_PREPARE_ATTEMPTS=3\n", log_text)
        self.assertIn("--pre-variant-command", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_forwards_prepare_context_to_gguf_preflight_plan(self):
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
                    "if [[ \"${1:-}\" == \"bash\" && \"${2:-}\" == \"-lc\" ]]; then",
                    "  IFS= read -r password_from_stdin || true",
                    "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                    "fi",
                    "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                ],
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/run_remote_optimization_sweep.sh",
                    "--run-prefix",
                    "unit-preflight-prepare",
                    "--variant",
                    "qwen3-vl-2b-instruct-q4-smoke",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_GGUF_PREFLIGHT="1",
                    JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
                    JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS="3",
                    JETSON_SSH_PASSWORD="secret-password",
                    FAKE_REMOTE_LOG=str(log_file),
                ),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertIn(
            "ARG=env\n"
            "ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n"
            "ARG=PYTHONPATH=src\n"
            "ARG=EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=1\n"
            "ARG=EDGE_VLM_MEMORY_PREPARE_ATTEMPTS=3\n"
            "ARG=EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=remote_wrapper_drop_caches\n"
            "ARG=python3\nARG=-m\nARG=edge_vlm.jetson_sweep\n",
            log_text,
        )
        self.assertIn("ARG=--plan-output\nARG=outputs/optimization_sweeps/unit-preflight-prepare/unit-preflight-prepare.preflight-plan.json\n", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_rejects_invalid_memory_prepare_attempts(self):
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
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_EXEC=str(fake_remote),
                    JETSON_REMOTE_SYNC="0",
                    JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT="1",
                    JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS="0",
                    JETSON_SSH_PASSWORD="secret-password",
                ),
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS must be a positive integer", result.stderr)

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
