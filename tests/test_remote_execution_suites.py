"""Remote execution, remote sweep, and remote suite contract tests."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path



class RemoteExecutionSuiteContractsTest(unittest.TestCase):

    def test_jetson_remote_exec_dry_run_sources_ignored_env_without_exposing_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.jetson"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/remote_exec.sh",
                    "git",
                    "status",
                    "--short",
                    "--branch",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "JETSON_REMOTE_DRY_RUN": "1",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("weizheng@192.168.1.12", result.stdout)
        self.assertIn("cd ~/code/jetson-vlm-lab", result.stdout)
        self.assertIn("git status --short --branch", result.stdout)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_jetson_remote_exec_requires_host_and_user(self):
        result = subprocess.run(
            ["bash", "scripts/jetson/remote_exec.sh", "git", "status"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                **os.environ,
                "JETSON_ENV_FILE": "/tmp/edge-vlm-missing-env-file",
                "JETSON_REMOTE_DRY_RUN": "1",
                "JETSON_SSH_HOST": "",
                "JETSON_SSH_USER": "",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("JETSON_SSH_HOST and JETSON_SSH_USER are required", result.stderr)

    def test_jetson_remote_probe_dry_run_sources_ignored_env_without_exposing_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.jetson"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", "scripts/jetson/remote_probe.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "JETSON_REMOTE_PROBE_DRY_RUN": "1",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("weizheng@192.168.1.12", result.stdout)
        self.assertIn("cd ~/code/jetson-vlm-lab", result.stdout)
        self.assertIn("edge-vlm-remote-probe", result.stdout)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_jetson_remote_probe_classifies_connect_timeout_without_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
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
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'ssh: connect to host 192.168.1.12 port 22: Connection timed out\\n' >&2",
                        "exit 255",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)
            result = subprocess.run(
                ["bash", "scripts/jetson/remote_probe.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                },
            )

        self.assertEqual(result.returncode, 255)
        self.assertIn("ssh_connect_timeout", result.stderr)
        self.assertIn("Connection timed out", result.stderr)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_jetson_remote_exec_can_use_askpass_without_sshpass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            log_file = tmp_path / "ssh.log"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            (fake_bin / "setsid").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "if [[ \"${1:-}\" == \"-w\" ]]; then shift; fi",
                        "exec \"$@\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (fake_bin / "ssh").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "password=\"$(${SSH_ASKPASS:?})\"",
                        "if [[ \"${password}\" != \"${EXPECTED_JETSON_PASSWORD:?}\" ]]; then",
                        "  echo 'askpass password mismatch' >&2",
                        "  exit 3",
                        "fi",
                        "printf 'ASKPASS_OK\\n' > \"${FAKE_SSH_LOG:?}\"",
                        "printf 'SSH_ASKPASS_REQUIRE=%s\\n' \"${SSH_ASKPASS_REQUIRE:-}\" >> \"${FAKE_SSH_LOG}\"",
                        "printf 'ARGS=%s\\n' \"$*\" >> \"${FAKE_SSH_LOG}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_bin / "setsid", 0o755)
            os.chmod(fake_bin / "ssh", 0o755)
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=192.168.1.12",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab",
                        "JETSON_SSH_PASSWORD=secret-password",
                        "JETSON_SSH_PASSWORD_HELPER=askpass",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "scripts/jetson/remote_exec.sh",
                    "git",
                    "pull",
                    "--ff-only",
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "EXPECTED_JETSON_PASSWORD": "secret-password",
                    "FAKE_SSH_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("ASKPASS_OK", log_text)
        self.assertIn("SSH_ASKPASS_REQUIRE=force", log_text)
        self.assertIn("weizheng@192.168.1.12", log_text)
        self.assertIn("cd ~/code/jetson-vlm-lab && git pull --ff-only", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", log_text)

    def test_remote_optimization_sweep_syncs_branch_and_forwards_pinned_sweep_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_REMOTE_LOG": str(log_file),
                },
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
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_BRANCH": "bench/qwen3-vl-instruct-q4",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 3)
        self.assertIn("ARG=git\nARG=fetch\nARG=origin\nARG=bench/qwen3-vl-instruct-q4\n", log_text)
        self.assertIn("ARG=git\nARG=checkout\nARG=--detach\nARG=FETCH_HEAD\n", log_text)
        self.assertNotIn("ARG=pull\n", log_text)
        self.assertIn("ARG=--variant\nARG=qwen3-vl-2b-instruct-q4-smoke\n", log_text)

    def test_remote_optimization_sweep_can_skip_git_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "remote.log"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
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
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "if [[ \"${1:-}\" == \"sudo\" ]]; then",
                        "  IFS= read -r password_from_stdin || true",
                        "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                        "fi",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_PREPARE_MAX_CLOCKS": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(log_text.count("CALL\n"), 2)
        self.assertIn("ARG=sudo\nARG=-S\nARG=sh\nARG=-c\n", log_text)
        self.assertIn("jetson_clocks && jetson_clocks --show > outputs/jetson_inspect/jetson-clocks-max-", log_text)
        self.assertIn("STDIN_BYTES=15\n", log_text)
        self.assertIn("ARG=bash\nARG=scripts/jetson/run_optimization_sweep.sh\n", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_prepare_max_clocks_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

            env = {
                **os.environ,
                "JETSON_REMOTE_EXEC": str(fake_remote),
                "JETSON_REMOTE_SYNC": "0",
                "JETSON_REMOTE_PREPARE_MAX_CLOCKS": "1",
            }
            env.pop("JETSON_SSH_PASSWORD", None)
            env.pop("JETSON_REMOTE_SUDO_PASSWORD", None)
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
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
                        "if [[ \"${1:-}\" == \"bash\" && \"${2:-}\" == \"-lc\" ]]; then",
                        "  IFS= read -r password_from_stdin || true",
                        "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                        "fi",
                        "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                    "FAKE_REMOTE_LOG": str(log_file),
                },
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
        self.assertIn("sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertIn("ARG=LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87\n", log_text)
        self.assertIn("ARG=PYTHONPATH=src\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertNotIn("sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", log_text)
        self.assertNotIn("|| exit 0", log_text)
        self.assertNotIn("secret-password", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_remote_optimization_sweep_drop_caches_requires_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_remote = Path(tmp) / "remote_exec.sh"
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

            env = {
                **os.environ,
                "JETSON_REMOTE_EXEC": str(fake_remote),
                "JETSON_REMOTE_SYNC": "0",
                "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
            }
            env.pop("JETSON_SSH_PASSWORD", None)
            env.pop("JETSON_REMOTE_SUDO_PASSWORD", None)
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
            fake_remote.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            os.chmod(fake_remote, 0o755)

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
                env={
                    **os.environ,
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT": "1",
                    "JETSON_SSH_PASSWORD": "secret-password",
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined with --pre-variant-command", result.stderr)

    def test_remote_current_defaults_suite_runs_locked_sweep_then_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "suite.log"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_current_defaults_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_CURRENT_DEFAULTS_RUN_PREFIX": "defaults-unit",
                    "JETSON_CURRENT_DEFAULTS_TRIAL_COUNT": "7",
                    "JETSON_CURRENT_DEFAULTS_MAX_TOKENS": "33",
                    "JETSON_CURRENT_DEFAULTS_FAKE_STREAM_MAX_FRAMES": "2",
                    "JETSON_CURRENT_DEFAULTS_MIN_LFB_BLOCKS": "199",
                    "JETSON_CURRENT_DEFAULTS_WAIT_TIMEOUT_S": "123",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
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
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_lightweight_model_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_LIGHTWEIGHT_RUN_PREFIX": "light-unit",
                    "JETSON_LIGHTWEIGHT_TRIAL_COUNT": "6",
                    "JETSON_LIGHTWEIGHT_MAX_TOKENS": "44",
                    "JETSON_LIGHTWEIGHT_FAKE_STREAM_MAX_FRAMES": "4",
                    "JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS": "177",
                    "JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S": "321",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = log_file.read_text(encoding="utf-8")

        self.assertIn("SWEEP\n", log_text)
        self.assertIn("ENV_PREPARE=1\n", log_text)
        self.assertIn("ENV_DROP=1\n", log_text)
        self.assertIn("ENV_SYNC=0\n", log_text)
        self.assertIn("SWEEP_ARG=--run-prefix\nSWEEP_ARG=light-unit\n", log_text)
        for variant_id in (
            "minicpm-q4-baseline-b128-u32-kvq8",
            "gemma-q4-baseline-gpu12-b512-u512-kvq8",
            "smolvlm2-256m-q8-smoke",
            "qwen3-vl-2b-thinking-q4-smoke",
            "qwen3-vl-2b-instruct-q4-smoke",
            "youtu-vl-4b-q4-thirdparty-smoke",
        ):
            self.assertIn(f"SWEEP_ARG=--variant\nSWEEP_ARG={variant_id}\n", log_text)
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
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            fake_remote = tmp_path / "remote_exec.sh"
            fake_sweep.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
                        "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_remote.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -Eeuo pipefail",
                        "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
                        "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_sweep, 0o755)
            os.chmod(fake_remote, 0o755)

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_tencent_text_suite.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_TENCENT_TEXT_RUN_PREFIX": "tencent-text-unit",
                    "JETSON_TENCENT_TEXT_TRIAL_COUNT": "4",
                    "JETSON_TENCENT_TEXT_MAX_TOKENS": "55",
                    "JETSON_TENCENT_TEXT_MIN_LFB_BLOCKS": "188",
                    "JETSON_TENCENT_TEXT_WAIT_TIMEOUT_S": "222",
                    "JETSON_REMOTE_SYNC": "0",
                    "JETSON_REMOTE_SWEEP": str(fake_sweep),
                    "JETSON_REMOTE_EXEC": str(fake_remote),
                    "FAKE_SUITE_LOG": str(log_file),
                },
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
