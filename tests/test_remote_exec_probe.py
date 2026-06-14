"""remote_exec.sh, remote_probe.sh, and SSH precheck contract tests."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import write_executable


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


class RemoteExecProbeContractsTest(unittest.TestCase):
    def test_jetson_remote_access_precheck_dry_run_sources_env_without_exposing_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.jetson"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PORT=2222",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab-bench",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                ["bash", "scripts/jetson/check_remote_access.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "JETSON_REMOTE_ACCESS_DRY_RUN": "1",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ssh_target=weizheng@100.95.31.18", result.stdout)
        self.assertIn("ssh_port=2222", result.stdout)
        self.assertIn("repo_dir=~/code/jetson-vlm-lab-bench", result.stdout)
        self.assertIn("icmp_probe=skipped_dry_run", result.stdout)
        self.assertIn("tcp_probe=skipped_dry_run", result.stdout)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)

    def test_jetson_remote_access_precheck_classifies_tcp_failure_without_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_executable(
                fake_bin / "ping",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'PING_ARGS=%s\\n' \"$*\" > \"${FAKE_PING_LOG:?}\"",
                ],
            )
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
                ["bash", "scripts/jetson/check_remote_access.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "JETSON_REMOTE_ACCESS_TCP_PROBE": "nc",
                    "FAKE_NC_LOG": str(tmp_path / "nc.log"),
                    "FAKE_PING_LOG": str(tmp_path / "ping.log"),
                },
            )
            nc_log = _read_text_if_exists(tmp_path / "nc.log")
            ping_log = _read_text_if_exists(tmp_path / "ping.log")

        self.assertEqual(result.returncode, 1)
        self.assertIn("icmp_probe=ok", result.stdout)
        self.assertIn("tcp_probe=tcp_connect_failed", result.stderr)
        self.assertIn("ssh_target=weizheng@100.95.31.18", result.stdout)
        self.assertIn("PING_ARGS=-c 1 -W 3 100.95.31.18", ping_log)
        self.assertIn("ARGS=-vz -w 5 100.95.31.18 22", nc_log)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", nc_log)
        self.assertNotIn("secret-password", ping_log)

    def test_jetson_remote_access_precheck_reports_tailnet_status_for_100x_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PASSWORD=secret-password",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_executable(
                fake_bin / "ping",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "exit 1",
                ],
            )
            write_executable(
                fake_bin / "tailscale",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'TAILSCALE_ARGS=%s\\n' \"$*\" > \"${FAKE_TAILSCALE_LOG:?}\"",
                    "printf 'backend state: NeedsLogin\\n' >&2",
                    "exit 1",
                ],
            )
            write_executable(
                fake_bin / "nc",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'nc: connect to 100.95.31.18 port 22 timed out\\n' >&2",
                    "exit 1",
                ],
            )

            result = subprocess.run(
                ["bash", "scripts/jetson/check_remote_access.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "JETSON_REMOTE_ACCESS_TCP_PROBE": "nc",
                    "FAKE_TAILSCALE_LOG": str(tmp_path / "tailscale.log"),
                },
            )
            tailscale_log = _read_text_if_exists(tmp_path / "tailscale.log")

        self.assertEqual(result.returncode, 1)
        self.assertIn("icmp_probe=failed", result.stdout)
        self.assertIn("tailnet_probe=tailscale_status_failed", result.stdout)
        self.assertIn("tcp_probe=tcp_connect_failed", result.stderr)
        self.assertIn("TAILSCALE_ARGS=status", tailscale_log)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", tailscale_log)

    def test_jetson_remote_access_precheck_can_use_powershell_tcp_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            powershell_log = tmp_path / "powershell.log"
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PASSWORD=secret-password",
                        f"JETSON_TAILSCALE_BIN={fake_bin / 'tailscale.exe'}",
                        f"JETSON_POWERSHELL_BIN={fake_bin / 'powershell.exe'}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_executable(
                fake_bin / "ping",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "exit 1",
                ],
            )
            write_executable(
                fake_bin / "tailscale.exe",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'TAILSCALE_ARGS=%s\\n' \"$*\" > \"${FAKE_TAILSCALE_LOG:?}\"",
                ],
            )
            write_executable(
                fake_bin / "nc",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'NC_ARGS=%s\\n' \"$*\" > \"${FAKE_NC_LOG:?}\"",
                    "printf 'nc linux namespace timeout\\n' >&2",
                    "exit 1",
                ],
            )
            write_executable(
                fake_bin / "powershell.exe",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'POWERSHELL_ARGS=%s\\n' \"$*\" > \"${FAKE_POWERSHELL_LOG:?}\"",
                ],
            )

            result = subprocess.run(
                ["bash", "scripts/jetson/check_remote_access.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "FAKE_NC_LOG": str(tmp_path / "nc.log"),
                    "FAKE_TAILSCALE_LOG": str(tmp_path / "tailscale.log"),
                    "FAKE_POWERSHELL_LOG": str(powershell_log),
                },
            )
            nc_log = _read_text_if_exists(tmp_path / "nc.log")
            tailscale_log = _read_text_if_exists(tmp_path / "tailscale.log")
            powershell_text = _read_text_if_exists(powershell_log)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("icmp_probe=failed", result.stdout)
        self.assertIn("tailnet_probe=tailscale_status_ok", result.stdout)
        self.assertIn("tcp_probe=ok", result.stdout)
        self.assertIn("tcp_probe_transport=powershell", result.stdout)
        self.assertIn("NC_ARGS=-vz -w 5 100.95.31.18 22", nc_log)
        self.assertIn("TAILSCALE_ARGS=status", tailscale_log)
        self.assertIn("Test-NetConnection", powershell_text)
        self.assertIn("100.95.31.18", powershell_text)
        self.assertIn("22", powershell_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", powershell_text)

    def test_jetson_remote_access_precheck_reports_tcp_ok_even_when_icmp_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_SSH_PORT=22",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_executable(
                fake_bin / "ping",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'PING_ARGS=%s\\n' \"$*\" > \"${FAKE_PING_LOG:?}\"",
                    "exit 1",
                ],
            )
            write_executable(
                fake_bin / "nc",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ARGS=%s\\n' \"$*\" > \"${FAKE_NC_LOG:?}\"",
                ],
            )

            result = subprocess.run(
                ["bash", "scripts/jetson/check_remote_access.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "FAKE_NC_LOG": str(tmp_path / "nc.log"),
                    "FAKE_PING_LOG": str(tmp_path / "ping.log"),
                },
            )
            nc_log = _read_text_if_exists(tmp_path / "nc.log")
            ping_log = _read_text_if_exists(tmp_path / "ping.log")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("icmp_probe=failed", result.stdout)
        self.assertIn("tcp_probe=ok", result.stdout)
        self.assertIn("PING_ARGS=-c 1 -W 3 100.95.31.18", ping_log)
        self.assertIn("ARGS=-vz -w 5 100.95.31.18 22", nc_log)

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
            write_executable(
                fake_remote,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ssh: connect to host 192.168.1.12 port 22: Connection timed out\\n' >&2",
                    "exit 255",
                ],
            )
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
            write_executable(
                fake_bin / "setsid",
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "if [[ \"${1:-}\" == \"-w\" ]]; then shift; fi",
                    "exec \"$@\"",
                ],
            )
            write_executable(
                fake_bin / "ssh",
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
                ],
            )
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

    def test_jetson_remote_exec_can_use_custom_ssh_binary_and_ignore_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env.jetson"
            fake_ssh = tmp_path / "windows-ssh.exe"
            log_file = tmp_path / "ssh.log"
            write_executable(
                fake_ssh,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ARGS=%s\\n' \"$*\" > \"${FAKE_SSH_LOG:?}\"",
                ],
            )
            env_file.write_text(
                "\n".join(
                    [
                        "JETSON_SSH_HOST=100.95.31.18",
                        "JETSON_SSH_USER=weizheng",
                        "JETSON_REPO_DIR=~/code/jetson-vlm-lab-bench",
                        "JETSON_SSH_PASSWORD=secret-password",
                        "JETSON_SSH_PASSWORD_HELPER=none",
                        f"JETSON_SSH_BIN={fake_ssh}",
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
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "JETSON_ENV_FILE": str(env_file),
                    "FAKE_SSH_LOG": str(log_file),
                },
            )
            log_text = _read_text_if_exists(log_file)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("weizheng@100.95.31.18", log_text)
        self.assertIn("cd ~/code/jetson-vlm-lab-bench && git status --short", log_text)
        self.assertNotIn("SSH_ASKPASS", log_text)
        self.assertNotIn("secret-password", result.stdout)
        self.assertNotIn("secret-password", result.stderr)
        self.assertNotIn("secret-password", log_text)


if __name__ == "__main__":
    unittest.main()
