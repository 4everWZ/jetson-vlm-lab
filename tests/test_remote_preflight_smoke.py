"""run_remote_preflight_smoke.sh contract tests."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._remote_execution_helpers import isolated_remote_env, write_executable


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


class RemotePreflightSmokeContractsTest(unittest.TestCase):
    def test_remote_preflight_smoke_runs_access_probe_and_dry_run_sweep_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "smoke.log"
            fake_access = tmp_path / "check_remote_access.sh"
            fake_probe = tmp_path / "remote_probe.sh"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            write_executable(
                fake_access,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ACCESS\\n' >> \"${FAKE_SMOKE_LOG:?}\"",
                ],
            )
            write_executable(
                fake_probe,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'PROBE\\n' >> \"${FAKE_SMOKE_LOG:?}\"",
                ],
            )
            write_executable(
                fake_sweep,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'SWEEP\\n' >> \"${FAKE_SMOKE_LOG:?}\"",
                    "printf 'ENV_ACCESS_PREFLIGHT=%s\\n' \"${JETSON_REMOTE_ACCESS_PREFLIGHT:-}\" >> \"${FAKE_SMOKE_LOG}\"",
                    "printf 'ENV_GGUF_PREFLIGHT=%s\\n' \"${JETSON_REMOTE_GGUF_PREFLIGHT:-}\" >> \"${FAKE_SMOKE_LOG}\"",
                    "printf 'ENV_GGUF_PREFLIGHT_FAIL=%s\\n' \"${JETSON_REMOTE_GGUF_PREFLIGHT_FAIL:-}\" >> \"${FAKE_SMOKE_LOG}\"",
                    "for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_SMOKE_LOG}\"; done",
                ],
            )

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_preflight_smoke.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_ACCESS_CHECK=str(fake_access),
                    JETSON_REMOTE_PROBE=str(fake_probe),
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    JETSON_REMOTE_SMOKE_RUN_PREFIX="unit-smoke",
                    JETSON_REMOTE_SMOKE_VARIANT="gemma-q4-baseline-gpu12-b512-u512-kvq8",
                    FAKE_SMOKE_LOG=str(log_file),
                ),
            )
            log_text = _read_text_if_exists(log_file)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log_text.splitlines()[:3], ["ACCESS", "PROBE", "SWEEP"])
        self.assertIn("ENV_ACCESS_PREFLIGHT=1\n", log_text)
        self.assertIn("ENV_GGUF_PREFLIGHT=1\n", log_text)
        self.assertIn("ENV_GGUF_PREFLIGHT_FAIL=0\n", log_text)
        self.assertIn("ARG=--dry-run\n", log_text)
        self.assertIn("ARG=--run-prefix\nARG=unit-smoke\n", log_text)
        self.assertIn("ARG=--variant\nARG=gemma-q4-baseline-gpu12-b512-u512-kvq8\n", log_text)
        self.assertIn("remote_preflight_smoke=ok", result.stdout)

    def test_remote_preflight_smoke_stops_when_access_check_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "smoke.log"
            fake_access = tmp_path / "check_remote_access.sh"
            fake_probe = tmp_path / "remote_probe.sh"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            write_executable(
                fake_access,
                [
                    "#!/usr/bin/env bash",
                    "set -Eeuo pipefail",
                    "printf 'ACCESS\\n' >> \"${FAKE_SMOKE_LOG:?}\"",
                    "printf 'tcp_probe=tcp_connect_failed\\n' >&2",
                    "exit 1",
                ],
            )
            write_executable(fake_probe, ["#!/usr/bin/env bash", "set -Eeuo pipefail", "printf 'PROBE\\n' >> \"${FAKE_SMOKE_LOG:?}\""])
            write_executable(fake_sweep, ["#!/usr/bin/env bash", "set -Eeuo pipefail", "printf 'SWEEP\\n' >> \"${FAKE_SMOKE_LOG:?}\""])

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_preflight_smoke.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_ACCESS_CHECK=str(fake_access),
                    JETSON_REMOTE_PROBE=str(fake_probe),
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    FAKE_SMOKE_LOG=str(log_file),
                ),
            )
            log_text = _read_text_if_exists(log_file)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(log_text, "ACCESS\n")
        self.assertIn("tcp_probe=tcp_connect_failed", result.stderr)

    def test_remote_preflight_smoke_can_skip_access_check_for_proxy_ssh_setups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "smoke.log"
            fake_access = tmp_path / "check_remote_access.sh"
            fake_probe = tmp_path / "remote_probe.sh"
            fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
            write_executable(fake_access, ["#!/usr/bin/env bash", "set -Eeuo pipefail", "printf 'ACCESS\\n' >> \"${FAKE_SMOKE_LOG:?}\""])
            write_executable(fake_probe, ["#!/usr/bin/env bash", "set -Eeuo pipefail", "printf 'PROBE\\n' >> \"${FAKE_SMOKE_LOG:?}\""])
            write_executable(fake_sweep, ["#!/usr/bin/env bash", "set -Eeuo pipefail", "printf 'SWEEP\\n' >> \"${FAKE_SMOKE_LOG:?}\""])

            result = subprocess.run(
                ["bash", "scripts/jetson/run_remote_preflight_smoke.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=isolated_remote_env(
                    JETSON_REMOTE_ACCESS_CHECK=str(fake_access),
                    JETSON_REMOTE_PROBE=str(fake_probe),
                    JETSON_REMOTE_SWEEP=str(fake_sweep),
                    JETSON_REMOTE_SMOKE_SKIP_ACCESS_CHECK="1",
                    FAKE_SMOKE_LOG=str(log_file),
                ),
            )
            log_text = _read_text_if_exists(log_file)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log_text.splitlines()[:2], ["PROBE", "SWEEP"])
        self.assertNotIn("ACCESS", log_text)


if __name__ == "__main__":
    unittest.main()
