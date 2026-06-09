"""Helpers shared by remote execution contract tests."""

import os


def isolated_remote_env(**overrides):
    env = {**os.environ, "JETSON_ENV_FILE": os.devnull}
    env.pop("JETSON_SSH_PASSWORD", None)
    env.pop("JETSON_REMOTE_SUDO_PASSWORD", None)
    env.update(overrides)
    return env


def write_executable(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def write_remote_arg_logger(path, *, stdin_mode=None):
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "printf 'CALL\\n' >> \"${FAKE_REMOTE_LOG:?}\"",
    ]
    if stdin_mode == "sudo":
        lines.extend(
            [
                "if [[ \"${1:-}\" == \"sudo\" ]]; then",
                "  IFS= read -r password_from_stdin || true",
                "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                "fi",
            ]
        )
    elif stdin_mode == "bash_lc":
        lines.extend(
            [
                "if [[ \"${1:-}\" == \"bash\" && \"${2:-}\" == \"-lc\" ]]; then",
                "  IFS= read -r password_from_stdin || true",
                "  printf 'STDIN_BYTES=%s\\n' \"${#password_from_stdin}\" >> \"${FAKE_REMOTE_LOG}\"",
                "fi",
            ]
        )
    lines.append("for arg in \"$@\"; do printf 'ARG=%s\\n' \"$arg\" >> \"${FAKE_REMOTE_LOG}\"; done")
    write_executable(path, lines)


def write_fake_suite_scripts(tmp_path):
    fake_sweep = tmp_path / "run_remote_optimization_sweep.sh"
    fake_remote = tmp_path / "remote_exec.sh"
    write_executable(
        fake_sweep,
        [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            "printf 'SWEEP\\n' >> \"${FAKE_SUITE_LOG:?}\"",
            "printf 'ENV_PREPARE=%s\\n' \"${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
            "printf 'ENV_DROP=%s\\n' \"${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-}\" >> \"${FAKE_SUITE_LOG}\"",
            "printf 'ENV_SYNC=%s\\n' \"${JETSON_REMOTE_SYNC:-}\" >> \"${FAKE_SUITE_LOG}\"",
            "printf 'ENV_QWEN3_SELECTOR=%s\\n' \"${JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR:-}\" >> \"${FAKE_SUITE_LOG}\"",
            "printf 'ENV_QWEN3_FALLBACK_MIN_LFB=%s\\n' \"${JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_MIN_LFB_BLOCKS:-}\" >> \"${FAKE_SUITE_LOG}\"",
            "for arg in \"$@\"; do printf 'SWEEP_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
        ],
    )
    write_executable(
        fake_remote,
        [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            "printf 'REMOTE\\n' >> \"${FAKE_SUITE_LOG:?}\"",
            "for arg in \"$@\"; do printf 'REMOTE_ARG=%s\\n' \"$arg\" >> \"${FAKE_SUITE_LOG}\"; done",
        ],
    )
    return fake_sweep, fake_remote
