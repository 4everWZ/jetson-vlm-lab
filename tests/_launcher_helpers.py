"""Helpers shared by launcher contract tests."""

import os
import subprocess


def launcher_env(**overrides):
    env = os.environ.copy()
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def run_bash(arguments, *, env):
    return subprocess.run(
        ["bash", *arguments],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )


def run_launcher(launcher_path, *arguments, env):
    return run_bash([launcher_path, *arguments], env=env)


def write_executable(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)
