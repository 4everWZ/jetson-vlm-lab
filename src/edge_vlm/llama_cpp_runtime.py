"""Probe llama.cpp Docker runtime metadata and multimodal server support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _text_tail(value: str | None, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def docker_image_metadata(image: str | None) -> dict[str, Any]:
    if not image:
        return {
            "image": None,
            "inspect_ok": False,
            "inspect_error": "LLAMA_CPP_DOCKER_IMAGE not set in sweep environment",
        }
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "image": image,
            "inspect_ok": False,
            "inspect_error": "docker command not found",
        }
    if result.returncode != 0:
        return {
            "image": image,
            "inspect_ok": False,
            "inspect_error": _text_tail(result.stderr or result.stdout, max_chars=1000),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "image": image,
            "inspect_ok": False,
            "inspect_error": f"invalid docker inspect JSON: {exc}",
        }
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return {
            "image": image,
            "inspect_ok": False,
            "inspect_error": "docker inspect returned no image object",
        }
    image_obj = payload[0]
    config = image_obj.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    labels = labels if isinstance(labels, dict) else {}
    return {
        "image": image,
        "inspect_ok": True,
        "image_id": image_obj.get("Id"),
        "created": image_obj.get("Created"),
        "repo_digests": image_obj.get("RepoDigests") if isinstance(image_obj.get("RepoDigests"), list) else [],
        "llama_cpp_ref": labels.get("org.opencontainers.image.version"),
        "source_revision": labels.get("org.opencontainers.image.revision"),
        "source": labels.get("org.opencontainers.image.source"),
        "base_image": labels.get("org.opencontainers.image.base.name"),
    }


def llama_server_probe_default() -> dict[str, Any]:
    return {
        "llama_server_probe_ok": False,
        "llama_server_probe_error": None,
        "llama_server_found": None,
        "llama_server_path": None,
        "llama_server_help_ok": None,
        "llama_server_supports_mmproj": None,
        "llama_server_multimodal_markers": [],
    }


def parse_probe_bool(value: str | None) -> bool | None:
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def docker_image_runtime_probe(
    image: str | None,
    llama_server_cmd: str | None = None,
    docker_gpu_args: str | None = None,
) -> dict[str, Any]:
    probe = llama_server_probe_default()
    if not image:
        probe["llama_server_probe_error"] = "LLAMA_CPP_DOCKER_IMAGE not set in sweep environment"
        return probe
    probe_script = """
set -Eeuo pipefail
server_path=""
server_cmd=()
if [[ -n "${EDGE_VLM_LLAMA_SERVER_CMD:-}" ]]; then
  read -r -a server_cmd <<< "${EDGE_VLM_LLAMA_SERVER_CMD}"
  if [[ ${#server_cmd[@]} -gt 0 ]]; then
    server_path="${server_cmd[0]}"
  fi
else
  if command -v llama-server >/dev/null 2>&1; then
    server_path="$(command -v llama-server)"
  elif [[ -x /usr/local/bin/llama-server ]]; then
    server_path="/usr/local/bin/llama-server"
  elif [[ -x /opt/llama.cpp/build/bin/llama-server ]]; then
    server_path="/opt/llama.cpp/build/bin/llama-server"
  elif command -v server >/dev/null 2>&1; then
    server_path="$(command -v server)"
  fi
  if [[ -n "${server_path}" ]]; then
    server_cmd=("${server_path}")
  fi
fi

if [[ -z "${server_path}" ]]; then
  printf 'llama_server_found=0\\n'
  printf 'llama_server_path=\\n'
  printf 'llama_server_help_ok=\\n'
  printf 'llama_server_supports_mmproj=\\n'
  printf 'llama_server_multimodal_markers=\\n'
  exit 0
fi

printf 'llama_server_found=1\\n'
printf 'llama_server_path=%s\\n' "${server_path}"
help_output=""
if help_output="$("${server_cmd[@]}" --help 2>&1)"; then
  printf 'llama_server_help_ok=1\\n'
else
  printf 'llama_server_help_ok=0\\n'
  printf 'llama_server_supports_mmproj=\\n'
  printf 'llama_server_multimodal_markers=\\n'
  exit 0
fi

markers=()
supports_mmproj=0
if grep -Fq -- "--mmproj" <<< "${help_output}"; then
  markers+=("--mmproj")
  supports_mmproj=1
fi
if grep -Fq -- "mmproj" <<< "${help_output}"; then
  markers+=("mmproj")
fi
if [[ "${supports_mmproj}" == "1" ]]; then
  printf 'llama_server_supports_mmproj=1\\n'
else
  printf 'llama_server_supports_mmproj=0\\n'
fi
printf 'llama_server_multimodal_markers=%s\\n' "$(IFS=,; printf '%s' "${markers[*]}")"
"""
    env = None
    if llama_server_cmd:
        env = {
            **os.environ,
            "EDGE_VLM_LLAMA_SERVER_CMD": str(llama_server_cmd),
        }
    gpu_args_text = str(docker_gpu_args or "--runtime nvidia").strip()
    gpu_args = gpu_args_text.split() if gpu_args_text else []
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", *gpu_args, "--entrypoint", "/bin/bash", image, "-lc", probe_script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        probe["llama_server_probe_error"] = "docker command not found"
        return probe
    if result.returncode != 0:
        probe["llama_server_probe_error"] = _text_tail(result.stderr or result.stdout, max_chars=1000)
        return probe
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    probe.update(
        {
            "llama_server_probe_ok": True,
            "llama_server_found": parse_probe_bool(parsed.get("llama_server_found")),
            "llama_server_path": parsed.get("llama_server_path") or None,
            "llama_server_help_ok": parse_probe_bool(parsed.get("llama_server_help_ok")),
            "llama_server_supports_mmproj": parse_probe_bool(parsed.get("llama_server_supports_mmproj")),
            "llama_server_multimodal_markers": [
                marker
                for marker in (parsed.get("llama_server_multimodal_markers") or "").split(",")
                if marker
            ],
        }
    )
    return probe


def runtime_is_multimodal_ready(runtime: dict[str, Any]) -> bool:
    return (
        runtime.get("llama_server_probe_ok") is True
        and runtime.get("llama_server_found") is True
        and runtime.get("llama_server_help_ok") is True
        and runtime.get("llama_server_supports_mmproj") is True
    )


def runtime_metadata(
    image: str | None,
    llama_server_cmd: str | None = None,
    docker_gpu_args: str | None = None,
) -> dict[str, Any]:
    metadata = docker_image_metadata(image)
    metadata.update(docker_image_runtime_probe(image, llama_server_cmd, docker_gpu_args))
    metadata["multimodal_ready"] = runtime_is_multimodal_ready(metadata)
    return metadata


def build_runtime_probe_artifact(
    *,
    image: str,
    output_path: str | Path,
    llama_server_cmd: str | None = None,
    docker_gpu_args: str | None = None,
) -> dict[str, Any]:
    artifact = {
        "kind": "llama_cpp_runtime_probe",
        **runtime_metadata(image, llama_server_cmd=llama_server_cmd, docker_gpu_args=docker_gpu_args),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe llama.cpp Docker image runtime support.")
    subparsers = parser.add_subparsers(dest="command")
    probe_parser = subparsers.add_parser(
        "probe-image",
        help="Write Docker image metadata and exact llama-server --mmproj support probe JSON",
    )
    probe_parser.add_argument("--image", required=True, help="llama.cpp Docker image tag or digest")
    probe_parser.add_argument("--output", required=True, help="Runtime probe JSON output path")
    probe_parser.add_argument("--llama-server-cmd", help="Optional llama-server command/path inside the image")
    probe_parser.add_argument(
        "--docker-gpu-args",
        default="--runtime nvidia",
        help="Docker GPU/runtime args for the probe container; use an empty string to disable",
    )
    args = parser.parse_args(argv)
    if args.command != "probe-image":
        parser.print_help()
        return 2
    artifact = build_runtime_probe_artifact(
        image=args.image,
        output_path=args.output,
        llama_server_cmd=args.llama_server_cmd,
        docker_gpu_args=args.docker_gpu_args,
    )
    print(
        json.dumps(
            {
                "image": args.image,
                "multimodal_ready": artifact["multimodal_ready"],
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
