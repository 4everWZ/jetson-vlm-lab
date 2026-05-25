"""Model config loading for edge VLM scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by configs/models/*.yaml.

    This intentionally supports only nested mappings with scalar values. If a
    config needs lists or advanced YAML features, install/use PyYAML in the
    project environment and this loader will defer to it.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            raise ValueError(f"unsupported YAML list at line {lineno}")
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"indentation must use multiples of two spaces at line {lineno}")
        if ":" not in stripped:
            raise ValueError(f"expected key: value at line {lineno}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key at line {lineno}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation at line {lineno}")
        current = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return loaded
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"model config not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a JSON object")
        config = loaded
    else:
        config = _load_yaml(config_path)

    server = config.setdefault("server", {})
    if not isinstance(server, dict):
        raise ValueError("server config must be a mapping")
    host = os.environ.get("VLM_SERVER_HOST")
    port = os.environ.get("VLM_SERVER_PORT")
    if host or port:
        host_value = host or "127.0.0.1"
        port_value = port or "8080"
        server["base_url"] = f"http://{host_value}:{port_value}/v1"
    return config


def config_model_name(config: dict[str, Any]) -> str:
    model = config.get("model", {})
    if isinstance(model, dict):
        name = model.get("name") or model.get("alias") or model.get("model_ref")
        if name:
            return str(name)
    raise ValueError("model.name, model.alias, or model.model_ref is required")


def config_base_url(config: dict[str, Any]) -> str:
    server = config.get("server", {})
    if isinstance(server, dict) and server.get("base_url"):
        return str(server["base_url"]).rstrip("/")
    return "http://127.0.0.1:8080/v1"


def config_supports_images(config: dict[str, Any]) -> bool:
    capabilities = config.get("capabilities", {})
    if isinstance(capabilities, dict):
        return bool(capabilities.get("image", False))
    return False
