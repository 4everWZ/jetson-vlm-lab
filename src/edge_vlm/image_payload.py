"""Image payload helpers for OpenAI-compatible chat requests."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_user_content(prompt: str, image_path: str | Path | None = None) -> str | list[dict[str, Any]]:
    if image_path is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
    ]
