"""Image payload helpers for OpenAI-compatible chat requests."""

from __future__ import annotations

import base64
import mimetypes
import time
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


def image_to_data_url_with_timing(image_path: str | Path) -> tuple[str, dict[str, Any]]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    mime_started = time.perf_counter()
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_detect_s = time.perf_counter() - mime_started
    if not mime_type:
        mime_type = "application/octet-stream"

    read_started = time.perf_counter()
    image_bytes = path.read_bytes()
    image_read_s = time.perf_counter() - read_started

    encode_started = time.perf_counter()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    base64_encode_s = time.perf_counter() - encode_started

    build_started = time.perf_counter()
    data_url = f"data:{mime_type};base64,{encoded}"
    data_url_build_s = time.perf_counter() - build_started

    return data_url, {
        "image_bytes": len(image_bytes),
        "mime_detect_s": mime_detect_s,
        "image_read_s": image_read_s,
        "base64_encode_s": base64_encode_s,
        "data_url_build_s": data_url_build_s,
    }


def build_user_content(prompt: str, image_path: str | Path | None = None) -> str | list[dict[str, Any]]:
    content, _timing = build_user_content_with_timing(prompt, image_path)
    return content


def build_user_content_with_timing(
    prompt: str,
    image_path: str | Path | None = None,
) -> tuple[str | list[dict[str, Any]], dict[str, Any]]:
    if image_path is None:
        return prompt, {
            "image_bytes": 0,
            "mime_detect_s": 0.0,
            "image_read_s": 0.0,
            "base64_encode_s": 0.0,
            "data_url_build_s": 0.0,
        }
    data_url, timing = image_to_data_url_with_timing(image_path)
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ], timing
