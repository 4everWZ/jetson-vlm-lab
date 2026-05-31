"""Small OpenAI-compatible client for llama-server experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import config_base_url, config_model_name, config_supports_images, load_model_config
from .image_payload import build_user_content_with_timing


@dataclass(frozen=True)
class CompletionResult:
    ok: bool
    text: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    latency_s: float
    error: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, config: dict[str, Any], timeout_s: float = 120.0) -> "OpenAICompatClient":
        return cls(base_url=config_base_url(config), model=config_model_name(config), timeout_s=timeout_s)

    def build_chat_payload(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        stream: bool = False,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        payload, _timings = self._build_chat_payload_with_timing(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            image_path=image_path,
        )
        return payload

    def _build_chat_payload_with_timing(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        stream: bool = False,
        image_path: str | Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload_started = time.perf_counter()
        content, timings = build_user_content_with_timing(prompt, image_path)
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }, {
            **timings,
            "payload_build_s": time.perf_counter() - payload_started,
        }

    @staticmethod
    def _serialize_payload(request_payload: dict[str, Any], timings: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        updated = dict(timings)
        serialize_started = time.perf_counter()
        body = json.dumps(request_payload).encode("utf-8")
        updated["json_serialize_s"] = time.perf_counter() - serialize_started
        updated["request_body_bytes"] = len(body)
        return body, updated

    def complete(
        self,
        *,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.2,
        stream: bool = False,
        image_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> CompletionResult:
        request_payload, timings = self._build_chat_payload_with_timing(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            image_path=image_path,
        )
        started = time.perf_counter()
        if dry_run:
            _body, timings = self._serialize_payload(request_payload, timings)
            timings["http_request_s"] = 0.0
            return CompletionResult(
                ok=True,
                text=f"[dry run] would POST {self.base_url}/chat/completions",
                request=request_payload,
                response={"dry_run": True},
                latency_s=time.perf_counter() - started,
                timings=timings,
            )
        if stream:
            return self._streaming_complete(request_payload, started, timings)
        return self._non_streaming_complete(request_payload, started, timings)

    def _non_streaming_complete(
        self,
        request_payload: dict[str, Any],
        started: float,
        timings: dict[str, Any],
    ) -> CompletionResult:
        url = f"{self.base_url}/chat/completions"
        body, timings = self._serialize_payload(request_payload, timings)
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
            method="POST",
        )
        try:
            http_started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
            timings["http_request_s"] = time.perf_counter() - http_started
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            timings.setdefault("http_request_s", time.perf_counter() - started)
            return CompletionResult(
                ok=False,
                text="",
                request=request_payload,
                response=None,
                latency_s=time.perf_counter() - started,
                error=str(exc),
                timings=timings,
            )

        try:
            parse_started = time.perf_counter()
            parsed = json.loads(raw)
            text = _extract_text(parsed)
            timings["response_parse_s"] = time.perf_counter() - parse_started
            return CompletionResult(
                ok=True,
                text=text,
                request=request_payload,
                response=parsed,
                latency_s=time.perf_counter() - started,
                timings=timings,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            timings.setdefault("response_parse_s", 0.0)
            return CompletionResult(
                ok=False,
                text="",
                request=request_payload,
                response={"raw": raw},
                latency_s=time.perf_counter() - started,
                error=f"could not parse chat response: {exc}",
                timings=timings,
            )

    def _streaming_complete(
        self,
        request_payload: dict[str, Any],
        started: float,
        timings: dict[str, Any],
    ) -> CompletionResult:
        payload = dict(request_payload)
        payload["stream"] = True
        url = f"{self.base_url}/chat/completions"
        body, timings = self._serialize_payload(payload, timings)
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
            method="POST",
        )
        chunks: list[str] = []
        try:
            http_started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    parsed = json.loads(data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        chunks.append(str(text))
            timings["http_request_s"] = time.perf_counter() - http_started
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            timings.setdefault("http_request_s", time.perf_counter() - started)
            return CompletionResult(
                ok=False,
                text="".join(chunks),
                request=payload,
                response=None,
                latency_s=time.perf_counter() - started,
                error=str(exc),
                timings=timings,
            )
        return CompletionResult(
            ok=True,
            text="".join(chunks),
            request=payload,
            response={"stream": True},
            latency_s=time.perf_counter() - started,
            timings=timings,
        )


def _extract_text(response: dict[str, Any]) -> str:
    choice = response["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        if text:
            return text
    reasoning_content = message.get("reasoning_content", "")
    if isinstance(reasoning_content, str):
        return reasoning_content
    if isinstance(reasoning_content, list):
        return "".join(str(part.get("text", "")) for part in reasoning_content if isinstance(part, dict))
    return str(content)


def _print_jsonl(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one prompt to an OpenAI-compatible local VLM server.")
    parser.add_argument("--config", required=True, help="Path to configs/models/*.yaml")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--image", default=None, help="Optional image path")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_model_config(args.config)
    if args.image and not config_supports_images(config):
        parser.error("selected model config has capabilities.image=false; choose an image-capable config")
    client = OpenAICompatClient.from_config(config)
    result = client.complete(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
        image_path=args.image,
        dry_run=args.dry_run,
    )
    _print_jsonl(
        [
            {
                "ok": result.ok,
                "latency_s": result.latency_s,
                "text": result.text,
                "error": result.error,
            }
        ]
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
