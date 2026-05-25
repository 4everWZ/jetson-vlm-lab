# Client And Benchmark Development Spec

## Goals & Boundaries

The Python package provides a small local client and harness layer. It does not own model loading, tokenizer internals, tensor shapes, or backend-specific preprocessing beyond OpenAI-compatible request construction.

Not implemented in the current version.

- Backend-specific low-level multimodal marker handling for non-chat endpoints.
- OpenAI SDK dependency.
- Markdown summary generation.

## Interfaces / Responsibilities

- `edge_vlm.client.OpenAICompatClient`: Sends `/v1/chat/completions` requests and supports dry-run payload validation.
- `edge_vlm.image_payload`: Converts local images into data URLs and builds content parts.
- `edge_vlm.benchmark.run_benchmark`: Reads JSONL cases and writes JSONL results.
- `edge_vlm.fake_stream.run_fake_stream`: Iterates a sorted image folder and writes one response record per frame.
- `edge_vlm.config.load_model_config`: Loads YAML/JSON model configs with environment overrides for host and port.

## Code Mapping

- Source: `src/edge_vlm/*.py`
- Tests: `tests/test_edge_vlm.py`
- Configs: `configs/models/*.yaml`
- Cases: `configs/benchmark/prompt_cases.jsonl`

## Tradeoffs

- The client uses standard-library HTTP to avoid adding project dependencies in this scaffold.
- The image payload uses OpenAI-style `image_url` content because llama.cpp documents this for `/v1/chat/completions`; server capability still needs runtime verification.
- YAML parsing supports PyYAML when present and a narrow fallback parser for current config files.

## Verification

Run:

```bash
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
PYTHONPATH=src conda run -n transformers python -m py_compile src/edge_vlm/*.py
```
