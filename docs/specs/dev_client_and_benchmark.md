# Client And Benchmark Development Spec

## Goals & Boundaries

The Python package provides a small local client and harness layer. It does not own model loading, tokenizer internals, tensor shapes, or backend-specific preprocessing beyond OpenAI-compatible request construction.

Not implemented in the current version.

- Backend-specific low-level multimodal marker handling for non-chat endpoints.
- OpenAI SDK dependency.

## Interfaces / Responsibilities

- `edge_vlm.client.OpenAICompatClient`: Sends `/v1/chat/completions` requests and supports dry-run payload validation.
- `edge_vlm.image_payload`: Converts local images into data URLs and builds content parts.
- `edge_vlm.benchmark.run_benchmark`: Reads JSONL cases, writes JSONL results, and can write a Markdown summary for the current run.
- `edge_vlm.fake_stream.run_fake_stream`: Iterates a sorted image folder and writes one response record per frame.
- `edge_vlm.config.load_model_config`: Loads YAML/JSON model configs with environment overrides for host and port.

The benchmark treats `input_type=fake_stream` as a marker case and records an instruction to run `edge_vlm.fake_stream`. It does not expand a folder inside the normal benchmark loop.

## Code Mapping

- Source: `src/edge_vlm/*.py`
- Tests: `tests/test_edge_vlm.py`
- Configs: `configs/models/*.yaml`
- Cases: `configs/benchmark/prompt_cases.jsonl`
- Sample assets: `data/sample_images/`, `data/sample_stream/`

## Tradeoffs

- The client uses standard-library HTTP to avoid adding project dependencies in this scaffold.
- The image payload uses OpenAI-style `image_url` content because llama.cpp documents this for `/v1/chat/completions`; server capability still needs runtime verification.
- YAML parsing supports PyYAML when present and a narrow fallback parser for current config files.
- Fake-stream execution is separate from the benchmark loop so frame-level failure handling can be explicit and resumable.
- Markdown summaries are optional sidecar files generated from records written in the current run; JSONL remains the raw source of truth.

## Verification

Run:

```bash
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
PYTHONPATH=src conda run -n transformers python -m py_compile src/edge_vlm/*.py
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-dryrun.md \
  --dry-run
```
