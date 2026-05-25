# Edge VLM Architecture

## Architecture

The project uses a thin-client architecture:

```text
WSL development workspace
  -> scripts choose runtime and model config
  -> llama-server owns inference
  -> edge_vlm client sends OpenAI-compatible requests
  -> benchmark/fake-stream tools write JSONL observations
  -> selected files sync to Jetson
```

`llama-server` is the backend boundary. This avoids guessing tokenizer internals, image tensor layouts, or model-specific preprocessing in this repository.

## Runtime Boundary

The Python package never loads VLM weights. It only:

- reads config,
- builds text or image chat payloads,
- sends HTTP requests,
- logs structured results.

Backend-specific support is represented by config capabilities and launch scripts, not hidden code branches.

## WSL / Jetson Split

WSL:

- inspect references,
- edit code/config/docs,
- run unit tests and dry-run benchmarks,
- build llama.cpp when appropriate,
- convert or stage models if disk space allows.

Jetson:

- receive minimal runtime package,
- load model files from NVMe/external storage,
- run `llama-server`,
- run benchmark/fake-stream tools,
- collect `tegrastats`.

## Data Flow

1. A model YAML selects model alias, backend, capabilities, and runtime defaults.
2. A run script starts `llama-server`.
3. The client reads the same model YAML and builds request payloads.
4. Benchmarks write one JSON object per case.
5. Results remain local under `outputs/`, which is ignored by Git.

## Unsupported Paths

Not implemented in the current version.

- TensorRT engine generation.
- Direct PyTorch Transformers hot path.
- Ollama/NanoLLM runtime wrappers.
- Real camera capture.
- Non-OpenAI-compatible llama.cpp `/completion` multimodal marker payloads.
