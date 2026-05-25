# WSL To Jetson Integration Spec

## End-To-End Assembly View

WSL owns development, reference inspection, Python client work, config editing, and first dry-run validation. Jetson owns only the edge runtime execution and benchmark collection.

```text
configs/models/*.yaml
        |
        v
scripts/wsl or scripts/jetson start llama-server
        |
        v
src/edge_vlm client sends OpenAI-compatible requests
        |
        v
outputs/*.jsonl benchmark records
```

## Integration Dependencies

- WSL: Conda env `transformers` for Python validation, plus optional local llama.cpp build.
- Jetson: Docker and NVIDIA runtime for the provided Docker scripts, or an equivalent local llama.cpp build.
- Models: Gemma uses the prepared Q8_0 GGUF plus `mmproj` baseline on WSL; MiniCPM-V 4.6 defaults to local converted GGUF plus `mmproj`.

## Validation Checklist

- Shell scripts pass `bash -n`.
- Python package passes unit tests and syntax compile.
- Configs parse.
- `scripts/common/check_server.sh` passes against a running server.
- Benchmark dry run writes JSONL.
- Real benchmark run records observed server output.
- Jetson run records `tegrastats` alongside benchmark JSONL.

## Benchmark / Regression Entry Points

- Unit tests: `PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v`
- Dry benchmark: `python -m edge_vlm.benchmark --dry-run ...`
- Fake stream: `python -m edge_vlm.fake_stream --dry-run ...`
- Server health: `scripts/common/check_server.sh`

## Known Hard Boundaries

- No performance claim is valid until a real model/server run completes.
- No Jetson support claim is valid until the Jetson Docker/runtime path is observed.
- MiniCPM-V 4.6 conversion and mmproj compatibility must be verified on the exact llama.cpp revision used for runtime.
- Gemma BF16-to-Q4 local quantization is not a WSL acceptance path on this host; use the prepared Q8_0 artifacts or an externally prepared lower-bit artifact.

## Final Acceptance Status

Partially implemented. The scaffold, scripts, configs, tests, docs, and Gemma Q8_0 local artifacts exist. Real WSL inference requires a passing server smoke test; Jetson inference remains unverified.
