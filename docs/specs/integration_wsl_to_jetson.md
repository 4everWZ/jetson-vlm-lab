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

- WSL: Conda env `transformers` for Python validation, plus local llama.cpp CPU fallback and CUDA build paths.
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
- MiniCPM-V 4.6 metadata-only inspection has observed HF file sizes and local llama.cpp conversion signals, but conversion and mmproj runtime compatibility must still be verified on the exact llama.cpp revision used for runtime.
- Gemma BF16-to-Q4 local quantization is not a WSL acceptance path on this host; use the prepared Q8_0 artifacts or an externally prepared lower-bit artifact.

## Final Acceptance Status

Partially implemented. The scaffold, scripts, configs, tests, docs, and Gemma Q8_0 local artifacts exist. Gemma Q8_0 text-only WSL smoke passed on the local CPU fallback llama.cpp build with low-memory settings. WSL CUDA build/run wrappers exist; the CUDA build is verified with `CMAKE_CUDA_ARCHITECTURES=86`, `BUILD_JOBS=8`, and `GGML_CUDA=ON`. Gemma Q8_0 WSL CUDA text smoke passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, and one server slot; the shared benchmark harness recorded three successful text cases, two missing-image failures, and the fake-stream marker case. MiniCPM-V 4.6 metadata-only inspection passed without downloading weights. Image requests, MiniCPM-V 4.6 conversion/runtime, Jetson inference, and broader performance remain unverified.
