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
        |
        v
optional Markdown summary
```

## Integration Dependencies

- WSL: Conda env `transformers` for Python validation, plus local llama.cpp CPU fallback and CUDA build paths.
- Jetson: Docker and NVIDIA runtime for the provided Docker scripts, or an equivalent local llama.cpp build.
- Models: Gemma uses prepared Q8_0 or Q4_K_M GGUF plus `mmproj` artifacts on WSL; MiniCPM-V 4.6 uses official pre-built Q4_K_M GGUF plus F16 `mmproj` artifacts from `openbmb/MiniCPM-V-4.6-gguf`.

## Validation Checklist

- Shell scripts pass `bash -n`.
- Python package passes unit tests and syntax compile.
- Configs parse.
- `scripts/common/check_server.sh` passes against a running server.
- Benchmark dry run writes JSONL.
- Benchmark dry run can write an optional Markdown summary.
- Shared image and fake-stream sample assets are present for out-of-box payload checks.
- Real Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 WSL CUDA benchmark records observed text and sample-image server output.
- Real Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 WSL CUDA fake-stream runs record observed frame output.
- Jetson run records `tegrastats` alongside benchmark JSONL.

## Benchmark / Regression Entry Points

- Unit tests: `PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v`
- Dry benchmark: `python -m edge_vlm.benchmark --dry-run --summary-output ...`
- Fake stream: `python -m edge_vlm.fake_stream --dry-run ...`
- Server health: `scripts/common/check_server.sh`
- Jetson launcher self-check: `JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh`

## Known Hard Boundaries

- No performance claim is valid until a real model/server run completes.
- No Jetson support claim is valid until the Jetson Docker/runtime path is observed.
- MiniCPM-V 4.6 local HF checkpoint conversion and local quantization are not WSL acceptance paths on this memory-constrained host; use the official pre-built GGUF artifacts.
- Gemma local quantization is not a WSL acceptance path on this host; use the prepared Q8_0 artifacts or the downloaded pre-built Q4_K_M artifact.
- Gemma Q8 WSL CUDA image requests require the observed `LLAMA_BATCH_SIZE=512` and `LLAMA_UBATCH_SIZE=512` setting on this llama.cpp build; the lower text-only `LLAMA_UBATCH_SIZE=32` setting triggered a non-causal attention assertion.

## Final Acceptance Status

Partially implemented. The scaffold, scripts, configs, tests, docs, shared sample assets, optional benchmark Markdown summaries, and ignored local model artifacts exist. Gemma Q8_0 text-only WSL smoke passed on the local CPU fallback llama.cpp build with low-memory settings. WSL CUDA build/run wrappers exist; the CUDA build is verified with `CMAKE_CUDA_ARCHITECTURES=86`, `BUILD_JOBS=8`, and `GGML_CUDA=ON`. Gemma Q8_0, Gemma Q4_K_M, and MiniCPM-V 4.6 Q4_K_M WSL CUDA text/sample-image benchmarks passed with one server slot; real fake-stream runs against one committed sample frame also passed. Jetson inference and broader performance remain unverified.
