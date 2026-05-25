# Implementation Plan

This is the current implementation plan for the WSL-first Jetson VLM workflow. It complements the APEX-aligned docs in `docs/specs/`, `docs/design/`, and `docs/matrix_edge_vlm_workflow.md`.

## Scope

Implemented in this iteration:

- WSL and Jetson script templates for llama.cpp `llama-server`.
- Runtime model configs for Gemma 4 E2B-it and MiniCPM-V 4.6.
- A small Python package for OpenAI-compatible chat requests, image payload construction, benchmark runs, and fake image stream runs.
- Benchmark prompt cases and JSONL logging.
- Migration, benchmark, runtime, reference, design, and matrix documentation.

Not implemented in the current version:

- Real model download or conversion.
- Real WSL llama.cpp build execution.
- Real Jetson runtime execution.
- TensorRT, TensorRT-LLM, NanoLLM, Ollama, vLLM, or custom kernels.
- Camera access or live video stream capture.

## File Responsibilities

- `scripts/wsl/build_llama_cpp.sh`: Checks for `git` and `cmake`, optionally shallow-clones llama.cpp, and builds `llama-server`, `llama-cli`, and `llama-mtmd-cli`.
- `scripts/wsl/run_gemma4_e2b_llama.sh`: Launches Gemma 4 E2B-it through `llama-server -hf`.
- `scripts/wsl/run_minicpmv46_llama.sh`: Launches MiniCPM-V 4.6 through local GGUF plus `mmproj`, or through an explicitly supplied `MODEL_REF`.
- `scripts/jetson/*.sh`: Docker-oriented Jetson launchers and `tegrastats` monitor.
- `configs/models/*.yaml`: Editable runtime model facts and capabilities.
- `configs/benchmark/prompt_cases.jsonl`: Shared WSL/Jetson benchmark cases.
- `src/edge_vlm/client.py`: OpenAI-compatible local server client.
- `src/edge_vlm/image_payload.py`: Base64 data URL image payload construction.
- `src/edge_vlm/benchmark.py`: JSONL benchmark runner.
- `src/edge_vlm/fake_stream.py`: Folder-based fake stream runner.

## Execution Steps

1. Verify the Python package with `PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v`.
2. Build or point to llama.cpp on WSL with `LLAMA_CPP_DIR=/path/to/llama.cpp scripts/wsl/build_llama_cpp.sh`.
3. Start one model server on WSL.
4. Run `scripts/common/check_server.sh`.
5. Run a dry-run benchmark first, then a real benchmark once server health is confirmed.
6. Copy only source/config/script/docs files to Jetson, not reference repos or WSL build output.
7. Start the Jetson Docker runtime and collect `tegrastats` during the benchmark.

## Verification Strategy

Minimum verification for this scaffold:

- Python unit tests in the `transformers` Conda environment.
- Python syntax compile for `src/edge_vlm/*.py`.
- CLI help for Python modules.
- `bash -n` for shell scripts.
- Config JSONL/YAML parse checks.
- Git ignore check confirming reference repos are ignored.

Real inference verification is intentionally separate and requires a running `llama-server` plus model files.
