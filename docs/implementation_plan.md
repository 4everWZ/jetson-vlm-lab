# Implementation Plan

This is the current implementation plan for the WSL-first Jetson VLM workflow. It complements the APEX-aligned docs in `docs/specs/`, `docs/design/`, and `docs/matrix_edge_vlm_workflow.md`.

## Scope

Implemented in this iteration:

- WSL and Jetson script templates for llama.cpp `llama-server`.
- Runtime model configs for Gemma 4 E2B-it and MiniCPM-V 4.6.
- Gemma 4 E2B-it Q8_0 and Q4_K_M preparation paths using pre-quantized GGUF files.
- No Gemma local quantization path in the normal WSL workflow because BF16-to-Q4 exceeded this host's memory budget and Q8-to-Q4 re-quantization was stopped by user request.
- A metadata-only MiniCPM-V 4.6 inspection path plus a download path for official pre-built Q4_K_M GGUF and F16 mmproj artifacts.
- A small Python package for OpenAI-compatible chat requests, image payload construction, benchmark runs, and fake image stream runs.
- Benchmark prompt cases and JSONL logging.
- Migration, benchmark, runtime, reference, design, and matrix documentation.

Not implemented in the current version:

- Gemma local BF16-to-Q4 or Q8-to-Q4 quantization on this constrained WSL host.
- A MiniCPM-V 4.6 local HF checkpoint conversion or local quantization path.
- Real Jetson runtime execution.
- TensorRT, TensorRT-LLM, NanoLLM, Ollama, vLLM, or custom kernels.
- Camera access or live video stream capture.

## File Responsibilities

- `scripts/wsl/build_llama_cpp.sh`: Checks for `git` and `cmake`, optionally shallow-clones llama.cpp, and builds `llama-server`, `llama-cli`, and `llama-mtmd-cli`.
- `scripts/wsl/prepare_gemma4_e2b_q8.sh`: Downloads official Gemma Q8_0 model and mmproj GGUF files without local quantization.
- `scripts/wsl/prepare_gemma4_e2b_q4.sh`: Downloads pre-built Gemma Q4_K_M model and mmproj GGUF files without local quantization.
- `scripts/wsl/inspect_minicpmv46_hf.sh`: Checks official MiniCPM-V 4.6 pre-built GGUF repo metadata without downloading model files.
- `scripts/wsl/prepare_minicpmv46_q4.sh`: Downloads official MiniCPM-V 4.6 Q4_K_M model and F16 mmproj GGUF files without local conversion or quantization.
- `scripts/wsl/run_gemma4_e2b_llama.sh`: Launches Gemma 4 E2B-it through local GGUF plus `mmproj`, or through `llama-server -hf`.
- `scripts/wsl/run_minicpmv46_llama.sh`: Launches MiniCPM-V 4.6 through local GGUF plus `mmproj`, or through an explicitly supplied `MODEL_REF`.
- `scripts/jetson/resolve_llama_cpp_image.sh`: Resolves the Jetson llama.cpp container image from `LLAMA_CPP_DOCKER_IMAGE`, `autotag llama_cpp`, or the dusty-nv fallback image.
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
3. Prepare Gemma artifacts with `scripts/wsl/prepare_gemma4_e2b_q8.sh` or `scripts/wsl/prepare_gemma4_e2b_q4.sh`.
4. Inspect MiniCPM-V 4.6 with `scripts/wsl/inspect_minicpmv46_hf.sh`, then download its official pre-built GGUF artifacts with `scripts/wsl/prepare_minicpmv46_q4.sh`.
5. Start one model server on WSL with small context and one server slot.
6. Run `scripts/common/check_server.sh`.
7. Run a dry-run benchmark first, then a real benchmark once server health is confirmed.
8. Copy only source/config/script/docs files to Jetson, not reference repos or WSL build output.
9. Start the Jetson Docker runtime with a dusty-nv `llama_cpp` image and collect `tegrastats` during the benchmark.

## Verification Strategy

Minimum verification for this scaffold:

- Python unit tests in the `transformers` Conda environment.
- Python syntax compile for `src/edge_vlm/*.py`.
- CLI help for Python modules.
- `bash -n` for shell scripts.
- Config JSONL/YAML parse checks.
- Git ignore check confirming reference repos are ignored.

Real inference verification is intentionally separate and requires a running `llama-server` plus model files. In this workspace, WSL CUDA real inference has passed for Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 text/sample-image benchmarks plus one-frame fake-stream runs. Jetson execution remains a hardware-bound verification step.
