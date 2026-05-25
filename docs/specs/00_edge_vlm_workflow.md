# Edge VLM Workflow Overview

## Purpose

Provide a WSL-first engineering workflow for experimenting with MiniCPM-V 4.6 and Gemma 4 E2B-it on Jetson Orin-class edge devices.

## Scope

In scope:

- llama.cpp `llama-server` as the first backend.
- GGUF model configs.
- OpenAI-compatible client code.
- Shared benchmark cases and JSONL logs.
- Folder-based fake image stream prototype.
- Jetson migration scripts and docs.

Out of scope for the current version:

- Downloading large models.
- Proving performance on WSL or Jetson.
- TensorRT, TensorRT-LLM, vLLM, NanoLLM, Ollama implementation, or custom kernels.
- Real camera integration.
- Porting Ascend-specific OrangePi code.

## Success Criteria

- Developers can build or point to llama.cpp on WSL without global installs.
- Developers can launch script templates for Gemma and MiniCPM-V.
- The Python client can build text and image OpenAI-compatible payloads.
- The benchmark harness logs each case as JSONL with success/failure.
- Jetson migration docs clearly state what to copy, what not to copy, and what remains unverified.

## Decomposition

- Client and benchmark behavior: `docs/specs/dev_client_and_benchmark.md`
- Runtime architecture: `docs/design/edge_vlm_architecture.md`
- WSL to Jetson integration: `docs/specs/integration_wsl_to_jetson.md`
- Coverage matrix: `docs/matrix_edge_vlm_workflow.md`
- Runtime choices: `docs/runtime_matrix.md`
- Benchmark protocol: `docs/benchmark_protocol.md`
- Migration guide: `docs/migration_wsl_to_jetson.md`
