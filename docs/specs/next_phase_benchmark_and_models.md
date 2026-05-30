# Next Phase: Benchmark Infra And Model Expansion

## Goal

Build a reproducible Jetson benchmark loop before adding more model families. New models must be compared against the same run metadata, repeated trials, power/thermal context, and prompt cases as the existing MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 smoke paths.

## Phase 1: Formal Benchmark Infra

Implement and use a formal benchmark path around the existing `edge_vlm.benchmark` runner.

Required fields:

- `run_id` on every JSONL record.
- `trial_index` and `case_index` on every JSONL record.
- JSON manifest sidecar with model/config paths, benchmark arguments, device label, runtime environment, success/failure counts, and Jetson profile pointers.
- Optional Markdown summary remains a readable sidecar; JSONL plus manifest remain the raw source of truth.
- Jetson wrapper script captures `tegrastats` when available and records `nvpmodel`, `jetson_clocks`, `uname`, and Docker version outputs under a profile directory.

Acceptance:

- Dry-run wrapper works without Jetson hardware or `tegrastats`.
- Real Jetson runs record `tegrastats` next to benchmark JSONL.
- Existing benchmark cases remain compatible.
- Existing smoke output is not reinterpreted as formal performance.

## Phase 2: Lightweight Model Expansion

After Phase 1 is usable, add new models only when they satisfy the repo's low-friction rule:

- GGUF model artifact is available.
- Multimodal projector / `mmproj` path is available or the repo documents equivalent llama.cpp multimodal loading.
- `llama-server` or `llama-mtmd-cli` can load it without local conversion on the memory-constrained WSL host.
- WSL dry-run and payload checks pass before Jetson smoke.
- Jetson support is claimed only after real JSONL plus manifest exists.

Initial candidate order:

| Candidate | Purpose | Source | Initial status |
|---|---|---|---|
| SmolVLM2 256M / 500M | Lowest-resource image baseline and latency floor | `ggml-org/SmolVLM2-256M-Video-Instruct-GGUF`, `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF` | Candidate, not observed in this repo |
| Qwen3-VL-2B | New small Qwen VLM quality/speed comparison | `ggml-org/Qwen3-VL-2B-Instruct-GGUF`, `Qwen/Qwen3-VL-2B-Thinking-GGUF` | Candidate, not observed in this repo |
| Tencent Youtu-VL-4B | Tencent small VLM candidate for Chinese/image reasoning comparison | `tencent/Youtu-VL-4B-Instruct-GGUF` | Candidate, not observed in this repo |
| InternVL3 1B / 2B | Compact OpenGVLab comparison point | `ggml-org/InternVL3-1B-Instruct-GGUF`, `ggml-org/InternVL3-2B-Instruct-GGUF` | Candidate, not observed in this repo |
| Moondream2 | Very small VLM behavior/latency check | `ggml-org/moondream2-20250414-GGUF` | Candidate, not observed in this repo |

Model-specific scripts should be added one family at a time. Do not broaden runtime claims from one candidate to another.

## Phase 3: llama.cpp Acceleration Sweep

Run parameter sweeps only after Phase 1 has at least one formal MiniCPM-V 4.6 Q4 baseline and one formal Gemma Q4 baseline.

Use `docs/specs/jetson_optimization_loop.md` and `scripts/jetson/run_optimization_sweep.sh` for reproducible sweeps. A faster candidate is not promotable unless the optimization report marks its sanity guard as passing.

Sweep variables:

- `N_GPU_LAYERS`
- `CTX_SIZE`
- `LLAMA_BATCH_SIZE`
- `LLAMA_UBATCH_SIZE`
- KV cache type
- mmproj offload on/off
- warmup on/off and cold-start separation
- pinned container image / llama.cpp artifact version

TensorRT, TensorRT-LLM, NanoLLM, Ollama, vLLM, and custom kernels stay deferred until the formal llama.cpp records show a specific bottleneck worth paying integration cost for.

## References Checked

- llama.cpp multimodal documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- SmolVLM2 256M GGUF: https://hf.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF
- SmolVLM2 500M GGUF: https://hf.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF
- Qwen3-VL 2B Instruct GGUF: https://hf.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF
- Qwen3-VL 2B Thinking GGUF: https://hf.co/Qwen/Qwen3-VL-2B-Thinking-GGUF
- Tencent Youtu-VL-4B Instruct GGUF: https://hf.co/tencent/Youtu-VL-4B-Instruct-GGUF
- InternVL3 1B GGUF: https://hf.co/ggml-org/InternVL3-1B-Instruct-GGUF
- InternVL3 2B GGUF: https://hf.co/ggml-org/InternVL3-2B-Instruct-GGUF
- Moondream2 GGUF: https://hf.co/ggml-org/moondream2-20250414-GGUF
