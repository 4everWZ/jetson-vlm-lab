# OrangePi MiniCPM-V 4.6 Reference Notes

Reference inspected:

- Repository: `https://github.com/lvyufeng/minicpm-v-4.6-orangepi`
- Local clone: `tmp/references/minicpm-v-4.6-orangepi`
- Inspected commit: `490d039`

## What The Repo Does

The OrangePi project is a from-scratch MiniCPM-V 4.6 inference engine for Ascend 310B hardware on Orange Pi AIPro. It moves the hot inference path into C++/AscendC and leaves Python responsible for CPU-side tokenizer, chat-template handling, image preprocessing, and UI/session orchestration.

The key Python boundary is in `src/python/minicpmv/session.py`. A persistent engine subprocess owns NPU execution; Python prepares token IDs and image tensors, writes a small request bundle, then streams decoded tokens back from the engine. The Gradio and CLI layers call this session wrapper rather than directly owning model execution.

The native side has a strong diagnostics culture. The README lists benches for decode, prefill, decode sub-ops, matmul throughput, step kernels, NPU bandwidth, host DDR bandwidth, and correctness tests against reference outputs. The test suite also validates lower-level operations and end-to-end-ish slices such as prefill from embeddings, autoregressive decode, lm_head, and vision patch embedding.

## Portable Ideas For Jetson

- Keep Python out of the hot path. Python can own orchestration, request construction, logging, and image selection, while `llama-server` owns inference.
- Use a persistent server or engine process instead of one process per request.
- Separate UI/client logic from model runtime logic. This project follows that through `src/edge_vlm/client.py`, benchmark runners, and shell launchers.
- Start with constrained deployment cases: batch 1, one image, small context, Q4 quantization, and fixed benchmark prompts.
- Measure prefill/decode and image cases separately before making broad performance claims.
- Keep correctness/reference checks distinct from performance checks.
- Treat single-image and short-context behavior as the first acceptance path before multi-image, video, long context, or real camera streams.

## Not Portable To Jetson

The following are Ascend-specific and must not be ported into this Jetson workflow:

- AscendC custom operators and kernel code.
- CANN, ACL, aclnn runtime calls, and custom OPP installation.
- OrangePi AIPro build scripts and environment setup.
- Cube-unit matmul implementation details.
- Ascend-specific NPU memory, stream, and operator scheduling assumptions.

Jetson runtime work should start with llama.cpp CUDA or a Jetson-compatible llama.cpp container, not by adapting Ascend code.

## Useful Engineering Patterns

- Native engine boundary: Python sends normalized request data to a persistent native process.
- Explicit resource modes: the reference can skip vision loading for text-only use, saving accelerator memory.
- Microbenchmarks first: the project measures individual decode sub-ops instead of relying only on end-to-end latency.
- Regression alignment: engine outputs are compared against PyTorch/HF references, especially for logits and vision features.
- Clear constraints: the README calls out single-batch greedy decode and vision slicing limitations rather than hiding them.

## Risks And Limitations

- The OrangePi numbers are not Jetson numbers. They measure Ascend 310B behavior and custom kernels.
- MiniCPM-V 4.6 image behavior depends on processor slicing, image token placement, and mmproj support. This project has observed a WSL CUDA llama.cpp request path with official pre-built GGUF artifacts, but Jetson behavior still needs device-side verification.
- The reference repo hardcodes model-shape knowledge in native code. That is useful for edge optimization, but this project should avoid model-shape assumptions until a backend exposes stable contracts.
- Multi-slice and high-resolution image handling can affect both correctness and latency. First-version benchmarks should mark image cases as single-image tests, not exhaustive VLM evaluation.

## Concrete Lessons For This Project

1. Build WSL tools first, then move only scripts, configs, client package, and benchmarks to Jetson.
2. Use llama.cpp as the first runtime boundary so custom CUDA/TensorRT work is deferred until there is a measured reason.
3. Keep benchmark output raw JSONL, with explicit success/failure and error text per case.
4. Do not claim Jetson performance or image correctness until a Jetson `llama-server` run has produced observed logs.
5. Keep model configs editable. MiniCPM-V 4.6 currently uses official pre-built GGUF plus mmproj artifacts; do not reintroduce local conversion as the default path without a measured reason.
