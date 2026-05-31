# Runtime Matrix

This matrix records the WSL-first and Jetson-Orin-targeted backend strategy. Status is intentionally conservative: a path is "implemented" only when this repository contains runnable scripts/client support. Runtime claims are limited to the exact smoke tests that have been observed.

| Backend | Model Format | WSL Support | Jetson Support | Image Support | Expected Memory Pressure | Implementation Complexity | Current Status | Notes / Blockers |
|---|---|---|---|---|---|---|---|---|
| llama.cpp `llama-server` | GGUF plus optional `mmproj` | Supported by scripts under `scripts/wsl/`, with CPU fallback and CUDA build/run wrappers | Supported by Docker-oriented scripts under `scripts/jetson/` using the canonical Jetson llama.cpp container | Supported when model/mmproj exposes multimodal capability | Medium for Gemma E2B Q8_0/Q4_K_M with small context; medium-high for MiniCPM-V 4.6 with vision; context size drives KV memory and GPU offload drives VRAM | Low-medium | First supported path | Local CPU fallback build, WSL CUDA build, model download scripts, and shared sample images are present. Gemma Q8, Gemma Q4, and MiniCPM-V 4.6 Q4 WSL CUDA text/sample-image benchmark plus fake-stream checks have passed. MiniCPM-V 4.6 Q4 and Gemma Q4 have Jetson formal/repeat evidence with `tegrastats`, startup timing, current-default comparison reports, locked-clock promotion rules, and three-frame fake-stream checks. Jetson Q8, camera input, and long-run behavior remain unverified. |
| llama.cpp `llama-mtmd-cli` | GGUF plus `mmproj` | Documented reference path | Possible if binary/container exists | Image-first CLI support | Similar to server path | Low | Documented only | Useful for manual backend smoke tests, but project client targets OpenAI-compatible server. |
| Ollama | Ollama model bundle / GGUF-backed | Likely usable on WSL | Possible but not first target | Model-dependent | Medium | Low if model exists; opaque runtime packaging | Notes only | Not implemented because llama.cpp exposes the runtime details and OpenAI-compatible API directly. |
| NanoLLM / Jetson AI Lab containers | Backend-specific model packaging | Not WSL-first | Potentially strong Jetson path | Model-dependent | Potentially lower operational overhead on Jetson | Medium | Notes only | Consider after llama.cpp baseline. Need current Jetson container verification before use. |
| vLLM | HF / quantized formats, not GGUF-first | Strong on larger Linux GPUs | Usually heavy for Orin Nano class devices | Model-dependent | High | Medium-high | Not implemented | Avoid for first pass due to memory/storage pressure and non-GGUF focus. |
| TensorRT / TensorRT-LLM | TensorRT engines | Build/debug on WSL is limited for Jetson target | Potential high performance | Vision tower integration would need model-specific work | Potentially efficient after engine build; high conversion cost | High | Not implemented | Deferred until llama.cpp measurements show a specific bottleneck and conversion path is verified. |
| Custom CUDA kernels | Custom native code | Development possible on WSL with CUDA | Targetable on Jetson | Only after model-specific design | Unknown | Very high | Not implemented | Explicitly out of scope for first version. OrangePi custom Ascend kernels are not portable. |

## Current Priority

1. llama.cpp GGUF on WSL with `llama-server`.
2. llama.cpp GGUF on Jetson through the canonical artifact-copy llama.cpp
   image or another Jetson-compatible build that passes the replacement gate.
3. Profiling-backed model/artifact/runtime comparisons under
   `docs/specs/next_phase_infra_and_model_strategy.md`.
4. Optional runtime lanes such as JPS/NanoLLM, TensorRT-LLM, Ollama, and
   Transformers only after profiling shows a bottleneck and an equivalent
   artifact path.
5. No custom CUDA or TensorRT kernels until a separate design proves the
   model-specific bottleneck and acceptance path.

## Model Notes

- Gemma 4 E2B-it: the verified WSL Q8 baseline is `configs/models/gemma4_e2b_q8.yaml` with pre-quantized `Q8_0` model and mmproj files from `ggml-org/gemma-4-E2B-it-GGUF`. The lower-memory Q4 path is `configs/models/gemma4_e2b_q4.yaml` with pre-built `Q4_K_M` artifacts from `mradermacher/gemma-4-E2B-it-GGUF`. Both Q8 and Q4 have passed WSL CUDA text, committed sample-image benchmark, and fake-stream checks with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, and one server slot. Gemma Q4 Jetson formal repeats settled on `CTX_SIZE=512`, `N_GPU_LAYERS=12`, batch/ubatch 512, `q8_0` KV cache, `--no-warmup`, and mmproj on GPU. The newer canonical `d749821db3bd` image passed the 10-trial current-defaults suite, but no tested Gemma flag candidate is ahead of the baseline. Local BF16-to-Q4 quantization exceeded this WSL memory budget, and Q8-to-Q4 re-quantization was stopped by user request because memory was insufficient. The earlier Gemma `LLAMA_UBATCH_SIZE=32` text-only setting triggered a llama.cpp non-causal attention assertion on image requests.
- MiniCPM-V 4.6: the WSL baseline is `configs/models/minicpmv46_q4.yaml` with official pre-built `Q4_K_M` model and F16 mmproj artifacts from `openbmb/MiniCPM-V-4.6-gguf`. WSL CUDA text, committed sample-image benchmark, and fake-stream checks passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=128`, `LLAMA_UBATCH_SIZE=32`, and one server slot. Jetson formal repeats keep the same context/offload/batch shape plus `q8_0` KV cache and `--no-warmup`; no tested MiniCPM flag candidate is ahead of the baseline. Local HF checkpoint conversion and local quantization are not baseline paths for this memory-constrained WSL host.
