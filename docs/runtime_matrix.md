# Runtime Matrix

This matrix records the WSL-first and Jetson-Orin-targeted backend strategy. Status is intentionally conservative: a path is "implemented" only when this repository contains runnable scripts/client support. Runtime claims are limited to the exact smoke tests that have been observed.

| Backend | Model Format | WSL Support | Jetson Support | Image Support | Expected Memory Pressure | Implementation Complexity | Current Status | Notes / Blockers |
|---|---|---|---|---|---|---|---|---|
| llama.cpp `llama-server` | GGUF plus optional `mmproj` | Supported by scripts under `scripts/wsl/`, with CPU fallback and CUDA build/run wrappers | Supported by Docker-oriented scripts under `scripts/jetson/` | Supported when model/mmproj exposes multimodal capability | Medium for Gemma E2B Q8_0 with small context; medium-high for MiniCPM-V 4.6 with vision; context size drives KV memory and GPU offload drives VRAM | Low-medium | First supported path | Local CPU fallback build, Gemma Q8_0 artifacts, and shared sample images are present. Gemma Q8_0 text smoke previously passed on WSL CPU fallback with `CTX_SIZE=512`, `N_GPU_LAYERS=0`, one server slot, and no warmup. WSL CUDA build is verified under `tmp/llama.cpp/build-cuda` with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, and `BUILD_JOBS=8`. Gemma Q8 WSL CUDA text and sample-image benchmark passed with `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `LLAMA_BATCH_SIZE=512`, `LLAMA_UBATCH_SIZE=512`, and one server slot. MiniCPM-V, Jetson, and broader performance remain unverified. |
| llama.cpp `llama-mtmd-cli` | GGUF plus `mmproj` | Documented reference path | Possible if binary/container exists | Image-first CLI support | Similar to server path | Low | Documented only | Useful for manual backend smoke tests, but project client targets OpenAI-compatible server. |
| Ollama | Ollama model bundle / GGUF-backed | Likely usable on WSL | Possible but not first target | Model-dependent | Medium | Low if model exists; opaque runtime packaging | Notes only | Not implemented because llama.cpp exposes the runtime details and OpenAI-compatible API directly. |
| NanoLLM / Jetson AI Lab containers | Backend-specific model packaging | Not WSL-first | Potentially strong Jetson path | Model-dependent | Potentially lower operational overhead on Jetson | Medium | Notes only | Consider after llama.cpp baseline. Need current Jetson container verification before use. |
| vLLM | HF / quantized formats, not GGUF-first | Strong on larger Linux GPUs | Usually heavy for Orin Nano class devices | Model-dependent | High | Medium-high | Not implemented | Avoid for first pass due to memory/storage pressure and non-GGUF focus. |
| TensorRT / TensorRT-LLM | TensorRT engines | Build/debug on WSL is limited for Jetson target | Potential high performance | Vision tower integration would need model-specific work | Potentially efficient after engine build; high conversion cost | High | Not implemented | Deferred until llama.cpp measurements show a specific bottleneck and conversion path is verified. |
| Custom CUDA kernels | Custom native code | Development possible on WSL with CUDA | Targetable on Jetson | Only after model-specific design | Unknown | Very high | Not implemented | Explicitly out of scope for first version. OrangePi custom Ascend kernels are not portable. |

## Initial Priority

1. llama.cpp GGUF on WSL with `llama-server`.
2. llama.cpp GGUF on Jetson through Docker or a Jetson-compatible build.
3. Optional notes for Ollama/NanoLLM after baseline logs exist.
4. No custom CUDA or TensorRT kernels in the first version.

## Model Notes

- Gemma 4 E2B-it: the WSL baseline is `configs/models/gemma4_e2b_q8.yaml` with official pre-quantized `Q8_0` model and mmproj files from `ggml-org/gemma-4-E2B-it-GGUF`. Local BF16-to-Q4 quantization was observed to exceed this WSL memory budget and is guarded behind `ALLOW_HIGH_MEMORY_QUANTIZE=1`. CPU fallback text-only smoke has passed. CUDA build uses `scripts/wsl/build_llama_cpp_cuda.sh`; runtime offload starts at `N_GPU_LAYERS=32` because VRAM, not WSL host RAM, is the limiting resource. WSL CUDA text and sample-image benchmark passed with the same Gemma Q8 artifacts after setting `LLAMA_BATCH_SIZE=512` and `LLAMA_UBATCH_SIZE=512`. The earlier `LLAMA_UBATCH_SIZE=32` text-only setting triggered a llama.cpp non-causal attention assertion on image requests.
- MiniCPM-V 4.6: current llama.cpp docs describe conversion from `openbmb/MiniCPM-V-4_6` into a language-model GGUF and a separate `mmproj` GGUF. This project therefore defaults to local `MODEL_PATH` and `MMPROJ_PATH` for MiniCPM-V 4.6 unless the user supplies a verified GGUF repo through `MODEL_REF`. Full WSL preparation is guarded behind `ALLOW_MINICPM_FULL_PREPARE=1`; use `scripts/wsl/inspect_minicpmv46_hf.sh` first to inspect file metadata and conversion-script signals without downloading model weights.
