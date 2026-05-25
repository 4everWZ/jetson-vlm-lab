# Tradeoffs

## TRD-001: Standard-Library Client Instead Of OpenAI SDK

Decision: use `urllib.request` instead of adding the OpenAI Python SDK as a dependency.

Reason: the repository started empty and the user asked not to install dependencies globally. The standard-library client is enough for local llama-server requests, dry runs, and JSONL benchmark logging.

Consequence: streaming support is minimal and only handles basic SSE `data:` chunks. If later work needs richer OpenAI API compatibility, add a project dependency file and document the target environment.

## TRD-002: MiniCPM-V 4.6 Defaults To Local GGUF Plus MMPROJ

Decision: Gemma uses a documented `-hf` GGUF repo default, while MiniCPM-V 4.6 defaults to local `MODEL_PATH` and `MMPROJ_PATH`.

Reason: llama.cpp documents MiniCPM-V 4.6 conversion from the official HF checkpoint into separate language-model and mmproj GGUF files. Assuming a public prebuilt GGUF repo would be less defensible than requiring explicit local paths or a user-supplied `MODEL_REF`.

Consequence: the MiniCPM-V launcher needs model conversion or staged files before it runs.

## TRD-003: Docker-First Jetson Scripts

Decision: Jetson scripts use Docker/NVIDIA runtime by default.

Reason: the Jetson should not be the primary development machine, and Docker keeps runtime setup more reproducible than Conda-heavy local installs.

Consequence: users without Docker/NVIDIA runtime configured must either fix that first or adapt scripts to a native llama.cpp build.

## TRD-004: Gemma Low-Memory Baseline Uses Pre-Quantized Q8_0

Decision: use official pre-quantized Gemma 4 E2B-it Q8_0 GGUF artifacts as the WSL baseline instead of local BF16-to-Q4 quantization.

Reason: local BF16-to-Q4 quantization was observed to be killed on this WSL host while processing a large embedding tensor. Reducing quantization threads limits CPU concurrency but does not remove that tensor-level peak memory requirement.

Consequence: `configs/models/gemma4_e2b_q8.yaml` and `scripts/wsl/prepare_gemma4_e2b_q8.sh` are the default WSL path. `scripts/wsl/prepare_gemma4_e2b_q4.sh` remains available only with `ALLOW_HIGH_MEMORY_QUANTIZE=1` for larger machines or externally managed conversion hosts.
