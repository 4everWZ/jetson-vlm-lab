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

## TRD-004: Gemma Uses Pre-Quantized GGUF Artifacts

Decision: use pre-quantized Gemma 4 E2B-it GGUF artifacts for both Q8 and Q4 paths instead of local quantization.

Reason: local BF16-to-Q4 quantization was observed to be killed on this WSL host while processing a large embedding tensor, and later Q8-to-Q4 re-quantization was stopped at the user's request because memory was insufficient. Downloading known GGUF artifacts keeps the WSL workflow reproducible and avoids using this host as a conversion machine.

Consequence: `scripts/wsl/prepare_gemma4_e2b_q8.sh` downloads `ggml-org/gemma-4-E2B-it-GGUF` Q8_0 artifacts, and `scripts/wsl/prepare_gemma4_e2b_q4.sh` downloads `mradermacher/gemma-4-E2B-it-GGUF` Q4_K_M artifacts. Gemma local quantization is not a WSL acceptance path on this machine.

## TRD-005: MiniCPM Full Preparation Requires Explicit Opt-In

Decision: keep MiniCPM-V 4.6 full HF download, F16 GGUF conversion, and Q4 quantization behind `ALLOW_MINICPM_FULL_PREPARE=1`.

Reason: the previous WSL OOM shows this host should not treat large model conversion or quantization as a safe default. Metadata inspection can still validate repository file sizes and llama.cpp conversion-script signals without downloading weights.

Consequence: `scripts/wsl/inspect_minicpmv46_hf.sh` is the default MiniCPM feasibility step. `scripts/wsl/prepare_minicpmv46_q4.sh` remains available only for a larger conversion machine or a deliberately accepted high-memory run.
