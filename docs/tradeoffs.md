# Tradeoffs

## TRD-001: Standard-Library Client Instead Of OpenAI SDK

Decision: use `urllib.request` instead of adding the OpenAI Python SDK as a dependency.

Reason: the repository started empty and the user asked not to install dependencies globally. The standard-library client is enough for local llama-server requests, dry runs, and JSONL benchmark logging.

Consequence: streaming support is minimal and only handles basic SSE `data:` chunks. If later work needs richer OpenAI API compatibility, add a project dependency file and document the target environment.

## TRD-002: MiniCPM-V 4.6 Uses Official Pre-Built GGUF Plus MMPROJ

Decision: MiniCPM-V 4.6 uses the official `openbmb/MiniCPM-V-4.6-gguf` repository for the default Q4_K_M model and F16 mmproj artifacts.

Reason: a public official pre-built GGUF repository is available and avoids using this memory-constrained WSL host as a conversion or quantization machine.

Consequence: `scripts/wsl/prepare_minicpmv46_q4.sh` downloads `MiniCPM-V-4_6-Q4_K_M.gguf` and `mmproj-model-f16.gguf`, while the WSL and Jetson launchers default to the same staged file layout.

## TRD-003: Docker-First Jetson Scripts

Decision: Jetson scripts use Docker/NVIDIA runtime by default.

Reason: the Jetson should not be the primary development machine, and Docker keeps runtime setup more reproducible than Conda-heavy local installs.

Consequence: users without Docker/NVIDIA runtime configured must either fix that first or adapt scripts to a native llama.cpp build.

## TRD-004: Gemma Uses Pre-Quantized GGUF Artifacts

Decision: use pre-quantized Gemma 4 E2B-it GGUF artifacts for both Q8 and Q4 paths instead of local quantization.

Reason: local BF16-to-Q4 quantization was observed to be killed on this WSL host while processing a large embedding tensor, and later Q8-to-Q4 re-quantization was stopped at the user's request because memory was insufficient. Downloading known GGUF artifacts keeps the WSL workflow reproducible and avoids using this host as a conversion machine.

Consequence: `scripts/wsl/prepare_gemma4_e2b_q8.sh` downloads `ggml-org/gemma-4-E2B-it-GGUF` Q8_0 artifacts, and `scripts/wsl/prepare_gemma4_e2b_q4.sh` downloads `mradermacher/gemma-4-E2B-it-GGUF` Q4_K_M artifacts. Gemma local quantization is not a WSL acceptance path on this machine.

## TRD-005: Local MiniCPM Conversion Is Not A Baseline Path

Decision: do not expose MiniCPM-V 4.6 HF checkpoint download, F16 GGUF conversion, or local Q4 quantization as the repository's default preparation path.

Reason: local conversion/quantization is unnecessary now that official pre-built GGUF artifacts are available, and the user explicitly stopped local quantization because this WSL host is memory constrained.

Consequence: old local-conversion artifacts are treated as residue and were removed from ignored `models/` storage. `scripts/wsl/inspect_minicpmv46_hf.sh` inspects the official pre-built GGUF repo, and `scripts/wsl/prepare_minicpmv46_q4.sh` downloads existing artifacts only.
