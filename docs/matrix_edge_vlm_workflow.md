# Edge VLM Workflow Implementation Matrix

| Requirement ID | Original Intent | Current Status | Implementation Pointer | Verification Pointer | Notes |
|---|---|---|---|---|---|
| R1 | Inspect current repo and preserve existing structure | Implemented | Initial repo had only `.vscode/settings.json`; new structure created | `git log --oneline`; repo status | No prior project code existed. |
| R2 | Inspect OrangePi reference and record lessons | Implemented | `docs/reference_notes/orangepi_minicpmv46_notes.md` | Manual reference inspection at commit `490d039` | Ascend-specific code documented as non-portable. |
| R3 | Use llama.cpp GGUF as first backend | Implemented | `docs/runtime_matrix.md`, `scripts/wsl/*.sh`, `scripts/jetson/*.sh` | `bash -n scripts/...`; local `llama-server --version` with `LD_LIBRARY_PATH` | Gemma Q8 artifacts are present. Real inference requires smoke-test evidence before performance claims. |
| R4 | Keep WSL and Jetson scripts separate | Implemented | `scripts/wsl/`, `scripts/jetson/`, `scripts/common/` | `bash -n` | Docker path is Jetson-oriented. |
| R5 | Keep runtime configs separate from code | Implemented | `configs/models/*.yaml` | Config parse check in `transformers` env | `.gitignore` fixed to track configs. |
| R6 | Implement OpenAI-compatible client | Implemented | `src/edge_vlm/client.py`, `src/edge_vlm/image_payload.py` | `tests/test_edge_vlm.py` | Uses standard-library HTTP. |
| R7 | Treat image support as backend-sensitive | Implemented | `capabilities.image`, client/benchmark checks | Unit tests and docs | Runtime server capability still must be observed. |
| R8 | Implement benchmark harness and cases | Implemented | `src/edge_vlm/benchmark.py`, `configs/benchmark/prompt_cases.jsonl` | Unit tests; dry-run command | `fake_stream` case is a marker for the fake-stream runner. Markdown summary not implemented. |
| R9 | Implement fake stream prototype | Implemented | `src/edge_vlm/fake_stream.py` | Unit tests; dry-run CLI | Folder input only; no camera. Continues after per-frame errors unless `--stop-on-error` is set. |
| R10 | Create Jetson migration docs and scripts | Implemented | `docs/migration_wsl_to_jetson.md`, `scripts/jetson/*.sh` | `bash -n`; doc review | Jetson execution unverified. |
| R11 | Avoid cloning references into source tree | Implemented | `.gitignore` ignores `tmp/`; clones under `tmp/references/` | `git status --ignored` | Reference repos not committed. |
| R12 | Validate without overstating runtime support | In progress | Tests, syntax checks, dry-run paths, Gemma Q8 artifact checks | Final validation commands and Gemma server smoke test | Gemma Q8 model/mmproj downloaded. MiniCPM conversion and Jetson runtime are still unverified. |
