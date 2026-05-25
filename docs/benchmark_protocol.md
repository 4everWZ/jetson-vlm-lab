# Benchmark Protocol

The benchmark harness is designed to run the same prompt cases on WSL and Jetson. It records observations; it does not claim accurate performance unless the server and model actually ran.

## Cases

`configs/benchmark/prompt_cases.jsonl` contains:

- `text_cn_short`
- `text_en_reasoning_short`
- `text_code_short`
- `image_caption_single`
- `image_safety_scene_single`
- `fake_stream_folder_sample`

Image paths are placeholders under `data/`. Add small local sample images before running image cases. Do not commit large or private media.

`fake_stream_folder_sample` is a marker case in the shared prompt list. The benchmark runner records it as a reminder to use the fake-stream harness; folder iteration itself is handled by `python -m edge_vlm.fake_stream`.

## Raw Output

The benchmark writes JSONL records with:

- `model`
- `backend`
- `quantization`
- `model_ref`
- `device`
- `prompt_case_id`
- `input_type`
- `image_path`
- `start_time`
- `end_time`
- `latency_s`
- `tokens`
- `tokens_per_sec`
- `success`
- `error`
- `output_excerpt`

Token counts are recorded only when the backend response exposes usage fields. If token counts are missing, `tokens` and `tokens_per_sec` remain null.

## WSL Dry Run

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl-dryrun.jsonl \
  --dry-run
```

Dry run validates payload construction and logging only. It does not contact a server and must not be used as a performance result.

## WSL Real Run

Start a server first, then check health:

```bash
scripts/wsl/run_gemma4_e2b_llama.sh
scripts/common/check_server.sh
```

In a second terminal:

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl
```

For MiniCPM-V 4.6, inspect metadata before attempting local conversion on WSL:

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

Then provide local converted GGUF files only after a deliberate high-memory preparation run or external conversion:

```bash
MODEL_PATH=/path/to/ggml-model-Q4_K_M.gguf \
MMPROJ_PATH=/path/to/mmproj-model-f16.gguf \
CTX_SIZE=512 \
N_GPU_LAYERS=0 \
scripts/wsl/run_minicpmv46_llama.sh
```

## Fake Stream Run

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "For this frame, describe the most important object or activity in one sentence." \
  --output outputs/fake_stream/gemma4-e2b-q8-wsl.jsonl
```

Add `--dry-run` to validate folder iteration and JSONL logging without contacting a server.

## Jetson Run

On Jetson, start `tegrastats` in one terminal:

```bash
scripts/jetson/monitor_tegrastats.sh
```

Start the model server in another terminal, then run the same benchmark command with `EDGE_VLM_DEVICE=jetson-orin`.

```bash
EDGE_VLM_DEVICE=jetson-orin PYTHONPATH=src python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-jetson.jsonl
```

## Reporting Rules

- Report dry-run logs as payload/logging validation only.
- Report WSL and Jetson logs separately.
- Include model ref, quantization, context size, `N_GPU_LAYERS`, Jetson power mode, storage location, and whether image cases succeeded.
- Do not compare Jetson and WSL as equivalent hardware. Use WSL for development correctness and Jetson for edge runtime observations.
