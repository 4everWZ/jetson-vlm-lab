# Jetson Formal Benchmark Results - 2026-05-30

This document records the first formal Jetson benchmark run after adding the
manifest-based benchmark workflow. Raw JSONL, manifests, profiles, fake-stream
outputs, and tegrastats logs were kept under ignored `outputs/` paths and are
not tracked by git.

## Git Tracking

| Step | Git scope | Status |
|---|---|---|
| Formal benchmark infra | `src/edge_vlm/benchmark.py`, `scripts/jetson/run_formal_benchmark.sh`, docs, tests | Tracked in commit `b3b5ea2` on branch `bench/formal-jetson-infra` |
| Raw benchmark artifacts | `outputs/benchmarks/`, `outputs/fake_stream/`, `outputs/tegrastats/` | Ignored local artifacts |
| Consolidated result tables | `docs/benchmarks/jetson_formal_20260530.md` | Tracked doc |
| Secrets and local SSH helpers | `.env`, `.env.*` | Ignored; password was not written into the repo |

## Environment

| Field | Value |
|---|---|
| Jetson repo used for execution | `~/code/jetson-vlm-lab-bench` |
| Model and weight root | `~/code/jetson-vlm-lab/models` |
| Docker image | `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87` |
| Benchmark client | `python3 scripts/jetson/run_formal_benchmark.sh` |
| Trial count | 3 |
| Max tokens | 64 |
| Temperature | 0 |
| Benchmark cases per trial | 3 text, 2 image, 1 fake-stream placeholder |
| One-frame fake-stream check | Run separately with `scripts/jetson/run_fake_stream.sh` |

The benchmark manifest records the benchmark client configuration. The actual
llama.cpp server launch parameters below came from the server launch commands
and should be treated as the source of truth for server-side runtime settings.

## Server Launch Settings

| Model | Config | Key model paths | Server parameters |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `configs/models/minicpmv46_q4.yaml` | `models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf`, `models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf` | `CTX_SIZE=512`, `N_GPU_LAYERS=32`, `--parallel 1`, `--batch-size 128`, `--ubatch-size 32`, `--cache-type-k q8_0`, `--cache-type-v q8_0`, `--no-warmup` |
| Gemma 4 E2B-it Q4 | `configs/models/gemma4_e2b_q4.yaml` | `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf`, `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf` | `CTX_SIZE=512`, `N_GPU_LAYERS=12`, `-fit off`, `--parallel 1`, `--batch-size 512`, `--ubatch-size 512`, `--cache-type-k q8_0`, `--cache-type-v q8_0`, `--no-warmup` |

## Health Check Snapshot

| Model | `/v1/models` id | Capabilities | Parameters | Model size bytes |
|---|---|---|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpmv46-q4` | `completion`, `multimodal` | 752,161,600 | 518,145,904 |
| Gemma 4 E2B-it Q4 | `gemma4-e2b-it-q4` | `completion`, `multimodal` | 4,647,450,147 | 3,412,060,300 |

## Summary

| Model | Run id | Records | Success | Text avg latency s | Text avg tok/s | Image avg latency s | Image avg tok/s | One-frame fake-stream latency s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `minicpmv46-q4-jetson-formal-20260530` | 18 | 18/18 | 1.452 | 44.201 | 1.562 | 41.588 | 2.125 |
| Gemma 4 E2B-it Q4 | `gemma4-e2b-q4-jetson-formal-20260530` | 18 | 18/18 | 8.953 | 7.152 | 9.391 | 6.914 | 10.565 |

MiniCPM-V 4.6 Q4 is the faster current baseline on this Jetson setup. Gemma 4
E2B-it Q4 loaded and completed all formal cases, but its first server start
failed after the MiniCPM run with CUDA allocation errors while tegrastats showed
low contiguous free blocks. Dropping page cache improved `lfb` and the same
Gemma command then loaded successfully. This makes contiguous-memory state a
benchmark precondition worth recording before larger model sweeps.

## Per-Case Averages

| Model | Case | Input | Runs | Avg latency s | Avg tok/s | Min latency s | Max latency s |
|---|---|---|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | `text_cn_short` | text | 3 | 1.514 | 42.573 | 1.418 | 1.697 |
| MiniCPM-V 4.6 Q4 | `text_en_reasoning_short` | text | 3 | 1.427 | 44.862 | 1.416 | 1.440 |
| MiniCPM-V 4.6 Q4 | `text_code_short` | text | 3 | 1.417 | 45.167 | 1.407 | 1.428 |
| MiniCPM-V 4.6 Q4 | `image_caption_single` | image | 3 | 1.576 | 41.280 | 1.425 | 1.876 |
| MiniCPM-V 4.6 Q4 | `image_safety_scene_single` | image | 3 | 1.548 | 41.895 | 1.415 | 1.809 |
| Gemma 4 E2B-it Q4 | `text_cn_short` | text | 3 | 8.960 | 7.150 | 8.659 | 9.355 |
| Gemma 4 E2B-it Q4 | `text_en_reasoning_short` | text | 3 | 8.999 | 7.115 | 8.729 | 9.188 |
| Gemma 4 E2B-it Q4 | `text_code_short` | text | 3 | 8.901 | 7.190 | 8.880 | 8.928 |
| Gemma 4 E2B-it Q4 | `image_caption_single` | image | 3 | 9.396 | 6.937 | 8.118 | 11.192 |
| Gemma 4 E2B-it Q4 | `image_safety_scene_single` | image | 3 | 9.385 | 6.890 | 8.649 | 10.784 |

## Per-Trial Averages

| Model | Trial | Tokenized cases | Avg latency s | Avg tok/s |
|---|---:|---:|---:|---:|
| MiniCPM-V 4.6 Q4 | 1 | 5 | 1.650 | 39.296 |
| MiniCPM-V 4.6 Q4 | 2 | 5 | 1.420 | 45.086 |
| MiniCPM-V 4.6 Q4 | 3 | 5 | 1.420 | 45.085 |
| Gemma 4 E2B-it Q4 | 1 | 5 | 9.861 | 6.548 |
| Gemma 4 E2B-it Q4 | 2 | 5 | 8.631 | 7.422 |
| Gemma 4 E2B-it Q4 | 3 | 5 | 8.893 | 7.200 |

`fake_stream_folder_sample` is included in each formal trial as a placeholder
case without token metrics. The actual fake-stream latency was measured
separately with one frame from `data/sample_stream/frame_001.png`.

## Next Benchmark Work

1. Add lightweight candidates only after the same formal wrapper is used for
   all existing baselines.
2. Use the manifest plus server-launch parameter table for every new model.
3. Record `tegrastats`, `lfb`, Docker image digest or tag, and exact model paths
   before running larger models.
4. Add Tencent `Youtu-VL-4B-Instruct-GGUF` after the current baseline tables are
   stable, then compare it against MiniCPM-V 4.6 Q4 and Gemma 4 E2B-it Q4 under
   the same 3-trial protocol.
