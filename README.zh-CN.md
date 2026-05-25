# Jetson VLM Lab

这是一个面向 Jetson Orin / Orin Nano 的 WSL 优先边缘 VLM 实验工作流。

本项目的组织原则是：开发、参考仓库阅读、客户端代码、benchmark 脚本和迁移文档先在 WSL 上完成；Jetson 只接收最小运行包、配置、脚本、benchmark 代码，以及放在 NVMe 或外置存储上的模型文件。

## 当前状态

- 第一条支持路线是 llama.cpp `llama-server` + GGUF。
- Python 命令使用本地 Conda 环境：`conda run -n transformers python`。
- 当前工作区已经有本地 llama.cpp CPU-only 构建：`tmp/llama.cpp/build/bin`。
- WSL 能看到 `nvidia-smi`，但没有 `nvcc`，所以当前实测 llama.cpp 构建不是 CUDA 构建。
- Gemma BF16 GGUF 源文件已经下载过，但 BF16 到 Q4_K_M 的本地量化被 WSL 内存压力杀掉；不完整 Q4 文件已经清理。
- 低内存 Gemma 主线改为官方已量化 Q8_0 GGUF：`scripts/wsl/prepare_gemma4_e2b_q8.sh`。当前工作区里，Q8_0 model 和 Q8_0 mmproj 文件已经在被 Git 忽略的 `models/` 目录下。
- Gemma Q8_0 text-only smoke 已经在本地 CPU-only llama.cpp 构建上通过，参数是 `CTX_SIZE=512`、`N_GPU_LAYERS=0`、`LLAMA_THREADS=2`、单 server slot、关闭 warmup。这只是功能烟测，不是性能结论。
- MiniCPM-V 4.6 metadata-only 检查已通过，且没有下载权重：HF 已知文件总量 2.44 GiB，其中 `model.safetensors` 是 2,600,957,528 bytes。本地 llama.cpp conversion modules 注册了 `MiniCPMV4_6ForConditionalGeneration`，包含 MiniCPM-V 4.6 projector metadata，并且转换入口暴露 `--mmproj`。转换和运行仍未验证。
- Jetson 推理还没有在本仓库中实测完成。

`full access` 不能降低模型量化的峰值内存。对于内存有限的 WSL，更合理的办法是用已量化 GGUF、缩小 context、CPU-only smoke test、单 server slot，而不是反复尝试 BF16->Q4。除非换到更大的机器，否则不要在这里重试本地 BF16-to-Q4 量化。

## 目录结构

```text
configs/models/                  模型运行配置
configs/benchmark/               共享 benchmark prompt cases
docs/                            设计、迁移、benchmark、参考笔记
scripts/wsl/                     WSL 构建、准备、运行脚本
scripts/jetson/                  Jetson Docker 启动和监控脚本
scripts/common/                  共享辅助脚本
src/edge_vlm/                    OpenAI-compatible 客户端和实验 harness
tests/                           轻量 contract tests
tmp/references/                  被忽略的参考仓库 clone
models/                          被忽略的本地模型文件
outputs/                         被忽略的 benchmark 日志
```

## WSL 设置

使用本地 `transformers` Conda 环境。不要把项目依赖装进 Conda `base`、系统 Python 或全局 Python。

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

构建或复用 llama.cpp：

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
```

构建脚本不会安装系统包。只有检测到 `nvcc`，或者你显式配置了可用 CUDA toolkit 时，才会启用 CUDA。

## 准备 Gemma 4 E2B-it

低内存 WSL 路线：

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

这个脚本会把官方 Q8_0 模型和 mmproj GGUF 下载到被 Git 忽略的 `models/` 目录。它不会执行 BF16 到 Q4 的本地量化。

文件存在后，用保守参数启动 WSL server：

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
CTX_SIZE=512 \
N_GPU_LAYERS=0 \
LLAMA_THREADS=2 \
LLAMA_THREADS_BATCH=2 \
LLAMA_BATCH_SIZE=128 \
LLAMA_UBATCH_SIZE=32 \
LLAMA_PARALLEL=1 \
LLAMA_SERVER_EXTRA_ARGS='--no-ui --no-warmup' \
scripts/wsl/run_gemma4_e2b_llama.sh
```

另开一个终端：

```bash
scripts/common/check_server.sh
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl
```

## MiniCPM-V 4.6

MiniCPM-V 4.6 当前按 llama.cpp 本地转换路线管理。在这台 WSL 上，先只检查元数据；确认存储和内存足够之前，不要启动完整下载、转换和量化流程：

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

本工作区观察到的元数据：HF 已知文件总量 2.44 GiB，`model.safetensors` 是 2,600,957,528 bytes。本地 llama.cpp tree 有 MiniCPM-V 4.6 conversion registration 和 projector metadata，但这只是可行性信号。

完整制备流程默认有保护，因为它会下载较大的 HF checkpoint，生成 F16 GGUF，再量化到 Q4_K_M。只有在内存和磁盘足够的机器上才显式打开：

```bash
ALLOW_MINICPM_FULL_PREPARE=1 scripts/wsl/prepare_minicpmv46_q4.sh
```

MiniCPM-V 4.6 仍然是未验证路线；只有当当前 llama.cpp 版本完成转换，并且真实 `llama-server` 请求成功后，才算可用。

本地文件存在后再启动：

```bash
MODEL_PATH=$PWD/models/MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/MiniCPM-V-4_6/mmproj-model-f16.gguf \
CTX_SIZE=512 \
N_GPU_LAYERS=0 \
LLAMA_THREADS=2 \
LLAMA_THREADS_BATCH=2 \
LLAMA_BATCH_SIZE=128 \
LLAMA_UBATCH_SIZE=32 \
LLAMA_PARALLEL=1 \
scripts/wsl/run_minicpmv46_llama.sh
```

## Benchmark

只验证 payload 和日志格式的 dry run：

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

Fake stream dry run：

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "Describe this frame." \
  --output outputs/fake_stream/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

图像 case 需要你在 `data/` 下放本地小样例图片；不要提交大文件或私有图片。

## Jetson 路线

复制 source、configs、scripts、docs 和 tests 到 Jetson。不要复制 WSL build 目录、Conda 环境、参考仓库或无关 benchmark 输出。

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '.vscode/' \
  --exclude 'tmp/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  ./ jetson:/home/jetson/edge-vlm-lab/
```

模型建议放 NVMe 或外置存储：

```bash
sudo mkdir -p /mnt/nvme/models
sudo chown "$USER:$USER" /mnt/nvme/models
```

Jetson 上用 Docker 跑 Gemma：

```bash
MODEL_DIR=/mnt/nvme/models \
CTX_SIZE=2048 \
N_GPU_LAYERS=99 \
scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

如果模型文件已经预放在 Jetson，本地路径必须在 `MODEL_DIR` 下，并同时传 `MODEL_PATH` 和 `MMPROJ_PATH`。

## 参考文档

- OrangePi MiniCPM-V 4.6 参考笔记：`docs/reference_notes/orangepi_minicpmv46_notes.md`
- Runtime matrix：`docs/runtime_matrix.md`
- Benchmark protocol：`docs/benchmark_protocol.md`
- WSL 到 Jetson 迁移：`docs/migration_wsl_to_jetson.md`
- APEX implementation matrix：`docs/matrix_edge_vlm_workflow.md`
