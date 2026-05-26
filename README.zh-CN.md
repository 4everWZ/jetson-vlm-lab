# Jetson VLM Lab

这是一个 WSL 优先的边缘 VLM 工作流，用于先在 WSL 上验证 GGUF 视觉语言模型，再把最小运行包迁移到 Jetson Orin / Orin Nano。

默认路径是务实的：下载现成 GGUF、启动 `llama-server`、跑共享 benchmark，然后只把源码、配置、脚本、文档和模型文件放到 Jetson 存储上。本机模型量化不是这台 WSL 的常规流程。

## 项目内容

- 第一条支持路线是 llama.cpp `llama-server` + GGUF。
- WSL 和 Jetson 共享同一个 Python client 与 benchmark harness。
- WSL 脚本和 Jetson 脚本分离。
- 运行配置放在 `configs/`，不写死在源码里。
- OrangePi MiniCPM-V 4.6 仓库只作为工程参考，不移植 Ascend 专用代码。
- 文档包含从 WSL 验证迁移到 Jetson 运行的路径。

## 当前验证状态

| 项 | 状态 |
|---|---|
| Python 环境 | 使用本地 Conda 环境：`conda run -n transformers python`。 |
| llama.cpp CPU build | 已存在于 `tmp/llama.cpp/build/bin`；GPU 访问不可用时可作为 fallback。 |
| llama.cpp CUDA build | 已在本地 `tmp/llama.cpp/build-cuda` 验证，使用 `GGML_CUDA=ON`、`CMAKE_CUDA_ARCHITECTURES=86` 和 `BUILD_JOBS=8`。 |
| WSL GPU 可见性 | `nvcc` 可用。full access 下 `nvidia-smi` 能看到 RTX 3060 Laptop GPU；沙箱命令可能看不到 NVML。 |
| Gemma 4 E2B-it Q8 | 官方已量化 Q8_0 model 和 mmproj 文件已在被 Git 忽略的 `models/` 存储里。 |
| Gemma 4 E2B-it Q4 | 使用 `mradermacher/gemma-4-E2B-it-GGUF` 的现成 `Q4_K_M` GGUF。WSL CUDA 文本、样例图 benchmark 和一帧 fake-stream 已通过，端口为 `VLM_SERVER_PORT=18083`。 |
| Gemma Q8 WSL CUDA smoke | 文本和样例图 benchmark 已通过，参数为 `CTX_SIZE=512`、`N_GPU_LAYERS=32`、`LLAMA_BATCH_SIZE=512`、`LLAMA_UBATCH_SIZE=512`、单 server slot、`VLM_SERVER_PORT=18081`。wrapper 默认参数真实运行写入了 `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl` 和 `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`。 |
| MiniCPM-V 4.6 | 已下载 `openbmb/MiniCPM-V-4.6-gguf` 的官方现成 `Q4_K_M` model 和 F16 mmproj 文件，存放在被 Git 忽略的 `models/` 目录。WSL CUDA 文本、样例图 benchmark 和一帧 fake-stream 已通过，端口为 `VLM_SERVER_PORT=18082`。 |
| Jetson runtime | 脚本和文档已准备，但本仓库还没有实测 Jetson 推理。 |

dry run 和 server startup 不能当作性能结果。性能结论必须来自真实模型/server 跑出的 benchmark JSONL。当前已观察到的 runtime 支持只覆盖 WSL CUDA 上的 Gemma Q8、Gemma Q4 和 MiniCPM-V 4.6 Q4；不验证 Jetson runtime 或泛化性能。

## 目录结构

```text
configs/models/                  模型运行配置
configs/benchmark/               共享 benchmark prompt cases
docs/                            设计、迁移、benchmark、矩阵和参考笔记
scripts/wsl/                     WSL 构建、准备、运行脚本
scripts/jetson/                  Jetson Docker 启动和监控脚本
scripts/common/                  共享辅助脚本
src/edge_vlm/                    OpenAI-compatible client 和 benchmark 代码
tests/                           轻量 contract tests
tmp/references/                  被忽略的参考仓库 clone
models/                          被忽略的本地模型文件
outputs/                         被忽略的 benchmark 日志
```

## 前置条件

- Windows host 上的 WSL。
- 名为 `transformers` 的 Conda 环境用于 Python 命令。
- `git` 和 `cmake` 用于 llama.cpp build。
- CUDA build 需要 WSL 中有 CUDA toolkit；当前工作区可用 `nvcc` 12.0。
- 如果沙箱命令无法访问 NVML，GPU runtime check 需要 full access shell。

不要把项目依赖装进 Conda `base`、系统 Python 或全局 Python。

## WSL 快速开始

先跑 contract tests：

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

下载现成模型 artifacts。Gemma Q8 是已验证的 WSL CUDA baseline；Gemma Q4 用于内存/存储压力更大的情况；MiniCPM Q4 是已验证的较小 WSL CUDA VLM 路线：

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
scripts/wsl/prepare_gemma4_e2b_q4.sh
scripts/wsl/prepare_minicpmv46_q4.sh
```

如果本地 build 目录不存在，先构建 llama.cpp：

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
scripts/wsl/build_llama_cpp_cuda.sh
```

启动已验证的 Gemma Q8 CUDA baseline：

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
VLM_SERVER_PORT=18081 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

另开终端，对这个 server 跑真实 benchmark：

```bash
VLM_SERVER_PORT=18081 scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=18081 conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl.md \
  --max-tokens 64 \
  --temperature 0
```

dry-run 只用于在没有 server 时验证 payload 和 JSONL 日志：

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-dryrun.md \
  --dry-run
```

## llama.cpp 构建

CPU fallback build，适合 GPU runtime 访问不可用时使用：

```bash
CLONE_LLAMA_CPP=1 scripts/wsl/build_llama_cpp.sh
```

这台 WSL 的 CUDA build：

```bash
scripts/wsl/build_llama_cpp_cuda.sh
```

CUDA wrapper 默认：

- `LLAMA_CPP_BUILD_DIR=$PWD/tmp/llama.cpp/build-cuda`
- `BUILD_JOBS=8`
- `CMAKE_CUDA_ARCHITECTURES=86`

这台 12 GiB RAM + 6 GiB swap 的 WSL 机器默认保持 `BUILD_JOBS=8`。当前主线不需要继续上调构建并发，除非后续明确重新调参。

## 模型文件

Q8 baseline：

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

下载：

- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf`
- `models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf`

Q4 低内存选项：

```bash
scripts/wsl/prepare_gemma4_e2b_q4.sh
```

下载：

- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf`
- `models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf`

这些脚本下载现成 GGUF，不运行 `llama-quantize`。

## 运行 Gemma 4 E2B-it

GPU 访问不可用时，用 CPU fallback：

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
scripts/wsl/run_gemma4_e2b_llama.sh
```

WSL CUDA 路线：

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

Q4 使用同一个 launcher，只换 artifact 和 alias：

```bash
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

CUDA launcher 默认 `CTX_SIZE=512`、`N_GPU_LAYERS=32`、`LLAMA_BATCH_SIZE=512`、`LLAMA_UBATCH_SIZE=512`、两个线程、单 server slot、关闭 warmup。512 batch/ubatch 是这版 llama.cpp 上 Gemma Q8 图像路径的实测可用设置；更低的 32 ubatch text-only 设置会触发 llama.cpp 图像断言。如果内存和显存还有余量，可以显式提高：

```bash
N_GPU_LAYERS=48 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
N_GPU_LAYERS=99 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

另开终端用 `nvidia-smi` 观察显存。这里用 8-10 GiB WSL host memory 做 build/runtime 是正常的；边界是不要把进程推到 OOM。`BUILD_JOBS` 主要消耗主机内存和 CPU，所以默认值按这台 12 GiB RAM + 6 GiB swap 的 WSL 设置得更积极。`N_GPU_LAYERS` 消耗 GPU 显存，所以后者要按 RTX 3060 Laptop 的 6 GiB 显存来调。

跑 Q4 时，benchmark 用 `configs/models/gemma4_e2b_q4.yaml` 记录。

## MiniCPM-V 4.6

先检查官方现成 GGUF repo metadata，不下载权重：

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

下载官方现成 Q4_K_M model 和 F16 mmproj 文件：

```bash
scripts/wsl/prepare_minicpmv46_q4.sh
```

下载文件：

- `models/MiniCPM-V-4.6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf`
- `models/MiniCPM-V-4.6-gguf/mmproj-model-f16.gguf`

文件存在后运行 WSL CUDA 路线：

```bash
VLM_SERVER_PORT=18082 \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

然后跑 benchmark：

```bash
VLM_SERVER_PORT=18082 scripts/common/check_server.sh
PYTHONPATH=src VLM_SERVER_PORT=18082 conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/minicpmv46_q4.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/minicpmv46-q4-wsl-cuda.jsonl \
  --summary-output outputs/benchmarks/minicpmv46-q4-wsl-cuda.md \
  --max-tokens 64 \
  --temperature 0
```

WSL CUDA smoke 已通过文本、已提交样例图和一帧 fake-stream。Jetson runtime 仍未实测。

## Fake Stream

文件夹图片流 dry run：

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.fake_stream \
  --config configs/models/gemma4_e2b_q8.yaml \
  --image-dir data/sample_stream \
  --prompt "Describe this frame." \
  --output outputs/fake_stream/gemma4-e2b-q8-dryrun.jsonl \
  --dry-run
```

仓库已在 `data/sample_images/` 和 `data/sample_stream/` 放入小型非私有样例图，clone 后即可做 dry run 和 payload 检查。不要提交大文件或私有图片。

## Jetson 迁移

复制 source、configs、scripts、docs 和可选 tests。不要复制 WSL build 目录、Conda 环境、参考仓库或无关 benchmark 输出。

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

用 `JETSON_DRY_RUN=1` 可以只打印 Docker 命令，不要求当前机器有 Docker 或 Jetson 硬件：

```bash
JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

完整 checklist 见 [docs/migration_wsl_to_jetson.md](docs/migration_wsl_to_jetson.md)。

## 文档

- [Runtime matrix](docs/runtime_matrix.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [WSL to Jetson migration](docs/migration_wsl_to_jetson.md)
- [Implementation plan](docs/implementation_plan.md)
- [APEX workflow matrix](docs/matrix_edge_vlm_workflow.md)
- [OrangePi MiniCPM-V 4.6 notes](docs/reference_notes/orangepi_minicpmv46_notes.md)
