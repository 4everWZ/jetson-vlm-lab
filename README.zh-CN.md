# Jetson VLM Lab

这是一个面向 Jetson Orin / Orin Nano 的 WSL 优先边缘 VLM 实验工作流。

本仓库的原则是：开发、参考仓库阅读、客户端代码、benchmark harness 和迁移文档先在 WSL 上完成。Jetson 只接收尽量小的运行包：源码、配置、脚本、benchmark 代码，以及放在 NVMe 或外置存储上的模型文件。

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
| llama.cpp CPU build | 已存在于 `tmp/llama.cpp/build/bin`；Gemma Q8 text-only smoke 之前用低内存 CPU 参数通过。 |
| llama.cpp CUDA build | 已在本地 `tmp/llama.cpp/build-cuda` 验证，使用 `GGML_CUDA=ON`、`CMAKE_CUDA_ARCHITECTURES=86` 和 `BUILD_JOBS=8`。 |
| WSL GPU 可见性 | `nvcc` 可用。full access 下 `nvidia-smi` 能看到 RTX 3060 Laptop GPU；沙箱命令可能看不到 NVML。 |
| Gemma 4 E2B-it | 官方已量化 Q8_0 model 和 mmproj 文件已在被 Git 忽略的 `models/` 存储里。 |
| Gemma Q8 WSL CUDA smoke | 文本和样例图 benchmark 已通过，参数为 `CTX_SIZE=512`、`N_GPU_LAYERS=32`、`LLAMA_BATCH_SIZE=512`、`LLAMA_UBATCH_SIZE=512`、单 server slot、`VLM_SERVER_PORT=18081`。真实运行写入了 `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-ub512.jsonl` 和 `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-real.jsonl`。 |
| Gemma BF16 到 Q4 | 不作为这台 WSL 的 baseline；本地量化曾被内存压力 kill。用已量化 GGUF 或更大的转换机器。 |
| MiniCPM-V 4.6 | metadata-only 检查已通过；本地转换和运行仍未验证。 |
| Jetson runtime | 脚本和文档已准备，但本仓库还没有实测 Jetson 推理。 |

dry run 和 server startup 不能当作性能结果。性能结论必须来自真实模型/server 跑出的 benchmark JSONL。当前 CUDA smoke 只验证了 WSL 上 Gemma Q8 文本和已提交样例图推理，不验证 MiniCPM-V 4.6、Jetson runtime 或泛化性能。

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

先跑测试：

```bash
cd /home/lawrence/code/pythonCurriculum/jetson/jetson-vlm-lab
PYTHONPATH=src conda run -n transformers python -m unittest discover -s tests -v
```

如果 Gemma Q8_0 文件还不存在，先准备模型：

```bash
scripts/wsl/prepare_gemma4_e2b_q8.sh
```

跑 benchmark dry run，验证 payload 和 JSONL 日志：

```bash
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-dryrun.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-dryrun.md \
  --dry-run
```

## llama.cpp 构建

CPU fallback build：

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

CUDA launcher 默认 `CTX_SIZE=512`、`N_GPU_LAYERS=32`、`LLAMA_BATCH_SIZE=512`、`LLAMA_UBATCH_SIZE=512`、两个线程、单 server slot、关闭 warmup。512 batch/ubatch 是这版 llama.cpp 上 Gemma Q8 图像路径的实测可用设置；更低的 32 ubatch text-only 设置会触发 llama.cpp 图像断言。如果内存和显存还有余量，可以显式提高：

```bash
N_GPU_LAYERS=48 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
N_GPU_LAYERS=99 scripts/wsl/run_gemma4_e2b_llama_cuda.sh
```

另开终端用 `nvidia-smi` 观察显存。这里用 8-10 GiB WSL host memory 做 build/runtime 是正常的；边界是不要把进程推到 OOM。`BUILD_JOBS` 主要消耗主机内存和 CPU，所以默认值按这台 12 GiB RAM + 6 GiB swap 的 WSL 设置得更积极。`N_GPU_LAYERS` 消耗 GPU 显存，所以后者要按 RTX 3060 Laptop 的 6 GiB 显存来调。

另开一个终端：

```bash
scripts/common/check_server.sh
PYTHONPATH=src conda run -n transformers python -m edge_vlm.benchmark \
  --config configs/models/gemma4_e2b_q8.yaml \
  --cases configs/benchmark/prompt_cases.jsonl \
  --output outputs/benchmarks/gemma4-e2b-q8-wsl.jsonl \
  --summary-output outputs/benchmarks/gemma4-e2b-q8-wsl.md
```

## MiniCPM-V 4.6

先做 metadata inspection，不下载权重：

```bash
scripts/wsl/inspect_minicpmv46_hf.sh
```

完整准备流程默认有保护，因为它会下载 HF checkpoint、生成 F16 GGUF、再量化到 Q4_K_M：

```bash
ALLOW_MINICPM_FULL_PREPARE=1 scripts/wsl/prepare_minicpmv46_q4.sh
```

只有本地转换后的 model 和 mmproj 文件存在后再运行：

```bash
MODEL_PATH=$PWD/models/MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/MiniCPM-V-4_6/mmproj-model-f16.gguf \
scripts/wsl/run_minicpmv46_llama_cuda.sh
```

MiniCPM-V 4.6 仍是未验证路线；只有选定 llama.cpp 版本完成转换且真实请求成功后，才算可用。

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
