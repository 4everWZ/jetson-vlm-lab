# Jetson VLM Lab

这是一个 WSL 优先的边缘 VLM 工作流，用于先在 WSL 上验证 GGUF 视觉语言模型，再把最小运行包迁移到 Jetson Orin / Orin Nano。

默认路径是务实的：下载现成 GGUF、启动 `llama-server`、跑共享 benchmark，然后只把源码、配置、脚本、文档和模型文件放到 Jetson 存储上。本机模型量化不是这台 WSL 的常规流程。

## Goal

- 持续迭代 2B 以内的 LLM/VLM 候选，优先使用现成 Q4 GGUF；当 Q4 artifact 缺失，或 strict Jetson gate 让 Q4 路线不可用时，再退到 Q8。
- Jetson 多模态默认 runtime 是自编译的官方 llama.cpp 镜像；dusty-nv `llama_cpp` 不是默认 VLM 路线，因为它没有提供本仓库已验证需要的多模态 `llama-server` 路径。
- 下一步优先做更底层的 infra：launcher、artifact 下载/恢复、profile capture、runtime image 复现和 multimodal load path；参数调整只能针对 profiling 指出的瓶颈做小范围验证。
- 不同任务使用独立 branch 或 worktree，验证成功后再合入 `main`。

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
| Gemma 4 E2B-it Q4 | 使用 `mradermacher/gemma-4-E2B-it-GGUF` 的现成 `Q4_K_M` GGUF。WSL CUDA 和 Jetson 文本、样例图 benchmark、一帧 fake-stream 已通过。 |
| Gemma Q8 WSL CUDA smoke | 文本和样例图 benchmark 已通过，参数为 `CTX_SIZE=512`、`N_GPU_LAYERS=32`、`LLAMA_BATCH_SIZE=512`、`LLAMA_UBATCH_SIZE=512`、单 server slot、`VLM_SERVER_PORT=18081`。wrapper 默认参数真实运行写入了 `outputs/benchmarks/gemma4-e2b-q8-wsl-cuda-image-wrapper-default.jsonl` 和 `outputs/fake_stream/gemma4-e2b-q8-wsl-cuda-wrapper-default.jsonl`。 |
| MiniCPM-V 4.6 | 已下载 `openbmb/MiniCPM-V-4.6-gguf` 的官方现成 `Q4_K_M` model 和 F16 mmproj 文件，存放在被 Git 忽略的 `models/` 目录。WSL CUDA 和 Jetson 文本、样例图 benchmark、一帧 fake-stream 已通过。 |
| Qwen3-VL 2B Instruct fallback 路线 | Jetson 上 strict `--min-lfb-blocks 150` 仍然会同时卡住 Q4 和 Q8，但共享的放宽 `100-LFB` 对比 `qwen3-instruct-q4q8-lfb100-20260609T091700Z` 已经让两条路线都真实跑通。Q4 仍然更快：34.865 text tok/s、31.958 image tok/s、1.827 s fake-stream；Q8 是 31.346 text tok/s、29.393 image tok/s、2.144 s fake-stream。Q8 仍然是 fallback，不是首选默认，但它的 prepare 增益明显更大（`lfb +88`、`MemAvailable +963.6 MB`）。 |
| Jetson runtime | MiniCPM-V 4.6 Q4 和 Gemma 4 E2B-it Q4 已通过 Jetson Docker launcher 产出 smoke 日志，使用自编译的官方 llama.cpp 镜像 `ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`。dusty-nv `llama_cpp` 不是默认路径，因为它没有提供本仓库需要的多模态 server 路线。 |

dry run 和 server startup 不能当作性能结果。性能结论必须来自真实模型/server 跑出的 benchmark JSONL。当前已观察到的 runtime 支持覆盖 WSL CUDA 上的 Gemma Q8、Gemma Q4、MiniCPM-V 4.6 Q4，以及 Jetson 上的 MiniCPM-V 4.6 Q4 和 Gemma Q4 smoke；不验证 Jetson Q8、camera input、长时间运行、电源/温度行为或泛化性能。

## Jetson 已观察 Smoke

被 Git 忽略的 `outputs/` 目录里有 2026-05-27 拷贝回来的 Jetson 日志。这些是 64-token 的短 smoke，覆盖已提交的文本/图片 cases，不是正式性能套件。

| 模型 | Jetson 命令形态 | Benchmark 输出 | 结果 |
|---|---|---|---|
| MiniCPM-V 4.6 Q4 | `CTX_SIZE=512`、`N_GPU_LAYERS=32`、batch `128`、ubatch `32`、KV cache `q8_0`、关闭 warmup | `outputs/benchmarks/minicpmv46-q4-jetson.jsonl` | benchmark 6/6 通过。文本平均 41.49 tok/s；图像平均 34.10 tok/s。一帧 fake-stream 3.27 s 通过。 |
| Gemma 4 E2B-it Q4 | `CTX_SIZE=512`、`N_GPU_LAYERS=12`、batch `512`、ubatch `512`、KV cache `q8_0`、关闭 warmup、mmproj 保持在 GPU | `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu12-mmproj-gpu.jsonl` | benchmark 6/6 通过。文本平均 6.84 tok/s；图像平均 5.91 tok/s。一帧 fake-stream 18.79 s 通过。 |
| Gemma 4 E2B-it Q4，对照 | `CTX_SIZE=512`、`N_GPU_LAYERS=12`、batch `256`、ubatch `256`、`--no-mmproj-offload`、关闭 warmup | `outputs/benchmarks/gemma4-e2b-q4-jetson-gpu.jsonl` | benchmark 6/6 通过，但图像平均降到 2.87 tok/s。这个 smoke 路线默认保留 mmproj 在 GPU，除非重新测试。 |

更早的 `outputs/benchmarks/gemma4-e2b-q4-jetson.jsonl` 是失败的启动/连接尝试：1 个 marker 成功、5 个 benchmark case connection refused。不要把它当作 runtime 成功引用。

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
Jetson GGUF launcher 会在启动 `llama-server` 前检查本机 model 和 mmproj
文件的 GGUF magic bytes。缓存文件如果没有通过这个检查，就删掉或恢复该文件后
重新运行 launcher，让 HF 路径重新下载。
如果只想先做不启动服务的 artifact preflight，可以在 Jetson 上运行：

```bash
scripts/jetson/check_gguf_artifacts.sh check \
  --artifact model /mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
  --artifact mmproj /mnt/nvme/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
  --output outputs/artifacts/gemma-q4.gguf-artifacts.json
```

manifest 会逐个 artifact 记录 `missing`、`invalid_magic` 或 `ok`，只要有一个
artifact 没准备好就返回非零。
remote suite wrapper 也会默认开启 advisory
`JETSON_REMOTE_GGUF_PREFLIGHT=1`，在 server startup 前写出
`outputs/optimization_sweeps/<run-prefix>/<run-prefix>.gguf-artifacts.json`。
如果希望缺失或损坏的缓存 artifact 直接中止，就设置
`JETSON_REMOTE_GGUF_PREFLIGHT_FAIL=1`。

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

WSL CUDA 和 Jetson smoke 已通过文本、已提交样例图和一帧 fake-stream。Jetson 结果只覆盖 [Jetson 已观察 Smoke](#jetson-已观察-smoke) 里的 Q4 文件和参数。

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

Jetson 上运行已观察的 MiniCPM-V 4.6 Q4 smoke 路线：

```bash
MODEL_DIR=$PWD/models \
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
CTX_SIZE=512 \
N_GPU_LAYERS=32 \
scripts/jetson/run_minicpmv46_llama_docker.sh \
  --parallel 1 \
  --batch-size 128 \
  --ubatch-size 32 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-warmup
```

Jetson 上运行已观察的 Gemma Q4 smoke 路线：

```bash
MODEL_DIR=$PWD/models \
LLAMA_CPP_DOCKER_IMAGE=ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
MODEL_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.Q4_K_M.gguf \
MMPROJ_PATH=$PWD/models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it.mmproj-Q8_0.gguf \
MODEL_ALIAS=gemma4-e2b-it-q4 \
CTX_SIZE=512 \
N_GPU_LAYERS=12 \
scripts/jetson/run_gemma4_e2b_llama_docker.sh \
  -fit off \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-warmup
```

Jetson 脚本默认使用已验证多模态 smoke 的自编译官方 llama.cpp 镜像：
`ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87`。dusty-nv
`llama_cpp` 不是默认路径，因为 2026-06-09 在这台 Jetson 上的直接证据仍然
说明它不满足本仓库的多模态 flag 路线：
`dustynv/llama_cpp:b5283-r36.4-cu128-24.04` 虽然能让 probe 找到
`/usr/local/bin/llama-server`，但真实的 Qwen3-VL 2B Instruct Q4 smoke
在 ready 前就因为 `error: invalid argument: --mmproj` 退出。如果需要指定
镜像，用 `LLAMA_CPP_DOCKER_IMAGE=...` 覆盖；只有明确想用
`autotag llama_cpp` 选择镜像时，才设置 `LLAMA_CPP_USE_AUTOTAG=1`。
Jetson sweep plan 现在会在镜像里探测 `llama-server --help`，记录 runtime
是否暴露真正的 `--mmproj`；如果图像输入 variant 对应的 runtime 明确不支持
这条多模态路径，就会直接以 `runtime_missing_mmproj_support` 跳过。
直连 VLM Docker launcher 在真实启动前也会跑同一套 probe，并且发生在
artifact 检查或下载之前。`JETSON_DRY_RUN=1` 仍然只打印 Docker 命令。
需要固定 probe JSON 路径用于审查时，设置 `LLAMA_CPP_RUNTIME_PROBE_OUTPUT=...`。
如果正在评估 runtime 镜像本身，可以在 sweep 前直接 probe：

```bash
PYTHONPATH=src python -m edge_vlm.llama_cpp_runtime probe-image \
  --image ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87 \
  --output outputs/jetson_inspect/llama_cpp_runtime_probe.json
```

这个 JSON 会记录 Docker image label、解析到的 `llama-server` 路径、精确
`--mmproj` 支持，以及 `multimodal_ready`。只有泛泛出现 `mmproj` 字样但没有
精确 `--mmproj` flag 时，不能算多模态可用证据。

对 Qwen3-VL 2B Instruct 这对 Q4/Q8，可以直接用 selector 入口做真正的
Q4-first / Q8 fallback 决策，而不是手工挑 smoke variant：

```bash
scripts/jetson/select_qwen3_instruct_variant.sh \
  --min-lfb-blocks 150 \
  --fallback-min-lfb-blocks 100 \
  --output outputs/jetson_inspect/qwen3-instruct-selector.json
```

selector 会写出结构化 JSON，里面包含 runtime probe、preflight sample、
每个候选的 GGUF artifact manifest、block reason 和最终选择的 variant id。
如果缓存 artifact 存在但没有通过 GGUF magic 检查，会被标成
`invalid_model_artifact` 或 `invalid_mmproj_artifact`，所以损坏的 Q4 文件仍
可以退到可用的 Q8 fallback。它作为独立入口时默认仍保持 primary 和
fallback 使用同一个严格 gate，除非明确在做 scoped fallback triage。

remote lightweight suite 现在也会自动使用这个 selector，而不是把
`qwen3-vl-2b-instruct-q4-smoke` 固定写死在 candidate 列表里。默认行为是
保留全局 `JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS=150` 作为 primary gate，只对
Qwen3 Instruct fallback 路线应用
`JETSON_LIGHTWEIGHT_QWEN3_FALLBACK_MIN_LFB_BLOCKS=100`。如果想回到完全手工
控制的 candidate 列表，可以把 `JETSON_LIGHTWEIGHT_QWEN3_SELECTOR=0`。
remote sweep wrapper 现在也会把这个 selector 决策带进 sweep manifest，
所以 `edge_vlm.optimization compare` 的 `Selection` 列能直接标出
auto-selected 的 Qwen3 路线，而不是只剩一个没有决策来源的静态 variant id。
comparison 表还会同时显示 `Required lfb`，把实际生效的 preflight gate 和
观测到的 `Preflight lfb` 并排放出来。再配合
`--ranking-min-lfb-blocks <strict-gate>` 时，report 还会增加
`Ranking precheck` 列，这样 relaxed fallback row 会保留在报告里，但不会被误读成
可直接 promotion 的 strict-gate 证据。
如果 ranking 或 export 决策还需要排除首次下载行，再加上
`--ranking-require-startup-precheck`，这样没有 cached-startup 证据的 row
也会直接失败 `Ranking precheck`。
如果 startup 时间本身要参与判断，再加上
`--startup-require-cached-artifacts`。这样 compare 会新增
`Startup precheck` 列，并且只把 profile phase timings 里明确记录
`artifact_check_or_download = cached` 的 row 当成可比较的 cached-startup
证据；首次下载的 row 仍然会保留在报告里，但不会再被误读成 cached startup
baseline。
这些 remote suite wrapper 现在还会在 sweep 结束后自动运行
`edge_vlm.sweep_quality_review`，并使用
`configs/benchmark/quality_review_policy.json`。它会把每个 run 的
`quality_review_json` 和 `quality_review_markdown` sidecar 写回 manifest
路径里，这样 `edge_vlm.optimization compare` 就能直接增加 `Quality review`
列，而不需要你再手工重跑一次 policy review。
如果 compare 是拿来做 promotion 审查，再加上
`--promotion-require-startup-precheck` 和
`--promotion-require-quality-review`，这样 `Promotion precheck` 就不再只看
机械条件，还会同时要求 cached-startup 证据和结构化
`Quality review` sidecar 通过。
如果还希望 remote promotion wrapper 在这层 gate 失败时直接退出非零，就显式开
`JETSON_CURRENT_DEFAULTS_FAIL_ON_PROMOTION_PRECHECK=1` 或
`JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK=1` 或
`JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK=1`。默认保持关闭，这样诊断性
run 仍然可以先产出 comparison report，而不会立刻变成硬失败。对于 text-only
row，`Promotion precheck` 会因为配置里声明了 `capabilities.image=false`
而跳过 fake-stream 要求。
这几条 remote wrapper 现在还会默认转发
`--startup-require-cached-artifacts`，所以它们生成的 compare 报告会一直带着
`Startup precheck`，把 cached startup 和首次下载行机械地区分开来。若希望
suite 在这条 cached-startup gate 失败时直接退出非零，再显式开
`JETSON_CURRENT_DEFAULTS_FAIL_ON_STARTUP_PRECHECK=1` 或
`JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK=1` 或
`JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK=1`；默认仍然是 `0`，这样诊断性
run 仍然能把 first-download 证据保留在报告里。
这些 wrapper 也会默认转发 `--ranking-require-startup-precheck`，所以
`Ranking precheck` 现在也只会放过已经通过 `Startup precheck` 的 row。若希望
suite 在 ranking gate 失败时直接退出非零，再显式开
`JETSON_CURRENT_DEFAULTS_FAIL_ON_RANKING_PRECHECK=1` 或
`JETSON_LIGHTWEIGHT_FAIL_ON_RANKING_PRECHECK=1` 或
`JETSON_TENCENT_TEXT_FAIL_ON_RANKING_PRECHECK=1`；默认同样保持 `0`。
如果后续的 ranking/export 自动化需要机器可读的 gate 结果，而不只是 Markdown
报告，再加上 `--eligibility-output
outputs/optimization_sweeps/<run-prefix>/comparison.eligibility.json`。这个
JSON sidecar 会把每一行的 `Startup precheck`、`Ranking precheck`、
`Promotion precheck` 状态，以及各 gate 的 eligible row 列表一起落盘。
这些 remote suite wrapper 现在也会默认生成
`comparison.eligibility.json`，和 `comparison.md` 放在同一个目录里。
如果还要把这个 compare sidecar 进一步变成下游直接消费的筛选结果，就运行
`python -m edge_vlm.optimization select-eligible --input
.../comparison.eligibility.json --gate ranking --output
.../ranking.selection.json`，或者把 gate 换成 `promotion` 输出
`promotion.selection.json`。这些 remote suite wrapper 现在默认也会做这两步，
这样 ranking/promotion 自动化就不需要再去解析 Markdown 表格。
如果要导出 scoped 的 `<=2B` 候选，还可以继续加
`--require-leq2b-candidate` 和 `--candidate-lane <vlm|text>`。现在
lightweight suite 会基于这两个过滤条件额外产出
`ranking.leq2b-vlm.selection.json` 和
`promotion.leq2b-vlm.selection.json`；Tencent text suite 也会额外产出
`ranking.leq2b-text.selection.json` 和
`promotion.leq2b-text.selection.json`，同时保留未过滤的
`ranking.selection.json` / `promotion.selection.json`。
再用 `python -m edge_vlm.optimization bundle-selections` 可以把这些 lane
artifact 合并成一个 `leq2b.candidate_bundle.json`；在 Jetson 上，
`scripts/jetson/build_remote_leq2b_candidate_bundle.sh` 可以读取
`JETSON_LEQ2B_VLM_SELECTION_DIR` 和 `JETSON_LEQ2B_TEXT_SELECTION_DIR`，并在远端写出同一个统一 bundle。
它默认还会从这个 bundle 导出 `leq2b.routes.json`；如果只想构建 bundle，可以设置
`JETSON_LEQ2B_BUILD_ROUTES=0`，也可以用 `JETSON_LEQ2B_ROUTE_GATE` /
`JETSON_LEQ2B_ROUTES_OUTPUT` 覆盖 route artifact 的 gate 和输出路径。
如果下游 router 需要按 lane 分组的主候选和备选候选，就继续运行
`python -m edge_vlm.optimization export-routes --input
.../leq2b.candidate_bundle.json --gate promotion --output
.../leq2b.routes.json`。这个 route export 会在每个 `comparison_group`
内部应用 `q4_first_q8_fallback`：同一组的 Q4 和 Q8 都通过当前 gate 时，
Q4 会成为该组的 primary，Q8 会作为 fallback metadata 写出。不同组之间仍保留
bundle 顺序，也不改写 ranking/promotion gate 语义。
如果 selector 在较宽松的 fallback gate 下选择了 Q8，wrapper 还会继续转发
`--variant-min-lfb-blocks qwen3-vl-2b-instruct-q8-smoke=...`，并把它记进
sweep plan 的 `variant_min_lfb_blocks`，这样真正执行 sweep 时不会又被更严格
的全局 `--min-lfb-blocks` 重新挡掉。

Jetson 远端连接参数在被 Git 忽略的 `.env.jetson` 中。远端 helper 会自动
读取它；不要把 SSH host、密码、token 或私有路径写进 tracked 文档。

用 `JETSON_DRY_RUN=1` 可以只打印 Docker 命令，不要求当前机器有 Docker 或 Jetson 硬件：

```bash
JETSON_DRY_RUN=1 scripts/jetson/run_gemma4_e2b_llama_docker.sh
```

正式 Jetson benchmark 时，先启动上面的某个模型 server，再跑 benchmark wrapper，让 JSONL、Markdown、manifest、profile 文件和可选 `tegrastats` 输出共用一个 run id：

```bash
EDGE_VLM_FORMAL_RUN_ID=minicpmv46-q4-jetson-formal-001 \
EDGE_VLM_CONFIG=configs/models/minicpmv46_q4.yaml \
EDGE_VLM_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.jsonl \
EDGE_VLM_SUMMARY_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.md \
EDGE_VLM_METADATA_OUTPUT=outputs/benchmarks/minicpmv46-q4-jetson-formal-001.manifest.json \
EDGE_VLM_TRIAL_COUNT=3 \
scripts/jetson/run_formal_benchmark.sh
```

用 `EDGE_VLM_FORMAL_DRY_RUN=1 EDGE_VLM_SKIP_TEGRASTATS=1` 可以在没有 server 或 Jetson 硬件时验证 wrapper 结构。

完整 checklist 见 [docs/migration_wsl_to_jetson.md](docs/migration_wsl_to_jetson.md)。

## 文档

- [Runtime matrix](docs/runtime_matrix.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [Next phase roadmap](docs/specs/next_phase_benchmark_and_models.md)
- [WSL to Jetson migration](docs/migration_wsl_to_jetson.md)
- [Implementation plan](docs/implementation_plan.md)
- [APEX workflow matrix](docs/matrix_edge_vlm_workflow.md)
- [OrangePi MiniCPM-V 4.6 notes](docs/reference_notes/orangepi_minicpmv46_notes.md)
