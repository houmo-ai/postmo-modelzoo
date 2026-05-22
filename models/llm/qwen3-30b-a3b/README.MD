# Qwen3-30B-A3B

本示例展示如何把 Qwen3-30B-A3B 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-30B-A3B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3`
- `model_size=30b-a3b`
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

tokenizer 和原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `Qwen3-30B-A3B`。

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen3-30B-A3B 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3 --model_size 30b-a3b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3 --model_size 30b-a3b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/qwen3-30b-a3b
```

推荐先下载原始模型，再执行：

```bash
python3 ptq.py --model-name qwen3 --model-size 30b-a3b
```

若需要启用 `gptqmodel`，可执行：

```bash
python3 ptq.py --model-name qwen3 --model-size 30b-a3b --gptqmodel
```

量化结果位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3 --model_size 30b-a3b
```

默认输出名称分别为：

- `qwen3-30b-a3b_prefill.hmm`
- `qwen3-30b-a3b_decode.hmm`

### 2.5 演示

Qwen3-30B-A3B 模型使用 python API 进行演示。

```bash
python3 demo.py --model_name qwen3 --model_size 30b-a3b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3`
- `MODEL_SIZE=30b-a3b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s demo
bash test.sh -s all
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 `prefill` 和 `decode` 模型。
- `demo`：下载预编译模型后执行 Python demo。
- 若环境中存在 `llm_perf`，`demo` 流程末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
存算一体（Computing-in-Memory，CIM）是一种新兴的计算架构技术，其核心思想是将数据存储与计算功能集成在同一物理位置，打破传统冯·诺依曼架构中“存储”与“计算”分离的瓶颈。相比传统架构，存算一体技术在多个方面展现出显著优势，主要体现在以下几个方面：

---

### 1. **大幅提升能效比（能效更高）**
- **问题**：传统架构中，数据需要在处理器和内存之间频繁搬运，功耗主要消耗在数据传输上（“内存墙”问题）。
- **优势**：存算一体技术在存储单元内部直接进行计算（如矩阵乘法、累加等），避免了大量数据搬运，显著降低功耗。
- **结果**：能效比可提升数十倍甚至上百倍，特别适合对功耗敏感的应用场景（如移动设备、边缘计算、物联网）。

---

### 2. **突破“内存墙”瓶颈，提升计算速度**
- **问题**：CPU与内存之间的带宽有限，数据传输成为性能瓶颈，尤其在处理大规模数据（如AI推理、图像处理）时。
- **优势**：计算在存储单元内完成，数据无需离开存储位置，极大减少了延迟和带宽压力。
- **结果**：显著提升处理速度，尤其在高并行、高密度计算任务中表现突出。

---

### 3. **支持高并行计算，适合AI与大数据应用**
- **优势**：存算一体天然支持大规模并行计算，特别适合执行神经网络中的矩阵-向量乘法（如卷积、全连接层）。
- **典型应用**：AI推理、深度学习、图像识别、自然语言处理等。
- **结果**：在AI芯片中，存算一体可实现“以数据为中心”的高效计算，大幅降低延迟和功耗。

---

### 4. **减少芯片面积与系统复杂度**
- **优势**：由于计算单元与存储单元集成，减少了传统架构中复杂的缓存层次、总线结构和控制逻辑。
- **结果**：芯片面积更小，系统集成度更高，有助于实现更紧凑、低功耗的芯片设计。

---

### 5. **降低系统延迟，提升响应速度**
- **优势**：数据“就近计算”，避免了长距离传输带来的延迟。
- **结果**：特别适用于实时性要求高的场景，如自动驾驶、工业控制、实时视频分析等。

---

### 6. **面向未来技术演进的潜力**
- 存算一体与新型存储技术（如RRAM、MRAM、PCM、SRAM等）高度兼容，可与先进制程（如3nm、2nm）结合。
- 为下一代智能芯片、类脑计算、存内计算芯片等提供技术基础。

---

### 典型应用场景
- AI推理芯片（如边缘AI芯片）
- 自动驾驶传感器处理
- 物联网终端设备
- 高性能计算（HPC）中的加速器
- 低功耗可穿戴设备

---

### 总结
存算一体技术通过“让数据在存储中计算”，从根本上重构了传统计算架构，解决了“内存墙”和“功耗墙”问题。其核心优势在于：**高能效、高速度、高并行性、低延迟**，是推动人工智能、边缘计算和下一代智能硬件发展的关键技术之一。

> ✅ 简言之：**存算一体 = 更快、更省电、更智能的计算新范式。**
随着材料、工艺和算法的持续进步，存算一体正从实验室走向产业化，有望成为未来计算架构的主流方向之一。
```

### 4.2 性能结果

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     30 tokens
  Output Length per Sample:    796 tokens
  Prefill Model Load Time: 6467.42ms
  Decode Model Load Time:   72.97ms
Prefill Stage Performance:
  Total Time:  193.65ms | Speed:  154.91 tokens/s
  Tokenization Time:   11.75ms
  Embedding Time:    2.85ms
  API SetInput Time:   0.73ms
  API Inference Time: 177.41ms | Prefill Speed:  169.10 tokens/s
  API GetOutput Time:  0.57ms
Decode Stage Performance:
  Total Time: 26950.99ms | Speed:   29.50 tokens/s
  Tokenization Time:  327.29ms
  Embedding Time:  131.23ms
  API SetInput Time:   0.62ms
  API Inference Time: 31.37ms | Decode Speed:   31.88 tokens/s
  API GetOutput Time:  0.50ms
Overall Performance Metrics:
  TTFT (Time To First Token):  193.65 ms
  TPOT (Time Per Output Token): 33.90 ms/token
  E2E Latency (End-to-End):     27.37 seconds
  E2E TPS (Throughput):         29.08 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
