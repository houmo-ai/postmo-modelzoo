# Qwen3-Next

本示例展示如何把 Qwen3-Next 模型编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-Next-80B-A3B-Instruct
- 来源: [https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct)
- 许可: Apache License 2.0 ([https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE))

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于 xh2 M50 双卡 24G 配置（`ndevice=2`，单设备显存 24G），不支持其他设备组合。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-next`
- `model_size=80b-a3b`
- `context_length=8k`
- `prefill_length=256`
- `ndevice=2`
- `ncore=2`

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

tokenizer 和原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `qwen3-next-80b-a3b`。

本示例**不提供量化脚本**，仅支持下载预编译模型进行编译和演示。

## 2 快速开始

目前 Qwen3-Next 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

本示例不提供量化脚本，因此不支持下载原始模型（`--type raw` 会抛出 `NotImplementedError`）。通过 `get_model.py` 下载预编译芯片模型用于编译和演示：

```bash
# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3-next --model_size 80b-a3b
```

### 2.3 编译

将预编译模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3-next --model_size 80b-a3b
```

默认输出名称分别为：

- `qwen3-next-80b-a3b_prefill.hmm`
- `qwen3-next-80b-a3b_decode.hmm`

> 编译仅支持在 x86_64 平台上执行。

### 2.4 演示

Qwen3-Next 模型使用 python API 进行演示。

```bash
python3 demo.py --model_name qwen3-next --model_size 80b-a3b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时（当前默认 `ndevice=2`），`prefill` / `decode` 路径默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3-next`
- `MODEL_SIZE=80b-a3b`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：编译 + 演示
bash test.sh -s all
```

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s build,demo or -s build -s demo.
  -name, --model_name     Model name. Default: qwen3-next.
  -size, --model_size     Model size. Default: 80b-a3b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh -s build
bash test.sh -s demo
bash test.sh -s all
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：本示例不支持量化，会打印 `Not support model quantization.`。
- `build`：调用 `build.py` 编译 `prefill` 和 `decode` 模型。
- `demo`：下载预编译模型后执行 Python demo。

## 4 参考结果

### 4.1 演示结果

```bash
- question:
请介绍一下存算一体技术的优势
- response:
存算一体（Compute-in-Memory, CIM）是一种颠覆传统冯·诺依曼架构的新型计算范式，其核心思想是将数据存储单元与计算单元融合在同一物理位置，从而在存储器内部直接完成数据处理。这种架构有效解决了传统计算机系统中“内存墙”（Memory Wall）和“功耗墙”（Power Wall）两大瓶颈问题。以下是存算一体技术的主要优势：

---

### 1. **突破“内存墙”瓶颈，大幅提升计算效率**
- **传统架构问题**：在冯·诺依曼架构中，CPU与内存分离，数据需在存储器与处理器之间频繁搬运，导致大量延迟和带宽瓶颈。
- **存算一体优势**：数据无需搬出存储器，直接在存储单元内进行运算（如矩阵乘加），显著减少数据搬运次数，提升吞吐率和计算速度，尤其适合高并行、大数据量的AI推理与训练任务。

> ✅ 举例：在神经网络推理中，权重和激活值存储在忆阻器或SRAM阵列中，直接进行点积运算，速度可提升10–100倍。

---

### 2. **大幅降低功耗**
- 数据搬运占现代芯片功耗的60%以上（据IBM研究）。
- 存算一体消除了大量数据传输，仅在必要时进行少量数据读写，从而显著降低动态功耗。
- 适用于边缘计算、物联网设备、可穿戴设备等对功耗敏感的场景。

> ✅ 实验表明：存算一体芯片在AI推理任务中功耗可降低90%以上。

---

### 3. **支持高并行计算，适合AI与类脑计算**
- 存算一体架构天然支持模拟计算（Analog Computing）和并行矩阵运算，非常适合深度学习中的卷积、全连接层等密集矩阵操作。
- 利用忆阻器（Memristor）、SRAM、DRAM、RRAM等器件的物理特性，可实现模拟域的乘加运算（MAC），实现“一次操作，百万并行”。

> ✅ 类比：如同人脑神经元“存储与计算一体”，存算一体是硬件层面的类脑计算实现。

---

### 4. **提升系统集成度与小型化**
- 存算一体减少了对外部存储器（如DDR）的依赖，可实现更紧凑的芯片设计。
- 有利于构建单芯片AI加速器，降低系统复杂度、成本和封装难度。

---

### 5. **支持新型非易失性存储器件**
- 存算一体常与新型存储器（如RRAM、PCM、MRAM、FeRAM）结合，这些器件具备非易失性、高密度、低功耗等特性，与计算功能融合后可实现“断电不丢数据+即时计算”，适合边缘智能和嵌入式AI。

---

### 6. **推动AI芯片架构革新**
- 传统GPU/TPU依赖高带宽内存（HBM），成本高、功耗大。
- 存算一体提供一种“低带宽、高能效”的替代路径，是下一代AI芯片（如类脑芯片、神经形态芯片）的核心技术之一。

---

### 应用场景举例：
| 领域 | 应用 |
|------|------|
| 人工智能 | 边缘AI推理、语音识别、图像分类 |
| 物联网 | 智能传感器、可穿戴设备 |
| 自动驾驶 | 实时感知与决策 |
| 云计算 | 低功耗AI服务器加速 |
| 量子计算辅助 | 模拟量子态演化 |

---

### 总结：
> **存算一体技术的核心优势 = 高效率 + 低功耗 + 高并行 + 小型化**
> 它不仅是架构优化，更是计算范式的革命，有望成为后摩尔时代突破算力瓶颈的关键技术，尤其在AI、边缘计算和智能物联网领域具有巨大潜力。

随着工艺进步（如3D堆叠、新材料）和算法协同设计的发展，存算一体正从实验室走向产业化，华为、英伟达、三星、Intel、清华大学、中科院等机构均已布局相关研发。
```

### 4.2 性能结果

====================================================================================================
Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     26 tokens
  Output Length per Sample:    908 tokens
  Prefill Model Load Time: 22419.70ms
  Decode Model Load Time: 4289.64ms
Prefill Stage Performance:
  Total Time:  357.75ms | Speed:   72.68 tokens/s
  Tokenization Time:    7.35ms
  Embedding Time:    6.93ms
  API SetInput Time:   1.34ms
  API Inference Time: 338.66ms | Prefill Speed:   76.77 tokens/s
  API GetOutput Time:  0.36ms
Decode Stage Performance:
  Total Time: 44669.71ms | Speed:   20.30 tokens/s
  Tokenization Time: 5085.52ms
  Embedding Time:  117.06ms
  API SetInput Time:   0.28ms
  API Inference Time: 43.11ms | Decode Speed:   23.20 tokens/s
  API GetOutput Time:  0.21ms
Overall Performance Metrics:
  TTFT (Time To First Token):  357.75 ms
  TPOT (Time Per Output Token): 49.25 ms/token
  E2E Latency (End-to-End):     45.29 seconds
  E2E TPS (Throughput):         20.05 tokens/s
====================================================================================================


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
