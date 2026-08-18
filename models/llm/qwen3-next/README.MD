# Qwen3-Next

本示例展示如何把 Qwen3-Next 模型量化、编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-Next-80B-A3B-Instruct
- 来源: [Qwen3-Next-80B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct)
- 许可: Apache License 2.0 ([https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE))

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于 xh2 M50 双卡 24G 配置（`ndevice=2`，单设备显存 24G），不支持其他设备组合。

当前目录使用 `config.yaml` 作为默认值单一真值来源。该配置同时被 `get_model.py`、`ptq.py`、`build.py` 和 `demo.py` 使用。当前默认模型为：

- `model_name=qwen3-next`
- `model_size=80b-a3b`
- `context_length=256k`
- `prefill_length=256`
- `ndevice=2`
- `ncore=2`

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

tokenizer 和原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `Qwen3-Next-80B-A3B-Instruct`。

### 资源要求

**后摩设备运行要求：**

- `80b-a3b`：仅支持 xh2 M50 双卡 24G 配置（`ndevice=2`，单设备显存 24G）。

**量化硬件要求：**

- `80b-a3b`：GPTQ 量化需要至少 48GB GPU 空闲显存，HMONNX 导出需要至少 200GB CPU 空闲内存。

**编译时 host 内存要求：**

- `80b-a3b`：需要至少 300GB CPU 空闲内存。

本例使用 GPTQ 对 Qwen3-Next 进行量化，并通过 Merak workflow 导出 HMONNX。量化和编译产物均不包含在仓库中。

### 用于校准的数据集

默认量化 workflow 使用 Qwen3-Next 专用的 GPTQ 校准数据：

- `gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl`
- 默认配置：`512` 个样本，序列长度 `1024`，文本字段为 `text`

该校准数据仅用于**研究和评估目的**，不作为本仓库的一部分进行分发。使用时请遵守其原始许可和相关使用条款。

## 2 快速开始

目前 Qwen3-Next 模型仅提供 python 脚本量化和编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或预编译芯片模型：

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3-next --model_size 80b-a3b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3-next --model_size 80b-a3b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

Qwen3-Next-80B-A3B-Instruct 的 GPTQ 量化需要至少 48GB GPU 空闲显存，后续 HMONNX 导出需要至少 200GB CPU 空闲内存。

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`：

```bash
source env.sh
cd models/llm/qwen3-next
python3 ptq.py --model_name qwen3-next --model_size 80b-a3b
```

`ptq.py` 默认执行 GPTQ 量化和 HMONNX 导出，默认使用 `config.yaml` 中的 workflow，最终将导出的 `hmquant` 内容移动到：

```text
output/${HOUMO_TARGET}/hmquant/
```

常用参数如下：

- `--model_dir`：显式指定 Hugging Face 原始模型目录。
- `--quant_config_path`：显式指定 Merak workflow YAML。
- `--output_dir`：指定最终 `hmquant` 输出目录，默认 `./output/${HOUMO_TARGET}/hmquant`。
- `--work_dir`：指定量化和导出的中间目录，默认 `./work_dirs`。
- `--bits`：覆盖 workflow 中的量化 bit 数，默认由 workflow 配置为 `4`。
- `--context-max-length`：覆盖导出上下文长度。
- `--export-from-quanted-model`：输入已经量化的 HF 模型时跳过量化阶段，直接执行导出。
- `--overwrite`：导出前删除已有的导出中间目录。
- `--dump-golden`：导出后生成 golden 数据。
- `--quick-test`：导出后运行快速 HMONNX 测试。

### 2.4 编译

将量化导出的 HMONNX 模型编译为在芯片上运行的模型。执行脚本：

编译前请确保系统具有至少 300GB CPU 空闲内存。

```bash
python3 build.py --model_name qwen3-next --model_size 80b-a3b
```

默认输出名称分别为：

- `qwen3-next-80b-a3b_prefill.hmm`
- `qwen3-next-80b-a3b_decode.hmm`

> 编译仅支持在 x86_64 平台上执行。

### 2.5 演示

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

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
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
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s demo
bash test.sh -s all
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：检查 GPU，下载原始模型并调用 `ptq.py` 执行量化导出。
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
  Input Length per Sample:     23 tokens
  Output Length per Sample:    447 tokens
  Prefill Model Load Time: 436096.38ms
  Decode Model Load Time: 2346.08ms
Prefill Stage Performance:
  Total Time:  296.49ms | Speed:   77.57 tokens/s
  Tokenization Time:   16.71ms
  Embedding Time:    3.31ms
  API SetInput Time:   4.07ms
  API Inference Time: 268.45ms | Prefill Speed:   85.68 tokens/s
  API GetOutput Time:  0.46ms
Decode Stage Performance:
  Total Time: 18220.40ms | Speed:   24.48 tokens/s
  Tokenization Time: 2680.59ms
  Embedding Time:   62.04ms
  API SetInput Time:   0.31ms
  API Inference Time: 34.26ms | Decode Speed:   29.19 tokens/s
  API GetOutput Time:  0.23ms
Overall Performance Metrics:
  TTFT (Time To First Token):  296.49 ms
  TPOT (Time Per Output Token): 40.85 ms/token
  E2E Latency (End-to-End):     19.25 seconds
  E2E TPS (Throughput):         23.22 tokens/s
====================================================================================================


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
