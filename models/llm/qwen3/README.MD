# Qwen3

本示例展示如何把 Qwen3 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-0.6B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-0.6B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3-1.7B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-1.7B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3-8B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-8B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3-14B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-14B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3`
- `model_size=8b`
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`

其中 `0.6b`、`1.7b`、`8b` 默认 `context_length=32k`，`14b` 默认 `context_length=16k`。

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

脚本默认会从 `modelscope_repo` 取最后一段作为 tokenizer 或模型目录默认值，例如 `qwen/qwen3-8b` 会推导为 `qwen3-8b`。

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

目前 Qwen3 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3 --model_size 0.6b
python3 get_model.py --type raw --model_name qwen3 --model_size 8b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3 --model_size 8b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/qwen3
```

推荐先下载原始模型，再执行：

```bash
python3 ptq.py --model_name qwen3 --model_size 8b
```

如需指定校准集，可显式传入：

```bash
python3 ptq.py --model_name qwen3 --model_size 8b --calib_data ~/your_data.jsonl
```

量化结果位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3 --model_size 8b
```

默认输出名称分别为：

- `qwen3-8b_prefill.hmm`
- `qwen3-8b_decode.hmm`

### 2.5 演示

Qwen3 模型使用 python API 进行演示。

```bash
python3 demo.py --model_name qwen3 --model_size 8b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3`
- `MODEL_SIZE=8b`
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
  -size, --model_size     Model size. Choices: 0.6b, 1.7b, 8b, 14b.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh --model_size 0.6b
bash test.sh -s quant --model_size 8b
bash test.sh -s build --model_size 8b
bash test.sh -s demo --model_size 8b
bash test.sh -s all --model_size 8b
bash test.sh -s demo --skip_download --model_size 8b
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 `prefill` 和 `decode` 模型。
- `demo`：下载预编译模型后执行 Python demo。
- 若环境中存在 `llm_perf`，`demo` 流程末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

```bash
User question:
请介绍一下存算一体技术的优势
Model response:
存算一体（Compute-in-Memory, CIM）技术是一种将存储与计算功能集成在同一物理结构中的新型计算架构，它突破了传统冯·诺依曼架构中存储与计算分离的限制，具有显著的优势。以下是存算一体技术的主要优势：

---

### 1. **降低能耗**
- **减少数据移动**：传统计算架构中，数据需要在存储器和处理器之间频繁搬运，这会消耗大量能量。存算一体技术将计算直接在存储单元中完成，大大减少了数据移动，从而显著降低能耗。
- **提升能效比**：由于减少了数据传输的能耗，存算一体技术在能效比（Energy Efficiency）方面表现优异，尤其适用于低功耗场景，如物联网（IoT）、边缘计算和人工智能（AI）设备。

---

### 2. **提高计算速度**
- **并行计算能力**：存算一体技术可以实现并行计算，多个计算任务可以同时在存储单元中进行，提升计算效率。
- **降低延迟**：由于计算直接在存储器中进行，避免了传统架构中数据搬运带来的延迟，从而加快了计算速度。

---

### 3. **适用于AI与大数据处理**
- **适合矩阵运算**：存算一体技术特别适合进行大规模矩阵运算，这在深度学习、神经网络和大数据处理中非常常见。
- **支持低精度计算**：存算一体技术通常基于非易失性存储器（如忆阻器、相变存储器、铁电存储器等），支持低精度计算（如二进制或低位整数），这在AI推理中可以显著降低计算资源需求。

---

### 4. **节省芯片面积**
- **集成度高**：存算一体技术将存储器与计算单元集成在同一芯片上，减少了对额外计算单元的需求，从而节省了芯片面积。
- **适合异构计算**：可以与CPU、GPU等其他计算单元协同工作，形成异构计算系统，提高整体性能。

---

### 5. **支持新型存储器技术**
- **兼容多种存储器**：存算一体技术可以与多种新型非易失性存储器（如忆阻器、相变存储器、存储器阵列等）结合，这些存储器具有低功耗、高密度、非易失性等优点。
- **推动存储器技术发展**：存算一体技术为新型存储器提供了新的应用场景，促进了存储器技术的创新与演进。

---

### 6. **适用于边缘计算与嵌入式系统**
- **低功耗、小体积**：存算一体技术适合用于边缘计算设备、嵌入式系统和移动设备，这些设备对功耗和体积有较高要求。
- **实时处理能力**：在边缘端进行本地计算，减少对云端的依赖，提高数据处理的实时性和安全性。

---

### 总结

存算一体技术通过将存储与计算功能融合，显著提升了计算效率、降低了能耗，并为人工智能、边缘计算、物联网等应用提供了新的解决方案。它被认为是未来计算架构的重要发展方向之一，尤其在高性能计算、低功耗计算和AI加速等领域具有广阔的应用前景。

如果你对存算一体技术在某个具体领域的应用感兴趣，也可以告诉我，我可以进一步为你介绍。
```

### 4.2 性能结果

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     30 tokens
  Output Length per Sample:    701 tokens
  Prefill Model Load Time: 2284.08ms
  Decode Model Load Time:  226.08ms
Prefill Stage Performance:
  Total Time:  130.42ms | Speed:  230.03 tokens/s
  Tokenization Time:   11.94ms
  Embedding Time:    0.57ms
  API SetInput Time:   3.86ms
  API Inference Time: 110.10ms | Prefill Speed:  272.47 tokens/s
  API GetOutput Time:  0.63ms
Decode Stage Performance:
  Total Time: 36907.97ms | Speed:   18.97 tokens/s
  Tokenization Time:  247.49ms
  Embedding Time:  100.48ms
  API SetInput Time:   0.62ms
  API Inference Time: 45.09ms | Decode Speed:   22.18 tokens/s
  API GetOutput Time:  0.49ms
Overall Performance Metrics:
  TTFT (Time To First Token):  130.42 ms
  TPOT (Time Per Output Token): 52.73 ms/token
  E2E Latency (End-to-End):     37.13 seconds
  E2E TPS (Throughput):         18.88 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
