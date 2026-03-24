# Qwen3(通义千问 3)

本示例展示如何把 Qwen3 模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

-   模型名称: Qwen3-30B-A3B
-   来源: https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B
-   许可: Apache License 2.0 （https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B/file/view/master/LICENSE?status=0）

预训练模型在运行时下载，工程发布**不包含**该模型。

## 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

-   数据集许可: CC BY-NC 4.0
-   来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

-   本数据集**不作为本仓库的一部分进行分发**。
-   本数据集**不受本项目 Apache License 2.0 许可的约束**。
-   任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
-   用户有责任确保在使用本数据集时遵守相关规定。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)。
本例只适用于 xh2。

## 2 快速开始

目前 Qwen3 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 get_model 脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型，下载后的模型存放在 qwen3-30b-a3b 目录：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）。

注意：提供的预编译芯片模型需运行在内存不少于 **24GB** 的后摩芯片中。

```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA12.2 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID，显存需要达到 24GB 以上，空闲内存需要达到 291GB 以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64.** /bin/bash
```

进入 docker 后 source 环境变量，并进入`models/llm/qwen3-30b-a3b`目录，执行 ptq.py 脚本。默认的 prefill 和 decode 上下文长度分别为 256 和 8192。

```bash
source env.sh
cd models/llm/qwen3-30b-a3b
python ptq.py
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- decoder
|-- prefill
|-- quant_embedding.pt
```

如需使用高阶量化提升精度，请打开gptqmodel量化权重（需要较长时间，大约超过5小时）

```bash
python ptq.py --gptqmodel
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括 2 个文件：
qwen3_prefill.hmm
qwen3_decode.hmm

### 2.3 演示

Qwen2 模型使用 python API 进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过 demo 中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

3. 注意：直接运行预编译好的 hmm 模型和执行量化+编译后产生的 hmm 模型执行 demo 后输出可能略有差异，属于正常现象。

```bash
python3 demo.py
```

### 2.4 一键评估

以上步骤可以通过 test.sh 脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
存算一体（Compute-in-Memory, CIM）是一种将计算功能直接集成在存储单元中的新型计算架构，旨在突破传统冯·诺依曼架构的“存储墙”和“功耗墙”限制。它在人工智能、边缘计算、物联网等高性能、低功耗应用中展现出显著优势。以下是存算一体技术的主要优势：

---

### 1. **提升计算效率，降低延迟**
- **传统架构**：数据需要在存储器和处理器之间频繁搬运，导致高延迟和低效率。
- **存算一体**：计算直接在存储单元内部完成，减少了数据搬运，显著降低了延迟，提高了计算效率。

---

### 2. **显著降低功耗**
- **传统架构**：数据搬运消耗大量能量，尤其是在大规模数据处理中。
- **存算一体**：由于减少了数据移动，功耗大幅降低，特别适合对能耗敏感的边缘设备和物联网终端。

---

### 3. **提高能效比（Energy Efficiency）**
- 存算一体技术在处理密集型计算任务（如神经网络推理）时，能效比远高于传统CPU或GPU，适合部署在功耗受限的场景中。

---

### 4. **支持大规模并行计算**
- 存算一体架构通常基于存储器的并行特性（如SRAM、DRAM、ReRAM等），能够实现大规模并行计算，适合处理矩阵运算、神经网络等任务。

---

### 5. **适用于AI和机器学习**
- 存算一体技术特别适合处理深度学习中的矩阵乘法和卷积操作，能够显著提升AI推理和训练的效率，降低硬件成本。

---

### 6. **减少对高速缓存和内存带宽的依赖**
- 传统架构依赖高速缓存和内存带宽来支撑计算，而存算一体直接在存储器中进行计算，减少了对这些资源的依赖。

---

### 7. **适合边缘计算和物联网**
- 在边缘设备中，存算一体技术可以实现本地数据处理，减少对云端的依赖，提高实时性和隐私保护。

---

### 8. **支持新型存储器技术**
- 存算一体技术可以与新型存储器（如ReRAM、PCM、MRAM等）结合，充分发挥其非易失性、低功耗、高密度等优势。

---

### 应用场景举例：
- **AI芯片**：如谷歌的TPU、华为的昇腾系列等。
- **边缘计算设备**：如智能摄像头、可穿戴设备。
- **物联网终端**：如传感器节点、智能家居设备。
- **自动驾驶**：用于实时图像识别和决策处理。

---

### 挑战与未来发展方向：
尽管存算一体技术优势显著，但目前仍面临一些挑战，如：
- 存储器的计算精度和可编程性限制；
- 与传统软件生态的兼容性问题；
- 制造工艺和成本控制问题。

未来，随着新型存储器的发展和算法优化，存算一体技术有望成为下一代计算架构的重要组成部分。

---

如果你对某个具体应用场景或技术细节感兴趣，我可以进一步展开说明。
```

### 3.2 性能结果

```bash
Total Input: 30 tokens, Output 668 tokens, Prefill Cost 1044.749 ms, Decode Cost 32407.680 ms
Prefill Speed: 28.72 tokens/s; Decode Speed: 20.58 tokens/s
TTFT (Time to First Token): 1044.749 ms
TPOT (Time Per Output Token): 48.587 ms/token
E2E Latency (End-to-End Latency): 33.454 seconds
E2E TPS (End-to-End Tokens Per Second): 19.97 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
