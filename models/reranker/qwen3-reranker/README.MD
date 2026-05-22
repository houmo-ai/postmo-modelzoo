# Qwen3(通义千问3)

本示例展示如何把Qwen3-Reranker模型量化和编译，部署到后摩芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: Qwen3-Reranker-8B
- 来源: https://modelscope.cn/models/Qwen/Qwen3-Reranker-8B
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

预训练模型在运行时下载，工程发布**不包含**该模型。

## 用于校准的数据集

本项目引用 T2Reranking 数据集仅用于**研究和评估目的**。

- 数据集许可: Apache License 2.0
- 来源: https://modelscope.cn/datasets/C-MTEB/T2Reranking

⚠️ 重要提示:
- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen3-Reranker-8B](https://modelscope.cn/models/Qwen/Qwen3-Reranker-8B)。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在Qwen3-Reranker-8B目录：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/reranker/qwen3-reranker`目录，执行ptq.py脚本。默认的prefill上下文长度分别为256和8192。

```bash
source env.sh
cd models/reranker/qwen3-reranker
python3 ptq.py
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- prefill
|-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
qwen3-reranker_prefill.hmm

### 2.5 演示

Qwen3模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过demo中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
queries =  "Explain gravity"
documents = "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."

scores:
[0.990234375]


### 3.2 性能结果

```bash
Total Input: 104 tokens, Prefill Cost 140.316 ms
Prefill Speed: 741.19 tokens/s
E2E Latency (End-to-End Latency): 0.144 seconds
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
