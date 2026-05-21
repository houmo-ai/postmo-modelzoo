# gte-Qwen2-1.5B-instruct

本示例展示如何把 gte-Qwen2-1.5B-instruct 模型量化和编译，部署到后摩芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: gte-Qwen2-1.5B-instruct
- 来源: https://www.modelscope.cn/models/iic/gte-Qwen2-1.5B-instruct
- 许可：https://www.apache.org/licenses/LICENSE-2.0

预训练模型在运行时下载，工程发布**不包含**该模型。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/iic/gte-Qwen2-1.5B-instruct](https://www.modelscope.cn/models/iic/gte-Qwen2-1.5B-instruct)。
本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=gte`
- `model_size=1.5b`
- `context_length=2k`
- `prefill_length=256`
- `ndevice=1`

## 2 快速开始

目前gte-Qwen2-1.5B-instruct模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和 wikitext2 数据集，原始模型默认下载到 `gte_Qwen2-1.5B-instruct` 目录：

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

进入docker后source环境变量，并进入`models/embedding/gte`目录，执行 `ptq.py` 脚本。默认的 `input_sequence_length` 和 `context_length` 分别来自 `config.yaml`，当前默认值是 `256` 和 `2k`。默认的量化方式为 `w8a8_sefp`。

```bash
source env.sh
cd models/embedding/gte
python3 ptq.py
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。默认输出名称为 `gte-1.5b_prefill.hmm`，执行脚本：

```bash
python3 build.py
```

### 2.5 演示

gte_Qwen2-1.5B-instruct 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：

  - `output/${HOUMO_TARGET}/gte-1.5b_prefill.hmm`
  - `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.6 一键评估

以上步骤可以通过 test.sh 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

test.sh 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: gte.
  -size, --model_size     Model size. Default: 1.5b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `gte` 和 `1.5b`。

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 显式指定模型名和模型规格（当前默认就是 gte 1.5b）
bash test.sh -s demo -name gte -size 1.5b

# 组合执行多个步骤
bash test.sh -s quant,build

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

## 3 参考结果

### 3.1 演示结果

```bash
[[71.75, 22.046875], [12.5859375, 62.84375]]
```

### 3.2 性能结果

```bash
Total Input: 177 tokens, Prefill Cost 188.270 ms
Prefill Speed: 4287.29 tokens/s
E2E Latency (End-to-End Latency): 0.189 seconds

```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
