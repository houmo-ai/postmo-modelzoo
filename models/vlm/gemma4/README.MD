# Gemma4

本示例展示如何把Gemma4模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: gemma-4-26B-A4B-it
- 来源: https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=gemma4`
- `model_size=26b-a4b`
- `context_length=2k`
- `ndevice=1`

## 2 快速开始

目前Gemma4模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name gemma4 --model_size 26b-a4b
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm --model_name gemma4 --model_size 26b-a4b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/vlm/gemma4` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd models/vlm/gemma4
python3 ptq.py --model-name gemma4 --model-size 26b-a4b
```

量化参数说明：
- `--model`: 模型目录路径，默认 `./gemma-4-26B-A4B-it`
- `--model-name`: 输出模型名，默认 `gemma4`
- `--model-size`: 输出模型规格，默认 `e2b`
- `--context-length`: 最大序列长度，默认 `2048`
- `--prefill-chunk-length`: 最大序列长度，默认 `256`

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name gemma4 --model_size 26b-a4b
```

默认输出名称分别为：

- `gemma4-26b-a4b_prefill.hmm`
- `gemma4-26b-a4b_decode.hmm`
- `gemma4-26b-a4b_visual.hmm`

### 2.5 演示

Gemma4模型使用python API进行演示。

1. 将编译好的模型放在output目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出结果。

```bash
# 文本模式
python3 demo.py --model_name gemma4 --model_size 26b-a4b --question "你好，请介绍一下你自己。"

# 图片模式
python3 demo.py --model_name gemma4 --model_size 26b-a4b --image ../../../data/pic/beach.jpeg --question "描述这张图片的内容"
```

演示参数说明：
- `--tokenizer_dir`: tokenizer目录路径
- `--embedding_path`: embedding权重路径
- `--prefill_path`: prefill模型路径
- `--decode_path`: decode模型路径
- `--vit_path`: visual模型路径
- `--model_name`: 模型名，默认 `gemma4`
- `--model_size`: 模型规格，默认 `26b-a4b`
- `--ndevice`: 设备数量，默认 `1`
- `--question`: 输入问题
- `--image`: 图片路径
- `--max-new-tokens`: 最大生成token数

## 3 一键评估

以上步骤可以通过 test.sh 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

当前脚本内部固定使用以下默认值：
- `MODEL_NAME=gemma4`
- `MODEL_SIZE=26b-a4b`
- `NDEVICE=1`

对应执行行为如下：
- `quant`：如未传 `--skip_download`，先下载 raw 模型，再执行 `python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"`
- `build`：执行 `python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"`
- `demo`：如未传 `--skip_download`，先下载预编译模型，再执行 `python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"`
- `demo`：如果环境中存在 `llm_perf`，会继续自动执行一组性能测试，默认使用 `output/${HOUMO_TARGET}/gemma4-26b-a4b_visual_448x448.hmm`

如果需要修改模型规格、设备数或 visual 尺寸，需要同步调整 `test.sh` 中的默认值，以及 `ptq.py`、`build.py`、`demo.py` 的相关参数。

test.sh 支持以下参数：

```bash
Usage: test.sh [options]
Options:
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size.
  -name, --model_name     Model name.
  --ndevice               Number of devices.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载 raw 模型并使用脚本内默认的 gemma4-26b-a4b 参数量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型；若环境中存在 llm_perf，还会追加跑性能用例
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 跳过模型下载，使用当前目录下已有的模型文件执行 demo
bash test.sh -s demo --skip_download
```

## 4 参考结果

### 4.1 演示结果

```bash
question: 你好，请介绍一下你自己。
response: 你好！我是 **Gemma 4**，是由 Google DeepMind 开发的大型语言模型。

作为一个开放权重的模型，我旨在通过自然语言交互来协助用户完成各种任务。以下是关于我的一些详细介绍：

### 1. 我的核心能力
* **文本处理**：我可以进行创作、翻译、总结、问答以及逻辑推理。
* **多模态理解**：除了文本，我还可以理解和处理**图像**信息。
* **音频处理**：在 Gemma 4 系列中，2B 和 4B 模型还具备处理**音频**输入的能力。
* **知识范围**：我的知识截止日期是 **2025 年 1 月**。

### 2. 我的输出形式
虽然我可以接收多种形式的输入（文本、图像、部分型号支持音频），但我**仅能生成文本**作为输出。我无法生成图像或音频文件。

### 3. 我的工作方式
* **工具使用**：除非在对话上下文中明确为我提供了特定的工具定义和端点，否则我无法直接访问互联网（如 Google 搜索）或使用外部工具。
* **基于上下文**：我非常擅长根据你提供的上下文信息进行回答，这有助于我提供更精准、更具针对性的帮助。

**你可以尝试这样问我：**
* “请帮我写一封正式的商务邮件。”
* “解释一下量子力学的基本概念。”
* “（上传一张图片）请描述一下这张图片里的内容。”

请问今天有什么我可以帮你的吗
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
