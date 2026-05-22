# GLM-OCR

本示例展示如何把 GLM-OCR 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: GLM-OCR
- 来源: https://www.modelscope.cn/models/zhipuAI/GLM-OCR
- 许可: MIT License

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=glm-ocr`
- `model_size=0.9b`
- `context_length=8k`
- `prefill_length=256`
- `image_size=672x672`
- `max_size_t=2`
- `ndevice=1`

## 2 快速开始

目前 GLM-OCR 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv

```bash
pip3 install -r requirements.txt
```

注：由于本示例依赖的 transformer 版本高于 python3.9 最高支持的 4.33.0 版本，因此无法在 python3.10 以下版本（如 ubuntu20.04）运行，如遇到找不到 transformer 库问题请自行安装 python3.10 以上版本。

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

如果需要自己量化，可选择下载原始模型：

```bash
python3 get_model.py --type raw --model_name glm-ocr --model_size 0.9b
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录：

```bash
python3 get_model.py --type hmm --model_name glm-ocr --model_size 0.9b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/ocr/glm-ocr` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd models/ocr/glm-ocr
python3 ptq.py --model_name glm-ocr --model_size 0.9b
```

常用参数：

- `--hf_model_dir`: 原始模型目录，默认从 `config.yaml` 的 `modelscope_repo` 推导为 `GLM-OCR`
- `--model_name`: 输出模型名前缀，默认 `glm-ocr`
- `--model_size`: 输出模型规格，默认 `0.9b`
- `--output_dir`: 量化输出目录，默认 `output/$HOUMO_TARGET/hmquant`
- `--image_path`: 导出视觉分支时使用的示例图片，默认 `../../../data/pic/ocr.jpeg`
- `--prompt`: 导出时使用的默认提示词，默认 `Text Recognition:`
- `--max_new_tokens`: 默认 `256`
- `--max_sequence_length`: 最大序列长度，默认由 `context_length=8k` 解析得到
- `--input_sequence_length`: prefill 输入长度，默认 `256`
- `--image_size_w`: 图像宽度，默认 `672`
- `--image_size_h`: 图像高度，默认 `672`
- `--max_size_t`: 默认 `2`

默认产物目录为 `output/$HOUMO_TARGET/hmquant`，其中会包含以下核心文件或子目录：

```text
output/xh2/hmquant
|-- prefill/
|-- decode/
|-- visual/
`-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name glm-ocr --model_size 0.9b
```

默认输出名称分别为：

- `glm-ocr-0.9b_prefill.hmm`
- `glm-ocr-0.9b_decode.hmm`
- `glm-ocr-0.9b_visual_672x672x2.hmm`

### 2.5 演示

GLM-OCR 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：
- `output/${HOUMO_TARGET}/glm-ocr-0.9b_prefill.hmm`
- `output/${HOUMO_TARGET}/glm-ocr-0.9b_decode.hmm`
- `output/${HOUMO_TARGET}/glm-ocr-0.9b_visual_672x672x2.hmm`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

2. 当 `ndevice > 1` 时，`prefill` 和 `decode` 会自动切换为 `.hmms`，视觉模型仍保持 `.hmm`。

3. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py --model_name glm-ocr --model_size 0.9b
```

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `glm-ocr` 和 `0.9b`。

对应执行行为如下：

- `quant`：如未传 `--skip_download`，先下载 raw 模型，再执行 `python3 ptq.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"`
- `build`：执行 `python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"`
- `demo`：当 `STEP=demo` 且未传 `--skip_download` 时，先下载预编译模型，再执行 `python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"`

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: glm-ocr.
  -size, --model_size     Model size. Default: 0.9b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

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

# 显式指定模型名和模型规格
bash test.sh -s demo -name glm-ocr -size 0.9b

# 组合执行多个步骤
bash test.sh -s quant,build

# 跳过模型下载，使用当前目录下已有模型文件进行演示
bash test.sh -s demo --skip_download
```

## 4 参考结果

### 4.1 演示结果

```bash
question:
Text Recognition:
response:
后摩智能
M50芯片
```

### 4.2 性能结果

#### M50 2core

```bash
# 输入 336x336 情况下
Output 156 tokens, Prefill Cost 36.223 ms
Prefill Speed: 4306.66 tokens/s
Output 8 tokens, Decode Cost 83.455 ms
Decode Speed: 95.86 tokens/s
TTFT (Time to First Token): 158.199 ms
TPOT (Time Per Output Token): 10.432 ms/token
E2E Latency (End-to-End Latency): 0.242 seconds
E2E TPS (End-to-End Tokens Per Second): 33.11 tokens/s
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
