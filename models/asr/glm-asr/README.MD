# GLM-ASR

本示例展示如何把 GLM-ASR-Nano-2512 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: GLM-ASR-Nano-2512
- 来源: https://www.modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512
- 参考仓库: https://github.com/zai-org/GLM-ASR
- 许可: Apache License 2.0 (https://github.com/zai-org/GLM-ASR/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=glm-asr`
- `model_size=nano-2512`
- `ndevice=1`

## 2 快速开始

目前 GLM-ASR 模型提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

注：由于本示例依赖的transformer版本高于python3.9最高支持的4.33.0版本，因此无法在python3.10以下版本（如ubuntu20.04）运行，如遇到找不到transformer库问题请自行安装python3.10以上版本。

### 2.2 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型或芯片模型。

如果仅查看演示结果，可直接下载预编译模型：

```bash
python3 get_model.py --type hmm
```

下载后的原始模型默认保存在当前目录下的 glm-asr-nano-2512：

```bash
python3 get_model.py --type raw
```

演示脚本默认读取仓库内的测试音频：

```bash
../../../data/audio/audio.mp3
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.2 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 models/asr/glm-asr 目录执行 ptq.py：

```bash
source env.sh
cd models/asr/glm-asr
python3 ptq.py --model_name glm-asr --model_size nano-2512
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

默认输出名称分别为：

- `glm-asr-nano-2512_encode.hmm`
- `glm-asr-nano-2512_prefill.hmm`
- `glm-asr-nano-2512_decode.hmm`

### 2.5 演示

GLM-ASR 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 读取 `model_name`、`model_size`、`ndevice`，并自动推导 `processor_path` 与 HMM 路径。

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

如需显式覆盖默认参数，命令如下：

```bash
python3 demo.py --model_name glm-asr --model_size nano-2512
```

如需显式传入模型路径，命令如下：

```bash
python3 demo.py \
  --encoder_path output/xh2/glm-asr-nano-2512_encode.hmm \
  --prefill_path output/xh2/glm-asr-nano-2512_prefill.hmm \
  --decode_path output/xh2/glm-asr-nano-2512_decode.hmm \
  --processor_path glm-asr-nano-2512 \
  --audio ../../../data/audio/audio.mp3
```

## 3 一键评估

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
  -size, --model_size     Model size. Default: nano-2512.
  -name, --model_name     Model name. Default: glm-asr.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `glm-asr` 和 `nano-2512`。

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant -name glm-asr -size nano-2512

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build -name glm-asr -size nano-2512

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo -name glm-asr -size nano-2512

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all -name glm-asr -size nano-2512

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download -name glm-asr -size nano-2512
```

## 4 参考结果

### 4.1 演示结果

```bash
Initializing GLM-ASR Inference via Tcim...
Loading encoder from: output/xh2/glm-asr-nano-2512_encode.hmm
encoder model loaded
Loading prefill from: output/xh2/glm-asr-nano-2512_prefill.hmm
prefill model loaded
Loading decode from: output/xh2/glm-asr-nano-2512_decode.hmm
decode model loaded
Loading processor and tokenizer from: GLM-ASR-Nano-2512
Loading token embedding from: output/xh2/hmquant/quant_embedding.pt
Processing Chunk 1/1 (2.4s)
Running Encoder...
Running Llm...
Encoder Time: 5329.015 ms
Prefill Cost: 45.320 ms
Decode Cost: 86.718 ms
============================================================
Transcription Result:
============================================================
每一天都要快乐哦
============================================================
Output 5 tokens, Decode Speed: 57.66 tokens/s
TTFT (Time to First Token): 50.196 ms
TPOT (Time Per Output Token): 17.344 ms/token
E2E Latency: 0.137 seconds
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。