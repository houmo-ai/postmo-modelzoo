# GLM-ASR

本示例展示如何把 GLM-ASR-Nano-2512 模型下载、量化、编译，并部署到后摩芯片设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: GLM-ASR-Nano-2512
- 来源: https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512
- 参考仓库: https://github.com/zai-org/GLM-ASR

预训练模型在运行时下载，工程发布不包含该模型，本例只适用于 xh2。

## 2 快速开始

目前 GLM-ASR 模型提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

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
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 models/asr/glm-asr 目录执行 ptq.py：

```bash
source env.sh
cd models/asr/glm-asr
python3 ptq.py --model-name glm-asr --model-size nano-2512
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

默认输出文件位于 `output/xh2/`：

- `glm-asr-nano-2512_encode.hmm`
- `glm-asr-nano-2512_prefill.hmm`
- `glm-asr-nano-2512_decode.hmm`

### 2.5 演示

如果使用 get_model.py 下载的预编译模型，可直接执行：

```bash
python3 demo.py
```

`demo.py` 默认会从 `config.yaml` 读取 `model_name`、`model_size`、`ndevice`，并自动推导 `processor_path` 与 HMM 路径；也可以显式覆盖：

```bash
python3 demo.py --model_name glm-asr --model_size nano-2512
```

如果使用 build.py 自行编译得到的模型，也可以直接使用默认路径；`demo.py` 默认查找的就是 build.py 生成的这组产物：

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
  --skip_download         Skip model download steps.
  -size, --model_size     Model size.
  -name, --model_name     Model name.
  -h, --help              Show this help message.
```

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

## 3 输出说明

- 性能统计，包括 Encoder Time、Prefill Cost、Decode Cost、TTFT、TPOT 和 E2E Latency

```bash
Encoder Time: 581.081 ms
Prefill Cost: 61.433 ms
Decode Cost: 755.838 ms
Output 48 tokens, Decode Speed: 63.51 tokens/s
TTFT (Time to First Token): 65.472 ms
TPOT (Time Per Output Token): 15.747 ms/token
```

## 4 免责声明

本示例依赖的模型与相关资源由第三方提供并维护。使用这些模型、数据或其他外部资源时，请遵循对应的许可证、使用条款和隐私政策，并自行评估由此带来的合规性与使用风险。