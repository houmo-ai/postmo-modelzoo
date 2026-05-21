# MiniCPM-o_2.6

本示例展示如何把 MiniCPMO 中的各个子模型量化和编译，部署到后摩 M50 芯片设备上。

## 特别事项

本项目使用从 ModelScope 下载的预训练模型、配置文件和脚本：

- 模型名称: MiniCPM-o-2_6
- 来源: https://modelscope.cn/openbmb/MiniCPM-o-2_6
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的预处理代码image_processing_minicpmv.py和preocessing_minicpmo.py已进行适配性修改，并包含在项目提交文件中。特此声明。

[TOC]

## 1 模型说明

本例使用的 MiniCPMO 各模块模型实现来源于 [https://modelscope.cn/openbmb/MiniCPM-o-2_6](https://modelscope.cn/openbmb/MiniCPM-o-2_6)。默认模型由 `config.yaml` 统一维护，当前默认组合为 `minicpmo` + `7b`。

## 2 快速开始

MiniCPMO支持Omni、vlm、llm、voice clone、instruction to speech和only tts等图像、本文、语音功能。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 get_model 脚本下载模型，支持获取原始模型或直接运行的预编译模型：

```bash
# 直接获取可直接运行的芯片模型
python3 get_model.py --type hmm

# 原始模型，需要量化+编译器编译后才能运行
python3 get_model.py --type raw

# 显式指定默认模型
python3 get_model.py --type raw --model_name minicpmo --model_size 7b
```

### 2.3 量化

当前仅支持w8a8基础量化，暂未提供其它高阶量化方案。
量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到48GB，主机空闲内存需要达到48GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64.** /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/omni/minicpmo` 目录执行 `ptq.py`。默认的大语言模型模块 `prefill_length=256`、`context_length=4k`，统一来自 `config.yaml`。

```bash
source env.sh
cd models/omni/minicpmo
```

```bash
python3 ptq.py
python3 ptq.py --model-name minicpmo --model-size 7b
```
量化完成的模型存放在 `output/$HOUMO_TARGET/hmquant` 目录。

### 2.4 编译

```bash
python3 build.py
python3 build.py --model_name minicpmo --model_size 7b
```

### 2.5 演示

MiniCPMO 使用 python API 进行演示。`demo.py` 默认会从 `config.yaml` 读取 `model_name` 与 `model_size`，并自动推导 `tokenizer_dir` 与各个 HMM 路径。

```bash
python3 demo.py
python3 demo.py --model_name minicpmo --model_size 7b
#默认为omni模式，如需体验llm
python3 demo.py --example_idx 1
#如需体验vlm
python3 demo.py --example_idx 2
#如需体验多张图的vlm
python3 demo.py --example_idx 3
#如需语音克隆
python3 demo.py --example_idx 4
#如需仅跑tts文本转语音，且声音为女声
python3 demo.py --example_idx 5
#如需仅跑tts文本转语音，且声音为男声
python3 demo.py --example_idx 6
#如遇到长文本，需要拆分多段来运行tts
python3 demo.py --example_idx 7
#如需体验指令到语音功能
python3 demo.py --example_idx 8
#如需体验语音对话功能
python3 demo.py --example_idx 9
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
  -size, --model_size     Model size.
  -name, --model_name     Model name.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前支持的模型组合为：

- `minicpmo` + `7b`

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant -name minicpmo -size 7b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build -name minicpmo -size 7b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo -name minicpmo -size 7b

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all -name minicpmo -size 7b

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download -name minicpmo -size 7b
```

## 3 参考结果

### 3.1 演示结果

```bash
Audio saved to output_omni.wav
画面中有一个穿着红色夹克和黑色裤子的人，他正在一个被雪覆盖的山地地形上滑雪。他正沿着一条清晰的滑雪道滑行，周围是高耸的雪山和多云的天空。
```

### 3.2 性能结果

```bash
Total Cost 18230.092 ms
Input Tokens: 1399, Output tokens: 50
Vision Cost: 5157.962 ms
Audio Cost: 385.364 ms
LLM Prefill Speed: 935.94 tokens/s
TTFT (Time to First Token): 7048.539 ms
TPOT (Time Per Output Token): 16.86 tokens/s
ViT+Whisper+LLM TPS (Tokens Per Second): 5.02 tokens/s
TTS Prefill Mean Time: 10.058 ms x 30 times
TTS Decoder Mean Time: 255.036 ms x 30 groups
TTS Dvae Cost: 41.207 ms
TTS Vocos Cost: 29.955 ms
Total Audio Duration Generated: 15.94 s
TTS Total Cost: 8137.064 ms
TTS Real-Time Factor(RTF): 0.510609
TTS Generate Speed: 1.96 x real-time
E2E Latency (End-to-End Latency): 18.230 seconds
```

### 3.3 备注

```bash
1. 由于tts对长文本容易产生异常（原算法即存在的现象），我们提供了一种文本拆分的方法。只需将chat()或chat_otts()调用时给定text_split=True，即可触发。同时，为控制多段语音中间自然过渡，可选择打开insert_silence来自动在多段之间插入静音间隔；
2. 部分生成的语音可能存在结尾有较长无效杂音的情况，目前算法正在持续优化和改善中，当前可通过打开trim_tail来根据音频能量自动截断无效尾音；
3. tts生成语音，后处理默认是没有打开后采样的，如需更丰富的回答可以选择打开后采样。打开方式：定义HMMiniCPMO时将tts_sample设为True即可。同时，后采样也能避免一些较长无效尾音、杂音、语音异常的问题（原始模型只有后采样方案，也是为了避免此类情况）。
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
