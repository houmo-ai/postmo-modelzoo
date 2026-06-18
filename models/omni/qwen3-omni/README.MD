# Qwen3-Omni

本示例展示如何把 Qwen3-Omni 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-Omni-30B-A3B-Instruct
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。
同时，`local_processor/` 目录下的 `processing_qwen3_omni_moe.py` 和 `image_processing_qwen2_vl.py` 基于 HuggingFace Transformers 库中的 Qwen3-Omni 预处理代码改编，遵循 Apache License 2.0 开源协议（Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team）。特此声明。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-omni`
- `model_size=30b-a3b`
- `context_length=4k`
- `prefill_length=256`
- `ndevice=1`

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

目前 Qwen3-Omni 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和 wikitext2 数据集：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型：

```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到48GB以上，主机空闲内存需要达到300GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/omni/qwen3-omni` 目录，使用 GPTQ 方案执行 ptq.py 脚本。默认的 `prefill_length` 和 `context_length` 分别来自 `config.yaml`，当前默认值是 `256` 和 `4k`。

```bash
source env.sh
cd models/omni/qwen3-omni
python3 ptq.py --model-name qwen3-omni --model-size 30b-a3b --gptqmodel
```

量化完成的模型存放在 `output/$HOUMO_TARGET/hmquant` 目录。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3-omni --model_size 30b-a3b
```

默认输出名称分别为：

- `qwen3-omni-30b-a3b_prefill.hmm`
- `qwen3-omni-30b-a3b_decode.hmm`
- `qwen3-omni-30b-a3b_visual.hmm`
- `qwen3-omni-30b-a3b_audio.hmm`
- `qwen3-omni-30b-a3b_talker_prefill.hmm`
- `qwen3-omni-30b-a3b_talker_decode.hmm`
- `qwen3-omni-30b-a3b_talker_prediction_prefill.hmm`
- `qwen3-omni-30b-a3b_talker_prediction_decode.hmm`
- `qwen3-omni-30b-a3b_text_projection.hmm`
- `qwen3-omni-30b-a3b_code2wav.hmm`

### 2.5 演示

Qwen3-Omni 模型使用 python API 进行演示。`demo.py` 默认从 `config.yaml` 读取 `model_name` 与 `model_size`，并自动推导 `tokenizer_dir` 与各个 HMM 路径。

```bash
# 默认为 omni 模式（图像 + 音频 + 文本 -> 文本 + 语音）
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b

# 如需多轮语音对话
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b --example_idx 1

# 如需图文描述（vlm）
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b --example_idx 2

# 如需乐曲赏析（music）
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b --example_idx 3

# 如需语音识别（asr）
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b --example_idx 4

# 如需语音翻译（英译中）
python3 demo.py --model_name qwen3-omni --model_size 30b-a3b --example_idx 5
```

当前 `demo.py` 支持以下 `example_idx` 模式：

| example_idx | 模式       | 说明                                                        |
|-------------|------------|-------------------------------------------------------------|
| 0           | omni       | 图像 + 音频 + 文本 -> 文本 + 语音（默认模式）               |
| 1           | conv       | 多轮语音对话，上下文保持，首轮无图像，次轮携带历史           |
| 2           | vlm        | 图像 + 文本 -> 文本 + 语音                                  |
| 3           | music      | 乐曲音频 + 文本 -> 文本 + 语音（听乐评乐）                  |
| 4           | asr        | 中文语音识别 -> 纯文本（无语音输出）                        |
| 5           | translate  | 英文语音翻译为中文文本（无语音输出）                        |

> **提示1**：长文本生成语音时，建议开启 `--talker_segment` 自动切分长文本来防止语音退化。使用 `--talker_segment_max_chars` 可控制每段最大字符数（默认 60），`--talker_segment_gap_ms` 可控制段间静音间隔（默认 120ms）。
> **提示2**：如需运行长上下文（如 `context_length=32k`）或者打开enable_audio_generation（语音生成），M50 芯片设备空闲内存需要达到48GB或使用单device大于等于18GB的2 device（推理demo.py时使用--run_device_num 2）。

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
  -name, --model_name     Model name. Default: qwen3-omni.
  -size, --model_size     Model size. Default: 30b-a3b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前支持的模型组合为：

- `qwen3-omni` + `30b-a3b`

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

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

## 4 参考结果

### 4.1 演示结果


[Result] text: ['我看到四张汽车的图片，同时听到了一阵咳嗽声。']
[Result] audio: /mnt/data/nan.xu/project/xh2_modelzoo/imodelzoo/models/omni/qwen3-omni/omni_output.wav

### 4.2 性能结果

====================================================================
              Qwen3-Omni HMM Inference Performance
====================================================================
  Input Length:    263 tokens  (audio=38, image=196)
  Output Length:     16 text tokens
--------------------------------------------------------------------
Encoders:
  Audio Encoder:  31.246 ms
  Visual Encoder: 195.082 ms
Thinker (text LLM):
  Prefill: 986.519 ms | Speed: 266.59 tokens/s
  TTFT (Time To First Token): 1212.846 ms
  Decode: 542.478 ms | TPOT: 36.165 ms/token | 27.65 tokens/s
Talker (speech codec):
  Talker Generate: 4883.063 ms | 40 codec steps
  Code2Wav: 868.464 ms
  Generated Audio: 3.200 s | RTF: 1.7974 (0.56x real-time)
--------------------------------------------------------------------


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
