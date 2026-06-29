# Gemma4

本示例展示如何把Gemma4模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例支持 Gemma4 系列以下规格：

> 注意：Gemma4 原模型具备视频理解能力，但当前示例仅覆盖文本、图片和音频输入，暂未提供视频处理支持。

- `26b-a4b`: MoE 图文模型，使用 `Gemma4MoE` 运行时。
- `31b`: Dense 图文模型，使用 `Gemma4` 运行时。
- `e2b` / `e4b`: E 系列图文音频模型，使用 `Gemma4E` 运行时。

模型实现来源于 ModelScope 的预训练模型：

| model_size | ModelScope 模型 | MTP assistant 模型 |
|---|---|---|
| `26b-a4b` | https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it | https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it-assistant |
| `31b` | https://www.modelscope.cn/models/google/gemma-4-31B-it | https://www.modelscope.cn/models/google/gemma-4-31B-it-assistant |
| `e2b` | https://www.modelscope.cn/models/google/gemma-4-E2B-it | https://www.modelscope.cn/models/google/gemma-4-E2B-it-assistant |
| `e4b` | https://www.modelscope.cn/models/google/gemma-4-E4B-it | https://www.modelscope.cn/models/google/gemma-4-E4B-it-assistant |

> 当前 runtime 的 MTP 仅支持 `26b-a4b`，因此实际使用 `--assistant-model` 时请指定 `26b-a4b` 对应的 assistant 模型目录；其他 assistant 链接仅作为模型来源说明。

许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值来源。当前默认模型为：

- `model_name=gemma4`
- `model_size=e2b`
- `context_length=128k`
- `ndevice=1`

可通过 `--model-name` / `--model-size` 覆盖默认值。`model_size` 支持 `26b-a4b`、`31b`、`e2b`、`e4b`。

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

MTP assistant 辅助模型可通过 `get_model.py --mtp` 一并下载：

```bash
# 下载原始模型 + MTP assistant 辅助模型
python3 get_model.py --type raw --model_name gemma4 --model_size 26b-a4b --mtp
```

也可通过 `modelscope` 命令单独下载 assistant 模型：

```bash
modelscope download --model google/gemma-4-26B-A4B-it-assistant --local_dir gemma-4-26B-A4B-it-assistant
```

下载完成后，可在量化导出 MTP 时通过 `--assistant-model` 指定该目录。

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

# 如需导出 MTP assistant 相关模型，目前仅 26b-a4b 支持，需要额外指定 assistant/draft HF 模型目录
python3 ptq.py --model-name gemma4 --model-size 26b-a4b --assistant-model /path/to/gemma4-assistant-model
```

量化参数说明：
- `--config`: 配置文件路径，默认当前目录下的 `config.yaml`
- `--model`: HF / ModelScope 模型目录路径；如果目录下存在 `quantize_config.json`，脚本会按已量化 GPTQModel HF 模型处理，跳过重新量化并直接进入导出流程
- `--model-name`: 输出模型名，默认来自 `config.yaml`，当前为 `gemma4`
- `--model-size`: 输出模型规格，默认来自 `config.yaml`，当前为 `e2b`；支持 `26b-a4b`、`31b`、`e2b`、`e4b`
- `--out-dir`: 输出目录，默认 `./output/${HOUMO_TARGET}`
- `--context-length`: 导出 HMONNX 使用的最大序列长度，默认 `2048`
- `--prefill-chunk-length`: prefill HMONNX 的 chunk 长度，默认 `256`
- `--seed`: 随机种子，默认 `42`
- `--debug`: 开启 workflow debug 模式
- `--assistant-model`: Gemma4 MTP assistant/draft HF 模型目录。指定该参数会开启 MTP 导出流程，目前仅支持 `--model-size 26b-a4b`；导出时会在量化目标模型的基础上额外导出 assistant 相关 HMONNX 产物，供后续 `build.py --mtp` 编译 assistant HMM 使用

`ptq.py` 的主要流程：

1. 根据 `--model-size` 选择 xh2modelzoo workflow 配置：
   - `26b-a4b` → `configs/gemma4_26b_a4b_full.yaml`
   - `31b` → `configs/gemma4_31b_full.yaml`
   - `e2b` → `configs/gemma4_e2b_autoround.yaml`
   - `e4b` → `configs/gemma4_e4b_full.yaml`
2. 创建 `AutoLLMWorkflow`。
3. 如果 `--model` 已经是量化模型目录（存在 `quantize_config.json`），使用 `algorithm=existing_hf` 的配置直接导出；否则先执行量化。
4. 导出 HMONNX 到临时目录 `output/${HOUMO_TARGET}/hmquant/exported_model`。
5. 在临时导出目录下查找 `hmquant_xh2*_YYYYMMDD` 目录，并将其中内容移动到 `output/${HOUMO_TARGET}/hmquant`，最后删除临时导出目录。移动时如果目标文件或目录已存在，会先删除再覆盖。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name gemma4 --model_size 26b-a4b

# 编译 MTP 模型，目前仅 26b-a4b 支持
python3 build.py --model_name gemma4 --model_size 26b-a4b --mtp
```

默认输出名称按模型规格生成，例如 `26b-a4b` 为：

- `gemma4-26b-a4b_prefill.hmm`
- `gemma4-26b-a4b_decode.hmm`
- `gemma4-26b-a4b_visual.hmm`

`e2b` / `e4b` 会额外编译 audio HMM：

- `gemma4-e2b_audio.hmm`
- `gemma4-e4b_audio.hmm`

`31b` 默认 `ndevice=2`，当设备数大于 1 时，LLM prefill/decode 输出后缀会从 `.hmm` 变为 `.hmms`。

编译参数说明：
- `--model_dir`: HMONNX / 量化产物目录，默认 `output/${HOUMO_TARGET}/hmquant`
- `--model_name`: 输出模型名，默认来自 `config.yaml`
- `--model_size`: 输出模型规格，默认来自 `config.yaml`
- `--output_dir`: HMM 输出目录，默认 `output/${HOUMO_TARGET}`
- `--ncore`: 编译核数，默认来自 `config.yaml` 或环境变量 `HOUMO_CORE_NUM`
- `--j`: 并行编译任务数，默认物理 CPU 核数
- `--context_length`: LLM context length，默认来自 `config.yaml`
- `--prefill_length`: prefill chunk 长度，默认来自 `config.yaml`
- `--batch`: decode batch，默认来自 `config.yaml`
- `--ndevice`: 多设备数量，默认来自 `config.yaml`
- `--stage`: `build`、`test`、`all`，默认 `build`
- `--flash_attention`: 两个整数，分别控制 LLM 和 ViT FlashAttention，默认 `2 1`
- `--enable_common_subgraph`: 开启 common subgraph 优化
- `--mtp`: 仅 `26b-a4b` 支持，开启 MTP 编译。开启后仍会编译 LLM decode HMM，但该 decode HMM 在 MTP runtime 中作为 target verify 使用；同时会额外编译 assistant draft HMM。其他模型规格暂不支持 MTP。

`build.py` 的主要流程：

1. 从 `output/${HOUMO_TARGET}/hmquant` 查找 `prefill`、`decode`、`visual` 子目录下的 HMONNX 文件。
2. 编译 LLM prefill HMM。
3. 非 MTP 模式下编译 LLM decode HMM 和 visual HMM。
4. `e2b` / `e4b` 非 MTP 模式下额外编译 audio HMM。
5. MTP 模式仅支持 `26b-a4b`。开启 `--mtp` 后，LLM decode HMM 仍会编译，但在 MTP runtime 中作为 target verify 使用；同时会额外编译 assistant draft HMM。

### 2.5 演示

Gemma4模型使用python API进行演示。

1. 将编译好的模型放在output目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出结果。

```bash
# 文本模式
python3 demo.py --model_name gemma4 --model_size e2b --question "你好，请介绍一下你自己。" --image "" --audio ""

# 图片模式
python3 demo.py --model_name gemma4 --model_size 26b-a4b --image ../../../data/pic/beach.jpeg --question "描述这张图片的内容"

# 音频模式，仅 e2b/e4b 支持
python3 demo.py --model_name gemma4 --model_size e2b --audio ../../../data/audio/0.wav --image "" --question "请描述这个音频的内容"

# MTP 文本模式，目前仅 26b-a4b 支持
python3 demo.py --model_name gemma4 --model_size 26b-a4b --mtp --question "你好，请介绍一下你自己。" --image "" --audio ""
```

演示参数说明：
- `--config`: 配置文件路径，默认 `config.yaml`
- `--tokenizer_dir`: tokenizer / processor 配置目录，默认 `output/${HOUMO_TARGET}/hmquant/hf_config`
- `--embedding_path`: LLM embedding 权重路径，默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--plib_embedding_path`: E 系列 per-layer input embedding 权重路径，默认 `output/${HOUMO_TARGET}/hmquant/per_layer_input_embedding.pt`
- `--prefill_path`: prefill HMM 路径，默认 `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `--decode_path`: decode HMM 路径，默认 `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`
- `--vit_path`: visual HMM 路径，默认 `output/${HOUMO_TARGET}/${model_name}-${model_size}_visual.hmm`
- `--audio_path`: audio HMM 路径，仅 `e2b` / `e4b` 使用，默认 `output/${HOUMO_TARGET}/${model_name}-${model_size}_audio.hmm`
- `--model_name`: 模型名，默认来自 `config.yaml`
- `--model_size`: 模型规格，默认来自 `config.yaml`
- `--ndevice`: 设备数量，默认来自 `config.yaml`
- `--question`: 输入问题
- `--image`: 图片路径；文件不存在时自动退化为文本模式
- `--audio`: 音频文件路径，仅 `e2b` / `e4b` 使用；文件不存在或 audio HMM 未加载时自动忽略音频
- `--max-new-tokens`: 最大生成 token 数，默认 `2048`
- `--enable-thinking`: 在 chat template 渲染阶段启用 thinking 模式
- `--mtp`: 启用 MTP speculative decoding，目前仅 `26b-a4b` 支持。该模式使用 decode HMM 作为 target verify，并加载 assistant HMM 生成候选 token；当前 runtime 按 no KV rollback 模式运行，即 partial reject 后不做 KV restore/replay。
- `--assistant_path`: MTP assistant HMM 路径，默认 `output/${HOUMO_TARGET}/${model_name}-${model_size}_assistant.hmm`，仅 `--mtp` 开启时生效

## 3 一键评估

以上步骤可以通过 test.sh 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

当前 `test.sh` 内部默认使用以下值：
- `MODEL_NAME=gemma4`
- `MODEL_SIZE=e2b`
- `NDEVICE=1`

对应执行行为如下：
- `quant`：如未传 `--skip_download`，先下载 raw 模型，再执行 `python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"`
- `build`：执行 `python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"`
- `demo`：如未传 `--skip_download`，先下载预编译模型，再执行 `python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"`

如果需要修改模型规格或设备数，可通过 `test.sh` 参数传入，无需修改脚本源码。

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
# 仅运行 quant，会先下载 raw 模型并使用脚本内默认的 gemma4-e2b 参数量化
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

## 4 HMM 内存占用参考

可使用 `readhmm` 查看编译后 HMM 的内存占用：

```bash
readhmm "output/${HOUMO_TARGET}/gemma4-e2b_prefill.hmm" --mem_usage
```

以下数据基于 `output/xh2` 下的一组已编译 HMM 统计，单位为二进制单位 GiB / MiB。`InOut(no KV)` 为 `mem_usage.inout` 中去除 KV cache 后的输入/输出占用，`Shared KV Cache` 为 LLM prefill/decode HMM 共享的 KV cache 输入占用；同一模型的 prefill 和 decode 使用同一份 KV cache，不需要把两行重复累加。`Workspace` 为 `mem_usage.workspace` 列表求和。`.hmms` 多设备模型的内存占用按单设备子模型统计。

| HMM | Context Length | ndevice | Total | Weights | Kernel | InOut(no KV) | Shared KV Cache | Workspace | Core | Args |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma4-26b-a4b_decode.hmm` | 131072 | 1 | 15.128 GiB | 13.647 GiB | 48.011 MiB | 188.008 MiB | 1.250 GiB | 0.182 MiB | 2 | 66 |
| `gemma4-26b-a4b_prefill.hmm` | 131072 | 1 | 17.718 GiB | 13.647 GiB | 192.284 MiB | 190.000 MiB | 1.312 GiB | 2.385 GiB | 2 | 67 |
| `gemma4-26b-a4b_visual.hmm` | - | 1 | 1.108 GiB | 580.271 MiB | 369.423 MiB | 6.570 MiB | 0.000 MiB | 178.454 MiB | 2 | 6 |
| `gemma4-31b_decode.hmms` | 262144 | 2 | 13.904 GiB | 8.503 GiB | 33.859 MiB | 375.263 MiB | 5.000 GiB | 0.613 MiB | 2 | 126 |
| `gemma4-31b_prefill.hmms` | 262144 | 2 | 24.204 GiB | 8.504 GiB | 689.624 MiB | 378.500 MiB | 5.125 GiB | 9.532 GiB | 2 | 127 |
| `gemma4-31b_visual.hmm` | - | 1 | 2.406 GiB | 583.018 MiB | 1.669 GiB | 7.937 MiB | 0.000 MiB | 163.735 MiB | 2 | 6 |
| `gemma4-e2b_audio.hmm` | - | 1 | 473.248 MiB | 380.184 MiB | 78.550 MiB | 2.972 MiB | 0.000 MiB | 11.543 MiB | 2 | 6 |
| `gemma4-e2b_decode.hmm` | 131072 | 1 | 1.964 GiB | 1.543 GiB | 39.768 MiB | 7.271 MiB | 384.000 MiB | 0.380 MiB | 2 | 37 |
| `gemma4-e2b_prefill.hmm` | 131072 | 1 | 2.020 GiB | 1.543 GiB | 80.391 MiB | 12.750 MiB | 384.000 MiB | 10.938 MiB | 2 | 37 |
| `gemma4-e2b_visual.hmm` | - | 1 | 416.669 MiB | 179.658 MiB | 64.887 MiB | 5.886 MiB | 0.000 MiB | 166.238 MiB | 2 | 6 |
| `gemma4-e4b_audio.hmm` | - | 1 | 476.259 MiB | 381.708 MiB | 78.571 MiB | 4.437 MiB | 0.000 MiB | 11.543 MiB | 2 | 6 |
| `gemma4-e4b_decode.hmm` | 131072 | 1 | 4.239 GiB | 2.883 GiB | 84.944 MiB | 23.027 MiB | 1.000 GiB | 256.693 MiB | 2 | 55 |
| `gemma4-e4b_prefill.hmm` | 131072 | 1 | 4.303 GiB | 2.883 GiB | 134.347 MiB | 29.875 MiB | 1.000 GiB | 265.750 MiB | 2 | 55 |
| `gemma4-e4b_visual.hmm` | - | 1 | 418.003 MiB | 180.422 MiB | 64.910 MiB | 6.433 MiB | 0.000 MiB | 166.238 MiB | 2 | 6 |

## 5 参考结果

### 5.1 演示结果

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

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
