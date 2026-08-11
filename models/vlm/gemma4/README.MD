# Gemma4

本示例展示如何把Gemma4模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例支持 Gemma4 系列以下规格：

> 注意：Gemma4 原模型具备视频理解能力，但当前示例仅覆盖文本、图片和音频输入，暂未提供视频处理支持。

- `26b-a4b` / `31b`: 图文模型，运行时统一使用 `Gemma4`。
- `e2b` / `e4b`: 图文音频模型，运行时使用 `Gemma4E`。

模型实现来源于 ModelScope 的预训练模型：

| model_size | ModelScope 模型 | MTP assistant 模型 |
|---|---|---|
| `26b-a4b` | https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it | https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it-assistant |
| `31b` | https://www.modelscope.cn/models/google/gemma-4-31B-it | https://www.modelscope.cn/models/google/gemma-4-31B-it-assistant |
| `e2b` | https://www.modelscope.cn/models/google/gemma-4-E2B-it | https://www.modelscope.cn/models/google/gemma-4-E2B-it-assistant |
| `e4b` | https://www.modelscope.cn/models/google/gemma-4-E4B-it | https://www.modelscope.cn/models/google/gemma-4-E4B-it-assistant |

> 当前 runtime 的 MTP 仅支持 `26b-a4b`。使用 MTP 配置进行量化和编译后，Demo 会根据模型目录中的 `*_assistant.hmm(s)` 产物自动启用 MTP；其他 assistant 链接仅作为模型来源说明。

许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `configs/*.yml` 作为下载、量化和编译的统一配置。`test.sh` 默认使用 `./configs/gemma4_e2b.yml`，可通过 `-c/--config` 选择其他配置：

| 配置文件 | 模型规格 | 量化方法 | Context Length | 输出目录 |
|---|---|---|---:|---|
| `configs/gemma4_e2b.yml` | `e2b` | `autoround` | 131072 | `output/e2b/${HOUMO_TARGET}` |
| `configs/gemma4_e4b.yml` | `e4b` | `gptq` | 131072 | `output/e4b/${HOUMO_TARGET}` |
| `configs/gemma4_26b-a4b.yml` | `26b-a4b` | `gptq` | 262144 | `output/26b-a4b/${HOUMO_TARGET}` |
| `configs/gemma4_26b-a4b-mtp.yml` | `26b-a4b[MTP]` | `gptq` | 262144 | `output/26b-a4b-mtp/${HOUMO_TARGET}` |
| `configs/gemma4_31b.yml` | `31b` | `gptq` | 262144 | `output/31b/${HOUMO_TARGET}` |

模型名称、规格、原始模型目录、设备数、量化方法、context length、prefill chunk length、组件开关和输出目录均由所选 YAML 配置决定。

## 2 快速开始

Gemma4 模型通过 HMATC CLI 完成量化和编译。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

`test.sh` 会在 Gemma4 目录下创建或复用 `gemma4_venv`，激活该虚拟环境，并安装 `requirements.txt`。如果环境变量 `HOUMO_EXAMPLES_PATH` 指向包含 `hmodel/gptqmodel/requirements.txt` 的示例仓库目录，脚本还会安装该 GPTQModel requirements 文件。首次运行 `test.sh` 需要可访问 Python 包源；也可以提前手动准备依赖：

```bash
python3 -m venv --system-site-packages gemma4_venv
source gemma4_venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载原始模型或预编译芯片模型。模型规格和下载参数从 `--config` 指定的 YAML 配置读取：

```bash
# 下载 e2b 原始模型，用于后续量化
python3 get_model.py --config ./configs/gemma4_e2b.yml --type raw

# 下载 e2b 预编译模型，用于直接运行 Demo
python3 get_model.py --config ./configs/gemma4_e2b.yml --type hmm

# 下载 26b-a4b 原始模型及其 MTP assistant 模型
python3 get_model.py --config ./configs/gemma4_26b-a4b-mtp.yml --type raw
```

原始模型默认下载到 `./models`。预编译模型默认解压到 `output/<model_size>/${HOUMO_TARGET}`。切换模型规格时，仅需替换配置文件。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。完成环境准备并进入 `models/vlm/gemma4` 后，通过 HMATC version 2 配置执行量化：

```bash
# 默认 e2b 配置
hmatc quant -c ./configs/gemma4_e2b.yml

# 26b-a4b MTP 配置
hmatc quant -c ./configs/gemma4_26b-a4b-mtp.yml
```

量化产物保存在配置 `save_dir` 对应的 `${HOUMO_TARGET}/hmquant` 目录，供后续编译和 Demo 使用。MTP 是否启用由配置中的 `quant.speculative_decode: mtp` 决定，无需额外传递 MTP 开关。

### 2.4 编译

使用同一份配置将量化产物编译为芯片模型：

```bash
# 默认 e2b 配置
hmatc build -c ./configs/gemma4_e2b.yml

# 26b-a4b MTP 配置
hmatc build -c ./configs/gemma4_26b-a4b-mtp.yml
```

编译参数由配置中的 `build` 段控制，主要包括 `ndevice`、`context_length`、`prefill_chunk_length`、`flash_attention`、`enable_common_subgraph` 以及各模型组件的构建开关。编译结果位于 `<save_dir>/${HOUMO_TARGET}`，prefill/decode 组件可能生成 `.hmm` 或 `.hmms`。

### 2.5 演示

Gemma4 模型使用 Python API 进行演示。将 HMM/HMMS 模型及其 `hmquant` 目录放在同一个模型目录下，然后执行：

```bash
# 文本模式；默认模型目录为 output/e2b/${HOUMO_TARGET}
python3 demo.py --question "你好，请介绍一下你自己。" --image "" --audio ""

# 图片模式
python3 demo.py --model /path/to/gemma4-model --image ../../../data/pic/beach.jpeg --audio "" --question "描述这张图片的内容"

# 音频模式
python3 demo.py --model /path/to/gemma4-e-model --audio ../../../data/audio/0.wav --image "" --question "请描述这个音频的内容"

# MTP 文本模式
python3 demo.py --model /path/to/gemma4-mtp-model --question "你好，请介绍一下你自己。" --image "" --audio ""
```

演示程序扫描 `--model` 目录第一层，自动发现 `*_prefill.hmm(s)`、`*_decode.hmm(s)`、`*_visual.hmm(s)`、`*_audio.hmm(s)` 和 `*_assistant.hmm(s)`。prefill 和 decode 为必需组件，同一类组件最多只能匹配一个文件；visual、audio 和 assistant 为可选组件。

模型目录还必须包含 `hmquant/quant_embedding.pt` 和 `hmquant/hf_config`。存在 audio 模型时还必须包含 `hmquant/per_layer_input_embedding.pt`，程序会使用 `Gemma4E`；否则使用 `Gemma4`。存在 assistant 模型时自动启用对应的 MTP 路径，无需显式传入模型规格或 MTP 开关。

演示参数说明：
- `--model`: 模型目录，默认 `output/e2b/${HOUMO_TARGET}`
- `--ndevice`: 此次运行允许使用的最大设备数，默认 `1`；实际设备列表为 `0` 到 `ndevice - 1`
- `--question`: 输入问题
- `--system_prompt`: system prompt
- `--image`: 图片路径，默认 `data/pic/beach.jpeg`；文件不存在或 visual 模型不存在时输出 warning 并忽略图片
- `--audio`: 音频路径，默认 `data/audio/0.wav`；文件不存在或 audio 模型不存在时输出 warning 并忽略音频
- `--max-new-tokens`: 最大生成 token 数，默认 `2048`
- `--enable-thinking`: 在 chat template 渲染阶段启用 thinking 模式

只运行文本输入时，应显式传入 `--image "" --audio ""`。输入文件会先按当前路径查找，再按 `${HOUMO_EXAMPLES_PATH}/<path>` 查找。

## 3 一键运行

以上步骤可以通过 `test.sh` 执行。脚本默认选择 `demo` 步骤和 `./configs/gemma4_e2b.yml`：

```bash
# 默认下载并运行 e2b 预编译模型
bash test.sh

# 使用指定配置完成下载、量化、编译和演示
bash test.sh -s all -c ./configs/gemma4_e2b.yml
```

各步骤行为如下：

- `quant`：未指定 `--skip_download` 时，先按配置下载 raw 模型，再执行 `hmatc quant -c <config>`。
- `build`：执行 `hmatc build -c <config>`。
- `demo`：单独运行该步骤且未指定 `--skip_download` 时，先按配置下载预编译模型，再执行 `demo.py`。
- `all`：依次执行 raw 模型下载、量化、编译和 Demo，不再额外下载预编译模型。

`test.sh` 支持以下参数：

```text
Usage: test.sh [options]
Options:
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
  -c, --config            Configuration file path relative to the Gemma4 directory.
                          Default: ./configs/gemma4_e2b.yml.
  --model                 Model directory passed to demo.py.
  --ndevice               Override the download device count and set the maximum devices for demo.py.
  --context_length        Override the context length used by get_model.py.
  --prefill_length        Override the prefill length used by get_model.py.
  --skip_download         Skip model download.
  --system_prompt         System prompt passed through to demo.py.
  -h, --help              Show this help message.
```

`-c/--config` 用于下载、量化和编译，Demo 的模型目录由 `demo.py` 的默认值或 `--model` 决定。`--context_length`、`--prefill_length` 和 `--ndevice` 会在下载 raw 或预编译模型时透传给 `get_model.py`，覆盖配置文件中的对应下载标识；其中 `--ndevice` 还会作为 Demo 使用的最大设备数。未指定这些参数时，继续使用 YAML 配置值。使用非 e2b 配置时，应显式传入对应模型目录：

```bash
# 仅量化 e4b
bash test.sh -s quant -c ./configs/gemma4_e4b.yml

# 编译 26b-a4b MTP
bash test.sh -s build -c ./configs/gemma4_26b-a4b-mtp.yml

# 下载并运行 31b 预编译模型
bash test.sh -s demo -c ./configs/gemma4_31b.yml --model output/31b/${HOUMO_TARGET}

# 使用已有 e2b 模型运行 Demo
bash test.sh -s demo --skip_download --model output/e2b/${HOUMO_TARGET}

# 完整执行 26b-a4b MTP 流程
bash test.sh -s all -c ./configs/gemma4_26b-a4b-mtp.yml --model output/26b-a4b-mtp/${HOUMO_TARGET}
```

## 4 HMM 内存占用参考

可使用 `readhmm` 查看编译后 HMM 的内存占用：

```bash
readhmm "output/e2b/${HOUMO_TARGET}/gemma4-e2b_prefill.hmm" --mem_usage
```

以下数据来自一组历史编译产物，单位为二进制单位 GiB / MiB，仅用于内存占用参考；表中的 context length 和 ndevice 以该组产物的实际编译参数为准，不一定与当前 `configs/*.yml` 完全一致。`InOut(no KV)` 为 `mem_usage.inout` 中去除 KV cache 后的输入/输出占用，`Shared KV Cache` 为 LLM prefill/decode HMM 共享的 KV cache 输入占用；同一模型的 prefill 和 decode 使用同一份 KV cache，不需要把两行重复累加。`Workspace` 为 `mem_usage.workspace` 列表求和。`.hmms` 多设备模型的内存占用按单设备子模型统计。

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
