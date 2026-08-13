# Fun-Audio-Chat

本示例展示如何使用 Fun-Audio-Chat-8B 的预编译芯片模型，在后摩芯片设备上完成音频理解、语音对话和端到端语音交互。当前目录暂不提供量化和编译流程，仅支持使用已有的 HMM 模型进行 Python 演示。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Fun-Audio-Chat-8B
- 来源: [FunAudioLLM/Fun-Audio-Chat-8B](https://www.modelscope.cn/models/FunAudioLLM/Fun-Audio-Chat-8B)
- 许可: Apache License 2.0 ([LICENSE](https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/LICENSE))

Fun-Audio-Chat 是面向低延迟语音交互的大型音频语言模型，支持音频输入、文本生成和语音生成。语音生成链路依赖配套的 CosyVoice3 token2wav 资源。

`hmm_xh2_funaudiochat-vad-token2wav.zip` 包含 VAD 和 token2wav 部分模型。这些模型是 Fun-Audio-Chat 的辅助模型，并非 Fun-Audio-Chat 主模型本体；如需支持 VAD 或 token2wav 相关能力，可以联系 demo 提供厂商。

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=funaudiochat`
- `model_size=8b`
- `quant_type=w8a8`
- `context_length=256`
- `prefill_length=256`
- `ncore=2`
- `ndevice=1`
- `batch=1`

`demo.py` 默认使用以下资源：

- 预训练模型处理器和 tokenizer: `Fun-Audio-Chat-8B/`
- 文本 embedding: `output/xh2/hmquant/quant_embedding.pt`
- 音频 embedding: `output/xh2/hmquant/quant_audio_embedding.pt`
- 语言模型 HMM: `output/xh2/funaudiochat-8b_prefill.hmm`、`output/xh2/funaudiochat-8b_decode.hmm`
- 音频理解 HMM: `output/xh2/funaudiochat-8b_audio_encoder.hmm`
- 语音生成 HMM: `output/xh2/funaudiochat-8b_audio_tower.hmm`、`output/xh2/funaudiochat-8b_audio_decoder_prefill.hmm`、`output/xh2/funaudiochat-8b_audio_decoder_decode.hmm`
- token2wav HMM: `output/xh2/cosyvoice3-0.5b-2512_flow_encoder.hmm`、`output/xh2/cosyvoice3-0.5b-2512_flow_spk.hmm`、`output/xh2/cosyvoice3-0.5b-2512_flow_decoder.hmm`、`output/xh2/cosyvoice3-0.5b-2512_hift_part1.hmm`、`output/xh2/cosyvoice3-0.5b-2512_hift_part2.hmm`
- E2E VAD HMM: `output/xh2/fsmd_vad.hmm`

### 资源要求

**后摩设备运行要求：**

- `Fun-Audio-Chat-8B`: 单卡12GB可运行

**量化硬件要求：**

- `Fun-Audio-Chat-8B`: 暂无数据。当前目录暂不提供量化脚本。

**编译时 host 内存要求：**

- `Fun-Audio-Chat-8B`: 暂无数据。当前目录暂不提供编译脚本。

## 2 快速开始

当前示例仅提供使用预编译 HMM 模型的 Python 推理方式。芯片模型只能在 Linux 环境下运行，量化和编译流程暂未纳入当前示例。

### 2.1 环境准备

建议使用 Python 3.12 虚拟环境。安装 Python 依赖：

```bash
python3 -m venv funaudiochat_venv
source funaudiochat_venv/bin/activate
pip3 install -r requirements.txt
```

模型运行还依赖仓库中的 `utils/python/houmo_engine`、后摩运行时和 `tcim_lite`。如果运行环境已经由仓库统一环境提供，可直接使用该环境；否则请先按照仓库环境说明配置后摩 Python 运行时。

### 2.2 获取模型和数据

通过 `get_model.py` 获取模型资源。默认下载预编译芯片模型和 token2wav 依赖：

```bash
export HOUMO_TARGET=xh2
python3 get_model.py --type hmm
```

如需从 ModelScope 下载预训练模型文件：

```bash
export HOUMO_TARGET=xh2
python3 get_model.py --type raw --source_type modelscope
```

`get_model.py` 默认将模型下载到当前目录，并将 HMM 资源解压到 `output/xh2/`。原始模型下载仅用于查看或后续开发，当前目录没有与之配套的量化和编译脚本。

获取预编译模型时，脚本还会自动获取 `hmm_xh2_funaudiochat-vad-token2wav.zip`，并将其中的 VAD、token2wav 辅助模型解压到 `output/xh2/`。如果只使用 `s2t` 模式，可以不加载 token2wav 相关模型；`s2s` 和 `e2e` 模式需要对应的辅助模型资源。

### 2.3 量化

当前目录暂不提供 `ptq.py`，暂不考虑量化，因此本章节没有可执行的量化命令。

### 2.4 编译

当前目录暂不提供 `build.py`，暂不考虑编译。演示默认直接加载 `output/xh2/` 下已经准备好的 HMM 文件。

### 2.5 演示

`demo.py` 提供三种推理模式：

- `s2t`: Speech-to-Text，仅输出文本。
- `s2s`: Speech-to-Speech，输出文本和语音，并保存一个 WAV 文件。
- `e2e`: 先使用 VAD 切分输入音频，再逐段进行语音对话，保存每一轮的输入音频、响应音频和 JSON 结果。

输入音频需要是单声道音频。脚本会将输入重采样到 16 kHz；`s2s` 和 `e2e` 的单段输入长度不能超过默认的 `126799` 个采样点，约 7.9 秒。生成语音采样率为 24 kHz。

使用默认输入音频运行文本识别：

```bash
python3 demo.py --stage s2t
```

指定输入音频运行语音对话：

```bash
python3 demo.py --stage s2s --audio_path /path/to/question.wav --output_wav result/response.wav
```

运行带 VAD 分段的端到端语音对话：

```bash
python3 demo.py --stage e2e --audio_path /path/to/questions.wav --e2e_output_dir result
```

常用推理参数如下：

- `--stage`: 推理模式，可选 `s2t`、`s2s`、`e2e`，默认 `s2t`。
- `--audio_path`: 输入音频路径；未指定时，`s2t`/`s2s` 默认使用 `$HOUMO_EXAMPLES_PATH/data/audio/question.wav`，`e2e` 默认使用 `$HOUMO_EXAMPLES_PATH/data/audio/questions.wav`。
- `--max_new_tokens`: 最大生成文本 token 数，默认 `2048`。
- `--temperature`: 语音采样温度，默认 `0.6`。
- `--top_k`: 语音采样的 top-k，默认 `20`。
- `--top_p`: 语音采样的 top-p，默认 `0.95`。
- `--repetition_penalty`: 语音重复惩罚系数，默认 `1.2`。
- `--device`: 起始 NPU 设备编号，默认 `0`。
- `--ndevice`: 使用的 NPU 数量，默认 `1`。
- `--perf`: 是否输出性能统计，默认启用。
- `--output_wav`: `s2s` 模式的响应 WAV 路径，默认 `result/turn_000_response.wav`。
- `--e2e_output_dir`: `e2e` 模式的输出目录，默认 `result/`。

## 3 一键评估

当前目录的 `test.sh` 暂时只支持 demo。默认会准备 Python 虚拟环境、下载预编译模型，并依次运行 `s2t`、`s2s` 和 `e2e` 演示：

```bash
bash test.sh
```

常用参数如下：

```bash
Usage: test.sh [options]
  -s, --step              运行步骤，当前仅支持 demo，默认 demo。
  -name, --model_name     模型名称，默认 funaudiochat。
  -size, --model_size     模型规格，默认 8b。
  --ndevice               NPU 数量，默认 1。
  --demo_mode             演示模式，可选 s2t、s2s、e2e；未指定时依次运行三种模式。
  --system_prompt         自定义 system prompt。
  --skip_download         跳过模型下载，使用当前目录已有资源。
  -h, --help              显示帮助信息。
```

示例：

```bash
# 依次运行 s2t、s2s 和 e2e 演示
bash test.sh

# 仅运行 s2t 演示
bash test.sh --demo_mode s2t

# 运行 s2s 演示
bash test.sh --demo_mode s2s

# 运行带 VAD 分段的 e2e 演示
bash test.sh --demo_mode e2e

# 使用已有模型，跳过下载
bash test.sh --skip_download --demo_mode s2t
```

`quant`、`build` 和 `all` 步骤当前不可用；传入这些步骤时脚本会直接报错退出。

## 4 参考结果

### 4.1 演示结果

以下结果来自当前目录已有的 `result/results.json`。输入音频为 `/hmdd/imodelzoo/data/audio/questions.wav`，VAD 检测到 3 个语音片段：

```text
turn 0: 0 ms - 3100 ms
generate_text: 你好，很高兴为你服务。
speech_token_count: 55

turn 1: 3620 ms - 7120 ms
generate_text: 是的，放AudioChat是一种端到端的语音大模型，它能够直接处理音频输入并生成语音输出，无需将音频转换为文本再转换回语音。
speech_token_count: 290

turn 2: 7650 ms - 10100 ms
generate_text: Fan Audio Chat能够进行语音交互，提供实时的音频聊天服务，支持多种语言的语音识别和合成，确保沟通的流畅性和自然性。
speech_token_count: 260
```

`e2e` 模式的输出文件包括：

- `result/turn_000_input.wav`、`result/turn_001_input.wav`、`result/turn_002_input.wav`
- `result/turn_000_response.wav`、`result/turn_001_response.wav`、`result/turn_002_response.wav`
- `result/results.json`

### 4.2 性能结果

当前目录未保存可核对的完整性能统计结果，暂无可提供的参考性能数据，待后续实测补充。运行 `demo.py` 时默认启用 `--perf`，程序会输出初始化、音频编码、语言模型 prefill/decode、语音生成和 VAD 等阶段的性能统计。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
