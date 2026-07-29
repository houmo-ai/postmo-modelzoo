# Qwen3-TTS

本目录展示如何将 Qwen3-TTS 12Hz 系列模型量化、编译并部署到后摩芯片设备上。当前覆盖的模型规格见第 1 节。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型，当前支持情况如下：

| 模型 | `model_size` | 来源 | 推理入口 | 主要用途 | 支持模式 |
|------|--------------|------|----------|----------|----------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | `0.6b-customvoice` | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | `demo.py` | 预置 speaker 语音合成 | `oneshot` / `streaming` |
| `Qwen3-TTS-12Hz-0.6B-Base` | `0.6b-base` | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `demo_base.py` | 基于参考音频和文本的 voice clone | `oneshot` / `streaming` |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `1.7b-customvoice` | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `demo.py` | 预置 speaker 语音合成 | `oneshot` / `streaming` |

- 许可: Apache License 2.0 (https://huggingface.co/Qwen/Qwen3-TTS/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-tts`
- `model_size=0.6b-customvoice`
- `context_length=2k`
- `prefill_length=256`
- `ncore=2`
- `ndevice=1`
- `quant_type=w8a16h1_sefp`

### 资源要求

**后摩设备运行要求：**

- `qwen3-tts-0.6b-customvoice`: 单卡 12 GB可运行
- `qwen3-tts-0.6b-base`: 单卡 12 GB可运行
- `qwen3-tts-1.7b-customvoice`: 单卡 12 GB可运行

**量化硬件要求：**

- `qwen3-tts-0.6b-customvoice`: 需要 GPU，显存最小 24 GB
- `qwen3-tts-0.6b-base`: 需要 GPU，显存最小 24 GB
- `qwen3-tts-1.7b-customvoice`: 需要 GPU，显存最小 26 GB

**编译时 host 内存要求：**

- `qwen3-tts-0.6b-customvoice`: 需要空闲 host 内存最小 18 GB
- `qwen3-tts-0.6b-base`: 需要空闲 host 内存最小 18 GB
- `qwen3-tts-1.7b-customvoice`: 需要空闲 host 内存最小 18 GB

## 2 快速开始

目前 Qwen3-TTS 模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 脚本职责

当前目录下主要脚本职责如下：

| 脚本 | 作用 |
|------|------|
| `get_model.py` | 根据 `config.yaml` 下载原始模型或预编译芯片模型 |
| `ptq.py` | 将原始模型导出并量化为 HMONNX / 量化辅助文件 |
| `build.py` | 将 HMONNX 编译为后摩芯片可运行的 HMM |
| `demo.py` | 运行 `0.6b-customvoice` 或 `1.7b-customvoice` 的 oneshot / streaming 推理 |
| `demo_base.py` | 运行 `0.6b-base` 的 oneshot / streaming voice clone 推理 |
| `qwen3_tts_runtime.py` | 两个 demo 共用的 HMM wrapper、路径推导、chunk buffer、流式 gap 统计等运行时工具 |
| `perf_utils.py` | 两个 demo 共用的性能统计 key、计时器和表格日志 |
| `test.sh` | 一键执行下载、量化、编译和 demo 步骤 |

### 2.1 环境准备

所有脚本当前只支持 `HOUMO_TARGET=xh2`，运行前需要先 source 后摩工具链环境，确保 `HOUMO_TARGET`、编译工具、运行时库等环境变量已生效：

```bash
source env.sh
```

安装示例运行所需的 Python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

如果需要执行量化，还需要安装 PTQ / ONNX 导出相关依赖：

```bash
pip3 install -r requirements_ptq.txt
```

部分音频处理依赖需要系统 sox 库。`test.sh` 在 demo 步骤中会自动执行安装；手动运行 demo 时可提前安装：

```bash
sudo apt update
sudo apt install sox libsox-dev -y
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型。`--model_size` 的支持值见第 1 节模型表。需要自行量化时下载原始模型；如果希望跳过量化和编译、直接运行演示，则下载预编译芯片模型：

```bash
# 下载原始模型
python3 get_model.py --type raw --model_name qwen3-tts --model_size <model_size>

# 下载预编译芯片模型
python3 get_model.py --type hmm --model_name qwen3-tts --model_size <model_size>
```

常用参数：

- `--type`：下载类型，`raw` 为原始 ModelScope/HF 模型，`hmm` 为预编译芯片模型；
- `--download_dir`：下载文件保存目录，默认当前目录；
- `--extract_dir`：解压目录，默认由下载工具决定；
- `--source_type`：下载源，可选 `jfrog` 或 `modelscope`，默认 `jfrog`；
- `--config`：配置文件路径，默认 `config.yaml`；
- `--model_name` / `--model_size`：模型名称和规格，未指定时使用 `config.yaml` 默认值；
- `--ndevice` / `--ncore` / `--context_length` / `--quant_type`：覆盖配置中用于匹配芯片模型包的参数。

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：
```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入模型目录，执行 `ptq.py` 脚本进行量化。`<model_size>` 的支持值见第 1 节模型表：
```bash
source env.sh
cd models/tts/qwen3-tts
python3 ptq.py --model_name qwen3-tts --model_size <model_size>
```

`ptq.py` 将依次导出以下子模型到 HMONNX 格式：
- talker（prefill + decode）
- code_predictor（prefill + decode）
- text_projection
- speech_tokenizer
- stateful_decoder
- speaker_encoder（仅 Base）
- speech_tokenizer_encoder（仅 Base）

可通过 `--models` 参数指定要量化的子模型（逗号分隔），例如：
```bash
python3 ptq.py --model_name qwen3-tts --model_size 0.6b-base --models talker,code_predictor
```

`ptq.py --models all` 是默认行为。对于 `0.6b-customvoice` 和
`1.7b-customvoice`，`all` 会自动跳过 `speaker_encoder` 和
`speech_tokenizer_encoder`；这两个子模型仅适用于 `0.6b-base`。如果显式指定
Base-only 子模型给 CustomVoice，会报错。

常用参数：

- `--output_dir`：量化产物输出目录，默认 `output/${HOUMO_TARGET}/hmquant`；
- `--work_dir`：中间工作目录前缀，脚本会自动追加模型 tag，例如 `work_dirs_qwen3-tts_0.6b-base`；
- `--debug`：保留中间工作目录，默认量化结束后会清理；
- `--dump_golden`：导出 golden 数据，便于验证；
- `--max_sequence_length`：Talker 最大上下文长度，默认 `2048`；
- `--input_sequence_length`：Talker prefill 长度，默认 `256`；
- `--code_predictor_max_sequence_length`：CodePredictor 最大长度，默认 `18`；
- `--code_predictor_input_sequence_length`：CodePredictor prefill 长度，默认 `2`；
- `--stateful_decoder_chunk_size`：StatefulDecoder 导出 chunk 大小，默认 `12`；
- `--frontend_audio_samples`：Base `speech_tokenizer_encoder` 静态输入采样点数，默认 `101760`；
- `--frontend_mel_frames` / `--frontend_mel_dim`：Base `speaker_encoder` 静态 mel 输入形状，默认 `400 x 128`。

量化完成的模型和辅助文件存放在 `output/${HOUMO_TARGET}/hmquant` 目录。demo 运行还会使用其中的 `quant_embedding.pt`、`quant_embedding_code_predictor.pt` 和 `decode_padding_shapes.json`。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。`<model_size>` 的支持值见第 1 节模型表：
```bash
python3 build.py --model_name qwen3-tts --model_size <model_size>
```

`0.6B CustomVoice` 默认输出名称包括：
- `qwen3-tts-0.6b-customvoice_text_projection.hmm`
- `qwen3-tts-0.6b-customvoice_code_predictor_prefill.hmm`
- `qwen3-tts-0.6b-customvoice_code_predictor_decode.hmm`
- `qwen3-tts-0.6b-customvoice_talker_prefill.hmm`
- `qwen3-tts-0.6b-customvoice_talker_decode.hmm`
- `qwen3-tts-0.6b-customvoice_speech_tokenizer.hmm`
- `qwen3-tts-0.6b-customvoice_stateful_decoder.hmm`

`1.7B CustomVoice` 的子模型组成相同，文件名前缀替换为 `qwen3-tts-1.7b-customvoice_`。

`Base` 默认输出名称包括：
- `qwen3-tts-0.6b-base_text_projection.hmm`
- `qwen3-tts-0.6b-base_code_predictor_prefill.hmm`
- `qwen3-tts-0.6b-base_code_predictor_decode.hmm`
- `qwen3-tts-0.6b-base_talker_prefill.hmm`
- `qwen3-tts-0.6b-base_talker_decode.hmm`
- `qwen3-tts-0.6b-base_speech_tokenizer.hmm`
- `qwen3-tts-0.6b-base_stateful_decoder.hmm`
- `qwen3-tts-0.6b-base_speaker_encoder.hmm`
- `qwen3-tts-0.6b-base_speech_tokenizer_encoder.hmm`

可通过 `--models` 参数指定要编译的子模型，例如：
```bash
python3 build.py --model_name qwen3-tts --model_size 0.6b-base --models talker_prefill talker_decode
```

常用参数：

- `--model_dir`：HMONNX / hmquant 输入目录，默认 `output/${HOUMO_TARGET}/hmquant`；
- `--output_dir`：HMM 输出目录，默认 `output/${HOUMO_TARGET}`；
- `--stage`：编译阶段，默认 `build`；
- `--context_length` / `--prefill_length` / `--ncore` / `--ndevice` / `--batch`：覆盖 `config.yaml` 中的编译配置；
- `--flash_attention`：两个整数，分别控制 LLM prefill/decode 和 non-LLM 子模型的 FlashAttention 开关，默认 `2 1`。

### 2.5 演示

Qwen3-TTS 模型使用 Python API 进行演示。`CustomVoice` 和 `Base` 两类模型均支持两种音频输出路径：

- `oneshot`：先完整生成 codec frames，再通过 `SpeechTokenizer` 一次性解码为 wav；
- `streaming`：边生成 codec frames 边通过 `StatefulDecoder` 按 chunk 输出音频。

两类模型的入口不同：

- `demo.py`：用于 `0.6b-customvoice` 和 `1.7b-customvoice`，输入为 `text + speaker + language`；
- `demo_base.py`：用于 `0.6b-base`，输入为 `text + language + ref_audio + ref_text`，用于 voice clone。

#### 2.5.1 CustomVoice 演示

`demo.py` 支持 `0.6b-customvoice` 和 `1.7b-customvoice`。脚本默认根据 `--model_name` / `--model_size` 从 `output/${HOUMO_TARGET}` 推导 HMM 路径，具体编译产物见 2.4。oneshot 模式需要 Talker、CodePredictor、TextProjection 和 SpeechTokenizer；streaming 模式会额外使用 `stateful_decoder_hmm`。

非流式执行示例：
```bash
python3 demo.py \
  --model_name qwen3-tts \
  --model_size 0.6b-customvoice \
  --mode oneshot \
  --output_wav output_custom_voice.wav
```

流式执行示例：
```bash
python3 demo.py \
  --model_name qwen3-tts \
  --model_size 0.6b-customvoice \
  --mode streaming \
  --output_wav output_custom_voice_streaming.wav
```

常用参数：

- `--text`：待合成的文本
- `--speaker`：说话人（`vivian`/`serena`/`uncle_fu`/`ryan`/`aiden`/`ono_anna`/`sohee`/`eric`/`dylan`）
- `--language`：语言（`Chinese`/`English`/`Japanese`/`Korean`/`French`/`German`/`Spanish`/`Italian`/`Portuguese`/`Russian`/`auto`）
- `--output_wav`：输出音频文件路径
- `--mode`：推理模式，`oneshot` 为非流式模式，`streaming` 为流式模式
- `--chunk_size`：流式模式下每个解码块的 codec 帧数，默认 `12`
- `--stateful_decoder_hmm`：流式模式使用的 Stateful Decoder HMM 路径，默认根据 `--model_name` / `--model_size` 推导
- `--ndevice`：使用的设备数量，默认 `1`
- `--seed`：随机种子，默认 `1024`
- `--no_progress`：关闭 tqdm 生成进度条，便于采集性能日志

#### 2.5.2 Base 演示

`Base` 模型用于 voice clone，使用 `demo_base.py` 运行。示例参考音频为 `${HOUMO_EXAMPLES_PATH}/data/audio/clone_1.wav`，参考文本为 `甚至出现交易几乎停滞的情况。`，可以通过 `--ref_audio` 和 `--ref_text` 显式指定。

`demo_base.py` 默认根据 `--model_name` / `--model_size` 从 `output/${HOUMO_TARGET}` 推导 HMM 路径，具体编译产物见 2.4。Base voice clone 除 Talker、CodePredictor、TextProjection 和音频解码器外，还需要 `speech_tokenizer_encoder_hmm` 和 `speaker_encoder_hmm`。

非流式执行示例：
```bash
python3 demo_base.py \
  --model_name qwen3-tts \
  --model_size 0.6b-base \
  --mode oneshot \
  --output_wav output_voice_clone.wav
```

流式执行示例：
```bash
python3 demo_base.py \
  --model_name qwen3-tts \
  --model_size 0.6b-base \
  --mode streaming \
  --output_wav output_voice_clone_streaming.wav
```

常用参数：

- `--text`：待合成的文本
- `--language`：语言（`Chinese`/`English`/`Japanese`/`Korean`/`French`/`German`/`Spanish`/`Italian`/`Portuguese`/`Russian`/`auto`）
- `--ref_audio`：参考音频文件路径
- `--ref_text`：参考音频对应的文本内容
- `--output_wav`：输出音频文件路径
- `--mode`：推理模式，`oneshot` 为非流式模式，`streaming` 为流式模式
- `--chunk_size`：流式模式下每个解码块的 codec 帧数，默认 `12`
- `--stateful_decoder_hmm`：流式模式使用的 Stateful Decoder HMM 路径，默认根据 `--model_name` / `--model_size` 推导
- `--ndevice`：使用的设备数量，默认 `1`
- `--seed`：随机种子，默认 `1024`
- `--no_progress`：关闭 tqdm 生成进度条，便于采集性能日志

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。当前脚本默认模型为 `qwen3-tts / 0.6b-customvoice`，默认执行 demo 步骤：
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
                          Supports comma-separated values or repeated flags.
  -name, --model_name     Model name. Default: qwen3-tts.
  -size, --model_size     Model size. Default: 0.6b-customvoice. Choices: 0.6b-customvoice, 0.6b-base, 1.7b-customvoice.
  --demo_mode             Demo mode. Default: oneshot. Choices: oneshot, streaming.
  -h, --help              Show this help message.
```

示例：
```bash
# 仅运行量化，会先下载原始模型再量化
bash test.sh -s quant

# 量化 + 编译
bash test.sh -s quant,build

# 运行 0.6B CustomVoice 流式 demo
bash test.sh -s demo --model_size 0.6b-customvoice --demo_mode streaming

# 运行 1.7B CustomVoice 非流式 demo
bash test.sh -s demo --model_size 1.7b-customvoice --demo_mode oneshot

# 运行 0.6B Base 非流式 voice clone demo
bash test.sh -s demo --model_size 0.6b-base --demo_mode oneshot

# 运行 0.6B Base 流式 voice clone demo
bash test.sh -s demo --model_size 0.6b-base --demo_mode streaming
```

## 4 参考结果

### 4.1 0.6B CustomVoice 演示结果

#### Oneshot 模式

```bash
Running Qwen3-TTS oneshot demo: model_name=qwen3-tts, model_size=0.6b-customvoice, hf_model_dir=./Qwen3-TTS-12Hz-0.6B-CustomVoice
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
Qwen3TTSCodecGenerator initialized successfully
SpeechTokenizer loaded: input_seq_len=300
Starting TTS generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
Using non-streaming mode for talker input embedding
Generating tokens:   7%|████████                                              | 136/2008 [00:07<01:48, 17.22it/s]Reached EOS token at step 137
Audio saved to output_custom_voice.wav | duration: 10.96s | inference: 9.29s | RTF: 0.8476
+----------------------------------------------------------------------------------------------------------+
|                                          Performance Breakdown:                                          |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
| Component          | Time(s) |    Pct | Count | Avg(ms) | Description                                    |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
|   embedding_prep   |    0.02 |   0.2% |       |         | Prepare model input embeddings                 |
|   talker           |    1.65 |  17.7% |       |         | All Talker model computation                   |
|     prefill        |    0.08 |   0.8% |     1 |   76.44 | Run the initial Talker forward pass            |
|     decode         |    1.37 |  14.8% |   137 |   10.01 | Generate codec-group-0 tokens autoregressively |
|     sampling       |    0.20 |   2.1% |   138 |    1.44 | Sample tokens from Talker logits               |
|   frame_prepare    |    0.02 |   0.2% |       |         | Prepare per-frame generation inputs            |
|   code_predictor   |    6.18 |  66.5% |       |         | All Code Predictor computation                 |
|     prepare        |    0.12 |   1.3% |       |         | Prepare Code Predictor inputs                  |
|     prefill        |    0.43 |   4.6% |   137 |    3.15 | Run Code Predictor prefill                     |
|     decode         |    4.54 |  48.9% |  1918 |    2.37 | Generate remaining codec groups                |
|     sampling       |    1.08 |  11.7% |  2055 |    0.53 | Sample tokens from Code Predictor logits       |
|   postprocess      |    0.00 |   0.0% |       |         | Assemble generated codec frames                |
|   speech_tokenizer |    1.31 |  14.1% |     1 | 1309.39 | Decode codec frames into a waveform            |
|   other            |    0.12 |   1.3% |       |         | Time outside individually tracked metrics      |
|   total            |    9.29 | 100.0% |       |         | End-to-end inference time                      |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
```

#### Streaming 模式

```bash
2026-07-13 02:23:08.450 | INFO     | __main__:main:962 - Running Qwen3-TTS streaming demo: model_name=qwen3-tts, model_size=0.6b-customvoice, hf_model_dir=./Qwen3-TTS-12Hz-0.6B-CustomVoice
W20260713 02:23:10.478791 1166112 __init__.py:108] Skipping import of cpp extensions due to incompatible torch version 2.8.0+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Device 0 -> Group 0, Chip 3
2026-07-13 02:23:13.673 | INFO     | qwen3_tts_runtime:__init__:149 - TextProjection loaded
Device 0 -> Group 0, Chip 3
2026-07-13 02:23:15.541 | INFO     | qwen3_tts_runtime:__init__:221 - CodePredictor loaded: prefill_length=2
Device 0 -> Group 0, Chip 3
2026-07-13 02:23:24.243 | INFO     | qwen3_tts_runtime:__init__:341 - Talker loaded, context_length=2048
2026-07-13 02:23:24.246 | INFO     | __main__:__init__:136 - Qwen3TTSCodecGenerator initialized successfully
Device 0 -> Group 0, Chip 3
2026-07-13 02:23:24.714 | INFO     | qwen3_tts_runtime:__init__:611 - StatefulDecoder loaded: chunk_size=12
2026-07-13 02:23:24.724 | INFO     | __main__:generate_streaming:876 - Starting streaming TTS for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
2026-07-13 02:23:24.735 | INFO     | __main__:_generate_codec_frames:622 - Using streaming mode for talker input embedding
2026-07-13 02:23:25.543 | INFO     | __main__:main:1028 - First audio chunk latency: 815.3ms
2026-07-13 02:23:25.543 | INFO     | __main__:main:1042 -   Chunk 1: 15360 samples (640ms audio)
2026-07-13 02:23:26.287 | INFO     | __main__:main:1047 -   Chunk 2: 23040 samples (960ms audio) | playback_gap: 104.0ms
2026-07-13 02:23:27.037 | INFO     | __main__:main:1047 -   Chunk 3: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:27.795 | INFO     | __main__:main:1047 -   Chunk 4: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:28.562 | INFO     | __main__:main:1047 -   Chunk 5: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:29.326 | INFO     | __main__:main:1047 -   Chunk 6: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:30.072 | INFO     | __main__:main:1047 -   Chunk 7: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:30.817 | INFO     | __main__:main:1047 -   Chunk 8: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:31.551 | INFO     | __main__:main:1047 -   Chunk 9: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:32.299 | INFO     | __main__:main:1047 -   Chunk 10: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:33.023 | INFO     | __main__:main:1047 -   Chunk 11: 23040 samples (960ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:33.153 | INFO     | __main__:_generate_codec_frames:763 - Reached EOS token at step 134
2026-07-13 02:23:33.210 | INFO     | __main__:main:1047 -   Chunk 12: 11520 samples (480ms audio) | playback_gap: 0.0ms
2026-07-13 02:23:33.255 | INFO     | __main__:main:1063 - Audio saved to output_custom_voice.wav | duration: 10.72s | inference: 8.49s | RTF: 0.7915 | chunks: 12 | first_chunk_latency: 815.3ms | playback_gap_chunks: 1 | max_playback_gap: 104.0ms | total_playback_gap: 104.0ms
2026-07-13 02:23:33.271 | INFO     | perf_utils:_log_footer:221 -
+----------------------------------------------------------------------------------------------------------+
|                                     Streaming Performance Breakdown:                                     |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
| Component          | Time(s) |    Pct | Count | Avg(ms) | Description                                    |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
|   embedding_prep   |    0.03 |   0.3% |       |         | Prepare model input embeddings                 |
|   talker           |    1.60 |  18.9% |       |         | All Talker model computation                   |
|     prefill        |    0.07 |   0.8% |     1 |   71.84 | Run the initial Talker forward pass            |
|     decode         |    1.33 |  15.7% |   134 |    9.92 | Generate codec-group-0 tokens autoregressively |
|     sampling       |    0.20 |   2.4% |   135 |    1.50 | Sample tokens from Talker logits               |
|   frame_prepare    |    0.02 |   0.2% |       |         | Prepare per-frame generation inputs            |
|   code_predictor   |    6.05 |  71.3% |       |         | All Code Predictor computation                 |
|     prepare        |    0.11 |   1.3% |       |         | Prepare Code Predictor inputs                  |
|     prefill        |    0.42 |   5.0% |   134 |    3.14 | Run Code Predictor prefill                     |
|     decode         |    4.43 |  52.3% |  1876 |    2.36 | Generate remaining codec groups                |
|     sampling       |    1.08 |  12.8% |  2010 |    0.54 | Sample tokens from Code Predictor logits       |
|   stateful_decoder |    0.68 |   8.0% |    12 |   56.34 | Decode generated codec chunks into waveforms   |
|   other            |    0.11 |   1.3% |       |         | Time outside individually tracked metrics      |
|   total            |    8.49 | 100.0% |       |         | End-to-end inference time                      |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
```

### 4.2 0.6B Base 演示结果

> **注：**Base streaming 的 `first_chunk_latency` 从参考音频处理和 StatefulDecoder 预热完成后开始计时，不是完整的端到端首音延时，也不能与 CustomVoice 的数值直接比较。详细说明见 4.4。

#### Oneshot 模式

```bash
Running Qwen3-TTS oneshot demo: model_name=qwen3-tts, model_size=0.6b-base, hf_model_dir=./Qwen3-TTS-12Hz-0.6B-Base
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
SpeechTokenizer loaded: input_seq_len=300
SpeechTokenizerEncoder loaded
SpeakerEncoder loaded: input_names=['mels'], output_names=['speaker_embedding'], input_frames=400, num_mels=128, input_dtype=<class 'numpy.float16'>
Qwen3TTSInference initialized successfully
Starting voice clone generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地
Reference audio frontend input: path=/data/wanyu.li/imodelzoo/data/audio/clone_1.wav, samples=100895, sr=24000, duration=4.2040s, peak=0.052004, tail_0p5s_rms=0.001986
Using non-streaming mode for talker input embedding
Generating tokens:   9%|███████                                         | 182/1947 [00:10<01:42, 17.29it/s]2026-07-13 Reached EOS token at step 183
Audio saved to output_voice_clone.wav | duration: 14.64s | inference: 11.88s | RTF: 0.8111
+----------------------------------------------------------------------------------------------------------------------+
|                                                Performance Breakdown:                                                |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
| Component                     | Time(s) |    Pct | Count | Avg(ms) | Description                                     |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
|   preparation                 |    0.08 |   0.6% |       |         | All input and prompt preparation                |
|     logits_processor          |    0.00 |   0.0% |       |         | Initialize sampling and logits processors       |
|     reference_audio           |    0.05 |   0.4% |       |         | All reference audio processing                  |
|       audio_load              |    0.02 |   0.2% |     1 |   18.52 | Load and preprocess reference audio             |
|       speech_tokenizer_encode |    0.02 |   0.2% |     1 |   23.01 | Encode reference audio into codec frames        |
|       speaker_embed           |    0.01 |   0.1% |     1 |    9.72 | Extract the reference speaker embedding         |
|     prompt_prepare            |    0.02 |   0.2% |     7 |    3.40 | All text and prompt embedding preparation       |
|   talker                      |    2.14 |  18.0% |       |         | All Talker model computation                    |
|     prefill                   |    0.07 |   0.6% |     1 |   72.68 | Run the initial Talker forward pass             |
|     decode                    |    1.83 |  15.4% |   183 |   10.00 | Generate codec-group-0 tokens autoregressively  |
|     sampling                  |    0.24 |   2.0% |   184 |    1.28 | Sample tokens from Talker logits                |
|   code_predictor              |    7.89 |  66.4% |       |         | All Code Predictor computation                  |
|     prefill                   |    0.58 |   4.9% |   183 |    3.16 | Run Code Predictor prefill                      |
|     decode                    |    6.03 |  50.8% |  2562 |    2.35 | Generate remaining codec groups                 |
|     sampling                  |    1.28 |  10.8% |  2745 |    0.47 | Sample tokens from Code Predictor logits        |
|   audio_decoder               |    1.31 |  11.0% |       |         | All waveform decoding                           |
|     speech_tokenizer          |    1.31 |  11.0% |     1 | 1309.68 | Decode codec frames into a waveform             |
|     stateful_ref_prime        |    0.00 |   0.0% |     0 |    0.00 | Prime decoder state with reference codec frames |
|     stateful_generated        |    0.00 |   0.0% |     0 |    0.00 | Decode generated codec chunks into waveforms    |
|   other                       |    0.46 |   3.9% |       |         | Time outside individually tracked metrics       |
|   total                       |   11.88 | 100.0% |       |         | End-to-end inference time                       |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
```

#### Streaming 模式

```bash
Running Qwen3-TTS streaming demo: model_name=qwen3-tts, model_size=0.6b-base, hf_model_dir=./Qwen3-TTS-12Hz-0.6B-Base
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
SpeechTokenizer loaded: input_seq_len=300
SpeechTokenizerEncoder loaded
SpeakerEncoder loaded: input_names=['mels'], output_names=['speaker_embedding'], input_frames=400, num_mels=128, input_dtype=<class 'numpy.float16'>
Qwen3TTSInference initialized successfully
StatefulDecoder loaded: chunk_size=12
Starting streaming voice clone generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地
Reference audio frontend input: path=/data/wanyu.li/imodelzoo/data/audio/clone_1.wav, samples=100895, sr=24000, duration=4.2040s, peak=0.052004, tail_0p5s_rms=0.001986
Using streaming mode for talker input embedding
First audio chunk latency: 812.3ms
  Chunk 1: 15360 samples (640ms audio)
  Chunk 2: 23040 samples (960ms audio) | playback_gap: 91.3ms
  Chunk 3: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 4: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 5: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 6: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 7: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 8: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 9: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 10: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 11: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 12: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 13: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 14: 23040 samples (960ms audio) | playback_gap: 0.0ms
Reached EOS token at step 177
  Chunk 15: 24960 samples (1040ms audio) | playback_gap: 0.0ms
Audio saved to output_voice_clone.wav | duration: 14.16s | inference: 11.36s | RTF: 0.8020 | chunks: 15 | first_chunk_latency: 812.3ms | playback_gap_chunks: 1 | max_playback_gap: 91.3ms | total_playback_gap: 91.3ms
+----------------------------------------------------------------------------------------------------------------------+
|                                           Streaming Performance Breakdown:                                           |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
| Component                     | Time(s) |    Pct | Count | Avg(ms) | Description                                     |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
|   preparation                 |    0.07 |   0.6% |       |         | All input and prompt preparation                |
|     logits_processor          |    0.00 |   0.0% |       |         | Initialize sampling and logits processors       |
|     reference_audio           |    0.04 |   0.4% |       |         | All reference audio processing                  |
|       audio_load              |    0.01 |   0.1% |     1 |   12.32 | Load and preprocess reference audio             |
|       speech_tokenizer_encode |    0.02 |   0.2% |     1 |   22.97 | Encode reference audio into codec frames        |
|       speaker_embed           |    0.01 |   0.1% |     1 |    9.35 | Extract the reference speaker embedding         |
|     prompt_prepare            |    0.03 |   0.2% |     7 |    3.68 | All text and prompt embedding preparation       |
|   talker                      |    2.09 |  18.4% |       |         | All Talker model computation                    |
|     prefill                   |    0.07 |   0.7% |     1 |   74.93 | Run the initial Talker forward pass             |
|     decode                    |    1.75 |  15.4% |   177 |    9.91 | Generate codec-group-0 tokens autoregressively  |
|     sampling                  |    0.26 |   2.3% |   178 |    1.45 | Sample tokens from Talker logits                |
|   code_predictor              |    7.63 |  67.2% |       |         | All Code Predictor computation                  |
|     prefill                   |    0.56 |   4.9% |   177 |    3.16 | Run Code Predictor prefill                      |
|     decode                    |    5.82 |  51.2% |  2478 |    2.35 | Generate remaining codec groups                 |
|     sampling                  |    1.25 |  11.0% |  2655 |    0.47 | Sample tokens from Code Predictor logits        |
|   audio_decoder               |    1.13 |  10.0% |       |         | All waveform decoding                           |
|     speech_tokenizer          |    0.00 |   0.0% |     0 |    0.00 | Decode codec frames into a waveform             |
|     stateful_ref_prime        |    0.28 |   2.5% |     5 |   56.39 | Prime decoder state with reference codec frames |
|     stateful_generated        |    0.85 |   7.5% |    15 |   56.54 | Decode generated codec chunks into waveforms    |
|   other                       |    0.44 |   3.9% |       |         | Time outside individually tracked metrics       |
|   total                       |   11.36 | 100.0% |       |         | End-to-end inference time                       |
+-------------------------------+---------+--------+-------+---------+-------------------------------------------------+
```

### 4.3 1.7B CustomVoice 演示结果

#### Oneshot 模式

```bash
Running Qwen3-TTS oneshot demo: model_name=qwen3-tts, model_size=1.7b-customvoice, hf_model_dir=./Qwen3-TTS-12Hz-1.7B-CustomVoice
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
Qwen3TTSCodecGenerator initialized successfully
SpeechTokenizer loaded: input_seq_len=300
Starting TTS generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
Using non-streaming mode for talker input embedding
Generating tokens:   7%|███████                                                       | 146/2008 [00:09<02:00, 15.44it/s]
Reached EOS token at step 146
Audio saved to output_custom_voice.wav | duration: 11.68s | inference: 10.98s | RTF: 0.9398
+----------------------------------------------------------------------------------------------------------+
|                                          Performance Breakdown:                                          |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
| Component          | Time(s) |    Pct | Count | Avg(ms) | Description                                    |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
|   embedding_prep   |    0.02 |   0.2% |       |         | Prepare model input embeddings                 |
|   talker           |    2.88 |  26.2% |       |         | All Talker model computation                   |
|     prefill        |    0.09 |   0.8% |     1 |   85.49 | Run the initial Talker forward pass            |
|     decode         |    2.56 |  23.3% |   146 |   17.55 | Generate codec-group-0 tokens autoregressively |
|     sampling       |    0.23 |   2.1% |   147 |    1.55 | Sample tokens from Talker logits               |
|   frame_prepare    |    0.02 |   0.2% |       |         | Prepare per-frame generation inputs            |
|   code_predictor   |    6.63 |  60.4% |       |         | All Code Predictor computation                 |
|     prepare        |    0.14 |   1.3% |       |         | Prepare Code Predictor inputs                  |
|     prefill        |    0.47 |   4.3% |   146 |    3.20 | Run Code Predictor prefill                     |
|     decode         |    4.87 |  44.4% |  2044 |    2.38 | Generate remaining codec groups                |
|     sampling       |    1.15 |  10.5% |  2190 |    0.53 | Sample tokens from Code Predictor logits       |
|   postprocess      |    0.00 |   0.0% |       |         | Assemble generated codec frames                |
|   speech_tokenizer |    1.30 |  11.9% |     1 | 1304.85 | Decode codec frames into a waveform            |
|   other            |    0.13 |   1.1% |       |         | Time outside individually tracked metrics      |
|   total            |   10.98 | 100.0% |       |         | End-to-end inference time                      |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
```

#### Streaming 模式

```bash
Running Qwen3-TTS streaming demo: model_name=qwen3-tts, model_size=1.7b-customvoice, hf_model_dir=./Qwen3-TTS-12Hz-1.7B-CustomVoice
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
Qwen3TTSCodecGenerator initialized successfully
StatefulDecoder loaded: chunk_size=12
Starting streaming TTS for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
Using streaming mode for talker input embedding
First audio chunk latency: 951.3ms
  Chunk 1: 15360 samples (640ms audio)
  Chunk 2: 23040 samples (960ms audio) | playback_gap: 193.1ms
  Chunk 3: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 4: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 5: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 6: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 7: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 8: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 9: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 10: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 11: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 12: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 13: 23040 samples (960ms audio) | playback_gap: 0.0ms
Reached EOS token at step 157
  Chunk 14: 9600 samples (400ms audio) | playback_gap: 0.0ms
Audio saved to output_custom_voice.wav | duration: 12.56s | inference: 11.15s | RTF: 0.8879 | chunks: 14 | first_chunk_latency: 951.3ms | playback_gap_chunks: 1 | max_playback_gap: 193.1ms | total_playback_gap: 193.1ms
+----------------------------------------------------------------------------------------------------------+
|                                     Streaming Performance Breakdown:                                     |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
| Component          | Time(s) |    Pct | Count | Avg(ms) | Description                                    |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
|   embedding_prep   |    0.02 |   0.2% |       |         | Prepare model input embeddings                 |
|   talker           |    3.07 |  27.5% |       |         | All Talker model computation                   |
|     prefill        |    0.09 |   0.8% |     1 |   89.08 | Run the initial Talker forward pass            |
|     decode         |    2.75 |  24.7% |   157 |   17.51 | Generate codec-group-0 tokens autoregressively |
|     sampling       |    0.23 |   2.1% |   158 |    1.47 | Sample tokens from Talker logits               |
|   frame_prepare    |    0.03 |   0.2% |       |         | Prepare per-frame generation inputs            |
|   code_predictor   |    7.11 |  63.8% |       |         | All Code Predictor computation                 |
|     prepare        |    0.13 |   1.2% |       |         | Prepare Code Predictor inputs                  |
|     prefill        |    0.50 |   4.5% |   157 |    3.19 | Run Code Predictor prefill                     |
|     decode         |    5.23 |  46.9% |  2198 |    2.38 | Generate remaining codec groups                |
|     sampling       |    1.25 |  11.2% |  2355 |    0.53 | Sample tokens from Code Predictor logits       |
|   stateful_decoder |    0.79 |   7.1% |    14 |   56.68 | Decode generated codec chunks into waveforms   |
|   other            |    0.13 |   1.1% |       |         | Time outside individually tracked metrics      |
|   total            |   11.15 | 100.0% |       |         | End-to-end inference time                      |
+--------------------+---------+--------+-------+---------+------------------------------------------------+
```

### 4.4 性能指标说明

#### Streaming 首音延时统计说明

- CustomVoice `demo.py`：`first_chunk_latency` 从 `stateful_decoder.create_state()` 完成后开始计时，到第一次 `yield` 非空 audio chunk 结束。该指标包含 CustomVoice streaming 模式下的 embedding 准备、Talker prefill、若干帧 Talker / CodePredictor 自回归生成、首个 StatefulDecoder chunk 解码，以及中间 Python 调度开销。

- Base `demo_base.py`：先完成 logits processor 构造、参考音频处理、Base ICL prompt 构造、`stateful_decoder.create_state()`，以及使用参考音频 `ref_code` 预热 StatefulDecoder state；之后才开始 `first_chunk_latency` 计时，直到第一次 `yield` 非空目标 audio chunk。因此该指标不包含 `prep_logits_processor`、`reference_audio`、`prompt_prepare` 和 `stateful_ref_prime`；这些耗时会计入总 `inference` 并在性能表中单独展示。

因此，Base streaming 的 `first_chunk_latency` 表示“参考音频和 decoder 预热完成后，到首个目标音频 chunk 输出”的延迟。比较端到端体验时，应同时查看 `inference`、`preparation.reference_audio`、`prompt_prepare`、`audio_decoder.stateful_ref_prime` 和 `first_chunk_latency`。

#### 当前结果解读

- 三组演示的 oneshot 和 streaming `RTF` 均小于 1，表明当前配置下整体生成速度快于实时播放。本次单次结果中，streaming RTF 为 `0.7915`（0.6B CustomVoice）、`0.8020`（0.6B Base）和 `0.8879`（1.7B CustomVoice）。
- 三组 streaming 结果都只在第 2 个 chunk 出现 1 次正向播放空挡，分别为 `104.0ms`、`91.3ms` 和 `193.1ms`；后续 chunk 的 `playback_gap` 均为 `0.0ms`。这说明首包后的稳态产出基本可以跟上播放，但从首个较短 chunk 切换到后续 chunk 时仍可能出现一次短暂空挡。
- `CodePredictor` 是当前主要耗时阶段，占总推理时间约 `60.4%`–`71.3%`；其中逐 codebook 的 `decode` 占比最高。`1.7B CustomVoice` 的 Talker decode 平均约 `17.5ms`，高于 0.6B 模型的约 `10.0ms`，是其首音延时和 RTF 更高的主要原因之一。
- 不同运行生成的 codec 帧数、音频时长和 chunk 数并不完全相同，因此不宜只比较 `inference` 总时间。评估子模型性能时，建议结合 `Count`、`Avg(ms)` 和 `Pct` 查看；比较用户体验时，建议同时查看 `RTF`、`first_chunk_latency` 和播放空挡指标。

#### 指标含义

- `duration`：生成音频的实际时长，单位为秒。
- `inference`：单次请求的端到端生成耗时，单位为秒。包含文本/参考音频准备、Talker、CodePredictor、音频解码器以及 Python 调度等开销，不包含 HMM 加载与初始化，也不包含最后统一调用 `sf.write()` 写入 wav 文件的耗时。
- `RTF`：Real-Time Factor，计算方式为 `inference / duration`。RTF 小于 1 表示推理速度快于实时播放。
- `first_chunk_latency`：流式模式首个非空音频 chunk 的延迟，CustomVoice 和 Base 的具体统计起点见本节“Streaming 首音延时统计说明”。
- `chunks`：流式模式输出的音频 chunk 数。
- `playback_gap`：从第二个 chunk 开始统计的播放空挡。如果从第一个 chunk 到达后立即播放，上一个 chunk 播放结束时当前 chunk 仍未到达，则该差值为正；`0.0ms` 表示当前 chunk 可以无缝接上。
- `playback_gap_chunks`：出现正向 `playback_gap` 的 chunk 数量。
- `max_playback_gap`：单次最大播放空挡。
- `total_playback_gap`：整段流式输出累计播放空挡。
- `Component`：推理阶段名称。
- `Time(s)`：该阶段累计耗时，单位为秒。
- `Pct`：该阶段耗时占端到端推理耗时的比例。
- `Count`：该阶段对应 HMM 或逻辑步骤的调用次数。
- `Avg(ms)`：该阶段单次调用平均耗时，单位为毫秒，计算方式为 `Time(s) * 1000 / Count`。

各阶段含义：

- `preparation`：Base 模型的前置准备总耗时，包括 logits processor 构造、参考音频处理和 prompt embedding 组装。
- `reference_audio`：Base 模型 voice clone 参考音频处理耗时，包括：
  - `audio_load`：加载参考音频并重采样到前端采样率；
  - `speech_tokenizer_encode`：通过 `speech_tokenizer_encoder` 将参考音频编码为 codec tokens；
  - `speaker_embed`：通过 `speaker_encoder` 提取参考音频 speaker embedding。
- `prompt_prepare`：Base 模型 prompt 组装中的较小步骤合计，包括文本 tokenization、special token embedding、codec prompt embedding、Talker role embedding、ICL prompt embedding 和最终拼接。
- `embedding_prep`：CustomVoice 模型的文本 tokenization、speaker/language 条件构造、TextProjection 以及 Talker 初始输入 embedding 拼接等准备工作。
- `talker`：Talker 主模型总耗时，包括：
  - `prefill`：处理初始上下文，初始化 Talker KV cache；
  - `decode`：逐帧生成每个音频帧的第一组 codec token，并推进 Talker KV cache；
  - `sampling`：Talker logits 输出后的 host 侧采样开销，包括 logits processor、softmax/multinomial 或 argmax，以及采样 token 写入历史。Talker sampling 次数通常为音频帧数加 1，包含首 token 采样。
- `frame_prepare`：CustomVoice 中 Talker 与 CodePredictor 之间的帧级 Python/Torch 准备开销，包括 token embedding、`predictor_input_embeds` 拼接、`codec_ids` 拼接、各 codebook embedding 求和，以及与 `trailing_text_hidden` 或 `tts_pad_embed` 相加。
- `code_predictor`：CodePredictor 总耗时，用于补齐每个音频帧中除第一组外的其余 codec codebook token，包括：
  - `prepare`：CustomVoice 每帧 CodePredictor prefill 前的准备开销；
  - `prefill`：每帧 CodePredictor 初始上下文 HMM 调用；
  - `decode`：CodePredictor 逐 codebook 自回归 decode HMM 调用；
  - `sampling`：CodePredictor logits 输出后的 host 侧采样开销。CodePredictor sampling 次数通常等于 `prefill` 次数加 `decode` 次数。
- `postprocess`：CustomVoice oneshot 模式下 codec 生成结束后的后处理开销，包括 `generated_codes` stack、EOS 位置查找、有效长度裁剪和 `talker_codes_list` 构造。
- `speech_tokenizer`：oneshot 模式下将完整 audio codes 解码为 waveform 的耗时。
- `stateful_decoder`：CustomVoice streaming 模式下按 chunk 将 audio codes 解码为 waveform 的耗时。
- `audio_decoder`：Base 模型音频解码总耗时。oneshot 使用 `speech_tokenizer`；streaming 包括 `stateful_ref_prime` 和 `stateful_generated`：
  - `stateful_ref_prime`：使用参考音频 codec 预热 StatefulDecoder state，并丢弃该阶段解码音频的耗时；
  - `stateful_generated`：使用预热后的 state 解码目标文本生成的音频 chunk 的耗时。
- `other`：端到端总耗时中未被上述阶段单独统计的剩余部分，主要包括 Python 循环调度、generator/list 迭代、少量 tensor/numpy 转换、计时和日志等开销。
- `total`：端到端推理总耗时，对应上方 `inference`。


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的关联、背书、担保或推荐。您在此进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- Qwen3-TTS-12Hz-0.6B-CustomVoice: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- Qwen3-TTS-12Hz-0.6B-Base: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- Qwen3-TTS-12Hz-1.7B-CustomVoice: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)


## 附录一：CustomVoice demo.py 推理流程说明

本附录用于说明当前 `demo.py` 如何在后摩芯片上完成 Qwen3-TTS CustomVoice 的端到端推理。

### 1. 整体数据流

`demo.py` 将“codec token 生成”和“codec token 到音频解码”拆成两层：

```text
输入 text + speaker + language
        │
        ▼
HF config / processor 解析文本和模型特殊 token
        │
        ▼
Text Embedding + TextProjection HMM
        │
        ▼
构造 Talker 输入 embedding
        │
        ▼
Talker Prefill HMM 初始化生成上下文和 KV cache
        │
        ▼
循环生成 codec frames：
  1. Talker 生成每帧第一组 codebook token
  2. CodePredictor 补齐该帧其余 codebook token
  3. Talker Decode 推进到下一帧
        │
        ▼
codec frames
        │
        ├─ oneshot: 收集完整 frames -> SpeechTokenizer HMM -> waveform
        │
        └─ streaming: CodeChunkBuffer 攒 chunk -> StatefulDecoder HMM -> audio chunk
        │
        ▼
最后统一写 output_wav，并打印性能统计
```

### 2. 主要模块职责

`Qwen3TTSCodecGenerator` 是 `demo.py` 的核心类，负责加载 HF 配置、TextProjection、Talker 和 CodePredictor，并生成 codec frames。音频解码器在 `main()` 中根据模式创建后传入：

| 模块 | 对应类 | 作用 |
|------|--------|------|
| text_projection | `Qwen3TTSTextProjectionInference` | 将文本 embedding 投影到 Talker 使用的隐藏空间 |
| talker | `Qwen3TTSTalkerInference` | 主自回归模型，生成每个音频帧的第一组 codec token，并维护 Talker KV cache |
| code_predictor | `Qwen3TTSCodePredictorInference` | 根据 Talker hidden state 和第一组 token，补齐同一音频帧的其余 codebook token |
| speech_tokenizer | `Qwen3TTSSpeechTokenizerInference` | oneshot 模式下将完整 codec token 序列解码为 24kHz waveform |
| stateful_decoder | `Qwen3TTSStatefulDecoderInference` | streaming 模式下按 chunk 解码 codec frames，并携带 decoder state 输出音频 |

Talker 和 CodePredictor 都拆分为 prefill / decode 两个 HMM。Prefill 处理初始上下文并建立 KV cache，decode 在后续 step 中复用 cache 做单步自回归推理。

`StatefulDecoder` 也通过显式 state 输入输出维护流式状态。history、latent buffer 和 KV cache 等输出会保留为 device tensor，并在下一轮 decode 时直接通过 `set_dev_input()` 复用，减少 host/device 拷贝。

### 3. 初始化与缓存

`Qwen3TTSCodecGenerator.__init__()` 会完成以下工作：

1. 通过 HF `AutoConfig` / `AutoProcessor` 加载模型配置和 tokenizer/processor。
2. 创建 `TextProjection`、`CodePredictor`、`Talker` 三类 HMM 推理对象。
3. 预分配若干 `np.int32` 输入 buffer，例如：
   - `_talker_prefill_valid_length_buf`
   - `_talker_prefill_current_length_buf`
   - `_talker_decode_valid_length_buf`
   - `_cp_prefill_valid_length_buf`
   - `_cp_prefill_current_length_buf`
   - `_cp_prefill_generation_steps_buf`
   - `_cp_decode_valid_length_buf`
   - `_cp_decode_generation_steps_buf`
4. 调用 `_init_static_embeddings()` 预计算或缓存请求间可复用的 embedding：
   - `tts_bos_embed`、`tts_eos_embed`、`tts_pad_embed`
   - `codec_pad_bos_embed`、`codec_bos_embed`
   - speaker embedding cache
   - codec pad embedding cache
   - language / no-language codec prefill embedding cache

这些缓存减少了每次推理和流式首包路径中的重复小算子和 Python 小对象创建。

### 4. 文本、说话人与语言条件构造

`generate_custom_voice()` 和 `generate_streaming()` 都复用 `_generate_codec_frames()` 生成 codec frames。主要输入包括：

- `text`：待合成文本；
- `language`：语言名称，例如 `Chinese`；
- `speaker`：预置说话人，例如 `vivian`。

在 `_generate_codec_frames()` 中，脚本会：

1. 使用 `build_assistant_text()` 将文本包装成 Qwen3-TTS 的 assistant prompt 格式：

   ```text
   <|im_start|>assistant
   ...文本内容...<|im_end|>
   <|im_start|>assistant
   ```

2. 使用 HF `AutoProcessor` 将文本转为 `input_ids`。
3. 通过 `_get_speaker_embedding()` 从缓存中取得 speaker embedding。
4. 通过 `_get_language_id()` 根据 `language` 和 `speaker` 得到语言 token。若 speaker 带方言标记，则会根据 `spk_is_dialect` 覆盖语言条件。
5. 使用预计算的 TTS special token embedding 和 codec 前缀 embedding 构造 Talker 输入。

### 5. Talker 输入 embedding

Talker 的输入不是单纯文本 token embedding，而是文本条件和 codec 条件组合后的 embedding。当前实现分为两种模式：

- `non_streaming_mode=True`：一次性把完整文本条件放入 Talker prefill，适合 oneshot，首包路径较长但文本条件完整。
- `non_streaming_mode=False`：只把开头文本放入 Talker prefill，剩余文本 hidden 放入 `trailing_text_hidden`，在每步 Talker decode 时逐步加入，适合 streaming，首包更短。

当前 `demo.py` 中：

- `generate_custom_voice()` 默认使用 `non_streaming_mode=True`；
- `generate_streaming()` 默认使用 `non_streaming_mode=False`。

最终会得到：

- `talker_input_embed`：Talker prefill 的初始输入；
- `trailing_text_hidden`：Talker decode 每一步附加的文本 hidden 条件；
- `tts_pad_embed`：文本条件耗尽后用于补齐的 pad embedding。

### 6. Talker 与 CodePredictor 自回归生成

#### 6.1 Talker Prefill

`_run_talker_prefill()` 将 `talker_input_embed` 按 Talker HMM 的 `prefill_length` 分块执行：

- `valid_length` 表示当前 chunk 在 KV cache 中的写入位置；
- `current_length` 表示当前 chunk 的有效长度；
- 只有最后一个 chunk 需要取回 `logits` 和 `past_hidden`。

最后一轮 prefill 输出的 logits 会通过 `_sample_next_token()` 采样得到第一个 codec token。

#### 6.2 CodePredictor 补齐 codebook

每个音频帧包含多组 codebook。Talker 只直接生成第一组 token，其余 token 由 `_run_code_predictor_generate()` 生成：

1. 将 Talker 的 `past_hidden` 和当前 token embedding 拼接为 `predictor_input_embeds`。
2. 对 `predictor_input_embeds` 做 padding，并 reset CodePredictor cache input。
3. 执行 CodePredictor prefill，采样剩余 codebook 的第一个 token。
4. 继续执行 CodePredictor decode，逐步采样剩余 codebook token。
5. 返回 `predictor_tokens`、各 codebook embedding 以及分项耗时。

CodePredictor 统计会拆成：

- `code_predictor_prepare`
- `code_predictor_prefill`
- `code_predictor_decode`
- `code_predictor_sampling`

#### 6.3 Talker Decode

每轮得到完整 `codec_ids` 后，生成器会：

1. `yield codec_ids` 给 oneshot 收集逻辑或 streaming chunk buffer；
2. 将上一帧各 codebook embedding 求和为下一步 codec 条件；
3. 根据 step 叠加 `trailing_text_hidden` 或 `tts_pad_embed`；
4. 调用 Talker decode 推进 KV cache；
5. 对 Talker logits 采样下一帧第一组 codebook token。

循环直到采样到 `codec_eos_token_id`、达到 `max_new_tokens`，或达到 Talker context 长度上限。

### 7. 采样与 logits processor

`demo.py` 使用 HuggingFace logits processor/warper 复现原模型生成策略，包括：

- `RepetitionPenaltyLogitsProcessor`
- `MinNewTokensLengthLogitsProcessor`
- `SuppressTokensLogitsProcessor`
- `TemperatureLogitsWarper`
- `TopKLogitsWarper`
- `TopPLogitsWarper`

Talker 和 CodePredictor 分别使用一套 processor：

- Talker 使用 `temperature/top_k/top_p/repetition_penalty`；
- CodePredictor 使用 `subtalker_temperature/subtalker_top_k/subtalker_top_p`。

当前采样统一走 `_sample_next_token()`：

```python
scores = logits_processor(input_ids, logits.to(dtype=torch.float32))
if do_sample:
    probs = F.softmax(scores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
else:
    next_token = torch.argmax(scores, dim=-1)
```

`main()` 会在正式推理计时前调用 `init_logits_processors()`，并将返回的 processors 传入 `generate_custom_voice()` 或 `generate_streaming()`，避免在 oneshot 推理计时或 streaming 首包路径中构造 Python processor 对象。

### 8. Oneshot 模式

oneshot 模式下，`main()` 创建 `Qwen3TTSSpeechTokenizerInference`，并调用：

```python
wavs, sr, perf, perf_count = codec_generator.generate_custom_voice(...)
```

`generate_custom_voice()` 内部流程：

1. 创建 `PerfTracker`。
2. 调用 `_generate_codec_frames()`，将生成器产出的 `codec_ids` 全部收集到 `generated_codes`。
3. 在 `postprocess` 阶段 stack `generated_codes`，根据第一组 codebook 中的 EOS token 找到有效长度，构造 `talker_codes_list`。
4. 调用 `speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])` 解码完整 waveform。
5. `perf.snapshot()` 返回性能统计字典。

`speech_tokenizer.decode()` 会把 `[seq, code_groups]` 形式的 codes 转为 HMM 需要的 `[1, code_groups, seq]`，再由 `chunked_decode()` 分块处理：

- 每块最多处理 `net_input_seq_len` 个 codec frame；
- 默认 `left_context_size=25`；
- 非首块解码后裁掉左上下文对应 waveform，避免重复音频；
- 最后拼接所有 waveform chunk。

`main()` 在 `generate_custom_voice()` 返回后计算 `inference_time`，再统一 `sf.write()` 写出 `output_wav`，最后调用 `log_oneshot_perf()` 打印性能表。

### 9. Streaming 模式

streaming 模式下，`main()` 创建 `Qwen3TTSStatefulDecoderInference`，并调用 `generate_streaming()`。生成器不会等待全部 codec frames 完成，而是边生成边解码音频 chunk：

1. `stateful_decoder.create_state()` 创建全零初始 state，并提前上传 history / KV cache 到设备。
2. 创建 `CodeChunkBuffer(chunk_size=args.chunk_size)`。
3. 从 `_generate_codec_frames()` 逐帧获取 `codec_ids`。
4. 每帧 `codec_ids.squeeze(0)` 后写入 buffer。
5. buffer 满 `chunk_size` 后，调用 `stateful_decoder.decode(chunk, decoder_state, is_final=False)`。
6. 若 decoder 返回非空音频，则 yield `(audio, 24000)`。
7. 生成结束后，对残余帧执行 final decode；如果最后一块刚好满 chunk，则发送 0 帧 final decode flush latent audio。

`StatefulDecoderState` 包含：

- `pre_conv_history`
- `latent_buffer`
- `conv_history`
- `kv_cache`
- `kv_valid_len`
- `skip_samples`
- `latent_audio`

每次 decode 后，新的 history / KV cache 以 device tensor 形式保存到下一轮 state，下一次 decode 直接作为 device input 复用。Python 侧只取回需要裁剪和返回的 waveform / valid samples。

### 10. 输出与性能统计

当前 `demo.py` 的 oneshot 和 streaming 都在 `main()` 中统一写出 wav，推理计时不包含最终 `sf.write()` 文件 I/O。Streaming 模式下，`main()` 会收集所有音频 chunk 后再 `np.concatenate()` 写文件。

`demo.py` 使用 `PerfTracker` 和统一的 `PerfKey` 管理性能统计。首音延时、播放空挡和各性能指标含义统一见 4.4。


## 附录二：Base demo_base.py 推理流程说明

本附录用于说明当前 `demo_base.py` 如何在后摩芯片上完成 `Qwen3-TTS-12Hz-0.6B-Base` 的 voice clone 推理。`demo_base.py` 和 `demo.py` 的 Talker / CodePredictor 自回归生成框架基本一致，主要差异在于 Base 模型没有固定 speaker 入口，而是通过 `ref_audio` 和 `ref_text` 构造 in-context learning prompt，并额外使用 `speech_tokenizer_encoder` 和 `speaker_encoder` 从参考音频中提取条件信息。

### 1. 整体数据流

`demo_base.py` 的端到端流程可以分为“参考音频条件构造”、“codec token 生成”和“codec token 到音频解码”三层：

```text
输入 text + language + ref_audio + ref_text
        │
        ├─ ref_audio -> load/resample -> SpeechTokenizerEncoder HMM -> ref_code
        │
        └─ ref_audio -> mel spectrogram -> SpeakerEncoder HMM -> speaker_embedding
        │
        ▼
ref_text + text -> tokenizer -> TextProjection HMM
        │
        ▼
构造 Base ICL prompt：
  参考文本 hidden + 参考音频 codec embedding + speaker/language 条件
        │
        ▼
Talker Prefill HMM 初始化生成上下文和 KV cache
        │
        ▼
循环生成目标音频 codec frames：
  1. Talker 生成每帧第一组 codebook token
  2. CodePredictor 补齐该帧其余 codebook token
  3. Talker Decode 推进到下一帧
        │
        ├─ oneshot:
        │    ref_code + generated_code -> SpeechTokenizer HMM -> waveform
        │    -> 裁掉参考音频对应 waveform -> output_wav
        │
        └─ streaming:
             ref_code -> StatefulDecoder 预热 state 并丢弃输出
             generated_code chunks -> StatefulDecoder HMM -> audio chunks
             -> 最后统一拼接写 output_wav
```

### 2. 主要模块职责

`Qwen3TTSInference` 是 `demo_base.py` 的核心类。相比 CustomVoice，Base 入口额外加载 `speech_tokenizer_encoder` 和 `speaker_encoder`，用于把参考音频转换为 prompt 条件。

| 模块 | 对应类 | 作用 |
|------|--------|------|
| text_projection | `Qwen3TTSTextProjectionInference` | 将文本 embedding 投影到 Talker 使用的隐藏空间 |
| talker | `Qwen3TTSTalkerInference` | 主自回归模型，生成每个音频帧的第一组 codec token，并维护 Talker KV cache |
| code_predictor | `Qwen3TTSCodePredictorInference` | 根据 Talker hidden state 和第一组 token，补齐同一音频帧的其余 codebook token |
| speech_tokenizer_encoder | `Qwen3TTSSpeechTokenizerEncoderInference` | 将参考音频编码成 `[T, num_code_groups]` 的 codec tokens，作为 Base ICL prompt 的参考音频部分 |
| speaker_encoder | `Qwen3TTSSpeakerEncoderInference` | 从参考音频 mel spectrogram 中提取 speaker embedding |
| speech_tokenizer | `Qwen3TTSSpeechTokenizerInference` | oneshot 模式下将完整 codec token 序列解码为 24kHz waveform |
| stateful_decoder | `Qwen3TTSStatefulDecoderInference` | streaming 模式下按 chunk 解码 codec frames，并携带 decoder state 输出音频 |

`demo_base.py` 只依赖 HMM、量化 embedding 文件和 HF config / processor，不加载原始 PyTorch 模型权重。

### 3. 参考音频处理

Base voice clone 的参考音频处理入口是 `encode_reference_audio(ref_audio)`：

1. 检查 `ref_audio` 文件是否存在。
2. `_load_reference_audio()` 通过 `torchaudio.load()` 读取音频，多声道时取均值，并重采样到 24kHz；日志会记录样本数、时长、峰值和末尾 0.5 秒 RMS。
3. `speech_tokenizer_encoder.encode(wav, sr)` 将参考音频补零到 HMM 的静态输入长度，同时构造 `padding_mask`，输出 `[T, num_code_groups]` 形式的 `ref_code`。
4. `extract_speaker_embedding(wav, sr)` 先计算 128 维 mel spectrogram，再补齐或裁剪到 SpeakerEncoder HMM 的静态帧数，最后输出 float32 `speaker_embedding`。
5. 将参考音频相关耗时写入 `PerfTracker`，包括：
   - `load`
   - `speech_tokenizer_encode`
   - `speaker_embed`

其中 `speech_tokenizer_encoder` 的 HMM 输入长度是静态的。如果参考音频超过模型输入长度，当前实现会打印 warning，并截断到 `net_input_samples` 后再编码；如果音频为空或编码后没有有效帧，则直接报错。

### 4. Base ICL Prompt 构造

`_prepare_voice_clone_prompt()` 负责把文本、语言、参考音频和参考文本组织成 Talker prefill 输入。它最终返回 `BasePromptContext`：

| 字段 | 含义 |
|------|------|
| `talker_input_embed` | Talker prefill 的完整输入 embedding |
| `trailing_text_hidden` | streaming embedding 组织方式下，后续 Talker decode 逐步追加的文本 hidden |
| `ref_code` | 参考音频的 codec tokens |
| `tts_pad_embed` | 文本条件耗尽后使用的 pad embedding |

Prompt 构造主要包括：

1. 使用 `build_assistant_text(text)` 构造目标文本 prompt。
2. 使用 `build_ref_text(ref_text)` 构造参考文本 prompt。
3. 根据 `language` 查找 codec language token；`auto` 不显式放入 language token。
4. 将 TTS bos / eos / pad token 依次送入文本 embedding 和 TextProjection，得到 `tts_bos_embed`、`tts_eos_embed`、`tts_pad_embed`。
5. 根据是否显式指定语言，组织 think/no-think token、language token、speaker embedding、codec pad / bos token，得到 codec prompt embedding。
6. 将目标文本 assistant role 开头的 3 个 token 送入文本 embedding 和 TextProjection，并与 codec prompt embedding 拼成起始上下文。
7. `_generate_icl_prompt()` 先将参考文本 token 和目标文本 token 在文本侧拼接，经 TextProjection 投影后追加 `tts_eos_embed`。
8. 参考音频的第 1 组 codebook 使用 Talker token embedding，其余 codebook 使用 CodePredictor token embedding；同一帧各组 embedding 求和，并在序列前加入 `codec_bos_id`。
9. 按 `non_streaming_mode` 组织文本 hidden 和参考 codec embedding，再与起始上下文拼接，形成最终 `talker_input_embed`。如果 prompt 长度已达 Talker context 上限，则直接报错。

### 5. Talker 输入 embedding 模式

`_generate_icl_prompt()` 支持两种 Talker input embedding 组织方式：

- `non_streaming_mode=True`：完整文本 hidden 先与 codec pad embedding 相加，再拼接“参考 codec embedding + `tts_pad_embed`”。返回的 `trailing_text_hidden` 为 `tts_pad_embed`，后续 Talker decode 不再逐步注入剩余文本 hidden。
- `non_streaming_mode=False`：文本 hidden 与参考 codec embedding 从头部对齐后相加。如果文本更长，未放入 prefill 的剩余 hidden 写入 `trailing_text_hidden`；如果 codec 更长，文本侧使用 `tts_pad_embed` 补齐。

当前 `demo_base.py` 中：

- `generate_voice_clone()` 默认使用 `non_streaming_mode=True`；
- `generate_voice_clone_streaming()` 默认使用 `non_streaming_mode=False`。

日志中会打印：

```text
Using non-streaming mode for talker input embedding
```

或：

```text
Using streaming mode for talker input embedding
```

用于区分当前 prompt embedding 的组织方式。

### 6. Codec Frame 生成

Base 的 codec frame 生成入口是 `_generate_codec_frames()`，流程与 CustomVoice 基本一致：

1. `Talker Prefill` 处理 `talker_input_embed`，初始化 Talker KV cache，并输出首帧第一组 codebook 的 logits。
2. `talker_sampling` 对 Talker logits 做 logits processor / warper 后采样得到第一组 token。
3. 每一帧调用 `CodePredictor`：
   - 根据 Talker `past_hidden` 和当前第一组 token 构造 `predictor_input_embeds`；
   - CodePredictor prefill 生成第二组 codebook token；
   - CodePredictor decode 继续生成剩余 codebook token；
   - `code_predictor_sampling` 统计 CodePredictor logits 采样耗时。
4. 拼出完整 `codec_ids` 并 `yield` 给 oneshot 收集逻辑或 streaming chunk buffer。
5. 将当前帧各 codebook embedding 求和，并叠加 `trailing_text_hidden` 或 `tts_pad_embed`，作为下一步 Talker decode 输入。
6. `Talker Decode` 推进到下一帧，并继续采样下一帧第一组 token。

循环会在以下情况停止：

- 采样到 `codec_eos_token_id`；
- 达到 `max_new_tokens`；
- 达到 Talker context 长度上限。

### 7. Oneshot 模式

oneshot 模式入口是 `generate_voice_clone()`，`main()` 在 `args.mode == "oneshot"` 时调用该接口。

核心流程如下：

1. 创建 `PerfTracker`。
2. 初始化 Talker / CodePredictor 的 logits processors。
3. 调用 `_prepare_voice_clone_prompt()` 构造 Base prompt。
4. 调用 `_generate_codec_frames()`，把所有生成的 `codec_ids` 收集到 `generated_codes`。
5. `_collect_talker_codes()` 根据第一组 codebook 中的 EOS token 裁掉无效帧。
6. 将 `prompt.ref_code` 拼接到生成 codes 前面：

   ```text
   codes_for_decode = ref_code + generated_codes
   ```

7. 调用 `speech_tokenizer.decode()` 一次性解码完整 waveform。
8. 按 `ref_len / total_len` 计算 waveform 的裁剪位置，移除参考 codec 对应的前缀音频，只保留目标文本生成音频。
9. 返回 `wavs, sr, perf_dict, perf_count`。

这里先拼接 `ref_code` 再裁掉参考音频，是为了让 oneshot SpeechTokenizer 解码时拥有与 prompt 一致的参考音频上下文，减少目标音频开头的衔接问题。

### 8. Streaming 模式

streaming 模式入口是 `generate_voice_clone_streaming()`，`main()` 在 `args.mode == "streaming"` 时创建 `Qwen3TTSStatefulDecoderInference` 并调用该接口。

Base streaming 相比 CustomVoice 多了一个关键步骤：用参考音频 `ref_code` 预热 StatefulDecoder state。

`_prime_stateful_decoder_with_ref_code()` 的流程是：

1. 创建 `CodeChunkBuffer(chunk_size=args.chunk_size)`。
2. 将 `ref_code` 逐帧写入 buffer。
3. 每当 buffer 满一个 chunk，就调用：

   ```python
   stateful_decoder.decode(ref_chunk, decoder_state, is_final=False)
   ```

4. 丢弃参考音频解码输出，只保留新的 `decoder_state`。
5. 对不足一个 chunk 的残余参考 codec，使用最后一帧补齐到 `chunk_size` 后继续预热。
6. 预热完成后，将 `decoder_state.skip_samples` 重置为 StatefulDecoder 的初始输出裁剪量，使该裁剪规则从目标音频开始生效。

预热完成后才开始生成目标音频 codec。Base streaming 首音延时的具体统计说明见 4.4。

目标音频生成阶段与 CustomVoice streaming 一致：

1. `_generate_codec_frames()` 逐帧生成目标 codec。
2. `CodeChunkBuffer` 攒够 `chunk_size` 后调用 `stateful_decoder.decode(..., is_final=False)`。
3. 若 decoder 返回非空音频，则 `yield (audio, 24000)`。
4. 生成结束后，对残余帧执行 final decode；如果没有残余帧，则发送 0 帧 final decode flush latent audio。
5. 第一个非空 audio chunk 产生时，记录从参考 codec 预热完成到该 chunk 输出的 `first_chunk_latency`。
6. `main()` 收集所有 audio chunks，记录 chunk 数量和 `playback_gap`，最后 `np.concatenate()` 后统一写 wav。

### 9. 输出与性能统计

当前 `demo_base.py` 的 oneshot 和 streaming 都在 `main()` 中统一写出 wav，请求推理计时都从进入生成接口前开始，因此包含 logits processor、参考音频、prompt、codec 生成和音频解码开销，但不包含模型初始化和最终 `sf.write()` 文件 I/O。

Streaming 的 `first_chunk_latency` 使用生成器内部的单独计时起点：它从参考音频 prompt 构造和 StatefulDecoder 预热完成后开始，到第一个非空目标音频 chunk 产生为止。因此该数值不包含参考音频处理、prompt 构造和 `stateful_ref_prime`；完整说明见 4.4。

Streaming 模式下，`main()` 会收集所有音频 chunk，校验采样率始终一致，并通过 `StreamingPlaybackGapTracker` 统计播放空挡；推理结束后再使用 `np.concatenate()` 拼接和写入文件。

`demo_base.py` 和 `demo.py` 共用 `PerfTracker` / `PerfKey` 管理性能统计。Base 的首音延时、播放空挡和各性能指标含义统一见 4.4。
