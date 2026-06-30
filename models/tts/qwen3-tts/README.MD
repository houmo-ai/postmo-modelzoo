# Qwen3-TTS

本目录展示如何将 Qwen3-TTS 0.6B 系列模型量化、编译并部署到后摩芯片设备上。当前覆盖两个 12Hz 模型：

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- `Qwen3-TTS-12Hz-0.6B-CustomVoice`
  - 来源: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
  - 用途: 使用预置 speaker 进行语音合成
- `Qwen3-TTS-12Hz-0.6B-Base`
  - 来源: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
  - 用途: 基于参考音频和参考文本进行 voice clone

- 许可: Apache License 2.0 (https://huggingface.co/Qwen/Qwen3-TTS/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前支持情况如下：

| 模型 | 推理入口 | 主要用途 | 当前支持状态 |
|------|----------|----------|--------------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | `demo.py` | 预置 speaker 合成 | 已支持非流式 `oneshot` 和流式 `streaming` 两种模式 |
| `Qwen3-TTS-12Hz-0.6B-Base` | `demo_base.py` | voice clone | 已支持非流式 `oneshot` 和流式 `streaming` 两种模式 |

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-tts`
- `model_size=0.6b-customvoice`
- `context_length=2k`
- `prefill_length=256`
- `ncore=2`
- `ndevice=1`
- `quant_type=w8a16h1_sefp`

当前示例支持的模型规格为：

- `0.6b-customvoice`
- `0.6b-base`

### 资源要求

**后摩设备运行要求：**

- `qwen3-tts-0.6b-customvoice`: 单卡12GB可运行
- `qwen3-tts-0.6b-base`: 单卡12GB可运行

**量化硬件要求：**

- `qwen3-tts-0.6b-customvoice`: 需要 GPU，显存最小 24GB
- `qwen3-tts-0.6b-base`: 需要 GPU，显存最小 24GB

**编译时 host 内存要求：**

- `qwen3-tts-0.6b-customvoice`: 需要空闲 host 内存最小 18GB
- `qwen3-tts-0.6b-base`: 需要空闲 host 内存最小 18GB

## 2 快速开始

目前 Qwen3-TTS 模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 脚本职责

当前目录下主要脚本职责如下：

| 脚本 | 作用 |
|------|------|
| `get_model.py` | 根据 `config.yaml` 下载原始模型或预编译芯片模型 |
| `ptq.py` | 将原始模型导出并量化为 HMONNX / 量化辅助文件 |
| `build.py` | 将 HMONNX 编译为后摩芯片可运行的 HMM |
| `demo.py` | 运行 `0.6b-customvoice` 的 oneshot / streaming 推理 |
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

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。`config.yaml` 中当前支持的 `model_size` 为：

- `0.6b-customvoice`
- `0.6b-base`

如果需要自己量化，可下载原始模型。

下载 `CustomVoice` 原始模型：
```bash
python3 get_model.py --type raw --model_name qwen3-tts --model_size 0.6b-customvoice
```

下载 `Base` 原始模型：
```bash
python3 get_model.py --type raw --model_name qwen3-tts --model_size 0.6b-base
```

如果仅查看演示结果，可下载芯片模型。

下载 `CustomVoice` 芯片模型：
```bash
python3 get_model.py --type hmm --model_name qwen3-tts --model_size 0.6b-customvoice
```

下载 `Base` 芯片模型：
```bash
python3 get_model.py --type hmm --model_name qwen3-tts --model_size 0.6b-base
```

常用参数：

- `--type`：下载类型，`raw` 为原始 ModelScope/HF 模型，`hmm` 为预编译芯片模型；
- `--download_dir`：下载文件保存目录，默认当前目录；
- `--extract_dir`：解压目录，默认由下载工具决定；
- `--config`：配置文件路径，默认 `config.yaml`；
- `--model_name` / `--model_size`：模型名称和规格，未指定时使用 `config.yaml` 默认值；
- `--ndevice` / `--ncore` / `--context_length` / `--quant_type`：覆盖配置中用于匹配芯片模型包的参数。

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：
```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入模型目录，执行 `ptq.py` 脚本进行量化：
```bash
source env.sh
cd models/tts/qwen3-tts
python3 ptq.py --model_name qwen3-tts --model_size 0.6b-customvoice
```

`ptq.py` 将依次导出以下子模型到 HMONNX 格式：
- talker（prefill + decode）
- code_predictor（prefill + decode）
- text_projection
- speech_tokenizer
- stateful_decoder
- speaker_encoder（仅 Base）
- speech_tokenizer_encoder（仅 Base）

量化 `Base` 模型时，示例命令如下：
```bash
python3 ptq.py --model_name qwen3-tts --model_size 0.6b-base
```

可通过 `--models` 参数指定要量化的子模型（逗号分隔），例如：
```bash
python3 ptq.py --model_name qwen3-tts --model_size 0.6b-base --models talker,code_predictor
```

`ptq.py --models all` 是默认行为。对于 `0.6b-customvoice`，`all` 会自动跳过 `speaker_encoder` 和 `speech_tokenizer_encoder`；这两个子模型仅适用于 `0.6b-base`。如果显式指定 Base-only 子模型给 CustomVoice，会报错。

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

将量化模型编译为在芯片上运行的模型。执行脚本：
```bash
python3 build.py --model_name qwen3-tts --model_size 0.6b-customvoice
```

`CustomVoice` 默认输出名称包括：
- `qwen3-tts-0.6b-customvoice_text_projection.hmm`
- `qwen3-tts-0.6b-customvoice_code_predictor_prefill.hmm`
- `qwen3-tts-0.6b-customvoice_code_predictor_decode.hmm`
- `qwen3-tts-0.6b-customvoice_talker_prefill.hmm`
- `qwen3-tts-0.6b-customvoice_talker_decode.hmm`
- `qwen3-tts-0.6b-customvoice_speech_tokenizer.hmm`
- `qwen3-tts-0.6b-customvoice_stateful_decoder.hmm`

编译 `Base` 模型时，示例命令如下：
```bash
python3 build.py --model_name qwen3-tts --model_size 0.6b-base
```

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
- `--cpp_backend`：默认从 `config.yaml` 读取，当前为 `v2`；
- `--flash_attention`：两个整数，分别控制 LLM prefill/decode 和 non-LLM 子模型的 FlashAttention 开关，默认 `2 1`。

对于 `0.6b-base`，`speech_tokenizer_encoder` 编译时会固定使用 `cpp_backend=v1`；其他子模型仍使用全局 `--cpp_backend` 或 `config.yaml` 中的默认值。

### 2.5 演示

Qwen3-TTS 模型使用 Python API 进行演示。`CustomVoice` 和 `Base` 两种模型均支持两种音频输出路径：

- `oneshot`：先完整生成 codec frames，再通过 `SpeechTokenizer` 一次性解码为 wav；
- `streaming`：边生成 codec frames 边通过 `StatefulDecoder` 按 chunk 输出音频。

两个模型的入口不同：

- `demo.py`：用于 `0.6b-customvoice`，输入为 `text + speaker + language`；
- `demo_base.py`：用于 `0.6b-base`，输入为 `text + language + ref_audio + ref_text`，用于 voice clone。

#### 2.5.1 CustomVoice 演示

`demo.py` 默认根据 `--model_name` / `--model_size` 从 `output/${HOUMO_TARGET}` 推导 HMM 路径，具体编译产物见 2.4。oneshot 模式需要 Talker、CodePredictor、TextProjection 和 SpeechTokenizer；streaming 模式会额外使用 `stateful_decoder_hmm`。

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

#### 2.5.2 Base 演示

`Base` 模型用于 voice clone，使用 `demo_base.py` 运行。脚本默认参考音频为 `./clone_1.wav`，参考文本为 `甚至出现交易几乎停滞的情况。`，也可以通过 `--ref_audio` 和 `--ref_text` 显式指定。

`demo_base.py` 默认根据 `--model_name` / `--model_size` 从 `output/${HOUMO_TARGET}` 推导 HMM 路径，具体编译产物见 2.4。Base voice clone 除 Talker、CodePredictor、TextProjection 和音频解码器外，还需要 `speech_tokenizer_encoder_hmm` 和 `speaker_encoder_hmm`。

非流式执行示例：
```bash
python3 demo_base.py \
  --model_name qwen3-tts \
  --model_size 0.6b-base \
  --mode oneshot \
  --ref_audio ./clone_1.wav \
  --ref_text "甚至出现交易几乎停滞的情况。" \
  --output_wav output_voice_clone.wav
```

流式执行示例：
```bash
python3 demo_base.py \
  --model_name qwen3-tts \
  --model_size 0.6b-base \
  --mode streaming \
  --ref_audio ./clone_1.wav \
  --ref_text "甚至出现交易几乎停滞的情况。" \
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
  -size, --model_size     Model size. Default: 0.6b-customvoice. Choices: 0.6b-customvoice, 0.6b-base.
  --demo_mode             Demo mode. Default: oneshot. Choices: oneshot, streaming.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：
```bash
# 仅运行量化，会先下载原始模型再量化
bash test.sh -s quant

# 量化 + 编译
bash test.sh -s quant,build

# 跳过下载，直接运行 demo（需已有芯片模型）
bash test.sh -s demo --skip_download

# 运行 CustomVoice 流式 demo
bash test.sh -s demo --model_size 0.6b-customvoice --demo_mode streaming

# 运行 Base 非流式 voice clone demo
bash test.sh -s demo --model_size 0.6b-base --demo_mode oneshot

# 运行 Base 流式 voice clone demo
bash test.sh -s demo --model_size 0.6b-base --demo_mode streaming
```

## 4 参考结果

### 4.1 CustomVoice 演示结果

#### Oneshot 模式

```bash
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
Qwen3TTSCodecGenerator initialized successfully
SpeechTokenizer loaded: input_seq_len=300
Starting TTS generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
Using non-streaming mode for talker input embedding
Reached EOS token at step 137
Audio saved to output_custom_voice.wav | duration: 10.96s | inference: 9.26s | RTF: 0.8446
========================================================================
Performance Breakdown:
  Component                      Time(s)    Pct  Count  Avg(ms)
  --------------------------------------------------------------
  embedding_prep                    0.02   0.2%
  talker                            1.66  17.9%
    prefill                         0.08   0.8%      1    77.78
    decode                          1.37  14.8%    137     9.98
    sampling                        0.21   2.3%    138     1.53
  frame_prepare                     0.02   0.2%
  code_predictor                    6.13  66.2%
    prepare                         0.11   1.2%
    prefill                         0.43   4.6%    137     3.13
    decode                          4.52  48.8%   1918     2.35
    sampling                        1.07  11.6%   2055     0.52
  postprocess                       0.00   0.0%
  speech_tokenizer                  1.32  14.2%      1  1316.77
  other                             0.11   1.2%
  --------------------------------------------------------------
  total                             9.26 100.0%
========================================================================
```

#### Streaming 模式

```bash
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
Qwen3TTSCodecGenerator initialized successfully
StatefulDecoder loaded: chunk_size=12
Starting streaming TTS for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地...
Using streaming mode for talker input embedding
First audio chunk latency: 835.0ms
  Chunk 1: 15360 samples (640ms audio)
  Chunk 2: 23040 samples (960ms audio) | playback_gap: 62.8ms
  Chunk 3: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 4: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 5: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 6: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 7: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 8: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 9: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 10: 23040 samples (960ms audio) | playback_gap: 0.0ms
  Chunk 11: 23040 samples (960ms audio) | playback_gap: 0.0ms
Reached EOS token at step 137
  Chunk 12: 17280 samples (720ms audio) | playback_gap: 0.0ms
Audio saved to output_custom_voice.wav | duration: 10.96s | inference: 8.54s | RTF: 0.7791 | chunks: 12 | first_chunk_latency: 835.0ms | playback_gap_chunks: 1 | max_playback_gap: 62.8ms | total_playback_gap: 62.8ms
========================================================================
Streaming Performance Breakdown:
  Component                      Time(s)    Pct  Count  Avg(ms)
  --------------------------------------------------------------
  embedding_prep                    0.02   0.2%
  talker                            1.63  19.1%
    prefill                         0.07   0.9%      1    74.23
    decode                          1.36  15.9%    137     9.93
    sampling                        0.19   2.3%    138     1.41
  frame_prepare                     0.02   0.2%
  code_predictor                    6.09  71.3%
    prepare                         0.11   1.3%
    prefill                         0.43   5.0%    137     3.12
    decode                          4.49  52.6%   1918     2.34
    sampling                        1.06  12.5%   2055     0.52
  stateful_decoder                  0.68   7.9%     12    56.43
  other                             0.10   1.2%
  --------------------------------------------------------------
  total                             8.54 100.0%
========================================================================
```

### 4.2 Base 演示结果

#### Oneshot 模式

```bash
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
SpeechTokenizer loaded: input_seq_len=300
SpeechTokenizerEncoder loaded
SpeakerEncoder loaded
Qwen3TTSInference initialized successfully
Starting voice clone generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地
Reference audio frontend input: path=./clone_1.wav, samples=100895, sr=24000, duration=4.2040s, peak=0.052004, tail_0p5s_rms=0.001986
Using non-streaming mode for talker input embedding
Reached EOS token at step 177
Audio saved to output_voice_clone.wav | duration: 14.16s | inference: 11.65s | RTF: 0.8229
========================================================================
Performance Breakdown:
  Component                      Time(s)    Pct  Count  Avg(ms)
  --------------------------------------------------------------
  preparation                       0.08   0.6%
    logits_processor                0.00   0.0%
    reference_audio                 0.05   0.4%
      audio_load                    0.02   0.2%      1    19.94
      speech_tokenizer_encode       0.02   0.2%      1    22.67
      speaker_embed                 0.01   0.1%      1     7.27
    prompt_prepare                  0.02   0.2%      7     3.56
  talker                            2.09  17.9%
    prefill                         0.07   0.6%      1    74.73
    decode                          1.77  15.2%    177     9.98
    sampling                        0.25   2.1%    178     1.39
  code_predictor                    7.70  66.1%
    prefill                         0.56   4.8%    177     3.14
    decode                          5.84  50.1%   2478     2.36
    sampling                        1.31  11.2%   2655     0.49
  audio_decoder                     1.32  11.3%
    speech_tokenizer                1.32  11.3%      1  1321.74
    stateful_ref_prime              0.00   0.0%      0     0.00
    stateful_generated              0.00   0.0%      0     0.00
  other                             0.46   4.0%
  --------------------------------------------------------------
  total                            11.65 100.0%
========================================================================
```

#### Streaming 模式

```bash
TextProjection loaded
CodePredictor loaded: prefill_length=2
Talker loaded, context_length=2048
SpeechTokenizer loaded: input_seq_len=300
SpeechTokenizerEncoder loaded
SpeakerEncoder loaded
Qwen3TTSInference initialized successfully
StatefulDecoder loaded: chunk_size=12
Starting streaming voice clone generation for text: 基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地
Reference audio frontend input: path=./clone_1.wav, samples=100895, sr=24000, duration=4.2040s, peak=0.052004, tail_0p5s_rms=0.001986
Using streaming mode for talker input embedding
First audio chunk latency: 813.0ms
  Chunk 1: 23040 samples (960ms audio)
  Chunk 2: 23040 samples (960ms audio) | playback_gap: 0.0ms
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
Reached EOS token at step 176
  Chunk 15: 23040 samples (960ms audio) | playback_gap: 0.0ms
Audio saved to output_voice_clone.wav | duration: 14.40s | inference: 11.27s | RTF: 0.7824 | chunks: 15 | first_chunk_latency: 813.0ms | playback_gap_chunks: 0 | max_playback_gap: 0.0ms | total_playback_gap: 0.0ms
========================================================================
Streaming Performance Breakdown:
  Component                      Time(s)    Pct  Count  Avg(ms)
  --------------------------------------------------------------
  preparation                       0.07   0.6%
    logits_processor                0.00   0.0%
    reference_audio                 0.05   0.4%
      audio_load                    0.01   0.1%      1    13.61
      speech_tokenizer_encode       0.02   0.2%      1    22.79
      speaker_embed                 0.01   0.1%      1     8.98
    prompt_prepare                  0.03   0.2%      7     3.83
  talker                            2.06  18.3%
    prefill                         0.08   0.7%      1    78.43
    decode                          1.74  15.5%    176     9.90
    sampling                        0.24   2.1%    177     1.35
  code_predictor                    7.57  67.2%
    prefill                         0.55   4.9%    176     3.14
    decode                          5.77  51.2%   2464     2.34
    sampling                        1.25  11.1%   2640     0.47
  audio_decoder                     1.13  10.0%
    speech_tokenizer                0.00   0.0%      0     0.00
    stateful_ref_prime              0.28   2.5%      5    56.63
    stateful_generated              0.85   7.5%     15    56.55
  other                             0.43   3.8%
  --------------------------------------------------------------
  total                            11.27 100.0%
========================================================================
```

### 4.3 性能指标说明

#### Streaming 首音延时统计口径

4.1 和 4.2 中的 streaming 结果里，Base 的 `first_chunk_latency` 低于 CustomVoice，主要原因是两种 demo 的首音延时统计起点不同，不能简单理解为 Base 的端到端首音链路一定更短。

- CustomVoice `demo.py`：`first_chunk_latency` 从 `stateful_decoder.create_state()` 完成后开始计时，到第一次 `yield` 非空 audio chunk 结束。该指标包含 CustomVoice streaming 模式下的 embedding 准备、Talker prefill、若干帧 Talker / CodePredictor 自回归生成、首个 StatefulDecoder chunk 解码，以及中间 Python 调度开销。
- Base `demo_base.py`：`first_chunk_latency` 从参考音频处理、Base ICL prompt 构造、`stateful_decoder.create_state()`、以及 `_prime_stateful_decoder_with_ref_code()` 参考音频 codec 预热全部完成后开始计时，到第一次 `yield` 非空目标 audio chunk 结束。该指标不包含 `prep_logits_processor`、`reference_audio`、`prompt_prepare` 和 `stateful_ref_prime`，这些耗时会体现在总 `inference` 和性能表中。

因此，Base streaming 的首音延时数字更小，主要是因为 Base 当前统计的是“目标音频生成阶段开始后到首个目标音频 chunk 输出”的延迟。比较端到端体验时，应同时查看 `inference`、`preparation.reference_audio`、`prompt_prepare`、`audio_decoder.stateful_ref_prime` 和 `first_chunk_latency`。

#### 指标含义

- `duration`：生成音频的实际时长，单位为秒。
- `inference`：端到端推理耗时，单位为秒。包含文本/参考音频准备、Talker、CodePredictor、音频解码器以及 Python 调度等开销，不包含最后统一 `sf.write()` 写 wav 的耗时。
- `RTF`：Real-Time Factor，计算方式为 `inference / duration`。RTF 小于 1 表示推理速度快于实时播放。
- `first_chunk_latency`：流式模式首个非空音频 chunk 的延迟，CustomVoice 和 Base 的具体统计起点见本节“Streaming 首音延时统计口径”。
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
  - `stateful_ref_prime`：Base streaming 用参考音频 codec 预热 StatefulDecoder state，并丢弃参考音频输出的耗时；
  - `stateful_generated`：Base streaming 解码生成音频 chunk 的耗时。
- `other`：端到端总耗时中未被上述阶段单独统计的剩余部分，主要包括 Python 循环调度、generator/list 迭代、少量 tensor/numpy 转换、计时和日志等开销。
- `total`：端到端推理总耗时，对应上方 `inference`。


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的关联、背书、担保或推荐。您在此进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- Qwen3-TTS-12Hz-0.6B-CustomVoice: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- Qwen3-TTS-12Hz-0.6B-Base: [https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)


## 附录一：CustomVoice demo.py 推理流程说明

本附录用于说明当前 `demo.py` 如何在后摩芯片上完成 `Qwen3-TTS-12Hz-0.6B-CustomVoice` 的端到端推理。`demo.py` 只负责 CustomVoice 模型；`Base` 模型的 voice clone 推理入口为 `demo_base.py`，运行方式见 2.5.2 和 4.2。

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

`demo.py` 使用 `PerfTracker` 和统一的 `PerfKey` 管理性能统计。首音延时、播放空挡和各性能指标含义统一见 4.3。


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
2. `_load_reference_audio()` 读取音频并统一到 24kHz 单声道。
3. `speech_tokenizer_encoder.encode(wav, sr)` 将参考音频编码为 `ref_code`。
4. `extract_speaker_embedding(wav, sr)` 调用 `speaker_encoder` 提取 `speaker_embedding`。
5. 将参考音频相关耗时写入 `PerfTracker`，包括：
   - `load`
   - `speech_tokenizer_encode`
   - `speaker_embed`

其中 `speech_tokenizer_encoder` 的 HMM 输入长度是静态的。如果参考音频超过模型输入长度，当前实现会截断到 `net_input_samples` 后再编码；如果编码后没有有效帧，则直接报错。

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
3. 根据 `language` 查找 codec language token。
4. 计算 `tts_bos_embed`、`tts_eos_embed`、`tts_pad_embed`。
5. 将 language token、speaker embedding、codec pad / bos token 组织成 codec prompt embedding。
6. 调用 `_generate_icl_prompt()` 拼接参考文本 hidden、目标文本 hidden 和参考音频 codec embedding。
7. 将 role embedding 与 ICL prompt embedding 拼接，形成最终 `talker_input_embed`。

### 5. Talker 输入 embedding 模式

`_generate_icl_prompt()` 支持两种 Talker input embedding 组织方式：

- `non_streaming_mode=True`：一次性把参考文本和目标文本 hidden 都放入 Talker prefill，再拼接参考音频 codec embedding。该模式适合 oneshot，文本条件完整，但 prefill 更长。
- `non_streaming_mode=False`：将文本 hidden 与参考音频 codec embedding 尽量对齐，未放入 prefill 的剩余文本 hidden 写入 `trailing_text_hidden`，后续在 Talker decode 时逐步叠加。该模式适合 streaming，可以缩短首包路径。

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
8. 根据参考 codec 帧占总 codec 帧的比例，裁掉 waveform 前半段参考音频对应的部分，只保留目标文本生成音频。
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

预热完成后才开始生成目标音频 codec。Base streaming 首音延时的具体统计口径见 4.3。

目标音频生成阶段与 CustomVoice streaming 一致：

1. `_generate_codec_frames()` 逐帧生成目标 codec。
2. `CodeChunkBuffer` 攒够 `chunk_size` 后调用 `stateful_decoder.decode(..., is_final=False)`。
3. 若 decoder 返回非空音频，则 `yield (audio, 24000)`。
4. 生成结束后，对残余帧执行 final decode；如果没有残余帧，则发送 0 帧 final decode flush latent audio。
5. `main()` 收集所有 audio chunks，最后 `np.concatenate()` 后统一写 wav。

### 9. 输出与性能统计

当前 `demo_base.py` 的 oneshot 和 streaming 都在 `main()` 中统一写出 wav，推理计时不包含最终 `sf.write()` 文件 I/O。Streaming 模式下，`main()` 会收集所有音频 chunk 后再 `np.concatenate()` 写文件。

`demo_base.py` 和 `demo.py` 共用 `PerfTracker` / `PerfKey` 管理性能统计。Base 的首音延时、播放空挡和各性能指标含义统一见 4.3。
