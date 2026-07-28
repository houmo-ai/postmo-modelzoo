# CosyVoice3 TTS C++ 推理示例

本示例展示如何使用C++在后摩系列芯片（XH2）的设备上运行CosyVoice3文本转语音（TTS）模型推理。当前支持Linux_x86_64、Android Native 和 OpenHarmony (OHOS) 环境，**Linux aarch64平台需要用户自行编译相关三方库**。

## 目录

- [CosyVoice3 TTS C++ 推理示例](#cosyvoice3-tts-c-推理示例)
  - [目录](#目录)
  - [1.工具说明](#1工具说明)
    - [1.1 文件及功能说明](#11-文件及功能说明)
    - [1.2 流程说明](#12-流程说明)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 用法说明](#14-用法说明)
  - [2.快速开始](#2快速开始)
    - [2.1 环境准备](#21-环境准备)
    - [2.2 编译](#22-编译)
    - [2.3 运行](#23-运行)
  - [3.参考结果](#3参考结果)
  - [4.免责声明](#4免责声明)

## 1.工具说明

### 1.1 文件及功能说明

文件目录如下：

| 文件名称 | 说明 |
| --- | --- |
| demo.cc | 主入口，拉起CosyVoice3 TTS推理流程 |
| hm_cosyvoice3.cc | 主文件，实现CosyVoice3完整推理流程 |
| hm_audio.cc | 音频预处理类实现（加载、重采样、mel频谱图） |
| hm_embedding.cc | Embedding层实现，用于token到向量转换 |
| hm_text_frontend.cc | 文本前端模块（规范化、分词、段落分割） |
| hm_speech_tokenizer.cc | 语音分词器，从音频提取语音token |
| hm_speaker_embedding.cc | 说话人嵌入提取，使用CampPlus模型 |
| hm_sampling.cc | 采样策略实现（nucleus采样、RAS采样） |
| hm_native_llm.cc | LLM模块，从文本生成语音token |
| hm_native_flow.cc | Flow解码器，将语音token转换为mel频谱图 |
| hm_native_vocoder.cc | HiFT声码器，将mel频谱图转换为音频波形 |
| hm_perf_monitor.cc | 性能监控和报告模块 |
| miniaudio_impl.cc | miniaudio库实现入口 |
| CMakeLists.txt | CMake编译配置文件 |
| build.sh | Linux平台编译脚本 |

功能支持度说明：
- 支持的芯片型号为：xh2
- 支持的大道软件版本要求：xh2≥v1.0.0
- 支持零样本语音克隆
- 支持跨语言合成
- 支持指令跟随合成
- 支持mel频谱图预处理（24kHz下80-bin，16kHz下128-bin）
- 支持FBANK特征提取（16kHz下80-bin）

### 1.2 流程说明

程序按以下流程进行：

1. 解析命令行参数，加载模型配置
2. 初始化文本前端（分词器）
3. 初始化音频处理器
4. 加载所有模型（语音分词器、说话人嵌入、LLM、Flow、声码器）
5. 从提示音频提取特征：
   - 语音token（用于LLM条件）
   - 说话人嵌入（用于声音克隆）
   - mel特征（用于Flow条件）
6. 将文本编码为token ID
7. 运行LLM推理生成语音token
8. 运行Flow解码器将语音token转换为mel频谱图
9. 运行HiFT声码器将mel频谱图转换为音频波形
10. 保存生成的音频到WAV文件

性能指标：

| 性能指标 | 说明 |
| --- | --- |
| LLM Total Cost | LLM总耗时，单位（ms） |
| LLM Prefill Speed | Prefill阶段速度，单位（tokens/s） |
| TTFT (Time to First Token) | 从输入到输出第一个token的耗时，单位（ms） |
| TPOT (Time Per Output Token) | Decode阶段平均每token耗时，单位（ms/token） |
| TTS Total Cost | Flow + Vocoder总耗时，单位（ms） |
| TTS RTF (Real-Time Factor) | 实时率（处理时间/音频时长），<1表示实时处理 |
| E2E Latency | 端到端总时间，单位（s） |

### 1.3 参数说明

命令行参数说明：

| 参数 | 说明 |
| --- | --- |
| --tokenizer_dir <path> | 分词器目录路径，默认：./Fun-CosyVoice3-0.5B-2512 |
| --model_dir <path> | 模型目录路径（.hmm文件），默认：./output/xh2 |
| --embedding_dir <path> | 嵌入目录路径（.hemm/.bin文件），默认：./output/xh2/hmquant |
| --output_dir <path> | 输出目录（生成的音频），默认：./results-cpp |
| --device_id <id> | TCIM设备ID，默认：0 |
| --help, -h | 打印帮助信息 |

### 1.4 用法说明

基本用法示例（在cosyvoice3目录下执行）：

```bash
# 基本用法
./bin/cosyvoice3-demo --tokenizer_dir ./Fun-CosyVoice3-0.5B-2512

# 指定模型路径
./bin/cosyvoice3-demo \
    --tokenizer_dir ./Fun-CosyVoice3-0.5B-2512 \
    --model_dir ./output/xh2 \
    --embedding_dir ./output/xh2/hmquant \
    --output_dir ./results-cpp

# 自定义输出目录
./bin/cosyvoice3-demo --output_dir ./my_output
```

## 2.快速开始

### 2.1 环境准备

进入houmo-examples根目录，检查并设置环境变量：

```bash
source env.sh
```

需要设置的环境变量：
- `HOUMO_TARGET`: 目标芯片类型，设置为 `xh2`
- `TCIM_RUNTIME_PATH`: TCIM Runtime库路径（必需）

对于Android平台交叉编译，还需设置：
- `NDK_PATH`: Android NDK路径

### 2.2 编译

**Linux平台编译：**

```bash
cd ${HOUMO_EXAMPLES_PATH}/models/tts/cosyvoice3/cpp
./build.sh
```

编译生成的可执行文件安装到 `../bin` 目录下。

**Android平台编译：**

```bash
# 设置NDK路径
export NDK_PATH=/path/to/android-ndk-r28c
export HOUMO_SDK_PATH=/path/to/houmo-drv-xh2
export TCIM_RUNTIME_PATH=/path/to/houmo-tcim-runtime-xh2

# 编译
./build_ndk.sh
```

**OpenHarmony (OHOS) 平台交叉编译：**

```bash
# 设置 OHOS SDK 和相关环境变量
export OHOS_SDK=/path/to/OpenHarmony/release/6.0-Release/linux/native
export HOUMO_TARGET=xh2
export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
export TCIM_RUNTIME_PATH=/path/to/houmo-sdk

# 方式一：使用预编译的三方音频库（需提前准备好 OHOS 版本的 .so 文件）
./build_OHOS.sh

# 方式二：从源码编译三方音频库（推荐，一步到位）
./build_OHOS.sh --audio-source
```

编译生成的可执行文件及依赖库安装到 `../bin` 目录下，目录结构如下：

```
../bin/
├── cosyvoice3-demo                     # 主程序
├── libkaldi-native-fbank-core.so       # FBANK 特征提取库
├── libsamplerate.so                    # 音频重采样库
├── libsndfile.so                       # 音频文件 I/O 库
├── libtcim_runtime_lite.so             # TCIM Runtime 库（需单独部署）
```

### 2.3 运行

确保以下文件已准备好：
- 分词器：`Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/tokenizer.json`
- 模型：output/xh2目录下的所有.hmm文件
- 嵌入：output/xh2/hmquant目录下的所有.bin文件
- 提示音频：`zero_shot_prompt.wav`（用于声音克隆演示）

在cosyvoice3目录下执行：

```bash
cd ${HOUMO_EXAMPLES_PATH}/models/tts/cosyvoice3

# 设置环境变量
export HOUMO_TARGET=xh2
export TCIM_BACKEND=Xh2HalBackend

# 运行推理
./bin/cosyvoice3-demo \
    --tokenizer_dir ./Fun-CosyVoice3-0.5B-2512 \
    --model_dir ./output/xh2 \
    --embedding_dir ./output/xh2/hmquant \
    --output_dir ./results-cpp
```

## 3.参考结果

预期输出：

```
==============================================
  CosyVoice3 TTS Demo - C++ Implementation
==============================================

--- Configuration ---
Tokenizer Dir: ./Fun-CosyVoice3-0.5B-2512
Model Dir: ./output/xh2
Embedding Dir: ./output/xh2/hmquant
Output Dir: ./results-cpp
Device ID: 0
Sample Rate: 24000
----------------------

--- Initializing CosyVoice3 ---
[HmTextFrontend] Tokenizer loaded from JSON: ./Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/tokenizer.json
[HmTextFrontend] Vocabulary size: 151924
[INFO] vocab_size: 151936, embedding_length: 896, prefill_length: 256
[INFO] vocab_size: 6761, embedding_length: 896, prefill_length: 6761
[INFO] vocab_size: 6561, embedding_length: 80, prefill_length: 1024

CosyVoice3 initialized successfully!

========================================
Demo 1: Zero-Shot Voice Cloning
========================================

=== Zero-Shot Voice Cloning ===
TTS Text: 下面为您朗诵一段绕口令...
Prompt WAV: ./zero_shot_prompt.wav
Speed: 1
Audio loaded: ./zero_shot_prompt.wav, duration: 3.48s
1 rounds of prefill needed for total sequence length 169

=== Zero-Shot Complete ===
Total audio samples: 347520
Audio duration: 14.48 seconds
RTF: 1.21737

=== Performance Summary ===
LLM: 4443.7 ms (prefill: 30.15 ms / 169 tokens, decode: 4409.79 ms / 362 tokens, TTFT: 33.91 ms)
Speaker Emb: 23.16 ms
Speech Tokenizer: 132.56 ms
Flow: 12762.7 ms (encoder: 4.05 ms, decoder: 12747.7 ms)
Vocoder: 421.18 ms
E2E Latency: 17.63 s
Audio Duration: 14.48 s
RTF: 1.21737
===========================

========================================
Demo 2: Cross-Lingual Synthesis
========================================

=== Cross-Lingual Synthesis ===
...
1 rounds of prefill needed for total sequence length 48

=== Performance Summary ===
LLM: 2491.06 ms (prefill: 28.52 ms / 48 tokens, decode: 2458.92 ms / 229 tokens, TTFT: 32.13 ms)
Flow: 12744.7 ms
Vocoder: 404.24 ms
E2E Latency: 15.64 s
Audio Duration: 9.16 s
RTF: 1.70742
===========================

========================================
Demo 3: Instruct2 Synthesis
========================================

=== Instruct2 Synthesis ===
...
1 rounds of prefill needed for total sequence length 45

=== Performance Summary ===
LLM: 1496 ms (prefill: 28.28 ms / 45 tokens, decode: 1464.36 ms / 137 tokens, TTFT: 31.63 ms)
Flow: 12748.7 ms
Vocoder: 403.64 ms
E2E Latency: 14.65 s
Audio Duration: 5.48 s
RTF: 2.67304
===========================

========================================
Overall Performance Summary
========================================

--- Aggregated Statistics ---
Average RTF: 1.87 (min: 1.22, max: 2.67)
Average LLM time: 2810.25 ms (min: 1496 ms, max: 4443.7 ms)
Average Flow time: 12752 ms
Average Vocoder time: 409.69 ms
Total audio generated: 29.12 s
Average Prefill Speed: 3001.91 tokens/s
Average Decode Speed: 87.12 tokens/s
============================

Saved audio files to: ./results-cpp/
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

本项目使用的第三方组件：
- tokenizers-cpp：HuggingFace分词器C++绑定
- kaldi-native-fbank：Kaldi FBANK特征提取C++实现
- LibrosaCpp：Librosa风格mel频谱图计算C++实现
- miniaudio：音频文件读取库
- libsamplerate：音频重采样库
- libsndfile：音频文件I/O库
