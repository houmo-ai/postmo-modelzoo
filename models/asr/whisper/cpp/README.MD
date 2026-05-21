# Whisper ASR C++ 推理示例

本示例展示如何使用C++在后摩系列芯片（XH2）的设备上运行Whisper自动语音识别（ASR）模型推理。当前支持Linux和Android Native环境。

## 目录

- [Whisper ASR C++ 推理示例](#whisper-asr-c-推理示例)
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

| 文件名称                  | 说明                                              |
| ------------------------- | ------------------------------------------------- |
| whisper_demo.cpp          | 主入口，拉起Whisper ASR推理流程                    |
| hm_whisper_infer.cpp      | 主文件，实现Whisper完整推理流程                    |
| hm_whisper_infer.hpp      | HmWhisperInfer类头文件                           |
| hm_whisper_audio.cpp      | 音频预处理类实现                                  |
| hm_whisper_audio.hpp      | 音频预处理类头文件                                |
| CMakeLists.txt            | CMake编译配置文件                                 |
| build.sh                  | Linux平台编译脚本                                 |
| build_ndk.sh              | Android平台编译脚本                               |
| tcim_runtime.cmake        | TCIM Runtime库查找配置                            |
| export_mel_filters.py     | 将asr模型的mel_filters参数转换为.h文件，方便cpp读取   |

功能支持度说明：
- 支持的芯片型号为：xh2。
- 支持的大道软件版本要求：xh2≥v1.0.0。
- 支持的模型：whisper-medium（80 mel bins）、whisper-large-v3-turbo（128 mel bins）。
- 支持长音频分块处理。
- 支持多语言自动识别。
- 支持mel频谱图预处理。
- **目前c++示例仅支持16kHz采样率音频输入。**

### 1.2 流程说明

程序按以下流程进行：

1. 解析命令行参数，加载模型配置
2. 初始化音频处理器
3. 加载Encode、Prefill、Decode三个模型
4. 加载音频文件并进行预处理（目前仅支持16kHz采样音频输入，计算mel频谱图）
5. 将长音频分块处理
6. 运行Encode模型处理音频特征
7. 检测语言类型
8. 运行Prefill模型处理初始token序列
9. 循环运行Decode模型生成输出token，直到生成结束token
10. 对输出token进行解码，返回最终识别文本

性能指标：
| 性能指标                                | 说明                                           |
| --------------------------------------- | ---------------------------------------------- |
| Decode Cost                             | Decode阶段总耗时，单位（ms）                   |
| Decode Speed                            | Decode阶段速度，单位（tokens/s）               |
| TTFT (Time to First Token)              | 从输入到输出第一个token的耗时，单位（ms）      |
| TPOT (Time Per Output Token)            | Decode阶段平均每token耗时，单位（ms/token）    |
| E2E Latency (End-to-End Latency)        | 端到端总时间，单位（s）                        |
| E2E TPS (End-to-End Tokens Per Second)  | 端到端每秒输出的token数，单位（tokens/s）      |
| RTF (Real-Time Factor)                  | 实时率（处理时间/音频时长），<1表示实时处理    |

### 1.3 参数说明

命令行参数说明：

| 参数                    | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| --audio_path <path>     | 输入音频文件路径，默认：audio.mp3                            |
| --model <name>          | 模型名称，支持：whisper-medium、whisper-large-v3-turbo，默认：whisper-medium |
| --encode_path <path>    | Encode模型路径（指定后将覆盖--model的默认路径）              |
| --decode_path <path>    | Decode模型路径（指定后将覆盖--model的默认路径）              |
| --prefill_path <path>   | Prefill模型路径（指定后将覆盖--model的默认路径）             |
| --tokenizer_path <path> | tokenizer.json路径（指定后将覆盖--model的默认路径）          |
| --chunk_size <n>        | 音频分块大小，单位秒，默认：30                               |
| --language <code|auto>  | 语言代码（如zh、en）或auto自动检测，默认：auto               |
| -h, --help              | 打印帮助信息                                                 |

### 1.4 用法说明

基本用法示例：

```bash
# 使用默认模型（whisper-medium）
./whisper-demo --audio_path audio.mp3

# 指定模型名称
./whisper-demo --audio_path audio.mp3 --model whisper-large-v3-turbo

# 指定语言
./whisper-demo --audio_path audio.mp3 --model whisper-large-v3-turbo --language zh

# 手动指定模型路径（覆盖--model的默认路径）
./whisper-demo --audio_path audio.mp3 \
    --encode_path models/whisper-medium_encode.hmm \
    --decode_path models/whisper-medium_decode.hmm \
    --prefill_path models/whisper-medium_prefill.hmm

# 自定义分块大小
./whisper-demo --audio_path audio.mp3 --chunk_size 10
```

## 2.快速开始

### 2.1 环境准备

进入houmo-examples根目录，检查并设置环境变量：

```bash
source env.sh
```

需要设置的环境变量：
- `HOUMO_TARGET`: 目标芯片类型，设置为 `xh2`
- `HOUMO_PATH`: Houmo SDK路径，包含include和lib目录
- `TCIM_RUNTIME_PATH`: TCIM Runtime库路径（可选，默认使用相对路径）

对于Android平台交叉编译，还需设置：
- `NDK_PATH`: Android NDK路径

### 2.2 编译

**Linux平台编译：**

```bash
cd /hmdd/imodelzoo/models/asr/whisper/cpp
./build.sh
```

编译生成的可执行文件在 `build/` 目录下。

**Android平台编译：**

```bash
# 设置NDK路径
export NDK_PATH=/path/to/android-ndk-r28c
export HOUMO_SDK_PATH=/path/to/houmo-drv-xh2
export TCIM_RUNTIME_PATH=/path/to/houmo-tcim-runtime-xh2

./build_ndk.sh
```

编译生成的可执行文件在 `build/` 目录下。


### 2.3 运行

确保以下文件已准备好：
- Encode模型：`output/xh2/{model}_encode.hmm`
- Decode模型：`output/xh2/{model}_decode.hmm`
- Prefill模型：`output/xh2/{model}_prefill.hmm`
- tokenizer.json：`{model}/tokenizer.json`

支持的模型：
- `whisper-medium`：标准模型，80 mel bins
- `whisper-large-v3-turbo`：大型turbo模型，128 mel bins

```bash
# 设置环境变量
export HOUMO_TARGET=xh2
export TCIM_BACKEND=Xh2HalBackend

# 使用 whisper-medium 模型（默认）
./build/whisper-demo --audio_path audio.mp3

# 使用 whisper-large-v3-turbo 模型
./build/whisper-demo --audio_path audio.mp3 --model whisper-large-v3-turbo
```

## 3.参考结果

```bash
./bin/whisper-demo --audio_path audio.mp3 --model whisper-medium
[2026-05-20 11:30:10.413] [tid 342089] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-05-20 11:30:10.413] [tid 342089] [warning] Empty backend name, use Xh2HalBackend instead.
[INFO] encoder model loaded
[INFO] prefill model loaded
[INFO] decoder model loaded
[INFO] Using 80 mel bins for audio processing
Audio loaded: audio.mp3, channels: 1, sample rate: 16000Hz, duration: 2.43187s, samples: 38910
[INFO] Pre-computing mel spectrograms for 1 chunks...
[INFO] Mel pre-processing completed in 0.424247s
[SUCCESS] transcription:
每一天都要快乐哦
[SUCCESS] Output 9 tokens, Decode Cost 164.095 ms
[SUCCESS] Decode Speed: 54.85 tokens/s
[SUCCESS] TTFT (Time to First Token): 655.118 ms
[SUCCESS] TPOT (Time Per Output Token): 18.233 ms/token
[SUCCESS] E2E Latency (End-to-End Latency): 0.821 seconds
[SUCCESS] E2E TPS (End-to-End Tokens Per Second): 10.96 tokens/s
[SUCCESS] RTF (Real-Time Factor): 0.0137 (lower is better, <1 means real-time)

 ./bin/whisper-demo --audio_path audio.mp3 --model whisper-large-v3-turbo
[2026-05-20 11:30:43.690] [tid 342110] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-05-20 11:30:43.690] [tid 342110] [warning] Empty backend name, use Xh2HalBackend instead.
[INFO] encoder model loaded
[INFO] prefill model loaded
[INFO] decoder model loaded
[INFO] Using 128 mel bins for audio processing
Audio loaded: audio.mp3, channels: 1, sample rate: 16000Hz, duration: 2.43187s, samples: 38910
[INFO] Pre-computing mel spectrograms for 1 chunks...
[INFO] Mel pre-processing completed in 0.424463s
[SUCCESS] transcription:
每一天都要快乐哦
[SUCCESS] Output 9 tokens, Decode Cost 56.494 ms
[SUCCESS] Decode Speed: 159.31 tokens/s
[SUCCESS] TTFT (Time to First Token): 455.209 ms
[SUCCESS] TPOT (Time Per Output Token): 6.277 ms/token
[SUCCESS] E2E Latency (End-to-End Latency): 0.513 seconds
[SUCCESS] E2E TPS (End-to-End Tokens Per Second): 17.54 tokens/s
[SUCCESS] RTF (Real-Time Factor): 0.0086 (lower is better, <1 means real-time)
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
