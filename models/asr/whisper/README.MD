# Whisper ASR

本示例展示如何把Whisper模型量化和编译，部署到后摩芯片的设备上。支持以下模型：

- **whisper-medium**：标准模型，80 mel bins
- **whisper-large-v3-turbo**：大型turbo模型，128 mel bins

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

| 模型名称 | 来源 | 许可 |
|---------|------|------|
| whisper-medium | [ModelScope](https://modelscope.cn/models/openai-mirror/whisper-medium) | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) |
| whisper-large-v3-turbo | [ModelScope](https://modelscope.cn/models/openai-mirror/whisper-large-v3-turbo) | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) |

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前Whisper-Medium模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型，以及模型类型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
# 下载 whisper-medium 芯片模型
python3 get_model.py --type hmm --model whisper-medium

# 下载 whisper-large-v3-turbo 芯片模型
python3 get_model.py --type hmm --model whisper-large-v3-turbo
```

如果需要自行编译模型，可以下载原始模型：
```bash
python3 get_model.py --type raw --model whisper-medium
python3 get_model.py --type raw --model whisper-large-v3-turbo
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash /bin/bash
```

进入docker后source环境变量，并进入`models/asr/whisper`目录，执行ptq.py脚本。默认的量化方式为w8a8_sefp。

```bash
source env.sh
cd models/asr/whisper

# 量化 whisper-medium
python3 ptq.py --model whisper-medium

# 量化 whisper-large-v3-turbo
python3 ptq.py --model whisper-large-v3-turbo
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
# 编译 whisper-medium
python3 build.py --model whisper-medium

# 编译 whisper-large-v3-turbo
python3 build.py --model whisper-large-v3-turbo
```

编译好的模型存放在 `output/xh2/` 目录，包括3个文件：
- `{model}_encode.hmm`
- `{model}_decode.hmm`
- `{model}_prefill.hmm`

### 2.5 演示

Whisper模型使用python API和c++示例进行演示，c++模型示例的详细介绍见cpp/README.MD文件。

**Python演示：**

```bash
# 使用 whisper-medium
python3 demo.py --audio audio.mp3 --tokenizer_path whisper-medium

# 使用 whisper-large-v3-turbo
python3 demo.py --audio audio.mp3 --tokenizer_path whisper-large-v3-turbo
```

**C++演示：**

```bash
# 使用 whisper-medium
./bin/whisper-demo --audio_path audio.mp3 --model whisper-medium

# 使用 whisper-large-v3-turbo
./bin/whisper-demo --audio_path audio.mp3 --model whisper-large-v3-turbo
```

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 whisper-medium 演示结果

```bash
python demo.py --tokenizer_path whisper-medium --audio audio.mp3
Device 0 -> Group 0, Chip 0
2026-05-20 11:26:12.797 | INFO     | __main__:__init__:293 - encode model loaded
2026-05-20 11:26:15.285 | INFO     | __main__:__init__:296 - prefill model loaded
2026-05-20 11:26:15.468 | INFO     | __main__:__init__:299 - decode model loaded
2026-05-20 11:26:16.109 | SUCCESS  | __main__:<module>:665 - transcription:
2026-05-20 11:26:16.534 | INFO     | __main__:asr:452 - Detected language id: 50260
每一天都要快乐哦
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:677 - Output 8 tokens, Decode Cost 167.267 ms
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:680 - Decode Speed: 47.83 tokens/s
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:681 - TTFT (Time to First Token): 547.600 ms
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:682 - TPOT (Time Per Output Token): 20.908 ms/token
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:685 - E2E Latency (End-to-End Latency): 0.715 seconds
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:688 - E2E TPS (End-to-End Tokens Per Second): 11.19 tokens/s
2026-05-20 11:26:16.818 | SUCCESS  | __main__:<module>:693 - RTF (Real-Time Factor): 0.2940 (lower is better, <1 means real-time)
```

### 3.2 whisper-large-v3-turbo 演示结果

```bash
python demo.py --tokenizer_path whisper-large-v3-turbo
Device 0 -> Group 0, Chip 0
2026-05-20 11:29:40.088 | INFO     | __main__:__init__:293 - encode model loaded
2026-05-20 11:29:41.498 | INFO     | __main__:__init__:296 - prefill model loaded
2026-05-20 11:29:41.542 | INFO     | __main__:__init__:299 - decode model loaded
2026-05-20 11:29:42.186 | SUCCESS  | __main__:<module>:665 - transcription:
2026-05-20 11:29:42.574 | INFO     | __main__:asr:452 - Detected language id: 50260
每一天都要快乐哦
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:677 - Output 8 tokens, Decode Cost 46.815 ms
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:680 - Decode Speed: 170.88 tokens/s
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:681 - TTFT (Time to First Token): 413.543 ms
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:682 - TPOT (Time Per Output Token): 5.852 ms/token
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:685 - E2E Latency (End-to-End Latency): 0.460 seconds
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:688 - E2E TPS (End-to-End Tokens Per Second): 17.38 tokens/s
2026-05-20 11:29:42.650 | SUCCESS  | __main__:<module>:693 - RTF (Real-Time Factor): 0.1893 (lower is better, <1 means real-time)
```

### 3.3 性能结果

**whisper-medium:**
```bash
Output 8 tokens, Decode Cost 167.267 ms
Decode Speed: 47.83 tokens/s
TTFT (Time to First Token): 547.600 ms
TPOT (Time Per Output Token): 20.908 ms/token
E2E Latency (End-to-End Latency): 0.715 seconds
E2E TPS (End-to-End Tokens Per Second): 11.19 tokens/s
RTF (Real-Time Factor): 0.2940 (lower is better, <1 means real-time)
```

**whisper-large-v3-turbo:**
```bash
Output 8 tokens, Decode Cost 46.815 ms
Decode Speed: 170.88 tokens/s
TTFT (Time to First Token): 413.543 ms
TPOT (Time Per Output Token): 5.852 ms/token
E2E Latency (End-to-End Latency): 0.460 seconds
E2E TPS (End-to-End Tokens Per Second): 17.38 tokens/s
RTF (Real-Time Factor): 0.1893 (lower is better, <1 means real-time)
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
