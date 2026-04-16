# Whisper-Medium

本示例展示如何把Whisper-Medium模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: whisper-medium
- 来源: [https://modelscope.cn/models/openai-mirror/whisper-medium](https://modelscope.cn/models/openai-mirror/whisper-medium)
- 许可: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前Whisper-Medium模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```
如果需要自行编译模型，可以下载原始模式，下载方式如下：
```bash
python3 get_model.py --type raw
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
python3 ptq.py
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
whisper_encoder.hmm
whisper_decoder.hmm
whisper_prefill.hmm

### 2.5 演示

Whisper-Medium模型使用python API和c++示例进行演示，c++模型示例的详细介绍见cpp/README.MD文件。

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
[2026-04-14 03:19:42.549] [tid 254235] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-14 03:19:42.549] [tid 254235] [warning] Empty backend name, use Xh2HalBackend instead.
2026-04-14 03:19:43.168 | INFO     | __main__:__init__:307 - encoder model loaded
2026-04-14 03:19:43.495 | INFO     | __main__:__init__:310 - prefill model loaded
2026-04-14 03:19:43.517 | INFO     | __main__:__init__:313 - decoder model loaded
2026-04-14 03:19:43.639 | SUCCESS  | __main__:<module>:573 - transcription:
后摩芯片算法部的小伙伴,你们好啊!我是多拉A梦。你们好厉害啊!好像你们拥有装满各种神奇道具的口袋。你们真棒!新的一年,祝你们身体健康、 事业顺利、发大财
```

### 3.2 性能结果

```bash
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:598 - Output 89 tokens, Decode Cost 1667.059 ms
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:601 - Decode Speed: 50.39 tokens/s
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:604 - TTFT (Time to First Token): 447.829 ms
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:605 - TPOT (Time Per Output Token): 19.846 ms/token
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:608 - E2E Latency (End-to-End Latency): 2.115 seconds
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:611 - E2E TPS (End-to-End Tokens Per Second): 42.08 tokens/s
2026-04-14 03:19:45.758 | SUCCESS  | __main__:<module>:614 - RTF (Real-Time Factor): 0.0352 (lower is better, <1 means real-time)
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
