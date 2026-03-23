# whisper-large-v3-turbo

本示例展示如何把whisper-large-v3-turbo模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: whisper-large-v3-turbo
- 来源: [https://modelscope.cn/models/openai-mirror/whisper-large-v3-turbo](https://modelscope.cn/models/openai-mirror/whisper-large-v3-turbo)
- 许可: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前Whisper-Turbo模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```
如果需要自行编译模型，可以下载原始模式，下载方式如下：
```bash
python3 get_model.py --type raw
```

### 2.2 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/asr/whisper-turbo`目录，执行ptq.py脚本。默认的量化方式为w8a8_sefp。

```bash
source env.sh
cd models/asr/whisper-turbo
python3 ptq.py
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
whisper_encoder.hmm
whisper_decoder.hmm
whisper_prefill.hmm

### 2.4 演示

Whisper-Turbo模型使用python API进行演示

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.5 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
厚魔星片三马步的小伙伴你们好呀我是多莱莫你们好厉害呀好像你们也有装满各种神奇道具的可待你们真棒新的一年祝你们身体健康事业顺利发大财
```

### 3.2 性能结果

```bash
Output 65 tokens, Decode Cost 430.304 ms
Decode Speed: 151.06 tokens/s
TTFT (Time to First Token): 595.871 ms
TPOT (Time Per Output Token): 6.620 ms/token
E2E Latency (End-to-End Latency): 1.026 seconds
E2E TPS (End-to-End Tokens Per Second): 63.34 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
