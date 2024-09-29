# Qwen mix(通义千问混合量化版)

本示例展示如何把Qwen模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 Qwen1.5 实现来源于[https://huggingface.co/Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)。

## 2 快速开始

目前Qwen模型仅提供python脚本编译方式。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或量化模型。

如果需要自己量化，可选择下载原始模型：

```bash
python3 get_model.py --type raw
```

如果仅编译芯片模型，可选择下载量化模型：

```bash
python3 get_model.py --type quant
```

### 2.2 量化（暂不支持）

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py38-x86.64 /bin/bash
```

进入docker后source环境变量，并进入/hmdd/models/llm/qwen_mix_multibatches目录，执行ptq.sh脚本。（注：可通过修改脚本中的MODEL_DIR修改模型位置，运行中需要连接网络以下载wikitext2数据集，如无法连接可打开注释下载wikitext2_cache.zip）

```bash
source env.sh
cd /hmdd/models/llm/qwen_mix_multibatches
bash ptq.sh
```

编译完成的模型存放在output/$HOUMO_TARGET/result目录：

```bash
.
|-- decoder
|-- decoder_head
|-- prefill
|-- prefill_head
|-- quant_embedding.pt
`-- weight.npy
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

### 2.3 演示

Qwen模型使用python API进行演示，同时输出性能结果。执行脚本：

```bash
python3 demo.py
```

### 2.4 一键评估

以上步骤可以通过perf.sh脚本一键执行：

```bash
bash perf.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
[session1]:
question:
请介绍一下存算一体技术的优势。
response:
存算一体技术是一种新兴的计算技术，它将存储和计算功能集成在一起，以提高计算效率和性能。以下是存算一体技术的一些优势：

1. 提高性能：存算一体技术通过将内存和处理器集成在同一块芯片上，减少了数据传输时间。相比于传统的CPU和内存分开设计，数据可以在处理器内部快速访问，减少了延迟，从而提高了计算速度。
...

total: 363 tokens, cost 76.478 s
prefill time: 883.224 ms, 1.13 tokens/s
decode average time: 203.654 ms, 4.91 tokens/s
end2end average time: 210.684 ms, 4.75 tokens/s

[session2]:
question:
你是谁？
response:
我是阿里云推出的一种超大规模语言模型，我叫通义千问。
total: 17 tokens, cost 5.749 s
prefill time: 309.794 ms, 3.23 tokens/s
decode average time: 213.097 ms, 4.69 tokens/s
end2end average time: 338.174 ms, 2.96 tokens/s

[session3]:
question:
1+1=?
response:
1+1=2
total: 5 tokens, cost 3.226 s
prefill time: 307.919 ms, 3.25 tokens/s
decode average time: 250.056 ms, 4.00 tokens/s
end2end average time: 645.260 ms, 1.55 tokens/s

[session4]:
question:
Introduce yourself in 200 words.
response:
Hello! I'm an AI language model, so I don't have a personal identity in the traditional sense. However, I'm designed to assist and communicate with humans. My name is Qwen, short for Question-Answering Wenfei, and I'm part of Alibaba Cloud, a leading technology company based in China.
...

total: 185 tokens, cost 40.682 s
prefill time: 307.939 ms, 3.25 tokens/s
decode average time: 204.404 ms, 4.89 tokens/s
end2end average time: 219.901 ms, 4.55 tokens/s

[session5]:
question:
人类的本质是什么?
response:
人类的本质是一个复杂且多维的问题，涉及到哲学、心理学、生物学、社会学等多个领域。从不同的角度和理论，人类的本质可以有不同的解释。
...

total: 347 tokens, cost 74.624 s
prefill time: 306.883 ms, 3.26 tokens/s
decode average time: 204.317 ms, 4.89 tokens/s
end2end average time: 215.055 ms, 4.65 tokens/s

[session6]:
question:
意识是大脑的产物还是独立于大脑存在的实体?
response:
这是一个哲学和神经科学领域长期争论的问题，目前并没有一个明确的答案。不同的观点和理论提出了不同的解释。
...

total: 177 tokens, cost 42.118 s
prefill time: 309.417 ms, 3.23 tokens/s
decode average time: 204.346 ms, 4.89 tokens/s
end2end average time: 237.952 ms, 4.20 tokens/s

```

### 3.2 性能结果

```bash
[summary]:
total: 1458 tokens, cost 76.450 s
prefill average time: 404.196 ms, 2.47 tokens/s
decode average time: 50.982 ms, 19.61 tokens/s
end2end average time: 52.435 ms, 19.07 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
