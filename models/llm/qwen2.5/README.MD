# Qwen2.5(通义千问2.5)

本示例展示如何把Qwen2.5模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 Qwen2 实现来源于[https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。

## 2 快速开始

目前Qwen2模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen2.5-7b-instruct-hf目录：

```bash
python3 get_model.py --type raw
python3 get_model.py --type dataset
```

如果仅编译芯片模型，可选择下载量化模型，下载后的模型存放在`output/$HOUMO_TARGET/hmquant`目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type quant
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

也可以直接下载以上所有模型和数据集（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py
```

### 2.2 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`/hmdd/models/llm/qwen2.5`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为128和4096，可通过ptq.py中的prefill_shape和cache_len参数修改。不同编译模式下支持的上下文长度组合为：
| cores | batch | prefill | decode |
| ----- | ----- | ------- | ------ |
| 4     | 1     | 256     | 2048   |
| 4     | 1     | 128     | 4096   |
| 2     | 1     | 128     | 2048   |
| 2     | 1     | 64      | 4096   |
| 4     | 4     | 128     | 2048   |

```bash
source env.sh
cd /hmdd/models/llm/qwen2.5
python3 ptq.py
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- decoder
|-- prefill
|-- quant_embedding.pt
`-- weight.npy
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
qwen2.5_prefill.hmm
qwen2.5_decode.hmm

### 2.3 演示

Qwen2模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过demo中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
存算一体技术，即存储计算一体化技术，是一种旨在提高计算效率和能效的技术。它通过将存储器和计算单元集成在一起，使得数据在进行计算时直接在存储器中完成处理，从而减少了数据在存储器和计算单元之间传输的时间和功耗。这种技术特别适用于机器学习和人工智能领域，因为它可以显著提高处理速度和降低能耗。以下是存算一体技术的一些主要优势：

1. **减少功耗**：通过减少数据在存储器和计算单元之间传输的时间，可以显著降低能耗。这对于移动设备和边缘设备尤为重要。

2. **提高处理速度**：由于减少了数据传输的时间，存算一体技术可以显著提高处理速度，特别是在处理大量数据时。

3. **提高能效比**：通过减少功耗，可以提高系统的能效比，这对于需要长时间运行的设备尤为重要。

4. **适用于机器学习**：存算一体技术特别适用于机器学习和人工智能领域，因为它可以显著提高处理速度和降低能耗，从而使得这些技术在移动设备和边缘设备上得到更广泛的应用。

5. **适用于大规模数据处理**：对于需要处理大量数据的应用，存算一体技术可以显著提高处理速度，从而提高系统的效率。

需要注意的是，存算一体技术仍处于研究和开发阶段，目前还存在一些挑战，例如如何实现高效的存储器和计算单元之间的集成，如何实现高效的计算算法等。
```

### 3.2 性能结果

```bash
total: 302 tokens, cost 35.182 s
prefill time: 370.612 ms, 2.70 tokens/s
decode average time: 115.651 ms, 8.65 tokens/s
end2end average time: 116.498 ms, 8.58 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
