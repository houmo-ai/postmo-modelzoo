# Qwen2(通义千问2)

本示例展示如何把Qwen2模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 Qwen2 实现来源于[https://huggingface.co/Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)。

## 2 快速开始

目前Qwen2模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen2-7b-instruct-hf目录：

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

进入docker后source环境变量，并进入`/hmdd/models/llm/qwen2`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为128和4096，可通过ptq.py中的prefill_shape和cache_len参数修改。不同编译模式下支持的上下文长度组合为：
| cores | batch | prefill | decode |
| ----- | ----- | ------- | ------ |
| 4     | 1     | 256     | 2048   |
| 4     | 1     | 128     | 4096   |
| 2     | 1     | 128     | 2048   |
| 2     | 1     | 64      | 4096   |
| 4     | 4     | 128     | 2048   |

```bash
source env.sh
cd /hmdd/models/llm/qwen2
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
qwen2_prefill.hmm
qwen2_decode.hmm

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
存算一体技术是一种将存储和计算功能集成在一起的计算技术，它在多个领域中具有显著的优势，特别是在人工智能、大数据处理、物联网、高性能计算和边缘计算等领域。以下是存算一体技术的一些主要优势：

1. **性能提升**：存算一体技术通过减少数据在存储和计算之间的传输，减少了延迟，提高了计算性能。在某些应用场景中，这可以显著提升系统的整体性能。

2. **能效提升**：通过减少数据传输和处理的能耗，存算一体技术可以显著降低系统的能耗，这对于需要长时间运行的系统尤为重要，比如在边缘计算和物联网设备中。

3. **成本降低**：在一些特定的场景下，存算一体技术可以减少对高速缓存、内存和存储的需求，从而降低系统的整体成本。

4. **数据安全性增强**：在边缘计算环境中，存算一体技术可以减少数据在网络中的传输，从而提高数据的安全性，减少数据泄露的风险。

5. **实时处理能力增强**：存算一体技术特别适合实时处理大量数据的场景，如实时分析、实时决策等，能够提供更快的响应时间和更高的处理效率。

6. **灵活性和可扩展性**：存算一体技术允许系统在不同的硬件平台上进行灵活部署，同时提供更好的可扩展性，能够根据需求进行动态调整。

7. **适应性强**：存算一体技术可以应用于多种不同的应用场景，包括但不限于人工智能、机器学习、大数据分析、图像处理、信号处理等，具有广泛的适应性。

存算一体技术的优势主要体现在提升性能、降低成本、增强安全性、提高实时处理能力、增强灵活性和可扩展性等方面，这些优势使得存算一体技术在多个领域中具有广泛的应用前景。
```

### 3.2 性能结果

```bash
total: 367 tokens, cost 42.672 s
prefill time: 361.431 ms, 2.77 tokens/s
decode average time: 115.599 ms, 8.65 tokens/s
end2end average time: 116.272 ms, 8.60 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
