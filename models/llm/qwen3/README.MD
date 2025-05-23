# Qwen3(通义千问3)

本示例展示如何把Qwen3模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 Qwen2 实现来源于[https://modelscope.cn/models/Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen3-8b目录：

```bash
python3 get_model.py --type raw
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

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`/hmdd/models/llm/qwen3`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为128和4096，可通过ptq.py中的prefill_shape和cache_len参数修改。不同编译模式下支持的上下文长度组合为：
| cores | batch | prefill | decode |
| ----- | ----- | ------- | ------ |
| 4     | 1     | 256     | 8192   |
| 4     | 1     | 256     | 4096   |
| 2     | 1     | 256     | 4096   |

```bash
source env.sh
cd /hmdd/models/llm/qwen3
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
qwen3_prefill.hmm
qwen3_decode.hmm

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
存算一体技术是一种将存储器和计算单元结合在一起的技术，它能够显著提高计算效率，降低能耗。以下是存算一体技术的一些主要优势：

1. **降低能耗**：传统的计算架构中，数据在计算单元和存储器之间频繁传输，这会导致大量的能耗。而存算一体技术通过将存储器和计算单元结合在一起，减少了数据传输的能耗，从而降低了整体能耗。

2. **提高计算效率**：在传统的计算架构中，数据需要从存储器传输到计算单元进行处理，然后再传输回存储器。这种频繁的数据传输会消耗大量的时间和能量。而在存算一体技术中，存储和计算可以同时进行，大大提高了计算效率。

3. **简化系统设计**：通过将存储器和计算单元结合在一起，可以简化整个系统的硬件设计，减少硬件复杂度，使得系统更容易实现和维护。

4. **适用于新型计算需求**：随着大数据、人工智能等领域的快速发展，对于计算能力的需求也在不断提高。存算一体技术能够更好地满足这些新型计算需求，提供更高效的数据处理能力。

5. **促进新型存储器技术的发展**：存算一体技术的发展促进了新型存储器技术的发展，这些新型存储器不仅具有高密度存储能力，还具有快速读写速度和低能耗的特点。

总之，存算一体技术通过将存储和计算功能结合在一起，不仅能够提高计算效率和降低能耗，还能够更好地满足新型计算需求，是未来计算技术发展的重要方向之一。
```

### 3.2 性能结果

```bash
total: 316 tokens, cost 24.357 s
prefill time: 561.781 ms, 1.78 tokens/s
decode average time: 75.536 ms, 13.24 tokens/s
end2end average time: 77.078 ms, 12.97 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
