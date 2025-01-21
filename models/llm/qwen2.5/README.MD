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

编译好的模型存放在当前目录，包括3个文件：
qwen2.5_prefill_body.hmm
qwen2.5_prefill_head.hmm
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
存算一体技术是一种将存储和计算功能集成在一起的架构，它在多个方面展现出显著的优势：

1. **性能提升**：存算一体架构通过将数据存储和处理操作在同一节点上进行，减少了数据传输的时间，从而显著提升了整体性能。这种架构减少了延迟，使得数据处理更加高效。

2. **成本效益**：通过减少硬件组件和网络需求，存算一体技术降低了总体拥有成本（TCO）。它通过整合存储和计算资源，减少了数据中心的占地面积和能源消耗，从而节省了基础设施成本。

3. **灵活性和可扩展性**：这种架构允许在存储和计算之间更自由地分配资源，可以根据应用需求进行动态调整。它支持水平扩展，即通过增加更多的存储或计算节点来增加整体容量和性能，而不需要重新设计整个系统。

4. **简化管理**：存算一体技术简化了系统管理，因为管理和维护的组件数量减少了。这降低了管理复杂度，提高了运维效率。

5. **数据安全**：在某些存算一体解决方案中，数据在存储和处理过程中保持在本地，减少了数据在网络传输中的风险，增强了数据保护和隐私。

6. **适应性**：存算一体架构能够更好地适应各种工作负载，无论是大数据分析、机器学习、实时处理还是其他计算密集型任务，都能提供高性能和高效率。

7. **能源效率**：通过减少数据移动和优化资源使用，存算一体技术提高了能源效率，有助于实现绿色数据中心的目标。

总之，存算一体技术通过整合存储和计算资源，提供了高性能、低成本、灵活可扩展、易于管理、安全可靠和能源效率高的解决方案，是现代数据中心和云计算环境中的重要技术趋势。
```

### 3.2 性能结果

```bash
total: 363 tokens, cost 47.154 s
prefill time: 288.581 ms, 3.47 tokens/s
decode average time: 129.447 ms, 7.73 tokens/s
end2end average time: 129.900 ms, 7.70 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
