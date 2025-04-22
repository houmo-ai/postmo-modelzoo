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

进入docker后source环境变量，并进入`/hmdd/models/llm/qwen2`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为256和4096，可通过ptq.py中的prefill_shape和cache_len参数修改。不同编译模式下支持的上下文长度组合为：
| cores | batch | prefill | decode |
| ----- | ----- | ------- | ------ |
| 4     | 1     | 256     | 8192   |
| 4     | 1     | 256     | 4096   |
| 2     | 1     | 256     | 4096   |

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
存算一体技术是指将存储和计算功能集成在同一硬件平台上，以提高数据处理效率和降低系统成本的一种技术。这种技术的优势主要体现在以下几个方面：

1. **性能提升**：存算一体技术通过将数据存储和处理过程在同一硬件上进行，减少了数据传输延迟，提高了数据处理的效率和速度。这对于大数据处理、实时分析等应用场景尤为重要。

2. **成本优化**：通过将存储和计算功能集成，可以减少硬件组件的数量，从而降低整体系统的成本。同时，这种设计通常能更有效地利用资源，减少能源消耗，进一步降低成本。

3. **灵活性和扩展性**：存算一体架构允许在存储和计算资源之间进行灵活调整，以适应不同的工作负载需求。这为系统提供了更好的扩展性，可以根据业务需求灵活调整资源分配。

4. **安全性增强**：在某些应用场景中，存算一体技术可以提供更紧密的数据控制，减少数据在不同组件间传输的风险，从而提高数据安全性和隐私保护。

5. **低延迟**：由于数据存储和处理在同一硬件上进行，减少了数据在不同组件间传输的时间，可以实现更低的延迟，这对于实时应用和高并发场景尤为重要。

6. **能源效率**：存算一体设计通常能更有效地管理能源使用，通过减少不必要的数据传输和处理步骤，实现更高效的能源利用。

存算一体技术在云计算、边缘计算、物联网、人工智能等领域都有广泛的应用前景，是未来技术发展的重要趋势之一。
```

### 3.2 性能结果

```bash
total: 316 tokens, cost 24.338 s
prefill time: 554.962 ms, 1.80 tokens/s
decode average time: 75.499 ms, 13.25 tokens/s
end2end average time: 77.019 ms, 12.98 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
