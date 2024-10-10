# Qwen2(通义千问2)

本示例展示如何把Qwen2模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 Qwen2 实现来源于[https://huggingface.co/Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)。

## 2 快速开始

目前Qwen2模型仅提供python脚本编译方式。

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

当前版本可用量化模型支持情况

| models                                 | 4core | 2core | prefill | decode |
| -------------------------------------- | ----- | ----- | ------- | ------ |
| hmquant_qwen2_256_2048_20240913.zip    | yes   | no    | 256     | 2048   |
| hmquant_qwen2_128_4096_20241010.zip    | yes   | no    | 128     | 4096   |

### 2.2 量化（暂不支持）

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py38-x86.64 /bin/bash
```

进入docker后source环境变量，并进入/hmdd/models/llm/qwen_mix目录，执行ptq.sh脚本。（注：可通过修改脚本中的MODEL_DIR修改模型位置，运行中需要连接网络以下载wikitext2数据集，如无法连接可打开注释下载wikitext2_cache.zip）

```bash
source env.sh
cd /hmdd/models/llm/qwen2
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
question:
请介绍一下存算一体技术的优势
response:
存算一体技术是一种将计算和存储紧密集成在一起的技术，它在多个领域中展现出显著的优势：

1. **性能提升**：存算一体技术能够减少数据在内存和存储之间的传输延迟，从而提升整体系统性能。由于数据处理和存储在同一物理位置进行，减少了数据传输的时间，使得处理速度更快。

2. **减少延迟**：在分布式系统中，数据传输延迟是影响系统性能的关键因素。存算一体技术通过减少数据移动，显著降低了延迟，提高了系统的响应速度。

3. **提高效率**：通过将计算和存储紧密集成，存算一体技术能够更有效地利用资源，减少能源消耗，提高整体系统的效率。

4. **简化架构**：存算一体技术简化了系统架构，减少了硬件和软件的复杂性，使得系统设计和维护更加容易。

5. **增强安全性**：在某些场景下，存算一体技术能够提供更好的数据安全性和隐私保护，因为数据在存储和处理时都位于同一物理位置，减少了数据在不同系统之间传输的风险。

6. **适应性强**：存算一体技术能够适应各种应用场景，从高性能计算到边缘计算，再到物联网和大数据分析，都显示出其强大的适应性和灵活性。

7. **成本效益**：通过减少硬件需求和优化资源使用，存算一体技术能够降低总体拥有成本（TCO），特别是在大规模部署和高可用性需求的场景中。

存算一体技术的这些优势使得它在云计算、数据中心、高性能计算、人工智能、边缘计算等多个领域具有广泛的应用前景。
```

### 3.2 性能结果

```bash
total: 331 tokens, cost 43.620 s
prefill time: 813.657 ms, 1.23 tokens/s
decode average time: 129.705 ms, 7.71 tokens/s
end2end average time: 131.781 ms, 7.59 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
