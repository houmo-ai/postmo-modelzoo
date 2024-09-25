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
存算一体技术是一种将计算和存储功能集成在同一芯片上的技术。这种技术的优势主要体现在以下几个方面：

1. **性能提升**：存算一体技术能够实现数据的就近处理，减少数据在内存和处理器之间的传输延迟，从而提升计算效率和性能。特别是在处理大规模数据集或高并发场景时，存算一体技术能够显著提升处理速度。

2. **功耗降低**：由于减少了数据传输的次数，存算一体技术能够有效降低系统的整体功耗。这对于数据中心和边缘计算设备来说尤为重要，有助于实现更节能的计算环境。

3. **成本节约**：通过减少硬件组件的数量，存算一体技术能够降低系统的整体成本。同时，由于性能的提升，可能需要更少的硬件资源来完成相同或更高的计算任务，进一步节省成本。

4. **简化系统设计**：存算一体技术使得系统设计更为简单，减少了系统架构的复杂性。这对于开发和维护系统来说是一个巨大的优势，能够加速产品的上市时间。

5. **适应性增强**：在某些应用场景中，存算一体技术能够提供更好的数据处理能力，特别是在需要实时处理和高响应速度的场景下，如自动驾驶、实时数据分析等。

6. **安全性提升**：通过将数据存储和处理紧密集成，存算一体技术有助于减少数据在传输过程中的暴露，从而提升数据安全性。

存算一体技术的这些优势使得它在云计算、人工智能、物联网等领域具有广泛的应用前景。随着技术的不断进步，存算一体技术有望在更多场景中发挥重要作用。

```

### 3.2 性能结果

```bash
total: 331 tokens, cost 38.525 s
prefill time: 332.867 ms, 3.00 tokens/s
decode average time: 115.720 ms, 8.64 tokens/s
end2end average time: 116.391 ms, 8.59 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
