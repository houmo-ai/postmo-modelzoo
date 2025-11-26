# Qwen3(通义千问3)

本示例展示如何把Qwen3模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen3-14B](https://modelscope.cn/models/Qwen/Qwen3-14B)。
本例只适用于xh2。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen3-14b目录：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.2 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/llm/qwen3`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为256和8192。

```bash
source env.sh
cd models/llm/qwen3-14b
python3 ptq.py
```

为帮助您的模型在量化后达到最佳性能，您可以使用自定义校准数据集，请按以下步骤操作：

步骤 1：准备校准数据
请准备一个 JSON Lines 格式的文件，其中包含您目标应用场景的文本数据。每行格式如下：

```json
{"text": "您的领域相关文本内容..."}
......
```

步骤 2：运行量化命令
将以下命令中的路径替换为您的数据路径后执行：
```bash
# 使用预设的校准数据集或者换成自己的jsonl文件
python3 ptq.py --calib_data ~/houmo-examples-xh2/hmodel/xh2/examples/xh_gen_data/*.jsonl
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
<think>
嗯，用户让我介绍一下存算一体技术的优势。首先，我需要明确什么是存算一体技术。存算一体，也就是存储和计算一体化，可能是指将存储和计算功能集成在一个系统中，这样可以减少数据在存储和计算单元之间的传输，从而提高效率。不过，可能用户指的是更具体的，比如在神经网络或AI芯片中的应用，比如存内计算（In-Memory Computing）或者存算一体架构。

接下来，我得考虑用户可能的背景。他们可能是学生、研究人员，或者企业技术人员，想了解这项技术在哪些方面有优势。可能他们正在做项目，或者想了解技术趋势，比如在AI、物联网或边缘计算中的应用。

用户的需求可能不仅仅是列举优势，还可能希望了解这些优势如何应用在实际场景中，或者与其他技术的对比。比如，传统架构中数据需要在存储和计算单元之间传输，导致延迟和能耗，而存算一体可以解决这些问题。所以，我需要分点说明优势，比如能效、速度、可扩展性、灵活性、降低延迟、支持复杂计算等。

另外，可能需要提到具体的应用场景，比如在AI芯片、物联网设备、边缘计算中的应用，以及如何提升性能。同时，可能还要提到技术挑战，比如设计复杂性、制造工艺等，但用户可能更关注优势，所以这部分可能需要简要提及。

还要注意术语的准确性，比如存算一体可能有不同的实现方式，比如存内计算、存内逻辑、存内模拟等，需要明确这些概念。不过用户可能不需要太深入的技术细节，所以保持简洁。

最后，确保回答结构清晰，分点列出优势，并给出实际例子，帮助用户理解。可能还需要比较传统架构和存算一体的差异，突出其优势。比如，传统架构的数据搬运瓶颈，存算一体如何解决，从而提升性能和能效。
</think>

存算一体技术（Compute-in-Memory, CIM）是一种将存储和计算功能集成在同一物理结构中的技术，旨在通过减少数据在存储和计算单元之间的传输，提升系统效率、降低能耗并优化性能。以下是其主要优势：

---

### **1. 提升计算效率**
- **减少数据搬运瓶颈**：传统架构中，数据需要在存储单元（如内存）和计算单元之间频繁传输，导致延迟和能耗。存算一体技术将计算直接嵌入存储单元，避免数据搬运，显著提升计算效率。
- **并行计算能力**：在存内计算中，数据可以直接在存储器中进行运算，支持并行处理，尤其适合处理大规模数据（如AI模型训练和推理）。

---

### **2. 降低能耗**
- **减少功耗**：数据搬运是计算过程中的主要能耗来源。存算一体技术通过减少数据移动，降低整体功耗，尤其适用于低功耗设备（如物联网终端、可穿戴设备）。
- **支持绿色计算**：在数据中心和边缘计算场景中，存算一体技术有助于降低碳排放，符合可持续发展需求。

---

### **3. 支持复杂计算任务**
- **加速AI与机器学习**：存算一体技术特别适合处理神经网络等复杂计算任务。例如，在存内模拟（In-Memory Computing）中，权重和数据可以直接在存储器中进行矩阵运算，显著提升AI推理速度。
- **高密度计算**：通过将计算逻辑嵌入存储器，存算一体技术可支持更高密度的计算任务，满足高性能计算需求。

---

### **4. 可扩展性与灵活性**
- **适应不同应用场景**：存算一体技术可根据需求调整存储和计算单元的配置，适用于从边缘设备到云端的多场景（如边缘AI、自动驾驶、智能传感器）。
- **兼容现有架构**：可与传统冯·诺依曼架构结合，通过混合架构实现性能与成本的平衡。

---

### **5. 降低系统复杂性**
- **简化硬件设计**：通过减少数据传输需求，存算一体技术降低了系统复杂性，简化了硬件设计和软件编程。
- **支持新型存储技术**：如忆阻器（ReRAM）、相变存储器（PCM）等，这些技术在存算一体架构中能更好地发挥其潜力。

---

### **6. 降低延迟**
- **实时响应需求**：在需要低延迟的场景（如自动驾驶、工业控制）中，存算一体技术能快速处理数据，满足实时性要求。
- **优化通信延迟**：在分布式系统中，减少数据传输延迟可提升整体系统性能。

---

### **应用场景**
- **AI芯片**：如TPU、GPU等加速器采用存算一体设计，提升AI模型训练和推理效率。
- **物联网设备**：在边缘端实现本地数据处理，减少对云端依赖。
- **数据中心**：降低能耗和延迟，提升大规模数据处理能力。
- **自动驾驶**：实时处理传感器数据，提升决策速度。

---

### **挑战与局限**
尽管存算一体技术具有显著优势，但其设计和实现仍面临挑战，如：
- **技术复杂性**：需要新型存储器和计算单元的协同设计。
- **制造工艺限制**：现有半导体工艺可能难以完全实现存算一体的集成。
- **算法适配**：需针对特定应用场景优化算法以充分利用存算一体特性。

---

### **总结**
存算一体技术通过整合存储与计算，解决了传统架构中的数据搬运瓶颈，显著提升了能效、速度和灵活性，是未来高性能计算、AI和边缘计算的重要方向。其优势在AI、物联网、自动驾驶等领域尤为突出，正在推动新一代计算架构的变革。
```

### 3.2 性能结果

```bash
total: 1199 tokens, cost 115.743 s
prefill time: 2331.709 ms, 0.429 tokens/s
decode average time: 148.15 ms, 6.75 tokens/s
end2end average time: 148.35 ms, 6.74 tokens/s
```

### 3.3 性能测试工具

使用perf.py工具可以测试llm的性能。运行指令如下

```bash
python perf.py

# usage
usage: perf.py [-h] [--tokenizer_dir TOKENIZER_DIR] [--embedding_path EMBEDDING_PATH] [--prefill_path PREFILL_PATH] [--decode_path DECODE_PATH] [--isq ISQ] [--osq OSQ]

options:
  -h, --help            show this help message and exit
  --tokenizer_dir TOKENIZER_DIR
                        tokenizer dir
  --embedding_path EMBEDDING_PATH
                        houmo embedding weight path
  --prefill_path PREFILL_PATH
                        houmo prefill model path
  --decode_path DECODE_PATH
                        houmo decode model path
  --isq ISQ             input seq length
  --osq OSQ             output seq length
```

```bash
total prefill count:1024
total decode count:1024
Batch: 1, Total Input: 1024 tokens, Output 1024 tokens, Prefill Cost: 1.650 seconds, Decode Cost: 67.361 seconds
Prefill Speed: 620.57 tokens/s; Decode Speed: 15.19 tokens/s
TTFT (Time to First Token): 1650.086 ms
TPOT (Time Per Output Token): 65.847 ms/token
E2E Latency (End-to-End Latency): 70.379 seconds
E2E TPS (End-to-End Tokens Per Second): 14.55 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
