# Qwen2.5(通义千问3)

本示例展示如何把Qwen2.5模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: Qwen2.5-7B
- 来源: https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct
- 许可: Apache License 2.0 （https://modelscope.cn/models/Qwen/Qwen2.5-8B/file/view/master/LICENSE?status=0）

预训练模型在运行时下载，工程发布**不包含**该模型。

## 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:
- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)。

## 2 快速开始

目前Qwen2.5模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen2.5-8b目录：

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
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/llm/qwen2.5`目录，执行ptq.py脚本。默认的prefill和decode上下文长度分别为256和8192。

```bash
source env.sh
cd models/llm/qwen2.5
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
qwen2.5_prefill.hmm
qwen2.5_decode.hmm

### 2.3 演示

Qwen2.5模型使用python API进行演示

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
User question:
请介绍一下存算一体技术的优势
Model response:
存算一体技术，即存储计算一体化技术，是一种将数据存储和数据处理功能集成在一起的新型计算架构。这种技术的优势主要包括以下几个方面：

1存储和计算资源的高度集成，可以减少数据传输带来的延迟和能耗，提高计算效率。在处理大规模数据时，存算一体技术可以显著提高处理速度，减少数据传输带来的能耗，提高能效比。

2. 降低能耗：存算一体技术可以减少数据在存储器和处理器之间传输的次数，从而降低能耗。这对于需要长时间运行的设备尤为重要，如数据中心、超级计算机等。

3. 支持更复杂的数据处理任务：存算一体技术可以支持更复杂的数据处理任务，如机器学习、人工智能等，这些任务通常需要大量的数据处理和存储。

通过将存储和计算功能集成在一起，存算一体技术可以更有效地处理这些任务。

4. 适应性更强：存算一体技术可以根据不同的应用场景和需求，灵活地调整存储和计算资源的比例，以满足不同的性能要求。这对于需要灵活调整计算资源的应用场景尤为重要，如边缘计算、物联网等。

存算一体技术是未来计算技术发展的重要方向之一，具有广泛的应用前景。
```

### 3.2 性能结果

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
Batch Size:      1
Input Length per Sample:     26 tokens
Output Length per Sample:    251 tokens
Prefill Model Load Time: 1585.67ms
Decode Model Load Time:  181.21ms
Prefill Stage Performance:
Total Time:  118.39ms | Speed:  219.61 tokens/s
Tokenization Time:   16.52ms
embedding Time:    3.23ms
API SetInput Time:   2.24ms
API Inference Time: 95.04ms | Prefill Speed:  273.56 tokens/s
API GetOutput Time:  0.22ms
Decode Stage Performance:
Total Time: 11146.76ms | Speed:   22.43 tokens/s
Tokenization Time:   25.84ms
Embedding Time:  204.11ms
API SetInput Time:   0.22ms
API Inference Time: 42.71ms | Decode Speed:   23.41 tokens/s
API GetOutput Time:  0.21ms
Overall Performance Metrics:
TTFT (Time To First Token):  118.39 ms
TPOT (Time Per Output Token): 44.59 ms/token
E2E Latency (End-to-End):     11.30 seconds
E2E TPS (Throughput):         22.21 tokens/s
====================================================================================================
```

### 3.3 性能测试工具

- 使用llm_perf性能评测工具测试模型性能。
- llm_perf工具详细使用说明参考${HOUMO_EXAMPLE_PATH}/tools/llm_perf目录的README.MD。
- 如果使用发布的容器，llm_perf工具已经预编译安装，可以直接通过llm_perf --help获取帮助。
- config.yaml为llm_perf执行测试的配置文件，用户可以按照需要自行修改。

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
