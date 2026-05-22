# Qwen2.5

本示例展示如何把 Qwen2.5 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen2.5-7B-Instruct
- 来源: [Qwen/Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)
- 许可: Apache License 2.0 (https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen2.5`
- `model_size=7b`
- `context_length=8k`
- `prefill_length=256`
- `ndevice=1`
- `batch=1`

标准流程编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

`get_model.py`、`ptq.py` 和 `demo.py` 默认会从 `modelscope_repo` 取最后一段作为本地模型目录和 `tokenizer_dir` 默认值，当前默认推导结果为 `Qwen2.5-7B-Instruct`。

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: [modelscope/wikitext](https://www.modelscope.cn/datasets/modelscope/wikitext)

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen2.5 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

如果需要自己量化，可选择下载原始模型和 wikitext 数据集，原始模型默认下载到 `Qwen2.5-7B-Instruct` 目录：

```bash
python3 get_model.py --type raw
```

如果仅查看单 batch 演示结果，可选择下载预编译模型：

```bash
python3 get_model.py --type hmm
```

如果需要运行 `demo_multibatch.py`，可下载 batch=2 对应的预编译模型：

```bash
python3 get_model.py --type hmm --batch 2
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/llm/qwen2.5` 目录，执行 `ptq.py` 脚本。默认量化参数会从 `config.yaml` 推导 `model_name=qwen2.5`、`model_size=7b`、`context_length=8k`、`input_sequence_length=256`：

```bash
source env.sh
cd models/llm/qwen2.5
python3 ptq.py --model-name qwen2.5 --model-size 7b
```

如需显式指定校准数据，也可以传入 `--calib_data`：

```bash
python3 ptq.py --model-name qwen2.5 --model-size 7b --calib_data ../../../hmodel/xh2/examples/xh_gen_data/gen_qwen2_7b_instruct.jsonl
```

量化完成的模型存放在 `output/${HOUMO_TARGET}/hmquant` 目录：

```bash
.
|-- prefill
|-- decoder
|-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。默认输出名称为 `qwen2.5-7b_prefill.hmm` 和 `qwen2.5-7b_decode.hmm`，执行脚本：

```bash
python3 build.py
```

默认输出名称分别为：

- `qwen2.5-7b_prefill.hmm`
- `qwen2.5-7b_decode.hmm`

如果需要生成多 batch 演示用模型，可显式指定 `--batch 2`：

```bash
python3 build.py --model_name qwen2.5 --model_size 7b --batch 2
```

### 2.5 演示

Qwen2.5 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：

   - `output/${HOUMO_TARGET}/qwen2.5-7b_prefill.hmm`
   - `output/${HOUMO_TARGET}/qwen2.5-7b_decode.hmm`
   - `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

2. 当 `--ndevice 2` 时，`demo.py` 会自动把 `prefill` 和 `decode` 默认后缀从 `.hmm` 切换为 `.hmms`。

3. 执行脚本，推理完成将输出模型回答和性能结果。

```bash
python3 demo.py
```

如果需要运行 batch=2 的多 batch 演示，可执行：

```bash
python3 demo_multibatch.py --model_name qwen2.5 --model_size 7b --forbid_flush
```

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen2.5`
- `MODEL_SIZE=7b`
- `NDEVICE=1`
- `BATCH=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 7b.
  -name, --model_name     Model name. Default: qwen2.5.
  -b, --batch             Batch size (requires model supports multi_batch feature).
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`、`demo_multibatch.py`。默认值分别是 `qwen2.5` 和 `7b`。

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 运行 batch=2 多 batch 演示
bash test.sh -s demo -b 2

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py`，并把 `BATCH` 透传到编译阶段。
- `demo`：当 `BATCH < 2` 时下载单 batch 预编译模型并执行 `demo.py`；若环境中存在 `llm_perf`，流程末尾还会自动执行性能测试。
- `demo`：当 `BATCH >= 2` 时下载对应 batch 的预编译模型并执行 `demo_multibatch.py --forbid_flush`。

## 4 参考结果

### 4.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
存算一体技术是一种将存储器和处理器集成在同一芯片上的技术，其主要优势包括：

1. **减少数据传输延迟**：在传统的计算架构中，数据需要在存储器和处理器之间频繁传输，这会带来显著的延迟。存算一体技术通过将存储器和计算单元集成在一起，可以大大减少数据传输的时间，提高计算效率。

2. **降低能耗**：由于减少了数据传输所需的能量，存算一体技术能够显著降低能耗。这对于需要长时间运行的设备（如数据中心、移动设备等）尤为重要。

3. **提高计算密度**：将存储器和计算单元集成在同一芯片上，可以实现更高的计算密度，即在相同面积的芯片上可以集成更多的计算资源，从而提高计算能力。

4. **适应人工智能和机器学习**：存算一体技术特别适合处理大规模数据和复杂计算任务，如人工智能和机器学习算法中的大量矩阵运算。通过优化存储和计算的协同工作，可以更高效地处理这些任务。

5. **灵活性和可扩展性**：存算一体技术允许更灵活地设计和调整存储和计算资源的比例，以适应不同的应用需求。此外，随着技术的发展，可以更容易地增加更多的计算或存储资源，以满足不断增长的计算需求。

尽管存算一体技术带来了许多潜在的优势，但其实际应用仍面临一些挑战，如如何高效地设计和实现这种集成架构，以及如何解决与之相关的热管理、可靠性等问题。随着技术的进步，这些问题有望得到解决，存算一体技术的应用前景将更加广阔。

```

### 4.2 性能结果

```bash
====================================================================================================
                           Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
    Batch Size:      1
    Input Length per Sample:     26 tokens
    Output Length per Sample:    332 tokens
    Prefill Model Load Time: 1795.20ms
    Decode Model Load Time:  165.11ms
Prefill Stage Performance:
    Total Time:  209.10ms | Speed:  124.34 tokens/s
    Tokenization Time:    6.79ms
    Embedding Time:    0.37ms
    API SetInput Time:   1.92ms
    API Inference Time: 195.76ms | Prefill Speed:  132.82 tokens/s
    API GetOutput Time:  0.57ms
Decode Stage Performance:
    Total Time: 16146.35ms | Speed:   20.50 tokens/s
    Tokenization Time:  129.31ms
    Embedding Time:   53.01ms
    API SetInput Time:   0.71ms
    API Inference Time: 39.74ms | Decode Speed:   25.16 tokens/s
    API GetOutput Time:  0.52ms
Overall Performance Metrics:
    TTFT (Time To First Token):  209.10 ms
    TPOT (Time Per Output Token): 48.78 ms/token
    E2E Latency (End-to-End):     16.41 seconds
    E2E TPS (Throughput):         20.23 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。