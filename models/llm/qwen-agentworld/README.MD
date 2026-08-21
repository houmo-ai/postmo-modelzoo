# Qwen-AgentWorld

本示例展示如何把 Qwen-AgentWorld 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen-AgentWorld-35B-A3B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen-AgentWorld-35B-A3B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Qwen-AgentWorld 是一个纯文本（language-only）大语言模型，不包含视觉处理模块。

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于 xh2。

当前目录使用 `config.yaml` 管理 `get_model.py`、`build.py`、`demo.py` 的默认值；`ptq.py` 的量化导出参数由 `config.yaml` 中 `workflow_config` 字段指向的 workflow yaml 管理。当前默认模型为：

- `model_name=qwen-agentworld`
- `model_size=35b-a3b`
- `context_length=2k`
- `prefill_length=256`
- `ndevice=1`

支持的模型组合为：

- `qwen-agentworld`: `35b-a3b`

标准流程编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

`demo.py` 默认从 `config.yaml` 中读取 `modelscope_repo`，并使用其最后一段作为 `tokenizer_dir` 默认值。

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen-AgentWorld 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen-agentworld --model_size 35b-a3b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen-agentworld --model_size 35b-a3b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/qwen-agentworld
```

推荐先通过 `get_model.py --type raw` 下载原始模型，再执行：

```bash
python3 ptq.py \
  --model_dir ./Qwen-AgentWorld-35B-A3B
```

`--model_dir` 指向 Hugging Face / ModelScope 原始模型目录。`ptq.py` 会自动从 `config.yaml` 读取 `workflow_config` 路径，执行 quant 和 export，并把 export 生成的 `hmquant*` 目录移动为 `output/${HOUMO_TARGET}/hmquant`，供后续 `build.py` 使用。

常用可选参数：

```bash
# 覆盖 workflow yaml 中的量化 bit
python3 ptq.py \
  --model_dir ./Qwen-AgentWorld-35B-A3B \
  --bits 8

# 显式指定 workflow yaml 路径
python3 ptq.py \
  --model_dir ./Qwen-AgentWorld-35B-A3B \
  --quant_config_path /path/to/workflow.yaml
```

量化导出输出位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

标准编译：

```bash
python3 build.py --model_name qwen-agentworld --model_size 35b-a3b
```

默认输出名称为：

- `qwen-agentworld-35b-a3b_prefill.hmm`
- `qwen-agentworld-35b-a3b_decode.hmm`

多设备场景下，demo 脚本会自动把 `prefill` / `decode` 默认模型后缀从 `.hmm` 切换为 `.hmms`。

### 2.5 演示

标准演示：

```bash
python3 demo.py --model_name qwen-agentworld --model_size 35b-a3b
```

也可以直接指定模型路径：

```bash
python3 demo.py \
  --prefill_path output/xh2/qwen-agentworld-35b-a3b_prefill.hmm \
  --decode_path output/xh2/qwen-agentworld-35b-a3b_decode.hmm \
  --tokenizer_dir output/xh2/hf_config/ \
  --embedding_path output/xh2/hmquant/quant_embedding.pt
```

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen-agentworld`
- `MODEL_SIZE=35b-a3b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + workflow 量化导出 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: qwen-agentworld.
  -size, --model_size     Model size. Default: 35b-a3b.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载原始模型，再调用 ptq.py 量化导出
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程
bash test.sh -s all

# 跳过下载，直接使用当前目录已有模型
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化导出。
- `build`：调用 `build.py` 编译。
- `demo`：下载预编译模型后执行 `demo.py`；若环境中存在 `llm_perf`，末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

#### Qwen-AgentWorld-35B-A3B 演示结果

```bash
Device 0 -> Group 0, Chip 1
2026-08-21 05:48:27.566 | INFO     | __main__:__init__:515 - prefill model loaded
2026-08-21 05:48:28.470 | INFO     | __main__:__init__:525 - decode model loaded
Input your instruction here: 介绍存算一体
2026-08-21 05:48:36.574 | SUCCESS  | __main__:chat:684 - question:
介绍存算一体
2026-08-21 05:48:36.806 | SUCCESS  | __main__:chat:786 - response:
存算一体（Processing-in-Memory, PIM）是一种新型的计算架构，旨在打破传统冯·诺依曼架构中计算单元和存储单元分离的设计，从而解决“内存墙”问题。在传统的计算机系统中，数据需要在处理器和内存之间频繁传输，这不仅增加了能耗，还限制了计算速度。而存算一体技术通过将计算功能直接集成到存储器中，使得数据可以在存储位置直接进行处理，大幅减少了数据移动的需求。

### 存算一体的主要特点

1. **减少数据移动**：传统架构中，数据在处理器和内存之间来回传输，导致延迟和能耗增加。存算一体技术将计算操作直接在存储器内部完成，减少了数据移动，提高了效率。
   
2. **降低能耗**：由于数据移动是传统架构中能耗的主要来源之一，存算一体技术通过减少数据移动，显著降低了系统的整体能耗。

3. **提高计算速度**：存算一体技术可以并行处理大量数据，从而提高计算速度，特别适用于需要处理大规模数据的场景，如人工智能、大数据分析等。

4. **简化系统设计**：存算一体技术可以减少系统中所需的组件数量，简化系统设计，降低制造成本。

### 存算一体的应用场景

1. **人工智能和机器学习**：在深度学习和神经网络中，大量的矩阵运算和数据处理需要高效的计算架构支持。存算一体技术可以显著提高这些计算任务的效率。

2. **大数据分析**：处理海量数据时，存算一体技术可以减少数据移动，提高数据处理速度，降低能耗。

3. **物联网（IoT）**：物联网设备通常需要在边缘进行数据处理，存算一体技术可以提供高效、低功耗的计算解决方案。

4. **高性能计算（HPC）**：在科学计算、模拟等领域，存算一体技术可以提高计算速度，降低能耗。

### 存算一体的挑战

尽管存算一体技术具有许多优势，但也面临一些挑战：

1. **技术成熟度**：目前，存算一体技术仍处于发展阶段，相关技术和产品尚未完全成熟。

2. **标准化问题**：存算一体技术需要新的硬件和软件支持，目前缺乏统一的标准和规范。

3. **成本问题**：存算一体技术的研发和制造成本较高，需要进一步降低成本才能广泛应用于市场。

4. **兼容性问题**：存算一体技术需要与现有的计算架构和软件生态兼容，这可能需要较大的技术投入。

### 总结

存算一体技术是一种具有巨大潜力的新型计算架构，能够有效解决传统冯·诺依曼架构中的“内存墙”问题，提高计算效率，降低能耗。随着技术的不断发展和应用场景的拓展，存算一体技术有望在未来成为计算领域的重要发展方向。

2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:653 - ====================================================================================================
2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:654 -                     Model Inference Performance Summary Report
2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:655 - ====================================================================================================
2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:658 - Configuration Details:
2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:659 -   Batch Size:      1
2026-08-21 05:49:02.592 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:660 -   Input Length per Sample:     27 tokens
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:663 -   Output Length per Sample:    569 tokens
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:670 -   Prefill Model Load Time: 48773.96ms
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:674 -   Decode Model Load Time:  903.88ms
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:741 - Prefill Stage Performance:
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:742 -   Total Time:  231.29ms | Speed:  116.74 tokens/s
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:746 -   Tokenization Time:   16.84ms
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:751 -   Embedding Time:    0.34ms
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:754 -   API SetInput Time:   2.36ms
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:757 -   API Inference Time: 208.43ms | Prefill Speed:  129.54 tokens/s
2026-08-21 05:49:02.593 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:760 -   API GetOutput Time:  0.49ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:765 - Decode Stage Performance:
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:766 -   Total Time: 25694.76ms | Speed:   22.11 tokens/s
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:770 -   Tokenization Time: 3921.86ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:775 -   Embedding Time:   55.26ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:778 -   API SetInput Time:   1.94ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:781 -   API Inference Time: 35.87ms | Decode Speed:   27.88 tokens/s
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:784 -   API GetOutput Time:  0.48ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:789 - Overall Performance Metrics:
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:790 -   TTFT (Time To First Token):  231.29 ms
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:791 -   TPOT (Time Per Output Token): 45.24 ms/token
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:792 -   E2E Latency (End-to-End):     32.90 seconds
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:793 -   E2E TPS (Throughput):         17.29 tokens/s
2026-08-21 05:49:02.594 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:795 - ====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
