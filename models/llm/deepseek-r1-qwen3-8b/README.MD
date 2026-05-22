# DeepSeek R1 Qwen3 8B

本示例展示如何把 DeepSeek R1 Qwen3 8B 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: DeepSeek-R1-0528-Qwen3-8B
- 来源: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=deepseek`
- `model_size=8b`
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`
- `batch=1`

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

tokenizer 默认从 `modelscope_repo` 最后一段推导，即 `DeepSeek-R1-0528-Qwen3-8B`。

## 2 快速开始

目前 DeepSeek R1 Qwen3 8B 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name deepseek --model_size 8b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name deepseek --model_size 8b
```

### 2.3 量化

`ptq.py` 需要 GPU 环境，并要求宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/deepseek-r1-qwen3-8b
```

默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

```bash
python3 ptq.py --model-name deepseek --model-size 8b
```

如果要使用自定义校准数据集，可直接传 `--calib_data`：

```bash
python3 ptq.py --model-name deepseek --model-size 8b --calib_data /path/to/custom.jsonl
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name deepseek --model_size 8b
```

默认输出名称分别为：

- `deepseek-8b_prefill.hmm`
- `deepseek-8b_decode.hmm`

### 2.5 演示

单 batch：

```bash
python3 demo.py --model_name deepseek --model_size 8b
```

多 batch：

```bash
python3 demo_multibatch.py --model_name deepseek --model_size 8b --forbid_flush
```

如果未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=deepseek`
- `MODEL_SIZE=8b`
- `NDEVICE=1`
- `BATCH=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
	-s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
													Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
	-size, --model_size     Model size. Default: 8b.
	-name, --model_name     Model name. Default: deepseek.
	-b, --batch             Batch size (requires model supports multi_batch feature).
	--ndevice               Number of devices.
	--skip_download         Skip model download steps.
	-h, --help              Show this help message.
```

示例：

```bash
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s all
bash test.sh -b 2
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py`，并透传 `--batch`。
- `demo`：`batch=1` 时执行 `demo.py`；`batch>=2` 时执行 `demo_multibatch.py --forbid_flush`。
- 单 batch demo 若环境中存在 `llm_perf`，流程末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

```bash
prefill model loaded
decode model loaded
question:
请介绍一下存算一体技术的优势
response:
<think>
嗯，用户想了解存算一体技术的优势。首先，我需要明确存算一体到底是什么，可能用户听说过但不太清楚具体含义。接着，得从技术角度分析它的优点，比如减少数据搬运、提升性能、降低延迟这些方面。然后要考虑实际应用中的好处，比如简化架构、节省空间、降低能耗，这些都是用户可能关心的点。

用户可能是技术人员或者对科技感兴趣的人，他们可能想了解这项技术如何在实际中应用，或者它带来的效率提升。深层需求可能包括寻找解决方案来优化现有系统，或者评估是否值得投资这项技术。需要确保回答不仅列出优势，还要解释为什么这些优势重要，比如在大数据处理中的应用。

还要注意用户可能没有明确提到的需求，比如成本效益、扩展性，或者与其他技术的对比。比如，用户可能想知道存算一体相比传统存储和计算分离的优势，或者它在云存储中的作用。因此，在回答中需要涵盖这些方面，同时避免过于技术化的术语，保持易懂。

另外，用户可能对技术的未来趋势感兴趣，所以可以提到它在AI、物联网等领域的潜力，以及如何支持实时数据处理。需要确保信息准确，同时结构清晰，分点说明，但避免使用Markdown，保持口语化。

最后，检查是否有遗漏的重要优势，比如数据安全、管理简化，或者对分布式系统的支持。确保回答全面，同时突出关键点，让用户能快速抓住重点。
</think>
存算一体技术是一种将数据存储和计算处理功能集成在同一物理单元或芯片上的技术。它的主要优势包括：

1. **减少数据搬运**：传统架构中，数据需要在存储和计算单元之间频繁传输，这会消耗大量时间和带宽。存算一体技术将计算靠近数据，甚至在同一芯片上完成，极大地减少了数据搬运的开销。

2. **提升性能**：由于计算和存储的集成，数据访问和处理的延迟显著降低，尤其是在需要快速响应的场景下，如实时分析、人工智能训练等，性能提升非常明显。

3. **降低延迟**：数据不需要从存储设备传输到处理器，计算可以直接在存储单元附近进行，从而减少了数据传输的延迟，提高了系统的响应速度。

4. **简化系统架构**：存算一体技术可以减少系统中存储和计算设备的数量，简化整体架构，降低系统的复杂性。

5. **节省空间和能耗**：集成设计减少了硬件设备的体积和数量，从而节省了空间。同时，减少了数据传输也降低了能耗，提高了能效比。

6. **提高吞吐量**：由于计算和存储的集成，数据处理的吞吐量可以显著提升，尤其是在处理大规模数据时，能够更快地完成计算任务。

7. **支持分布式计算**：存算一体技术可以更好地支持分布式计算环境，使得数据在分布式节点中能够高效地进行本地计算，减少了对网络传输的依赖。

8. **降低总体成本**：通过减少硬件设备的数量和简化系统架构，存算一体技术可以降低系统的总体拥有成本（TCO）。

9. **增强数据安全性**：数据在本地进行计算，减少了在传输过程中被窃取或篡改的风险，从而提高了数据的安全性。

10. **适用于大数据和AI场景**：存算一体技术在处理大数据、人工智能、机器学习等需要大量数据存储和实时计算的场景中表现出色，能够满足这些高负载应用的需求。

总的来说，存算一体技术通过将存储和计算功能集成在一起，显著提升了数据处理的效率和性能，同时降低了系统的复杂性和成本，适用于多种现代计算和存储密集型应用场景。

====================================================================================================
                            Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
	Batch Size:      1
	Input Length per Sample:     16 tokens
	Output Length per Sample:    761 tokens
	Prefill Model Load Time: 1622.67ms
	Decode Model Load Time:  144.51ms
Prefill Stage Performance:
	Total Time:  122.59ms | Speed:  130.52 tokens/s
	Tokenization Time:   11.08ms
	Embedding Time:    2.72ms
	API SetInput Time:   1.05ms
	API Inference Time: 103.16ms | Prefill Speed:  155.10 tokens/s
	API GetOutput Time:  0.57ms
Decode Stage Performance:
	Total Time: 40932.05ms | Speed:   18.57 tokens/s
	Tokenization Time:  295.38ms
	Embedding Time:  118.41ms
	API SetInput Time:   0.64ms
	API Inference Time: 45.37ms | Decode Speed:   22.04 tokens/s
	API GetOutput Time:  0.51ms
Overall Performance Metrics:
	TTFT (Time To First Token):  122.59 ms
	TPOT (Time Per Output Token): 53.86 ms/token
	E2E Latency (End-to-End):     41.14 seconds
	E2E TPS (Throughput):         18.50 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
