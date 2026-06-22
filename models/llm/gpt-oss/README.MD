# GPT-OSS

本示例展示如何把 GPT-OSS 模型编译并部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: gpt-oss-20b
- 来源: https://www.modelscope.cn/models/openai-mirror/gpt-oss-20b
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: gpt-oss-120b
- 来源: https://www.modelscope.cn/models/openai-mirror/gpt-oss-120b
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

**硬件要求：**
- `gpt-oss-20b`: 单卡24GB即可运行
- `gpt-oss-120b`: 需要在后摩 2chips x 30GB 以上的卡才能运行

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=gpt-oss`
- `model_size=20b`
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`

tokenizer 默认从 `modelscope_repo` 最后一段推导，即 `gpt-oss-20b`。

## 2 快速开始

目前 gpt-oss 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下编译和演示。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型
python3 get_model.py --type raw --model_name gpt-oss --model_size 20b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name gpt-oss --model_size 20b
```

### 2.3 编译

当前目录不提供 `ptq.py`。`build.py` 负责编译现有 `hmquant` 输出，默认输入目录为 `output/${HOUMO_TARGET}/hmquant`。

```bash
python3 build.py --model_name gpt-oss --model_size 20b
```

默认输出名称分别为：

- `gpt-oss-20b_prefill.hmm`
- `gpt-oss-20b_decode.hmm`

### 2.4 演示

```bash
python3 demo.py --model_name gpt-oss --model_size 20b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=gpt-oss`
- `MODEL_SIZE=20b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
bash test.sh
bash test.sh -s demo
bash test.sh -s build
bash test.sh -s demo --skip_download
```

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 20b.
  -name, --model_name     Model name. Default: gpt-oss.
  --ndevice               Number of devices.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前目录不提供 `ptq.py`，因此：

- `-s quant` 不支持

标准 demo 流程结束后，如果环境中存在 `llm_perf`，脚本会自动执行显式参数形式的性能测试。

## 4 参考结果

下述所示为 gpt-oss-20b 的参考结果

### 4.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
<|channel|>analysis<|message|>The user asks in Chinese: "请介绍一下存算一体技术的优势" meaning "Please introduce the advantages of integrated storage and computing technology." They want an explanation of the advantages of "存算一体技术" (storage-computing integration). We should respond in Chinese, explaining the advantages: improved performance, lower latency, reduced data movement, energy efficiency, cost savings, better scalability, etc. Also mention specific use cases: AI, big data, HPC, etc. Provide a concise but thorough answer.<|end|><|start|>assistant<|channel|>final<|message|>**存算一体技术（Storage‑to‑Compute Integration）**
是指将存储与计算资源在同一硬件平台或同一系统内实现紧密耦合、共享资源、协同调度的一种架构。相比传统的“存‑算‑网络”三层分离模式，存算一体技术在性能、能耗、成本和管理效率等方面具有显著优势。下面从几个关键维度进行阐述：

| 维度 | 传统模式的痛点 | 存算一体技术的优势 |
|------|----------------|--------------------|
| **数据访问延迟** | 计算节点需要频繁通过网络或磁盘I/O访问存储，导致高延迟 | 计算单元与存储共享同一芯片或同一机架，数据可直接在本地高速缓存或内存中访问，延迟降低至毫秒级甚至微秒级 |
| **带宽瓶颈** | 存储I/O带宽受限于磁盘、SSD或网络链路 | 通过在同一平台上集成NVMe‑SSD、HBM、或专用高速总线，带宽可达数百GB/s，满足大规模并行计算需求 |
| **能耗与成本** | 需要额外的网络设备、磁盘阵列、冷却系统，能耗高 | 统一硬件平台减少冗余，能耗下降30%–50%，同时硬件采购、运维成本下降 |
| **数据一致性与安全** | 需要复杂的分布式一致性协议，易出现数据不一致 | 本地存储与计算共享同一内存空间，天然实现一致性；可在芯片级实现加密、访问控制 |
| **开发与运维** | 需要多层软件栈（OS、文件系统、数据库、应用） | 通过统一的API或统一的管理平台，开发者可直接调用存储资源，运维人员只需管理单一系统 |
| **可扩展性** | 需要水平扩展存储与计算节点，网络拓扑复杂 | 通过模块化的存算一体节点（如NVMe‑to‑CPU、GPU+SSD一体化）可“一键”水平扩展，保持低延迟 |
| **适用场景** | 传统数据库、文件系统、云存储 | AI/ML训练、边缘计算、实时大数据分析、数据库加速、FPGA/ASIC加速等 |

---

### 典型应用场景

| 场景 | 需求 | 存算一体技术的价值 |
|------|------|--------------------|
| **AI/ML 训练** | 需要频繁读取/写入海量训练数据 | 通过本地SSD+GPU共享内存，显著降低I/O瓶颈，提升训练速度 |
| **实时大数据分析** | 低延迟查询、实时聚合 | 计算节点可直接访问本地存储，查询响应时间从秒级降到毫秒级 |
| **边缘计算** | 资源受限、网络不稳定 | 本地存算一体可在离线环境下完成完整的计算与存储，减少对云端的依赖 |
| **高性能计算 (HPC)** | 需要高带宽、低延迟的数据共享 | 通过统一的高速总线和共享内存，MPI通信与磁盘I/O无缝集成 |
| **数据库加速** | 需要高并发事务、低延迟访问 | 本地存算一体可将数据库引擎与存储层合并，减少磁盘访问次数 |

---

### 未来趋势

1. **芯片级集成**：CPU+SSD+GPU+AI加速器在同一芯片或同一模块上实现，进一步降低延迟。
2. **软件定义存算**：统一的资源管理平台（如Kubernetes + CSI）支持动态调度存算资源。
3. **边缘与云的无缝衔接**：边缘节点采用存算一体技术，云端通过统一API实现弹性扩展。
4. **安全与隐私**：在芯片级实现硬件加密、可信执行环境，满足合规性要求。

---

### 小结

存算一体技术通过打破传统的“存‑算‑网络”三层分离，实现在同一硬件平台上高效协同的数据存储与计算。它在降低延迟、提升带宽、节能降本、简化运维、增强安全等方面具有显著优势，已成为 AI/ML、边缘计算、实时分析、HPC 等多领域的关键技术之一。随着芯片技术和软件生态的进一步成熟，存算一体将成为未来数据中心与边缘计算的标准架构。
```

### 4.2 性能结果

```bash
====================================================================================================
                               Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
   Batch Size:      1
   Input Length per Sample:     90 tokens
   Output Length per Sample:   1192 tokens
Prefill Stage Performance:
  Total Time:  169.55ms | Speed:  530.83 tokens/s
  Tokenization Time:   22.82ms
  Embedding Time:    1.00ms
  API SetInput Time:   0.77ms
  API Inference Time: 140.00ms | Prefill Speed:  642.86 tokens/s
  API GetOutput Time:  0.58ms
Decode Stage Performance:
  Total Time: 43412.42ms | Speed:   27.43 tokens/s
  Tokenization Time:  384.41ms
  Embedding Time:  201.77ms
  API SetInput Time:   0.86ms
  API Inference Time: 28.41ms | Decode Speed:   35.20 tokens/s
  API GetOutput Time:  0.50ms
Overall Performance Metrics:
  TTFT (Time To First Token):  169.55 ms
  TPOT (Time Per Output Token): 36.45 ms/token
  E2E Latency (End-to-End):     43.70 seconds
  E2E TPS (Throughput):         27.28 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
