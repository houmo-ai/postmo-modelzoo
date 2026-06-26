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

**编译时 host 内存要求：**
- `gpt-oss-20b`: 需要空闲 host 内存最小 80GB
- `gpt-oss-120b`: 需要空闲 host 内存最小 350GB

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

### 4.1 gpt-oss-20b

#### 4.1.1 演示结果

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

#### 4.1.2 性能结果

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

### 4.2 gpt-oss-120b

#### 4.2.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
<|channel|>analysis<|message|>The user asks: "请介绍一下存算一体技术的优势".They want an introduction of the advantages of "存算一体技术". This is Chinese. "存算一体技术" likely refers to "compute-in-memory" (CIM) or "memory-compute integration", i.e., integrating storage and computation, like in-memory computing, processing-in-memory (PIM), or "存算一体" which is Chinese for "storage and computing integrated". They want a description of its advantages. Should be in Chinese. Provide a thorough explanation: what it is, its advantages: reduced data movement, lower latency, energy efficiency, higher bandwidth, better scalability, better performance for AI, edge computing, etc. Also mention challenges and future prospects. Provide bullet points. Should be concise but thorough.<|end|><|start|>assistant<|channel|>final<|message|>## 什么是“存算一体”技术？

“存算一体”（Compute‑In‑Memory，CIM）指的是把 **存储（Memory）** 与 **计算（Compute）** 功能深度融合在同一芯片或同一模块中，使数据在存储单元内部即可完成算术、逻辑或矩阵运算，而不必先搬运到传统的CPU/GPU 进行处理。常见实现方式包括：

- **基于 SRAM/DRAM 的位并行算术**（如 XNOR‑Net、Bit‑Serial 计算）
- **基于新型存储器的逻辑功能**（如 ReRAM、PCM、MRAM 的逻辑门）
- **片上微型算子阵列**（如 3D‑Stacked PIM、HBM‑PIM）
- **片上 AI 加速器**（如 HBM‑AI、HBM‑PIM）

---

## 存算一体的核心优势

| 序号 | 关键优势 | 具体表现 | 受益场景 |
|------|----------|----------|----------|
| **1** | **显著降低数据搬迁（Data Movement）** | - 数据在同一存储阵列内部完成计算，只需把结果搬出 <br>- 典型系统中 DRam↔CPU 之间的带宽瓶颈可削减 80% 以上 |大规模 AI 推理、图像/视频处理、数据库查询 |
| **2** | **能耗大幅下降** | - 只激活少量位/列，单次算子能耗可低至 0.1 fJ/operation <br>- 与 CPU/GPU 对比，整体功耗降低 5‑10× |边缘设备、物联网、移动终端 |
|**3**|**极高的并行度与吞吐量**|- 每个存储单元（bit/列）都可以并行执行相同算子（SIMD）<br>- 适合矩阵乘法、卷积等大规模并行计算 |深度学习推理、推荐系统、基因测序、金融风险模型 |
|**4**|**带宽与延迟优势**|- 内部算子完成后直接返回结果，单次延迟可在 **ns‑µs** 级 <br>- 传统系统受限于 DRAM 带宽（≈ 100 GB/s），CIM 可实现 **TB/s** 级等效带宽 |实时视频分析、自动驾驶感知、AR/VR 渲染 |
|**5**|**硬件资源利用率提升**|- 存储阵列本身的面积/功耗已经被利用为计算单元，整体芯片面积利用率提升 30‑50% |高密度数据中心、云端 AI 加速器 |
|**6**|**可扩展性与模块化**|- 通过 3D‑Stack 或 Chip‑on‑Chip 方式叠加，可轻松实现数十至上百层的计算层 <br>- 与现有 CPU/GPU 通过标准总线（PCIe、CCIX、CXL）协同工作 |异构计算平台、可插拔 AI 加速卡 |
|**7**|**降低系统复杂度**|- 省去大量的缓存层、总线桥、DMA 控制器等中间硬件，系统设计更简洁、可靠性更高 |嵌入式系统、航空航天、工业控制 |
|**8**|**适配新型模型**|- 天然支持二值/低位宽网络（BinaryNet、XNOR‑Net）<br>- 可直接实现稀疏矩阵乘、向量‑点积等新型算子 |边缘 AI、压缩模型、神经网络加速 |
|**9**|**成本潜在下降**|- 当存储成本继续下降（DDR5/LP‑DDR、HBM）时，同价位的芯片即可提供计算功能，整体系统成本下降 10‑30% |大规模部署的 AI 推理服务器、智能摄像头 |

---

## 为什么这些优势在实际场景中如此重要？

| 场景 | 关键痛点 | 存算一体带来的突破 |
|------|----------|------------------------|
| **AI 推理服务器** | 计算与内存带宽不匹配导致功耗飙升、延迟受限 |在同一芯片内部完成卷积/矩阵乘，显著降低功耗、提升吞吐，同等功耗下可部署更多模型实例 |
| **边缘/IoT** | 电池容量、散热、体积受限 |极低功耗、极小延迟的 PIM �块直接嵌入传感器或摄像头，几毫瓦即可完成推理 |
| **大数据分析** | 数据库查询需要频繁读取、过滤、聚合 |在存储阵列内部完成过滤、计数、聚合，只把最终结果返回，I/O 流量下降 90% 以上 |
| **高性能计算 (HPC)** | 传统 CPU/GPU 受限于内存带宽，导致“内存墙” |CIM 把算子搬到内存内部，突破内存墙，实现更高的 FLOPS/Watt |
| **实时视频/AR** | 超低延迟是关键 |内部算子完成卷积/特征提取后直接输出特征图，端到端延迟可压缩到 10 µs 以内 |

---

## 典型实现案例（截至 2024‑2025）

| 公司/项目 | 技术路线 | 关键指标 | 适用领域 |
|-----------|----------|----------|----------|
| **Microsoft & Intel** | 3D‑Stacked DRAM + 逻辑层（CIM） |>10 TOPS/W，延迟 < 100 ns|云端 AI 推理、搜索引擎 |
| **Samsung** | HBM‑PIM（基于 SRAM 逻辑）|>1 TB/s 等效带宽，功耗 0.5 pJ/bit|高性能计算、AI 训练加速 |
| **IBM** | ReRAM‑PIM（非易失性存储）|>0.1 fJ/operation，支持 8‑bit 乘加|边缘 AI、低功耗传感器 |
| **Cambridge Quantum** | 量子位存算一体（NV‑PIM）|>10⁶ OPS/µJ|密码学、量子机器学习 |
| **华为** |昇腾+存算一体（昇腾 PIM）|>5 TOPS/W，兼容 Ascend AI �架|边缘计算、智能摄像头 |

---

## 发展趋势与挑战

| 趋势 | 说明 |
|------|------|
| **工艺融合** | 3D‑TSV、微凸点（micro‑bump）技术让存储层与逻辑层的互连更紧凑，带宽提升 10‑20× |
| **低位宽算子** | 越来越多的 AI 模型采用 4‑bit、2‑bit甚至二值化算子，CIM 与这些模型天然匹配 |
| **标准化接口** | CXL、CCIX 等高速互连标准正在加入对 PIM 的原生支持，生态逐步统一 |

| 关键挑战 | 可能的解决路径 |
|-----------|----------------|
| **热管理** |采用高效散热微通道、热导管或液冷；在算法层面采用稀疏/低位宽模型降低功耗 |
| **编程模型** |发展专用编译器（如 TVM、MLIR）和抽象层（PIM‑API、CIM SDK），让开发者无需关注硬件细节 |
| **可靠性/容错** |在存储层加入 ECC、行/列冗余；采用自校准的自适应算子 |
| **成本** |随着 DRAM/LPDDR/HBm 成本下降，CIM 采用的门槛将进一步降低 |

---

## 小结

- **存算一体** 把“数据在存储里就能算”变成现实，根本上解决了 **“内存墙”** 与 **“数据搬迁能耗”** 两大瓶颈。
- 其 **低功耗、低延迟、高并行** 特性，使其在 **AI 推理、边缘计算、数据库加速、实时视频处理** 等场景中拥有显著优势。
- 随着 **3D‑Stack、低位宽算子、标准化互连** 的成熟，存算一体正从实验室走向大规模商用，预计在 **2025‑2027 年**会成为数据中心与边缘设备的重要计算加速器。

如果您想了解更具体的实现细例、编程模型或选型建议，欢迎进一步提问！
```

#### 4.2.2 性能结果

```bash
====================================================================================================
                    Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     90 tokens
  Output Length per Sample:   2199 tokens
Prefill Stage Performance:
    Total Time:  310.18ms | Speed:  290.16 tokens/s
    Tokenization Time:   24.39ms
    Embedding Time:    3.28ms
    API SetInput Time:   0.25ms
    API Inference Time: 278.86ms | Prefill Speed:  322.74 tokens/s
    API GetOutput Time:  0.27ms
Decode Stage Performance:
    Total Time: 94480.25ms | Speed:   23.26 tokens/s
    Tokenization Time:  879.83ms
    Embedding Time:  430.31ms
    API SetInput Time:   0.26ms
    API Inference Time: 34.52ms | Decode Speed:   28.97 tokens/s
    API GetOutput Time:  0.25ms
Overall Performance Metrics:
    TTFT (Time To First Token):  310.18 ms
    TPOT (Time Per Output Token): 42.98 ms/token
    E2E Latency (End-to-End):     95.04 seconds
    E2E TPS (Throughput):         23.14 tokens/s
====================================================================================================
```


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
