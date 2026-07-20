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
<|channel|>analysis<|message|>The user asks: "请介绍一下存算一体技术的优势".They want an introduction about the advantages of "存算一体技术". This is a Chinese term: "存算一体技术" likely refers to "compute-in-memory" (CIM) or "in-memory computing" where storage and computation are integrated, also known as "processing-in-memory" (PIM). They ask to introduce its advantages. So we need to provide a comprehensive answer in Chinese, describing the technology, its advantages, maybe categories, applications, challenges, etc. Should be helpful. No disallowed content. Provide a clear, structured answer.<|end|><|start|>assistant<|channel|>final<|message|>## 什么是“存算一体”技术？

**存算一体（Compute‑in‑Memory，CIM）**，又称 **Processing‑in‑Memory（PIM）**，是一类把 **存储单元（SR）与计算单元（ALU）深度融合**的新型芯片架构。传统计算机体系结构（冯·诺依曼架构）中，**CPU 与 DRAM 之间需要频繁的数据搬运**（即“存储-计算分离”），导致 **带宽瓶颈、能耗高、延迟大**。CIM 通过在存储单元内部直接完成算术、逻辑、矩阵乘等运算，**在“读‑写”的同一硬件单元里完成计算**，从而实现 **“存储‑计算合一”**。

---

## 存算一体的核心优势

| 序号 | 关键优势 | 具体表现 | 价值意义 |
|------|----------|----------|----------|
|1|**显著降低数据搬运（Data Movement）**|- 计算在存储阵列内部完成，几乎不需要把数据搬到CPU/GPU缓存<br>- 只需把指令或激活信号送到存储阵列 |降低总线带宽需求，减轻系统总线/IO压力 |
|2|**能耗大幅下降**|- 传统体系结构：一次存取 + 一次计算 → 2×能耗<br>- CIM：一次存取即完成计算 → **≈30%~90%**的能耗下降（取决于工艺与工作负载）|在移动设备、边缘AI、数据中心等对功耗敏感的场景中，可显著延长续航或降低散热 |
|3|**带宽瓶颈得到突破**|- 传统CPU/GPU 受限于 DRAM 带宽（≈100–200 GB/s）<br>- CIM 通过 **内部并行存储阵列**（上万至上百万位并行）实现 **TB/s 级内部带宽** |在大模型推理、图像/视频处理、搜索等带宽密集型任务中，系统吞�率提升 2–10× |
|4|**极高的并行度**|- 每个存储单元（如 SRAM、RR、ReRAM、FeFET）都可以并行执行算子<br>- 可实现 **数十万至数百万并行算子** |天然适配 **矩阵乘、卷积、向量点积**等深度学习核心算子，实现 **算子级加速** |
|5|**面积（Silicon Area）利用率提升**|- 传统 CPU/GU 需要独立的算术单元、寄存器、缓存等，占用大量硅片面积<br>- CIM 将存储阵列直接复用为计算单元，**存储单元本身即算子单元** |在同等硅片面积下，可实现 **更高的算力密度**（TOPs/mm²）|
|6|**延迟（Latency）显著降低**|- 传统体系结构：存取 → 传输 → 计算 → 写回（多级流水）<br>- CIM：**一次存取即完成计算**，单次操作延迟可从 **µs 级**降至 **ns 级** |对实时推理、边缘感知、自动驾驶等对时延极度敏感的场景尤为关键 |
|7|**可扩展性与模块化**|- 通过堆叠或二维/三维封装（HBM、3D‑TSV、Chip‑on‑Wafer）实现 **垂直/水平扩展**<br>- 兼容现有标准接口（e.g., LPMS, OpenCAPI, CXL） |可在服务器、边缘盒子、甚至嵌入式芯片中灵活部署 |
|8|**多功能融合**|- 除算术（加、减、乘、逻辑）外，还可实现 **模拟计算、矩阵乘、卷积、稀�稀矩阵乘、搜索/匹配、加密/解密**等 |在 AI、数据库、密码学、信号处理等多领域提供“一体化”解决方案 |
|9|**安全与隐私**|- 数据在存储单元内部完成计算，**不必在总线上暴露**，降低侧信道攻击风险 |对隐私敏感的金融、医疗、军工等行业尤重要 |
|10|**生态与软件堆栈**|- 越来越多的 **编译器、库（如 TVM、ONNX Runtime、OpenCL‑CIM）** 正在适配 CIM 硬件 |降低开发门檻，促进快速落地 |

---

## 典型应用场景

| 场景 | 关键需求 | CIM 能带来的价值 |
|------|----------|-------------------|
|**边缘 AI（智能摄像头、IoT、AR/VR）**|低功耗、低延迟、实时推理|功耗下降 50%+、响应时间从 ms 降到 < 10 µs|
|**数据中心加速器*|大模型推理、搜索/推荐、向量检索|带宽瓶颈解除，算力密度提升 3–5×，能耗降低 30%|
|**存�算一体的存储系统*|高吞�率、低能耗的数据库|在 **向量搜索、相似度计算** 中直接在存储阵列完成点积，查询延迟下降 10×|
|**密码学/同态加密*|安全计算、数据不离芯片|在存储单元内部完成同态运算，防止数据泄露|
|**模拟信号处理*|高带宽、低噪声|利用 **RR/FeFET 的模拟特性**实现乘加运算，降低模拟前端功耗|
|**高性能计算（HPC）*|大规模矩阵乘、稀疏运算|在 3D‑堆叠 CIM 中实现 **TB/s 级内部带宽**，显著提升 FLOPS/W|

---

## 技术实现路径（常见实现方式）

| 技术 | 典型器件 | 计算方式 | 备注 |
|------|----------|----------|------|
|**SRAM‑CIM**|6T/8T SRAM 单元|数字加法、位并行乘法|成熟工艺、速度快、功耗相对较高|
|**RR（Resistive RAM）**|ReRAM、PCM|电导变化实现乘累（Ohm‑Law）|适合模拟乘加、低功耗|
|**FeFET / 2‑T1R**|Ferroeroelectric FET|电荷存储 + 逻辑门|兼具非易失性与高速|
|**光学存算**|光学相位阵列|光学乘法（光场相乘）|适用于大规模矩阵乘、光学 AI|
|**混合型（CPU+CIM）**|CIM 加速卡、PCIe / CXL 插件|CPU 负责控制流、CIM 负责数据密集算|兼容现有软件生态|

---

## 与传统加速器（GPU/TPU）对比

| 对比维度 | GPU/TPU | CIM |
|----------|----------|------|
|**功耗**|高（数百瓦）|低（10–30 W）|
|**带宽瓶颈**|受限于外部 DRAM 带宽|内部存算带宽可达 TB/s|
|**面积/密度**|算力密度受限于缓存层级|算力密度提升 3–5×|
|**延迟**|受限于数据搬运流水线|单次运算延迟可在 ns 级|
|**可扩展性**|需要大量显存、PCIe 带宽|通过 3D‑TSV/堆��层轻松扩展|
|**成本**|高功耗导致散热、供电成本高|低功耗、低散热降低系统整体成本|

---

## 发展趋势与前景

1. **工艺成熟化**：随着 **FinFin、FD‑SOI、FD‑SOI** 以及 **极紫外（EUV）** 制程的成熟，CIM 单元的 **面积、功耗、可靠性** 正在快速提升。
2. **标准化接口**：**CXL 2.0/3.0**、**OpenCAPI**、**PCIe 6.0** 正在加入对 **存算一体**的原生支持，促进生态统一。
3. **软件��栈**：**编译器（LLVM‑CIM）**、**中间表示（MLIR）**、**高层框架（TensorFlow、PyTorch、ONNX）** 正在加入 **CIM 后端**，降低开发门槛。
4. **跨领域融合**：**光学‑CIM**、**量子‑CIM**、**神经形态‑CIM** 等前沿方向正在探索，可能实现 **光‑电‑磁‑量子** 多模计算。
5. **安全加速**：在 **同态加密、隐私计算**场景中，CIM 的“数据不离芯片」特性将成为 **硬件根信任** 的关键。

---

## 小结

- **存算一体技术通过把存储与计算深度融合，实现“一次存取即完成计算”。**
- 其 **优势**体现在 **带宽突破、能耗下降、延迟降低、并行度提升、面积利用率提升**等方面。
- **适用于边缘 AI、数据中心、数据库、密码学、模拟信号处理等多种场景**，并且 **与现有 CPU/GPU 生态兼容**，具备 **可扩展性、模块化和安全优势**。
- 随着工艺、标准、软件生态的同步进化，**存算一体**有望在 **AI、云计算、边缘计算、5G/6G、物联网**等领域成为 **下一代计算加速器**的核心技术。

如果您对某一具体实现（如 SRAM‑CIM、ReRAM‑CIM）或想了解 **如何在现有系统中部署**（如 PCIe 卡、CXL 模块），欢迎进一步提问！
```

#### 4.2.2 性能结果

```bash
====================================================================================================
                    Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     90 tokens
  Output Length per Sample:   2542 tokens
Prefill Stage Performance:
    Total Time:  320.23ms | Speed:  281.05 tokens/s
    Tokenization Time:   22.55ms
    Embedding Time:    3.80ms
    API SetInput Time:   0.32ms
    API Inference Time: 288.39ms | Prefill Speed:  312.08 tokens/s
    API GetOutput Time:  0.26ms
Decode Stage Performance:
    Total Time: 118306.92ms | Speed:   21.48 tokens/s
    Tokenization Time: 1179.10ms
    Embedding Time:  557.72ms
    API SetInput Time:   0.28ms
    API Inference Time: 37.09ms | Decode Speed:   26.96 tokens/s
    API GetOutput Time:  0.26ms
Overall Performance Metrics:
    TTFT (Time To First Token):  320.23 ms
    TPOT (Time Per Output Token): 46.56 ms/token
    E2E Latency (End-to-End):    118.97 seconds
    E2E TPS (Throughput):         21.37 tokens/s
====================================================================================================
```


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
