# gpt-oss

本示例展示如何把gpt-oss-20b模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: gpt-oss-20b
- 来源: https://modelscope.cn/models/openai-mirror/gpt-oss-20b
- 许可: Apache License 2.0 （https://modelscope.cn/models/openai-mirror/gpt-oss-20b/file/view/master/LICENSE?status=0）

预训练模型在运行时下载，工程发布**不包含**该模型。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/openai-mirror/gpt-oss-20b](https://www.modelscope.cn/models/openai-mirror/gpt-oss-20b)。
本例只适用于xh2。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载量化模型或芯片模型。

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
gpt-oss_prefill.hmm
gpt-oss_decode.hmm

### 2.4 演示

Qwen2模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.5 一键评估

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
<|channel|>analysis<|message|>The user asks: "请介绍一下存算一体技术的优势" which is Chinese: "Please introduce the advantages of integrated storage and computing technology." They want an explanation of the advantages of "存算一体技术" (storage-computing integration technology). They likely refer to "storage-computing integration" or "compute-in-storage" or "storage-computing integration" technology. They want advantages. So we need to explain what it is, and its advantages: improved performance, lower latency, better data locality, energy efficiency, cost savings, etc. Provide details. Also mention use cases: big data, AI, HPC, etc. Provide bullet points. Also mention challenges. But the user only wants advantages. So we can give a concise but thorough explanation. Use Chinese. Provide bullet points. Also mention that it reduces data movement, improves throughput, reduces network traffic, etc. Also mention that it can be used in edge computing, etc. Provide examples: NVMe over Fabrics, compute in storage, etc. Provide references to industry. Provide a summary. That is the answer.<|end|><|start|>assistant<|channel|>final<|message|>**存算一体技术（Storage‑in‑Compute / Compute‑in‑Storage）**  
是一种将计算单元（CPU、GPU、FPGA、ASIC 等）与存储介质（SSD、HDD、磁盘阵列等）紧密集成的架构。传统的“存储‑计算”分离模式需要在存储层和计算层之间频繁传输数据，导致延迟高、能耗大、网络拥塞等问题。存算一体技术通过在存储设备内部或与之紧密耦合的方式实现数据本地计算，显著提升系统整体性能与效率。

---

## 主要优势

| 优势 | 具体表现 | 业务价值 |
|------|----------|----------|
| **极低延迟** | 数据在本地完成计算，避免了跨网络或总线的传输 | 适用于实时分析、金融交易、工业控制等对延迟敏感的场景 |
| **高吞吐量** | 计算与存储共享同一总线，减少瓶颈 | 适合大规模日志处理、视频转码、机器学习训练等高并发任务 |
| **能耗降低** | 减少数据搬运，降低网络/总线功耗 | 适合数据中心、边缘计算、移动设备等对能耗敏感的环境 |
| **成本节约** | 统一硬件平台，减少多机房、网络设备投入 | 适合中小型企业、云服务商的成本优化 |
| **数据本地化** | 计算与数据在同一物理位置，提升安全性 | 适用于合规性要求高的行业（金融、医疗、政府） |
| **简化架构** | 统一管理、统一监控，降低运维复杂度 | 适合快速迭代、DevOps 场景 |
| **可扩展性** | 通过模块化存算一体设备快速扩容 | 适合弹性计算、云原生应用 |
| **支持多种计算单元** | CPU、GPU、FPGA、ASIC 可按需组合 | 适合多样化工作负载（AI 推理、图像处理、数据库查询） |

---

## 典型应用场景

| 场景 | 需求 | 存算一体技术如何赋能 |
|------|------|---------------------|
| **大数据分析** | 需要在海量日志中快速聚合、过滤 | 本地计算减少网络 I/O，提升查询速度 |
| **人工智能推理** | 需要低延迟、并行计算 | GPU/FPGA 与 SSD 直接协同，推理速度提升 2–3 倍 |
| **数据库/OLAP** | 需要高并发读写 | 本地索引、压缩、查询引擎直接在存储层运行 |
| **边缘计算** | 资源受限、网络不稳定 | 本地计算减少对云的依赖，提升可靠性 |
| **高性能计算 (HPC)** | 需要大规模并行 I/O | 通过 NVMe‑over‑Fabrics + 计算单元实现高带宽、低延迟 |

---

## 技术实现路径

1. **NVMe‑over‑Fabrics + 计算单元**  
   - 在 SSD 控制器中集成 GPU/FPGA，直接对存储块进行计算。  
2. **存储阵列内置计算节点**  
   - 传统磁盘阵列（RAID）中嵌入 CPU/FPGA，完成数据预处理、压缩、加密等。  
3. **全局统一编程模型**  
   - 通过统一 API（如 OpenCL、CUDA、OpenMP）在存储层编写计算任务，系统自动调度。  
4. **软硬件协同优化**  
   - 采用可编程存储控制器、可重构逻辑，动态匹配工作负载。  

---

## 未来趋势

- **AI/ML 专用存算一体芯片**：集成专用 AI 加速器与高性能存储，专为深度学习训练/推理设计。  
- **边缘 AI 存算一体设备**：小型化、低功耗的存算一体模块，支持 IoT、工业自动化。  
- **统一编程框架**：通过统一的编程模型（如 Apache Arrow、DataFusion）实现跨层计算。  
- **安全与合规**：在存算一体设备中集成硬件级加密、访问控制，满足 GDPR、HIPAA 等合规要求。  

---

### 小结

存算一体技术通过将计算与存储紧密耦合，显著降低数据搬运成本、提升性能与能效，并为多种业务场景提供更高的灵活性与安全性。随着 AI、边缘计算和云原生技术的快速发展，存算一体技术将成为数据中心、云服务商乃至终端设备的关键架构之一。
```

### 3.2 性能结果

```bash
Total Input: 90 tokens, Output 1359 tokens, Prefill Cost 616.634 ms, Decode Cost 80450.388 ms
Prefill Speed: 145.95 tokens/s; Decode Speed: 16.88 tokens/s
TTFT (Time to First Token): 676.548 ms
TPOT (Time Per Output Token): 59.242 ms/token
E2E Latency (End-to-End Latency): 92.205 seconds
E2E TPS (End-to-End Tokens Per Second): 14.74 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
