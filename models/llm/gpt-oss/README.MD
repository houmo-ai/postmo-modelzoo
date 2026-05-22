# GPT-OSS

本示例展示如何把 GPT-OSS 模型编译并部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: gpt-oss-20b
- 来源: https://www.modelscope.cn/models/openai-mirror/gpt-oss-20b
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

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

### 4.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
<|channel|>analysis<|message|>We need to answer in Chinese. The user asks: "请介绍一下存算一体技术的优势" meaning "Please introduce the advantages of integrated storage and computing technology." They want an explanation of the advantages of "存算一体技术" (storage-computing integration technology). We should explain what it is, its benefits: reducing latency, improving performance, energy efficiency, cost, data locality, etc. Also mention use cases: AI, big data, edge computing, etc. Provide a concise but thorough explanation. Use Chinese. Probably mention that it reduces data movement, improves throughput, reduces power consumption, simplifies architecture, etc. Also mention that it can be used in data centers, HPC, AI training, etc. Provide bullet points or paragraphs. Let's produce a good answer.<|end|><|start|>assistant<|channel|>final<|message|>
**存算一体技术（Integrated Storage‑Compute）** 是指将存储（如SSD、NAND、内存等）与计算单元（CPU、GPU、FPGA、ASIC等）在同一芯片或同一封装内实现高度集成的技术。相比传统的“存储‑计算分离”架构，存算一体技术在多方面具有显著优势：

| 维度 | 传统架构 | 存算一体技术 | 主要优势 |
|------|----------|--------------|----------|
| **数据访问延迟** | 存储与计算分离，数据需跨总线/网络传输 | 计算单元直接与存储介质相连 | 延迟降低 10‑100×，适合低时延应用（如实时AI、金融交易） |
| **带宽与吞吐** | 受总线/网络带宽限制 | 采用片上或近距离互连，带宽可达数 TB/s | 处理大规模数据时不再受限于外部总线 |
| **能耗** | 数据搬运耗电量高 | 省去搬运，能耗下降 30‑70% | 适合边缘/移动设备、数据中心节能 |
| **系统成本** | 需要额外的网络、缓存、IO 控制器 | 统一封装，硬件件数减少 | 维护成本与采购成本降低 |
| **可靠性** | 存储与计算分离导致多点故障 | 统一管理，故障定位更精准 | 故障恢复更快，系统稳定性提升 |
| **安全性** | 数据在网络中传输易被截获 | 数据在芯片内部流动，攻击面缩小 | 数据泄露风险降低 |
| **可扩展性** | 需要多块卡/模块叠加 | 通过多芯片堆叠或多核集成实现 | 规模化更灵活，易于升级 |
| **应用场景** | 传统服务器、云存储 | AI/ML 训练、图像/视频处理、边缘计算、IoT、工业控制 | 适用性更广，满足多样化需求 |

### 关键技术实现

1. **片上存储（On‑Chip Flash / 3D NAND）**  
   - 将闪存直接集成在CPU/GPU芯片内部，减少外部接口。  
   - 通过 3D NAND 结构实现高密度、低功耗的存储。

2. **近场互连（Near‑Field Interconnect）**  
   - 采用高速互连（如 HBM、HBM2、HBM3、Infinity Fabric）将存储与计算单元紧密连接。  
   - 通过多通道并行访问，提升带宽。

3. **混合内存层次（Hybrid Memory Hierarchy）**  
   - 在同一芯片内同时集成 SRAM、DRAM、NVM 等多种存储介质，形成多级缓存与持久化存储。  
   - 通过软件/硬件协同，动态调度数据在不同层级间迁移。

4. **专用加速器（ASIC/FPGA）**  
   - 在存算一体芯片中嵌入专用加速器，针对 AI 推理、加密、压缩等任务进行硬件加速。  
   - 与存储直接配合，进一步降低数据搬运成本。

### 典型应用案例

| 行业 | 典型需求 | 存算一体技术解决方案 |
|------|----------|----------------------|
| **人工智能** | 大规模模型训练与推理 | 通过 GPU/TPU 与高带宽存储集成，显著提升训练速度 |
| **视频处理** | 4K/8K 视频编码/解码 | 低延迟存取 + 近场加速器，支持实时流媒体 |
| **边缘计算** | IoT 设备、工业控制 | 低功耗、紧凑尺寸，满足设备端实时处理 |
| **金融交易** | 高频交易、风险评估 | 低延迟、可预测的性能，满足秒级决策 |
| **科研 HPC** | 大规模数值模拟 | 高带宽、低功耗，提升能效比 |

### 未来趋势

- **3D 堆叠技术**：将存储层与计算层垂直堆叠，进一步提升集成度与带宽。  
- **持久化内存（Persistent Memory）**：将 NVM 与 DRAM 混合，提供持久化存储与高速访问。  
- **AI 专用芯片**：将 AI 推理/训练加速器与存储集成，形成“一体化 AI 芯片”。  
- **系统级协同**：通过统一的编程模型（如统一内存地址空间）简化开发。

---

**总结**

存算一体技术通过将存储与计算紧密集成，显著降低数据搬运延迟、提升带宽、降低能耗与成本，并提升系统可靠性与安全性。它已成为 AI、边缘计算、云计算等领域的关键技术方向，未来将进一步推动高性能、低功耗计算平台的发展。
```

### 4.2 性能结果

```bash
====================================================================================================
                               Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
   Batch Size:      1
   Input Length per Sample:     90 tokens
   Output Length per Sample:   1351 tokens
Prefill Stage Performance:
   Total Time:  215.32ms | Speed:  417.98 tokens/s
   Tokenization Time:   24.77ms
   Embedding Time:   22.97ms
   API SetInput Time:   9.66ms
   API Inference Time: 149.15ms | Prefill Speed:  603.43 tokens/s
   API GetOutput Time:  0.58ms
Decode Stage Performance:
   Total Time: 54897.34ms | Speed:   24.59 tokens/s
   Tokenization Time:  564.31ms
   Embedding Time:  206.36ms
   API SetInput Time:   1.35ms
   API Inference Time: 30.85ms | Decode Speed:   32.42 tokens/s
   API GetOutput Time:  0.53ms
Overall Performance Metrics:
   TTFT (Time To First Token):  215.32 ms
   TPOT (Time Per Output Token): 40.66 ms/token
   E2E Latency (End-to-End):     55.27 seconds
   E2E TPS (Throughput):         24.45 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
