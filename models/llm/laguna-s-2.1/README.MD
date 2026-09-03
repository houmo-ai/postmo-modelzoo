# Laguna-S-2.1

本示例展示如何将 Laguna-S-2.1 量化、编译并部署到后摩 XH2 M50 设备上运行。

[TOC]

## 1 模型说明

本示例使用来自 ModelScope 的 Laguna-S-2.1 预训练模型：

- 模型名称：Laguna-S-2.1（118B-A8B）
- 来源：[poolside/Laguna-S-2.1](https://www.modelscope.cn/models/poolside/Laguna-S-2.1)
- 许可：[OpenMDW-1.1](https://modelscope.cn/models/poolside/Laguna-S-2.1/file/view/master/LICENSE.md)

模型代码、权重、配置和其他模型材料由第三方提供。本仓库不随源码发布模型权重；用户需要从官方来源自行获取，并遵守 OpenMDW-1.1 及相关使用条款。

本例只适用于 xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。默认配置为：

- `model_name=laguna-s-2.1`
- `model_size=118b-a8b`
- `context_length=256k`
- `prefill_length=256`
- `ndevice=2`
- `quant_type=w4a8`
- 量化 workflow：`laguna_s_2_1_full_autoround_expert_w4_rest_w8_g64_xh2a_w4a8.yaml`

### 资源要求

以下均指流程运行前的**空余**资源，实际需求还会受到系统、进程、驱动和并行度影响。

**后摩设备运行要求：**

- `118b-a8b`、W4A8：需要后摩 M50 **48GB × 2** 或 **24GB × 4** 设备运行推理示例。

**量化硬件要求：**

- `118b-a8b`、W4A8：需要至少 **48GB 空余 GPU 显存**。
- 量化过程还需要至少 **300GB 空余 CPU 内存**。

**编译时 host 内存要求：**

- `118b-a8b`、W4A8：需要至少 **512GB 空余 CPU 内存**。

资源不足可能导致量化或编译过程中出现 OOM；请在开始对应阶段前确认资源已经释放。

### 模型许可和使用限制

Laguna-S-2.1 遵循 [OpenMDW-1.1 License](https://modelscope.cn/models/poolside/Laguna-S-2.1/file/view/master/LICENSE.md)。如果分发模型材料的任何部分，需要同时保留该许可证以及模型材料中适用的版权和来源声明。模型使用者需要自行确认模型材料中可能涉及的其他权利、许可和使用限制，并遵守 Poolside 的[可接受使用政策](https://poolside.ai/legal/acceptable-use-policy)。

## 2 快速开始

芯片模型只能在 Linux 环境下编译和运行。量化和编译前请确保已准备好对应的模型目录、工具链和硬件资源。

### 2.1 环境准备

安装示例依赖，建议使用独立虚拟环境：

```bash
pip3 install -r requirements.txt
```

Linux 环境中先初始化仓库环境：

```bash
source env.sh
```

### 2.2 获取模型和数据

下载原始模型用于量化：

```bash
python3 get_model.py --type raw --model_name laguna-s-2.1 --model_size 118b-a8b
```

下载预编译模型用于直接运行 demo：

```bash
python3 get_model.py --type hmm --model_name laguna-s-2.1 --model_size 118b-a8b
```

模型下载到本地后，请确认 `Laguna-S-2.1/` 目录包含完整的 Hugging Face 模型文件。模型不随本仓库源码分发，用户自行下载时应遵守 OpenMDW-1.1。

### 2.3 量化

量化需要 GPU，并要求宿主机使用与 CUDA 12.8 兼容的 CUDA 和驱动。量化前需要至少 **300GB 空余 CPU 内存**和 **48GB 空余 GPU 显存**。启动 docker 时将 GPU 映射到容器内：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录初始化环境，再进入当前模型目录执行量化：

```bash
source env.sh
cd models/llm/laguna-s-2.1
python3 ptq.py --model_name laguna-s-2.1 --model_size 118b-a8b
```

仅运行量化且跳过已完成的模型下载：

```bash
cd models/llm/laguna-s-2.1
./test.sh -s quant --skip_download
```

量化结果默认位于：

```text
work_dirs/laguna_s_2_1_quantized/
```

### 2.4 编译

编译需要至少 **512GB 空余 CPU 内存**。将量化输出编译为后摩模型：

```bash
python3 build.py --model_name laguna-s-2.1 --model_size 118b-a8b
```

默认编译输出目录为：

```text
output/xh2/
```

主要输出文件名基于以下逻辑名称生成；单设备编译产物使用 `.hmm`，多设备编译产物使用工具链对应的 `.hmms` 形式：

- `laguna-s-2.1-118b-a8b_prefill.hmm`
- `laguna-s-2.1-118b-a8b_decode.hmm`

当 `ndevice=2` 或 `ndevice=4` 时，demo 会读取对应的多设备 `.hmms` 产物。

### 2.5 演示

当前 `demo.py` 支持 `ndevice=2` 或 `ndevice=4`，默认配置使用双设备模型。确保已经完成编译，并根据设备组合传入对应的设备数量：

```bash
python3 demo.py --model_name laguna-s-2.1 --model_size 118b-a8b --ndevice 2
```

使用 24GB × 4 的 M50 设备时执行：

```bash
python3 demo.py --model_name laguna-s-2.1 --model_size 118b-a8b --ndevice 4
```

模型推理示例需要 `48GB × 2` 或 `24GB × 4` 的后摩 M50 设备组合。

### 2.6 一键流程

```bash
# 默认执行 demo；需要已有预编译模型
bash test.sh

# 完整流程：下载原始模型、量化、编译并运行 demo
bash test.sh -s all

# 仅量化，跳过已完成的下载
bash test.sh -s quant --skip_download
```

`test.sh` 支持的主要参数：

```text
-s, --step              demo、build、quant 或 all
-name, --model_name     模型名称
-size, --model_size     模型规格
--skip_download         跳过下载步骤
--system_prompt         传递给 demo 的 system prompt
```

## 3 参考结果

### 3.1 演示结果

以下为 W4A8 量化模型在后摩 M50 设备上的参考性能记录。

### 3.2 性能结果

测试配置：Batch Size 为 1，输入长度为 54 tokens，输出长度为 730 tokens。以下数据来自 2026-09-02 的一次实测，仅作参考，实际结果会随设备、固件、runtime、编译配置和系统负载变化。

```text
Model Inference Performance Summary Report

Configuration Details:
  Batch Size:                         1
  Input Length per Sample:           54 tokens
  Output Length per Sample:         730 tokens
  Prefill Model Load Time:      260312.67 ms
  Decode Model Load Time:         1412.42 ms

Prefill Stage Performance:
  Total Time:                    418.85 ms
  Speed:                         128.92 tokens/s
  Embedding Time:                 2.79 ms
  API SetInput Time:              0.53 ms
  API Inference Time:           414.70 ms
  Prefill Speed:                130.21 tokens/s
  API GetOutput Time:             0.06 ms

Decode Stage Performance:
  Total Time:                  38156.33 ms
  Speed:                         19.11 tokens/s
  Embedding Time:              152.92 ms
  API SetInput Time:             0.18 ms
  API Inference Time:            51.82 ms
  Decode Speed:                  19.30 tokens/s
  API GetOutput Time:             0.11 ms

Overall Performance Metrics:
  TTFT:                         418.85 ms
  TPOT:                          52.34 ms/token
  E2E Latency:                   42.87 seconds
  E2E TPS:                       17.03 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

此外，Laguna-S-2.1 模型材料受 [OpenMDW-1.1 License](https://modelscope.cn/models/poolside/Laguna-S-2.1/file/view/master/LICENSE.md) 约束；该模型许可证与本仓库源码的 Apache License 2.0 相互独立。本仓库不对第三方模型材料授予额外的许可或权利。
