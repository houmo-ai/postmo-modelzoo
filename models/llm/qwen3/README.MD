# Qwen3(通义千问3) - 0.6B/1.7B/8B/14B 统一示例

本示例展示如何把Qwen3模型（支持0.6B、1.7B、8B和14B版本）量化和编译，部署到后摩芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称:  Qwen3-0.6B / Qwen3-1.7B / Qwen3-8B / Qwen3-14B
- 来源: https://modelscope.cn/models/Qwen/Qwen3-0.6B / https://modelscope.cn/models/Qwen/Qwen3-1.7B / https://modelscope.cn/models/Qwen/Qwen3-8B / https://modelscope.cn/models/Qwen/Qwen3-14B
- 许可: Apache License 2.0

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

本例使用的模型实现来源于：
- Qwen3-0.6B: https://modelscope.cn/models/Qwen/Qwen3-0.6B
- Qwen3-1.7B: https://modelscope.cn/models/Qwen/Qwen3-1.7B
- Qwen3-8B: https://modelscope.cn/models/Qwen/Qwen3-8B
- Qwen3-14B: https://modelscope.cn/models/Qwen/Qwen3-14B

本示例通过 `--model_size` 参数（0.6b/1.7b/8b/14b）来区分不同模型版本。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

**0.6B模型：**
```bash
python3 get_model.py --type raw --model_size 0.6b
```

**1.7B模型：**
```bash
python3 get_model.py --type raw --model_size 1.7b
```

**8B模型：**
```bash
python3 get_model.py --type raw --model_size 8b
```

**14B模型：**
```bash
python3 get_model.py --type raw --model_size 14b
```

如果仅查看演示结果，可选择下载芯片模型：
```bash
python3 get_model.py --type hmm --model_size 8b  # 或 0.6b / 1.7b / 14b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/llm/qwen3`目录，执行ptq脚本。

**0.6B模型：**
```bash
source env.sh
cd models/llm/qwen3
python3 ptq.py --model_size 0.6b
```

**1.7B模型：**
```bash
source env.sh
cd models/llm/qwen3
python3 ptq.py --model_size 1.7b
```

**8B模型：**
```bash
source env.sh
cd models/llm/qwen3
python3 ptq.py --model_size 8b
```

**14B模型：**
```bash
source env.sh
cd models/llm/qwen3
python3 ptq.py --model_size 14b
```

为帮助您的模型在量化后达到最佳性能，您可以使用自定义校准数据集：

```bash
python3 ptq.py --model_size 8b --calib_data ~/your_data.jsonl
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- decoder
|-- prefill
|-- quant_embedding.pt
`-- weight.npy
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。

**0.6B模型：**
```bash
python3 build.py --model_size 0.6b
```

**1.7B模型：**
```bash
python3 build.py --model_size 1.7b
```

**8B模型：**
```bash
python3 build.py --model_size 8b
```

**14B模型：**
```bash
python3 build.py --model_size 14b
```

编译好的模型存放在当前目录，包括2个文件：
- qwen3_prefill.hmm
- qwen3_decode.hmm

### 2.4 演示

Qwen3模型使用python API进行演示。

**0.6B模型：**
```bash
python3 demo.py --model_size 0.6b
```

**1.7B模型：**
```bash
python3 demo.py --model_size 1.7b
```

**8B模型：**
```bash
python3 demo.py --model_size 8b
```

**14B模型：**
```bash
python3 demo.py --model_size 14b
```

### 2.5 一键评估

以上步骤可以通过test.sh脚本一键执行：

**0.6B模型：**
```bash
bash test.sh --model_size 0.6b
```

**1.7B模型：**
```bash
bash test.sh --model_size 1.7b
```

**8B模型：**
```bash
bash test.sh --model_size 8b
```

**14B模型：**
```bash
bash test.sh --model_size 14b
```

## 3 参考结果

### 3.1 演示结果

两个模型输出内容相似，以下为示例：

```bash
question:
请介绍一下存算一体技术的优势
response:
存算一体技术（Compute-in-Memory, CIM）是一种将存储和计算功能集成在同一物理结构中的技术，旨在通过减少数据在存储和计算单元之间的传输，提升系统效率、降低能耗并优化性能...
```

### 3.2 性能结果

**Qwen3-0.6B:**
```bash
hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:     30 tokens
hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    196 tokens
hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time:  856.72ms
hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  178.91ms
hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
hmatc.utils.perf_infomations:show_summary:347 -   Total Time:   41.84ms | Speed:  716.97 tokens/s
hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   11.54ms
hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    4.28ms
hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:   0.77ms
hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 24.37ms | Prefill Speed: 1231.00 tokens/s
hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.39ms
hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 3634.80ms | Speed:   53.65 tokens/s
hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time:   50.68ms
hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   21.58ms
hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   0.65ms
hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 12.92ms | Decode Speed:   77.42 tokens/s
hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.47ms
hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):   41.84 ms
hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 18.64 ms/token
hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):      3.70 seconds
hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         53.01 tokens/s
```

**Qwen3-1.7B:**
```bash
hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:     30 tokens
hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    693 tokens
hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 10204.87ms
hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  325.14ms
hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
hmatc.utils.perf_infomations:show_summary:347 -   Total Time:   51.41ms | Speed:  583.59 tokens/s
hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   14.45ms
hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    3.00ms
hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:   1.63ms
hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 30.09ms | Prefill Speed:  996.91 tokens/s
hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.38ms
hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 15815.78ms | Speed:   43.75 tokens/s
hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time:  205.59ms
hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   82.94ms
hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   0.70ms
hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 16.46ms | Decode Speed:   60.77 tokens/s
hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.52ms
hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):   51.41 ms
hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 22.86 ms/token
hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     15.94 seconds
hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         43.47 tokens/s
```

**Qwen3-8B:**
```bash
hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:     30 tokens
hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    621 tokens
hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 36855.28ms
hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  159.15ms
hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  125.88ms | Speed:  238.32 tokens/s
hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   12.17ms
hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    0.39ms
hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:   2.56ms
hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 107.54ms | Prefill Speed:  278.97 tokens/s
hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.58ms
hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 35726.51ms | Speed:   17.35 tokens/s
hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time:  188.39ms
hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   76.22ms
hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   0.66ms
hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 50.90ms | Decode Speed:   19.65 tokens/s
hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.54ms
hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  125.88 ms
hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 57.62 ms/token
hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     35.92 seconds
hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         17.29 tokens/s
```

**Qwen3-14B:**
```bash
hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:     30 tokens
hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    728 tokens
hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 67933.86ms
hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  337.75ms
hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  210.32ms | Speed:  142.64 tokens/s
hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   11.87ms
hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    0.43ms
hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:   3.60ms
hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 193.68ms | Prefill Speed:  154.90 tokens/s
hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.28ms
hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 68261.80ms | Speed:   10.65 tokens/s
hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time:  188.00ms
hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   73.75ms
hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   0.56ms
hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 87.06ms | Decode Speed:   11.49 tokens/s
hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.55ms
hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  210.32 ms
hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 93.90 ms/token
hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     68.54 seconds
hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         10.62 tokens/s
```

### 3.3 性能测试工具

- 使用llm_perf性能评测工具测试模型性能。
- llm_perf工具详细使用说明参考${HOUMO_EXAMPLE_PATH}/tools/llm_perf目录的README.MD。
- 如果使用发布的容器，llm_perf工具已经预编译安装，可以直接通过llm_perf --help获取帮助。
- config-0.6b.yaml / config-1.7b.yaml / config-8b.yaml / config-14b.yaml 为llm_perf执行测试的配置文件，用户可以按照需要自行修改。

## 4 命令行参数说明

### get_model.py
- `--model_size`: 模型大小，0.6b、1.7b、8b 或 14b（默认：8b）
- `--type`: 下载类型，raw 或 hmm（默认：hmm）
- `--download_dir`: 下载目录（默认：当前目录）
- `--source_type`: 下载源，jfrog 或 modelscope（默认：jfrog）
- `--context_length`: 上下文长度（默认：8k）
- `--batch`: 批次大小（默认：1）
- `--ndevice`: 设备数量（默认：1）

### ptq.py
- `--model_size`: 模型大小，0.6b、1.7b、8b 或 14b（默认：8b）
- `--model`: 模型路径（默认根据 model_size 自动设置）
- `--calib_data`: 校准数据集路径（默认根据 model_size 自动设置）
- `--context-length`: 最大序列长度（默认：2048）
- `--input-sequence-length`: 输入序列长度（默认：256）
- `--quant-type`: 量化类型（默认：w4a8h0_ssfp）

### build.py
- `--model_size`: 模型大小，0.6b、1.7b、8b 或 14b（默认：8b）
- `--model_name`: 输出模型名称（默认：qwen3）
- `--batch`: 批次大小（默认：1）
- `--context_length`: 上下文长度（默认：2048）
- `--flash_attention`: Flash Attention优化（默认：2）
- `--device_kernel_split`: 设备内核分割（主要用于14B模型，默认：0）
- `--stage`: 构建阶段，build/test/all（默认：build）

### demo.py
- `--model_size`: 模型大小，0.6b、1.7b、8b 或 14b（默认：8b）
- `--tokenizer_dir`: tokenizer目录（默认根据 model_size 自动设置）
- `--prefill_path`: prefill模型路径
- `--decode_path`: decode模型路径
- `--ndevice`: 设备数量（默认：1）
- `--question`: 提问内容
- `--it`: 交互模式
- `--history`: 保留对话历史

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
