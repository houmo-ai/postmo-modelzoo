# Gemma4

本示例展示如何把Gemma4模型量化和编译，部署到后摩芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: gemma-4-26B-A4B-it
- 来源: https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it
- 许可: Apache License 2.0 （https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md）

预训练模型在运行时下载，工程发布**不包含**该模型。

## 数据集使用声明

本项目使用公开数据集进行量化校准。数据集采用Apache License 2.0开源协议授权。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it](https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it)。

## 2 快速开始

目前Gemma4模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化下载的原始模型。在当前目录下，执行脚本：

```bash
python3 ptq.py
```

量化参数说明：
- `--model`: 模型目录路径，默认 `./models/gemma-4-26B-A4B-it`
- `--context-length`: 最大序列长度，默认 2048
- `--nsamples`: 校准样本数，默认 512
- `--seqlen`: 校准序列长度，默认 1024
- `--mse`: MSE阈值，默认 2.4
- `--bits`: 量化位数，默认 4
- `--group-size`: 量化分组大小，默认 64
- `--calibration-jsonl`: 校准数据集路径，默认 `./calib_EBSS.jsonl`

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在output目录，包括3个文件：
- gemma4_26b-a4b_prefill.hmm
- gemma4_26b-a4b_decode.hmm
- gemma4_26b-a4b_visual.hmm

### 2.5 演示

Gemma4模型使用python API进行演示。

1. 将编译好的模型放在output目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出结果。

```bash
# 文本模式
python3 demo.py --question "你好，请介绍一下你自己。"

# 图片模式
python3 demo.py --image ../../../data/pic/beach.jpeg --question "描述这张图片的内容"
```

演示参数说明：
- `--tokenizer_dir`: tokenizer目录路径
- `--embedding_path`: embedding权重路径
- `--prefill_path`: prefill模型路径
- `--decode_path`: decode模型路径
- `--vit_path`: visual模型路径
- `--device`: 设备ID
- `--question`: 输入问题
- `--image`: 图片路径
- `--max-new-tokens`: 最大生成token数

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question: 你好，请介绍一下你自己。
response: 你好！我是 **Gemma 4**，是由 Google DeepMind 开发的大型语言模型。

作为一个开放权重的模型，我旨在通过自然语言交互来协助用户完成各种任务。以下是关于我的一些详细介绍：

### 1. 我的核心能力
* **文本处理**：我可以进行创作、翻译、总结、问答以及逻辑推理。
* **多模态理解**：除了文本，我还可以理解和处理**图像**信息。
* **音频处理**：在 Gemma 4 系列中，2B 和 4B 模型还具备处理**音频**输入的能力。
* **知识范围**：我的知识截止日期是 **2025 年 1 月**。

### 2. 我的输出形式
虽然我可以接收多种形式的输入（文本、图像、部分型号支持音频），但我**仅能生成文本**作为输出。我无法生成图像或音频文件。

### 3. 我的工作方式
* **工具使用**：除非在对话上下文中明确为我提供了特定的工具定义和端点，否则我无法直接访问互联网（如 Google 搜索）或使用外部工具。
* **基于上下文**：我非常擅长根据你提供的上下文信息进行回答，这有助于我提供更精准、更具针对性的帮助。

**你可以尝试这样问我：**
* “请帮我写一封正式的商务邮件。”
* “解释一下量子力学的基本概念。”
* “（上传一张图片）请描述一下这张图片里的内容。”

请问今天有什么我可以帮你的吗

2026-05-10 08:35:45.315 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:316 - ====================================================================================================
2026-05-10 08:35:45.315 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:317 -                     Model Inference Performance Summary Report
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:318 - ====================================================================================================
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:     20 tokens
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    336 tokens
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 120483.64ms
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  809.74ms
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:333 -   Vision Model Load Time: 6244.74ms
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  261.68ms | Speed:   76.43 tokens/s
2026-05-10 08:35:45.316 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   71.45ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    0.35ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:  64.78ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 121.58ms | Prefill Speed:  164.50 tokens/s
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  2.62ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 13198.44ms | Speed:   25.38 tokens/s
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:363 -   Tokenization Time: Skipped (No operation)
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   66.12ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   1.19ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 35.20ms | Decode Speed:   28.41 tokens/s
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  2.33ms
2026-05-10 08:35:45.317 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
2026-05-10 08:35:45.318 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  261.68 ms
2026-05-10 08:35:45.318 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 39.40 ms/token
2026-05-10 08:35:45.318 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     13.50 seconds
2026-05-10 08:35:45.318 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         24.89 tokens/s
2026-05-10 08:35:45.318 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:376 - ====================================================================================================
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
