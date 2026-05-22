# CoPaw-Flash

本示例展示如何将 CoPaw-Flash 模型量化、编译并部署到后摩鸿途芯片设备上，支持图片理解与图文问答场景。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称：CoPaw-Flash-9B
- 原始模型来源：https://modelscope.cn/models/AgentScope/CoPaw-Flash-9B
- 许可：Apache License 2.0

预训练模型在运行时下载，工程发布**不包含**该模型。

## 代码版权说明

本目录中的以下文件包含基于第三方开源实现的适配代码，后续如对外开源、分发或单独复用，需**保留原始版权声明、修改说明及 Apache License 2.0 许可信息**：

- `image_processing_copawflash.py`
- `vision_process_copawflash.py`

其中：
- `image_processing_copawflash.py` 基于 HuggingFace Transformers / Qwen2-VL 相关图像处理实现改写；
- `vision_process_copawflash.py` 为视觉预处理流程适配代码，开源时需明确标注其来源、修改内容与继承许可。

[TOC]

## 1 模型说明

本例使用的 CoPaw-Flash 模型实现来源于 AgentScope 发布的 CoPaw-Flash 系列模型，适配目标为后摩 xh2 芯片平台。

当前示例聚焦于单图输入的视觉理解与图文问答流程，整体包含以下三个子模型：

- Vision：视觉编码模型
- Prefill：首 token 预填充模型
- Decode：逐 token 解码模型

编译完成后，可通过 Python API 完成图片理解与多模态问答演示。

## 2 快速开始

目前 CoPaw-Flash 仅提供 Python 脚本量化与编译方式。芯片模型仅能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 Python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 脚本下载原始模型或已编译的芯片模型。

下载原始模型（用于量化 + 编译）：

```bash
python3 get_model.py --type raw
```

下载已编译芯片模型（仅用于演示）：

```bash
python3 get_model.py --type hmm
```

如需从 ModelScope 下载，可附加参数：

```bash
python3 get_model.py --type raw --source_type modelscope
```

默认示例图片使用 `${HOUMO_PIC_PATH}/beach.jpeg`，默认校准文本使用当前目录下的 `qwen3_8b_gen_data.jsonl`。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.2 兼容的 CUDA 和驱动。启动 docker 时请将 GPU 映射到容器内，并指定可用 GPU。**建议至少准备 24GB GPU 显存，以及 200GB 以上 CPU 空闲内存。** 参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，source 环境变量并进入 `models/llm/CoPaw-Flash` 目录：

```bash
source env.sh
cd models/llm/CoPaw-Flash
```

执行量化脚本：

```bash
python3 ptq.py --gptqmodel
```

如需指定原始模型目录、上下文长度或输入长度，可参考：

```bash
python3 ptq.py \
  --model /path/to/CoPaw-Flash-9B \
  --context-length 2048 \
  --input-sequence-length 256 \
  --gptqmodel
```

量化完成的模型存放在 `output/$HOUMO_TARGET/hmquant` 目录，典型结构如下：

```bash
output/$HOUMO_TARGET/hmquant/
├── prefill/
├── decode/
├── visual/
└── quant_embedding.pt
```

### 2.4 编译

将量化模型编译为可在芯片上运行的 `.hmm` 模型：

```bash
python3 build.py
```

如需指定上下文长度、prefill 长度或 FlashAttention 选项，可参考：

```bash
python3 build.py --context_length 8192 --prefill_length 256 --flash_attention 2 1
```

编译输出默认位于 `output/$HOUMO_TARGET` 目录，包括：

- `copaw-flash_prefill.hmm`
- `copaw-flash_decode.hmm`
- `copaw-flash_visual.hmm`

当 `--ndevice 2` 时，LLM 模型会生成对应的多设备模型文件。

### 2.5 演示

CoPaw-Flash 模型使用 Python API 进行演示。

直接运行默认示例：

```bash
python3 demo.py
```

指定图片进行单次推理：

```bash
python3 demo.py --image_path ../../../data/pic/beach.jpeg
```

开启交互模式：

```bash
python3 demo.py --it
```

开启多轮历史对话保留：

```bash
python3 demo.py --it --history
```

如需手动指定 tokenizer、embedding、vision/prefill/decode 模型路径，可附加对应参数。

### 2.6 一键评估

以上步骤可以通过 `test.sh` 脚本一键执行：

```bash
bash test.sh
```

默认流程会下载预编译模型并执行 demo。

如需走“原始模型下载 → 量化 → 编译 → 演示”完整流程，可执行：

```bash
bash test.sh -t compile
```

仅执行编译相关步骤：

```bash
bash test.sh -s build -t compile
```

仅执行演示：

```bash
bash test.sh -s demo
```

## 3 参考结果

### 3.1 演示结果

以下为单张图片问答的参考输出：

```bash
question:
分析这张图片中主要内容。
response:
这张图片呈现了一个宁静而富有情感的场景，主要内容如下：

- **主体人物与动物**：画面中有一位女性和一只大型犬（看起来像金毛寻回犬或类似品种）。女性留着长发，身穿格子衬衫和深色裤子，坐在沙地上。狗狗则戴着蓝色项圈，正抬头望着女性，姿态温顺。

- **互动关系**：女性似乎正在与狗狗互动——可能是在喂食、抚摸或轻声说话，营造出一种亲密、信任的氛围。

- **环境背景**：场景位于一片开阔的沙滩上，远处是模糊的海平线和柔和的天空，整体光线偏暗，色调沉静，可能是在黄昏或阴天拍摄，增强了画面的静谧感。

- **整体氛围**：图像传达出人与自然、人与动物之间和谐共处的温馨意境，带有一种孤独却不寂寞的诗意美感。

总结：这是一张表现人与宠物在自然环境中共享宁静时刻的照片，强调陪伴、信任与平和的情感联结。
```

### 3.2 性能结果

以下为同一次运行的性能统计摘要：

```bash
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:316 - ====================================================================================================
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:317 -                     Model Inference Performance Summary Report
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:318 - ====================================================================================================
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:    216 tokens
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    203 tokens
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:326 -   Number of Images:      1 images
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 43709.12ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time: 1104.04ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:333 -   Vision Model Load Time: 7247.46ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:338 - Vision Stage Performance:
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:339 -   Total Time:  262.85ms | Speed:    3.80 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:340 -   Preprocessing Time: 107.21ms | Speed:    9.33 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:341 -   API SetInput Time:   3.77ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:342 -   API Inference Time: 149.75ms | Speed:    6.68 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:343 -   API GetOutput Time:  1.34ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  465.60ms | Speed:  463.92 tokens/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   18.43ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    7.40ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:  11.27ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 156.47ms | Prefill Speed: 1380.47 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.64ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 12759.69ms | Speed:   15.83 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time: 2436.97ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   27.95ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   1.83ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 48.63ms | Decode Speed:   20.57 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.60ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  728.45 ms
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 63.17 ms/token
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     13.92 seconds
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         14.58 tokens/s
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:376 - ====================================================================================================
```

### 3.3 备注

- 当前 demo 已内置视觉、Prefill、Decode 各阶段性能统计；
- 若仅验证功能，建议优先下载预编译模型直接运行；
- 若需重新量化，请优先确认 GPU 显存、主机空闲内存以及 CUDA 版本满足要求。

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
