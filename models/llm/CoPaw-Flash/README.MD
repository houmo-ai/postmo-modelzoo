# CoPaw-Flash

本示例展示如何把 CoPaw-Flash 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: CoPaw-Flash-9B
- 来源: https://www.modelscope.cn/models/AgentScope/CoPaw-Flash-9B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=copaw-flash`
- `model_size=9b`
- `context_length=8k`
- `prefill_length=256`

视觉尺寸默认值为 `448x448x2`。

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`
- `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

tokenizer 和原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `CoPaw-Flash-9B`。

## 2 快速开始

目前 CoPaw-Flash 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name copaw-flash --model_size 9b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name copaw-flash --model_size 9b
```

如需从 ModelScope 下载，可附加：

```bash
python3 get_model.py --type raw --source_type modelscope --model_name copaw-flash --model_size 9b
```

### 2.3 量化

`ptq.py` 需要 GPU 环境，并要求宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/CoPaw-Flash
```

默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

`test.sh -s quant` 在执行量化前还会自动检查并安装 `hmodel/gptqmodel`，随后以 `--gptqmodel` 方式调用 `ptq.py`。

```bash
python3 ptq.py --model-name copaw-flash --model-size 9b --gptqmodel
```

如果要降低导出规模或使用自定义校准文本，可显式传参：

```bash
python3 ptq.py --model-name copaw-flash --model-size 9b --gptqmodel \
  --context-length 2048 \
  --input-sequence-length 256 \
  --calib_data /path/to/custom.jsonl
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name copaw-flash --model_size 9b
```

如果需要显式调整上下文长度或 FlashAttention：

```bash
python3 build.py --model_name copaw-flash --model_size 9b --context_length 8192 --prefill_length 256 --flash_attention 2 1
```

### 2.5 演示

默认单次 demo 会使用 `${HOUMO_PIC_PATH}/beach.jpeg` 作为输入图片；若环境变量未设置，则回退到仓库内 `data/pic/beach.jpeg`。

```bash
python3 demo.py --model_name copaw-flash --model_size 9b
```

指定图片：

```bash
python3 demo.py --model_name copaw-flash --model_size 9b --image_path ../../../data/pic/beach.jpeg
```

交互模式：

```bash
python3 demo.py --model_name copaw-flash --model_size 9b --it
```

如未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`、`vision_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=copaw-flash`
- `MODEL_SIZE=9b`

默认执行 `demo` 步骤：

```bash
bash test.sh
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s all
bash test.sh -s demo --skip_download
```

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 9b.
  -name, --model_name     Model name. Default: copaw-flash.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本行为说明：

- `quant`：可自动回溯或使用 `HOUMO_EXAMPLES_PATH` 定位仓库根目录，安装 `gptqmodel` 后执行 `ptq.py --gptqmodel`。
- `build`：调用 `build.py` 编译 `prefill`、`decode` 和 `visual` 模型。
- `demo`：下载预编译模型后执行 Python demo。
- 当前目录没有单独的 `llm_perf` 流程；性能信息由 `demo.py` 内部统计输出。

## 4 参考结果

### 4.1 演示结果

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

### 4.2 性能结果

以下为同一次运行的性能统计摘要：

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:    216 tokens
  Output Length per Sample:    203 tokens
  Number of Images:      1 images
  Prefill Model Load Time: 43709.12ms
  Decode Model Load Time: 1104.04ms
  Vision Model Load Time: 7247.46ms
Vision Stage Performance:
  Total Time:  262.85ms | Speed:    3.80 images/s
  Preprocessing Time: 107.21ms | Speed:    9.33 images/s
  API SetInput Time:   3.77ms
  API Inference Time: 149.75ms | Speed:    6.68 images/s
  API GetOutput Time:  1.34ms
Prefill Stage Performance:
  Total Time:  465.60ms | Speed:  463.92 tokens/s
  Tokenization Time:   18.43ms
  Embedding Time:    7.40ms
  API SetInput Time:  11.27ms
  API Inference Time: 156.47ms | Prefill Speed: 1380.47 tokens/s
  API GetOutput Time:  0.64ms
Decode Stage Performance:
  Total Time: 12759.69ms | Speed:   15.83 tokens/s
  Tokenization Time: 2436.97ms
  Embedding Time:   27.95ms
  API SetInput Time:   1.83ms
  API Inference Time: 48.63ms | Decode Speed:   20.57 tokens/s
  API GetOutput Time:  0.60ms
Overall Performance Metrics:
  TTFT (Time To First Token):  728.45 ms
  TPOT (Time Per Output Token): 63.17 ms/token
  E2E Latency (End-to-End):     13.92 seconds
  E2E TPS (Throughput):         14.58 tokens/s
====================================================================================================
```

本目录中的以下文件包含基于第三方开源实现的适配代码，后续如对外开源、分发或单独复用，需保留原始版权声明、修改说明及 Apache License 2.0 许可信息：

- `image_processing_copawflash.py`
- `vision_process_copawflash.py`

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
