# PaddleOCR-VL

本示例展示如何把 PaddleOCR-VL 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: PaddleOCR-VL
- 来源: https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=paddleocr-vl`
- `model_size=0.9b`
- `context_length=4k`
- `prefill_length=256`
- `max_size_w=448`
- `max_size_h=448`
- `ndevice=1`

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 PaddleOCR-VL 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

如果需要自己量化，可选择下载原始模型：

```bash
python3 get_model.py --type raw --model_name paddleocr-vl --model_size 0.9b
```

如果仅查看演示结果，可选择下载芯片模型：

```bash
python3 get_model.py --type hmm --model_name paddleocr-vl --model_size 0.9b
```

如需从 ModelScope 下载，可附加 `--source_type modelscope`：

```bash
python3 get_model.py --type raw --source_type modelscope --model_name paddleocr-vl --model_size 0.9b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/ocr/paddleocr-vl` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd models/ocr/paddleocr-vl
python3 ptq.py --model-name paddleocr-vl --model-size 0.9b
```

常用参数：

- `--model-name`: 输出模型名前缀，默认 `paddleocr-vl`
- `--model-size`: 输出模型规格，默认 `0.9b`
- `--context-length`: 最大序列长度，默认由 `config.yaml` 的 `4k` 解析得到
- `--input-sequence-length`: prefill 输入长度，默认 `256`
- `--quant-type`: 默认 `w8a8h0_ssfp`
- `--max-size-w`: 视觉输入宽度，默认 `448`
- `--max-size-h`: 视觉输入高度，默认 `448`
- `--nsamples`: wikitext 校准样本数
- `--seqlen`: wikitext 校准序列长度

默认产物目录为 `output/$HOUMO_TARGET/hmquant`，其中会包含以下核心文件或子目录：

```text
output/xh2/hmquant
|-- prefill/
|-- decode/
|-- visual/
`-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name paddleocr-vl --model_size 0.9b
```

默认输出名称分别为：

- `paddleocr-vl-0.9b_prefill.hmm`
- `paddleocr-vl-0.9b_decode.hmm`
- `paddleocr-vl-0.9b_visual_448x448.hmm`

如需显式调整上下文长度或 FlashAttention，可传入：

```bash
python3 build.py --model_name paddleocr-vl --model_size 0.9b --context_length 4096 --prefill_length 256 --flash_attention 2 1
```

### 2.5 演示

PaddleOCR-VL 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：
- `output/${HOUMO_TARGET}/paddleocr-vl-0.9b_prefill.hmm`
- `output/${HOUMO_TARGET}/paddleocr-vl-0.9b_decode.hmm`
- `output/${HOUMO_TARGET}/paddleocr-vl-0.9b_visual_448x448.hmm`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

2. 默认输入图片为 `${HOUMO_PIC_PATH}/ocr.jpeg`；若环境变量未设置，则回退到仓库内 `data/pic/ocr.jpeg`。默认任务为 `ocr`，对应提示词 `OCR:`。

3. 当 `ndevice > 1` 时，`prefill` 和 `decode` 会自动切换为 `.hmms`，视觉模型仍保持 `.hmm`。

4. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py --model_name paddleocr-vl --model_size 0.9b
```

指定图片：

```bash
python3 demo.py --model_name paddleocr-vl --model_size 0.9b --image_path ../../../data/pic/ocr.jpeg
```

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `paddleocr-vl` 和 `0.9b`。

对应执行行为如下：

- `quant`：如未传 `--skip_download`，先下载 raw 模型，再执行 `python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"`
- `build`：执行 `python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"`
- `demo`：当 `STEP=demo` 且未传 `--skip_download` 时，先下载预编译模型，再执行 `python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"`

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 0.9b.
  -name, --model_name     Model name. Default: paddleocr-vl.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载 raw 模型并使用脚本内默认的 paddleocr-vl-0.9b 参数量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 跳过模型下载，使用当前目录下已有模型文件执行 demo
bash test.sh -s demo --skip_download
```

## 4 参考结果

### 4.1 演示结果

```bash
question:
OCR:
images: ['/mnt/data/nan.xu/project/xh2_modelzoo/tmp_imodelzoo/imodelzoo/data/pic/ocr.jpeg']
processor image_grid_thw: tensor([[ 1, 32, 32]])
processor image token count: 256
image_embeds shape: torch.Size([256, 1024])
response:
后摩智能
M50芯片
```

### 4.2 性能结果

```bash
====================================================================================================
                    Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:    269 tokens
  Output Length per Sample:     10 tokens
  Number of Images:      1 images
  Prefill Model Load Time: 3484.83ms
  Decode Model Load Time:  118.31ms
  Vision Model Load Time: 7358.59ms
Vision Stage Performance:
  Total Time:  218.48ms | Speed:    4.58 images/s
  Preprocessing Time: 87.50ms | Speed:   11.43 images/s
  API SetInput Time:   0.57ms
  API Inference Time: 127.06ms | Speed:    7.87 images/s
  API GetOutput Time:  0.58ms
Prefill Stage Performance:
  Total Time:  262.86ms | Speed: 1023.37 tokens/s
  Tokenization Time:    5.36ms
  Embedding Time:    2.65ms
  API SetInput Time:   8.34ms
  API Inference Time: 24.74ms | Prefill Speed: 10873.40 tokens/s
  API GetOutput Time:  0.15ms
Decode Stage Performance:
  Total Time:   99.29ms | Speed:   90.64 tokens/s
  Tokenization Time:   29.07ms
  Embedding Time:    0.85ms
  API SetInput Time:   1.06ms
  API Inference Time:  6.44ms | Decode Speed:  155.25 tokens/s
  API GetOutput Time:  0.28ms
Overall Performance Metrics:
  TTFT (Time To First Token):  481.34 ms
  TPOT (Time Per Output Token): 11.03 ms/token
  E2E Latency (End-to-End):      0.44 seconds
  E2E TPS (Throughput):         22.77 tokens/s
====================================================================================================
```

### 4.3 备注

- 当前 demo 已内置视觉、Prefill、Decode 各阶段性能统计；
- 若仅验证功能，建议优先下载预编译模型直接运行；
- 若需重新量化，请优先确认 GPU 显存（24G空闲）、主机空闲内存（40G空闲）以及 CUDA 版本满足要求；
- 由于模型中存在大量reshape，如需修改context_length则需重新执行ptq.py和build.py。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
