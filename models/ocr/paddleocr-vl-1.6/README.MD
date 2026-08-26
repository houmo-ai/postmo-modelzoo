# PaddleOCR-VL

本示例展示如何把 PaddleOCR-VL-1.6 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: PaddleOCR-VL-1.6
- 来源: https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型、tokenizer 和相关权重在运行时下载，工程发布**不包含**这些第三方资产。

### 1.1 第三方组件与合规边界

- `PaddleOCR-VL-1.6` 的模型权重、tokenizer 和原始模型卡来自上游发布页；请以上游仓库和模型卡中列出的许可与使用条款为准。
- `image_processing.py` 和 `processing_paddleocr_vl.py` 基于 Hugging Face Transformers 5.15.0 的 PaddleOCR-VL processor 实现修改，保留上游 Apache-2.0 声明并标注了本项目修改内容。
- `demo.py` 中的表格后处理和 `spotting` 后处理可选依赖 `PaddleX`；该依赖不随本仓库分发，需由用户按其各自许可证单独安装和使用。

如需重新分发模型、数据集、PaddleX 或其他外部依赖，请先核验其原始许可证、商用限制和再分发条件。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=paddleocr-vl-1.6`
- `model_size=0.9b`
- `context_length=16k`
- `prefill_length=256`
- `max_size_w=896`
- `max_size_h=896`
- `ndevice=1`

### 推理能力

`demo.py` 支持以下任务：

- `ocr`：文本识别
- `table`：表格识别并转换为 HTML
- `formula`：公式识别
- `chart`：图表识别
- `spotting`：文本识别与定位
- `seal`：印章文字识别

模型使用固定视觉输入尺寸，默认配置为 `896x896`。`demo.py` 支持 `stretch` 和 `letterbox` 两种静态缩放方式。

`demo.py` 还补充了推理性能统计：

- 运行结束后输出 TTFT breakdown
- 运行结束后打印 `InferencePerformanceTracker` 汇总
- 记录视觉、Prefill 和 Decode 的 load / setinput / infer / getoutput 时间

## 2 功能验证与结果对比

本目录已完成 `ocr`、`table`、`formula`、`chart`、`spotting` 和 `seal` 六类功能验证。

- `demo.py` 支持 HMM 模型加载、图像预处理、Prefill、Decode 和六类任务后处理。
- 表格与 spotting 后处理在 PaddleX 可用时启用；不可用时会保留原始输出。

## 3 快速开始

本示例仅适用于 Linux 和 XH2。模型通过 Python 脚本完成量化、编译和 HMM 推理。

### 3.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv

```bash
pip3 install -r requirements.txt
```

运行依赖请参考当前目录的 `requirements.txt`。

### 3.2 获取模型

通过 `get_model.py` 下载原始模型或预编译 HMM 模型。

下载原始模型：

```bash
python3 get_model.py --type raw --model_name paddleocr-vl-1.6 --model_size 0.9b
```

下载预编译 HMM 模型：

```bash
python3 get_model.py --type hmm --model_name paddleocr-vl-1.6 --model_size 0.9b
```

### 3.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后执行：

```bash
source env.sh
cd models/ocr/paddleocr-vl-1.6
python3 ptq.py --model-name paddleocr-vl-1.6 --model-size 0.9b
```

量化产物默认位于 `output/$HOUMO_TARGET/hmquant`。

```text
output/xh2/hmquant
|-- prefill/
|-- decode/
|-- visual/
`-- quant_embedding.pt
```

### 3.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name paddleocr-vl-1.6 --model_size 0.9b
```

默认生成 visual、prefill 和 decode 三个 HMM 文件。可按需调整上下文长度和 prefill 长度：

```bash
python3 build.py --model_name paddleocr-vl-1.6 --model_size 0.9b --context_length 4096 --prefill_length 256 --flash_attention 2 1
```

### 3.5 HMM 推理

`demo.py` 使用 `tcim_lite` 加载 visual、prefill 和 decode HMM 模型。

执行脚本即可进行 HMM 推理。

```bash
python3 demo.py --model_name paddleocr-vl-1.6 --model_size 0.9b
```

指定图片：

```bash
python3 demo.py --model_name paddleocr-vl-1.6 --model_size 0.9b --image_path ../../../data/pic/ocr.jpeg
```

如需使用其他已编译静态分辨率，请同时指定匹配的 `--max_size_w`、`--max_size_h` 和 visual HMM 文件。

#### 3.5.1 性能示例

运行 `demo.py` 时会输出 TTFT 分阶段耗时和完整性能汇总：

```bash
python3 demo.py --task seal --image_path <image_path>
```

以下为单张 `896x896` 图片、输入长度 1037 tokens、生成 9 tokens 的一次实测结果。模型加载时间不计入端到端推理耗时，不同硬件状态和输入内容会影响结果。

| 指标 | 结果 |
| --- | ---: |
| Prefill 模型加载 | 129.04 ms |
| Decode 模型加载 | 28.36 ms |
| Vision 模型加载 | 18596.94 ms |
| TTFT | 4022.02 ms |
| Vision 推理 | 3857.93 ms |
| Prefill 推理 | 78.13 ms |
| Decode 推理 | 7.55 ms/token |
| TPOT | 9.03 ms/token |
| E2E Latency | 4.09 s |

TTFT 包含 prompt/tokenize、视觉处理和 Prefill，不包含模型加载时间。`Vision Infer` 是该输入下 TTFT 的主要耗时。

## 4 一键评估

以上步骤可以通过 `test.sh` 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `paddleocr-vl-1.6` 和 `0.9b`。

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
  -name, --model_name     Model name. Default: paddleocr-vl-1.6.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载 raw 模型并使用脚本内默认参数量化
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

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
