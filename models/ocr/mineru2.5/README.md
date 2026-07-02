# MinerU2.5

本示例展示如何把 MinerU2.5 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: MinerU2.5-Pro-2604-1.2B
- 来源: https://www.modelscope.cn/models/OpenDataLab/MinerU2.5-Pro-2604-1.2B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=mineru2.5-pro-2604`
- `model_size=1.2b`
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`

MinerU2.5 是基于 Qwen2-VL 架构的视觉语言模型，支持文档版面检测和文字识别。模型分为三个部分：

- **LLM（语言模型）**：负责文本生成，包含 prefill 和 decode 两个子模型
- **ViT（视觉模型）**：负责图像特征提取，采用多分桶（multi-bucket）策略以适配不同尺寸的图像输入
- **Token Embedding**：词嵌入层，以独立文件形式提供

编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`
- `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}.hmm`
- `${model_name}-${model_size}_visual_${h}x${w}.hmm`（各静态分桶）

脚本默认会从 `modelscope_repo` 取最后一段作为 tokenizer 或模型目录默认值，例如 `OpenDataLab/MinerU2.5-Pro-2604-1.2B` 会推导为 `MinerU2.5-Pro-2604-1.2B`。

### 视觉分桶（Visual Buckets）

MinerU2.5 的视觉模型采用多分桶策略，为不同尺寸的图像选择最优的视觉模型，以减少 padding 开销并提升识别精度。默认分桶配置如下：

| 类别 | 分桶尺寸 (H×W) | 适用场景 |
|------|---------------|---------|
| 中等水平内容 | 140×392 | 标题、短文本、说明文字、小表格 |
| | 196×560 | |
| | 280×784 | |
| | 392×1036 | |
| 长水平内容 | 112×1792 | PPT 文本条、公式、宽表格行 |
| | 168×1792 | |
| | 252×1792 | |
| | 392×2044 | |
| 非水平内容 | 560×560 | 方形/竖向内容 |
| | 1036×392 | |

其中 `1036×1036` 为 fallback 分桶，当输入图像不匹配任何静态分桶时使用。

## 2 快速开始

目前 MinerU2.5 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name mineru2.5-pro-2604 --model_size 1.2b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name mineru2.5-pro-2604 --model_size 1.2b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/ocr/mineru2.5
```

推荐先下载原始模型，再执行：

```bash
python3 ptq.py --model models/MinerU2.5-Pro-2604-1.2B
```

`ptq.py` 支持以下常用参数：

```bash
python3 ptq.py --model <模型路径> --chip-arch XH2a --context-length 4096 --prefill-chunk-length 256
```

量化过程会同时导出 LLM（prefill/decode）和 ViT（含所有静态分桶）模型，并生成 `mineru_visual_buckets.json` 分桶清单文件。

量化结果位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name mineru2.5-pro-2604 --model_size 1.2b
```

默认输出名称分别为：

- `mineru2.5-pro-2604-1.2b_prefill.hmm`
- `mineru2.5-pro-2604-1.2b_decode.hmm`
- `mineru2.5-pro-2604-1.2b_visual_1036x1036.hmm`
- `mineru2.5-pro-2604-1.2b_visual_<h>x<w>.hmm`（各静态分桶）

`build.py` 支持以下常用参数:

```bash
python3 build.py --model_name mineru2.5-pro-2604 --model_size 1.2b --flash_attention 2 1
```

其中 `--flash_attention` 接受两个参数：第一个为 LLM FlashAttention 开关（0/1/2），第二个为 ViT FlashAttention 开关（0/1）。

### 2.5 演示

MinerU2.5 模型使用 python API 进行演示，支持文档版面检测和文字识别两阶段提取。

```bash
python3 demo.py --model_name mineru2.5-pro-2604 --model_size 1.2b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`、`vit_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

可通过 `--image` 指定输入图像：

```bash
python3 demo.py --model_name mineru2.5-pro-2604 --model_size 1.2b --image ./data/0002.png
```

演示脚本会输出：

- 版面检测结果（JSON 格式的 block 列表，包含类型、边界框、旋转角度等）
- 带标注的版面检测图像（`<输入图像名>_layout_boxes.png`）

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=mineru2.5-pro-2604`
- `MODEL_SIZE=1.2b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Choices: 1.2b.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s demo
bash test.sh -s all
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 prefill、decode 和 visual（含所有分桶）模型。
- `demo`：下载预编译模型后执行 Python demo。

## 4 参考结果

### 4.1 演示结果

使用 `data/0002.png` 作为输入图像，演示脚本输出版面检测和文字识别结果：

```bash
[MinerU2.5] Layout  detect done in X.XXs — N blocks (M to extract)
[MinerU2.5] Extract [1/M] type=text size=WxH | X.XXs  result: ...
[MinerU2.5] Extract done in X.XXs (X.XXs layout + X.XXs recognition)
```

版面检测结果（JSON 格式）：

```json
[
  {"type": "text", "bbox": [x1, y1, x2, y2], "angle": null, "context": "识别的文字内容..."},
  {"type": "title", "bbox": [x1, y1, x2, y2], "angle": null, "context": "标题内容..."},
  {"type": "table", "bbox": [x1, y1, x2, y2], "angle": null, "context": null}
]
```

同时生成带标注的版面检测图像 `data/0002_layout_boxes.png`，其中不同类型的区域使用不同颜色标注。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
