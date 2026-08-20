# PaddleOCR V5

本示例展示 PaddleOCR V5 detection + recognition 模型在后摩 XH2 平台上的量化、编译和 HMM demo 运行流程。

[TOC]

## 1. 模型说明

本例使用的 PaddleOCR V5 实现来源于 PaddleOCR 开源项目：

- [PP-OCRv5 文档](https://github.com/PaddlePaddle/PaddleOCR/tree/46bfd1ef04e7545f33b2065525ffbc742f1a5930/docs/version3.x/algorithm/PP-OCRv5)

本目录包含两个模型：

- `paddleocr_v5_det`：文本检测模型
- `paddleocr_v5_rec`：文本识别模型

HMM demo 会先运行 det 模型得到文本框，再裁剪文本区域并运行 rec 模型完成识别。

## 2. 环境准备

```bash
cd imodelzoo/models/ocr/paddleocr_v5

# 如需手动安装依赖
python3 -m pip install -r requirements.txt
```

`test.sh` 会按 imodelzoo 约定自动创建临时 Python venv 并安装 `requirements.txt` 中的依赖。

## 3. 下载模型

`get_model.py` 支持下载三类模型：

| type | 说明 | 典型用途 |
| --- | --- | --- |
| `raw` | 原始 ONNX 模型和 det calibration input | 用于重新量化 |
| `hmm` | 已编译 HMM 模型 | 用于直接运行 demo |

### 3.1 下载原始 ONNX 模型

```bash
python3 get_model.py --type raw
```

下载内容包括：

```text
paddleocr_v5_det_sim.onnx
paddleocr_v5_rec_sim.onnx
paddleocr_v5_det_input.npy
```


### 3.2 下载已编译 HMM 模型

```bash
python3 get_model.py --type hmm
```

默认解压到：

```text
output/xh2/
```

解压后 demo 需要的 HMM 文件为：

```text
output/xh2/paddleocr_v5_det_xh2_b1_1core_O2.hmm
output/xh2/paddleocr_v5_rec_xh2_b6_1core_O2.hmm
```

也可以指定解压目录：

```bash
python3 get_model.py --type hmm --build_model_dir output/xh2
```

## 4. hmatc 配置

本目录新增两个 hmatc 配置文件：

- `det.yml`：检测模型量化和编译配置
- `rec.yml`：识别模型量化和编译配置

它们分别对应：

- `paddleocr_v5_det_sim.onnx`
- `paddleocr_v5_rec_sim.onnx`

配置里使用的输入尺寸和归一化参数与示例模型保持一致。

## 5. 量化与编译

如果已经安装好 hmatc 环境，可以直接在本目录运行：

```bash
hmatc quant -c det.yml
hmatc quant -c rec.yml
hmatc build -c det.yml
hmatc build -c rec.yml
```

量化输出默认保存到：

```text
output/xh2/hmquant/
```

编译输出默认保存到：

```text
output/xh2/
```

主要产物包括：

```text
output/xh2/paddleocr_v5_det_xh2_b1_1core_O2.hmm
output/xh2/paddleocr_v5_rec_xh2_b6_1core_O2.hmm
```

## 6. 运行 demo

如果只需要运行 demo，可以直接下载 HMM 模型：

```bash
python3 get_model.py --type hmm
python3 demo.py --image ./ocr.jpeg --perf
```

运行完成后会生成 OCR 可视化结果图：

```text
ocr_result.jpg
```

## 7. 性能和批量评估

### 7.1 端到端 demo 性能

`demo.py --perf` 统计的是 OCR demo 端到端耗时，包括 det/rec 前处理、推理、后处理、裁剪、解码和可视化等流程。

```bash
python3 get_model.py --type hmm
python3 demo.py --image ./ocr.jpeg --perf
```

说明：这里的 demo 性能和批量评估是模型目录内的手动能力；底层单模型 HMM 性能可以使用 `tcim_perf` / `hmatc perf` 单独测试 det 或 rec 模型。

## 8. test.sh 一键运行

`test.sh` 当前支持按 step 运行，支持的 step 为 `quant`、`build`、`demo`。默认 step 为 `demo`。

### 8.1 直接运行 demo

```bash
bash test.sh
```

等价于：

```bash
bash test.sh -s demo
```

执行逻辑：

1. 下载已编译 HMM 模型：`python3 get_model.py --type hmm`
2. 运行 HMM demo：`python3 demo.py --image ./ocr.jpeg --perf`

### 8.2 编译模型

```bash
bash test.sh -s build
```

执行逻辑：

1. 下载原始 ONNX 模型：`python3 get_model.py --type raw`
2. 量化：`hmatc quant -c det.yml` 和 `hmatc quant -c rec.yml`
3. 编译：`hmatc build -c det.yml` 和 `hmatc build -c rec.yml`

### 8.3 从原始模型量化

```bash
bash test.sh -s quant
```

执行逻辑：

1. 下载原始 ONNX 模型：`python3 get_model.py --type raw`
2. 运行量化：`hmatc quant -c det.yml` 和 `hmatc quant -c rec.yml`

### 8.4 说明

`test.sh` 暂不支持 `-s perf` 和 `-s eval` 参数。

如需端到端性能或批量评估，请直接运行 `demo.py`：

```bash
python3 demo.py --image ./ocr.jpeg --perf

### 8.5 跳过下载

如果本地已经有对应模型文件，可以使用 `--skip-download`：

```bash
bash test.sh -s demo --skip-download
```


## 9. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- PaddleOCR V5 模型：[https://github.com/PaddlePaddle/PaddleOCR/tree/46bfd1ef04e7545f33b2065525ffbc742f1a5930/docs/version3.x/algorithm/PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR/tree/46bfd1ef04e7545f33b2065525ffbc742f1a5930/docs/version3.x/algorithm/PP-OCRv5)
