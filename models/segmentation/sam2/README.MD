# SAM2

本示例展示如何把 SAM2 (Segment Anything Model 2) 模型量化和编译,部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 Meta 的 SAM2 预训练模型:

- 模型名称: SAM2.1 Hiera Small
- 来源: https://github.com/facebookresearch/sam2
- 许可: Apache License 2.0 (https://github.com/facebookresearch/sam2/blob/main/LICENSE)

预训练模型在运行时下载,工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录提供两种输入尺寸的配置:

**640x640 输入:**
- `encoder.yml`: 编码器配置
- `decoder.yml`: 解码器配置

### 模型架构

SAM2 包含两个主要组件:
- **Encoder (编码器)**: 提取图像特征,生成图像嵌入
- **Decoder (解码器)**: 根据提示点或框生成分割掩码

### 资源要求

**后摩设备运行要求:**
- 无

**量化硬件要求:**
- 无

**编译时 host 内存要求:**
- 无

## 2 快速开始

目前 SAM2 模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

量化编译直接使用模型转换docker，部署直接使用部署docker，无额外依赖

### 2.2 获取模型

SAM2 模型需要先导出为 ONNX 格式。

导出的 ONNX 模型存放在上级目录:
- `sam2.1s_encoder.onnx`: 编码器模型 (640x640)
- `sam2.1s_decoder.onnx`: 解码器模型 (640x640)

### 2.3 量化

将 ONNX 模型量化为后摩芯片支持的格式。使用配置文件进行量化:

```bash
hmatc quant -c encoder.yml
hmatc quant -c decoder.yml
```

量化完成的模型存放在 `output/${HOUMO_TARGET}/hmquant` 目录。

### 2.4 编译

将量化模型编译为在芯片上运行的模型:

```bash
hmatc build -c encoder.yml
hmatc build -c decoder.yml
```

默认输出名称为:
- `sam2.1s_encoder_xh2_b1_1core_O2.hmm`
- `sam2.1s_decoder_xh2_b1_1core_O2.hmm`

### 2.5 演示与精度验证

本示例使用 `demo.py` 进行演示和 COCO 精度验证。`demo.py` 现在只负责命令行入口与可视化，核心推理逻辑已拆分到 `sam2_engine.py` 中，便于复用和维护。

#### 2.5.1 当前功能点

- **统一后端接口**:
  - `BaseRuntimeModel`: 单模型推理抽象, `infer()` 内部统一调用 `set_input`、`run`、`get_output`。
  - `HMMRuntimeModel`: 使用 `tcim_lite` 加载和运行 `.hmm` 模型。
  - `ONNXRuntimeModel`: 使用 `onnxruntime` 加载和运行 `.onnx` 模型。
  - `SAM2Engine`: 组合 encoder/decoder, 对上层 demo 暴露统一的 `load()`、`infer()`、`infer_auto_all()`、`eval()`。

- **代码结构**:
  - `demo.py`: 命令行参数解析、交互/可视化入口、调用 `SAM2Engine`。
  - `sam2_engine.py`: Runtime 封装、SAM2 encoder/decoder 基础推理、mask 后处理。
  - `coco_segment_dataset.py`: COCO segmentation 数据集读取、评估流程和指标统计。

- **后端选择**:
  - `--backend hmm`: 使用编译后的 HMM 模型。
  - `--backend xh2`: `hmm` 的别名。
  - `--backend onnx`: 使用原始 ONNX 模型 `sam2.1s_encoder.onnx` 和 `sam2.1s_decoder.onnx`。

- **演示模式**:
  - `--mode 0`: segmentation数据集精度验证。
  - `--mode 1`: 交互式点击分割,在窗口中点击目标点。
  - `--mode 2`: 预设点分割,通过 `--point x y` 指定前景点。
  - `--mode 3`: 自动全图分割,通过网格采样提示点生成多个候选 mask 并按 IoU 去重。

- **输出命名**:
  - demo 可视化结果文件名包含后端名称,例如:
    - `demo_hmm_single_point_result.png`
    - `demo_onnx_single_point_result.png`
    - `demo_hmm_auto_all_result.png`
    - `demo_onnx_auto_all_result.png`
  - COCO eval 预测文件默认保存到:
    - `eval_hmm_results/pred_hmm_segm.json`
    - `eval_onnx_results/pred_onnx_segm.json`

- **COCO eval 支持部分数据统计**:
  - `--eval-num N`: 只评估前 N 张存在的 COCO val 图片,`0` 表示全部。
  - `--max-ann-per-image N`: 每张图最多评估 N 个 annotation,`0` 表示全部 annotation。
  - 当 COCO 数据不完整时,eval 会跳过缺失图片,精度只基于实际存在且成功读取的图片计算。

#### 2.5.2 运行示例

HMM 后端预设点分割:

```bash
cd models/segmentation/sam2
python3 demo.py \
  --backend hmm \
  --mode 2 \
  --image ${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg \
  --point 800 800
```

ONNX 后端预设点分割:

```bash
cd models/segmentation/sam2
python3 demo.py \
  --backend onnx \
  --mode 2 \
  --image ${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg \
  --point 800 800
```

自动全图分割:

```bash
cd models/segmentation/sam2
python3 demo.py \
  --backend hmm \
  --mode 3 \
  --image ${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg \
  --grid-size 8
```

COCO val2017 精度验证:

```bash
cd models/segmentation/sam2
python3 demo.py \
  --backend onnx \
  --mode 0 \
  --dataset_dir ${HOUMO_EXAMPLES_PATH}/data/datasets/coco2017 \
  --eval-num 32
```

限制每张图只评估部分 annotation:

```bash
cd models/segmentation/sam2
python3 demo.py \
  --backend hmm \
  --mode 0 \
  --dataset_dir ${HOUMO_EXAMPLES_PATH}/data/datasets/coco2017 \
  --eval-num 32 \
  --max-ann-per-image 3
```

#### 2.5.3 默认模型路径

`--backend hmm` / `--backend xh2` 默认读取:

- 编码器模型: `output/xh2/sam2.1s_encoder_xh2_b1_1core_O2.hmm`
- 解码器模型: `output/xh2/sam2.1s_decoder_xh2_b1_1core_O2.hmm`

`--backend onnx` 默认读取:

- 编码器模型: `sam2.1s_encoder.onnx`
- 解码器模型: `sam2.1s_decoder.onnx`

也可以通过 `--encoder` 和 `--decoder` 显式指定模型路径。

## 3 参考结果

### 3.1 量化精度

基于 640x640 输入尺寸的量化精度对比:

**Encoder 输出对比:**
- `image_embed`: ONNX vs hmquant 余弦相似度 0.997083, hmquant vs xh2 余弦相似度 0.998429
- `high0`: ONNX vs hmquant 余弦相似度 0.998027, hmquant vs xh2 余弦相似度 0.998787
- `high1`: ONNX vs hmquant 余弦相似度 0.994407, hmquant vs xh2 余弦相似度 0.995757

**Decoder 输出对比:**
- `masks`: 余弦相似度 0.999884
- `iou_preds`: 余弦相似度 0.999999

### 3.2 性能结果

**推理性能:**
```bash
推理平均延迟: 91 ms
QPS: 21
```

**量化编译性能:**

**Encoder 量化时间:** 约 324 秒
**Encoder 编译时间:** 约 623 秒
**Decoder 量化时间:** 约 142 秒
**Decoder 编译时间:** 约 214 秒

### 3.3 演示结果

演示结果图片展示了对图像进行分割的效果:

**单点提示分割:**
`demo_hmm_single_point_result.png`

**自动分割：**
`demo_hmm_auto_all_result.png`

## 4 免责声明

您明确了解并同意,以下链接中的软件、数据或者模型由第三方提供并负责维护,后摩算法团队不对其合法性、准确性、完整性、有效性、安全性做任何形式的保证。您因使用或依赖以下链接中的软件、数据或模型而导致的任何直接、间接、偶然、特殊、后续、惩罚性损害(包括但不限于数据丢失、利润损失、业务中断等)不承担任何责任。

- SAM2 模型: https://github.com/facebookresearch/sam2
- SAM2 预训练权重: https://dl.fbaipublicfiles.com/segment_anything_2/

您应当自行判断是否使用上述第三方提供的软件、数据或模型,并自行承担相关风险。如果您不同意上述声明,请不要使用相关软件、数据或模型。
