# SigLIP2 Large Patch16 256

本目录提供 `google/siglip2-large-patch16-256` 的导出、量化、编译、推理和 ImageNet 评测示例，仅支持 `HOUMO_TARGET=xh2`。

模型拆分为 vision 和 text 两个 encoder，输出均为 L2 归一化的 1024 维 embedding：

```python
similarity = image_embeds @ text_embeds.T
```

## 模型产物

| 产物 | 默认路径 |
| --- | --- |
| Vision ONNX | `output/xh2/hmquant/onnx/siglip2_large_patch16_256_vision.onnx` |
| Text ONNX | `output/xh2/hmquant/onnx/siglip2_large_patch16_256_text.onnx` |
| Text ONNX 权重 | `output/xh2/hmquant/onnx/siglip2_large_patch16_256_text.onnx.data` |
| Vision hmonnx | `output/xh2/hmquant/vision/hmquant_siglip2_large_patch16_256_vision_with_act.onnx` |
| Text hmonnx | `output/xh2/hmquant/text/hmquant_siglip2_large_patch16_256_text_with_act.onnx` |
| Tokenizer | `output/xh2/hmquant/hf_config` |
| Vision HMM | `output/xh2/siglip2-large-patch16-256_vision.hmm` |
| Text HMM | `output/xh2/siglip2-large-patch16-256_text.hmm` |

Text ONNX 的 FP32 权重超过 2 GiB，因此使用一个独立的 `.onnx.data` 文件，使用时需与 ONNX 文件放在同一目录。

## 文件说明

- `ptq.py`：加载本地 Hugging Face 模型，保存 tokenizer，导出 opset 17 ONNX，并转换为 hmonnx。
- `build.py`：将 vision/text hmonnx 编译为 HMM，仅支持 x86_64 编译环境。
- `model.py`：ONNX/HMM runner、图像预处理和文本 embedding 等公共逻辑。
- `demo.py`：使用 HMM 对单张图片执行 zero-shot 分类，输出 top-5。
- `eval_imagenet.py`：使用 ONNX 或 HMM 评测 ImageNet top-1/top-5。
- `perf.py`：测试 vision/text HMM 的 H2D、Infer 和 D2H 耗时。
- `get_model.py`：下载原始 Hugging Face 模型或已编译 HMM。
- `test.sh`：执行下载、量化、编译或 demo 流程。

## 快速开始

```bash
cd models/embedding/siglip2
./test.sh -s all
```

### 完整量化和编译

下载原始模型：

```bash
python3 get_model.py --type raw
```

导出 ONNX 并量化为 hmonnx：

```bash
python3 ptq.py
```

`ptq.py` 默认读取 `siglip2-large-patch16-256/`，主要参数如下：

```text
--model_dir <hf_model_dir>
--out_dir output/xh2/hmquant
--quant_type w8a8_sefp
--image_size 256
--seq_len 64
--resizer_input_size 1080 1920
--skip_vision / --skip_text / --overwrite
```

`ptq.py` 保留原始 ONNX 和 tokenizer 的现有目录，只将量化后的 hmonnx 按模型拆分：

```text
output/xh2/hmquant/
├── onnx/
├── hf_config/
├── vision/
└── text/
```

编译 HMM：

```bash
python3 build.py
```

默认生成：

```text
output/xh2/siglip2-large-patch16-256_vision.hmm
output/xh2/siglip2-large-patch16-256_text.hmm
```

### 单图 Demo

`demo.py` 仅使用 HMM。若本地没有 HMM，可先下载：

```bash
python3 get_model.py --type hmm
```

运行默认图片：

```bash
python3 demo.py
```

指定图片和标签文件：

```bash
python3 demo.py \
  --image /path/to/image.jpg \
  --labels /path/to/synset_1000.txt
```

输出格式：

```text
top1: <label>    <similarity>
...
top5: <label>    <similarity>
```

### 测试脚本

```bash
./test.sh -s quant
./test.sh -s build
./test.sh -s demo
```

不指定 `-s` 时默认执行 demo。单独执行 demo 时，脚本会先下载已编译 HMM。

## HMM 性能

`perf.py` 使用同一份输入重复推理，预处理不计入耗时：

- H2D：`set_input`
- Infer：`run` 和 `sync`
- D2H：`get_output`

```bash
python3 perf.py --warmup 5 --repeat 100
```

实测结果：

| Model | H2D (ms) | Infer (ms) | D2H (ms) | Total (ms) |
| --- | ---: | ---: | ---: | ---: |
| Vision | 3.65 | 17.94 | 0.11 | 21.71 |
| Text | 0.16 | 6.41 | 0.07 | 6.63 |

## ImageNet 评测

HMM 评测：

```bash
python3 eval_imagenet.py \
  --imagenet_dir /path/to/imagenet \
  --num 1000 \
  --vision_backend hmm \
  --text_backend hmm
```

ONNX 评测：

```bash
python3 eval_imagenet.py \
  --imagenet_dir /path/to/imagenet \
  --num 1000 \
  --vision_backend onnx \
  --text_backend onnx
```

`--num 0` 表示评测 `val.txt` 中所有可用图片。

前 1000 张 ImageNet 验证图片的测试结果：

| Vision | Text | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| HMM | HMM | 0.701000 | 0.866000 |
| ONNX | ONNX | 0.695000 | 0.873000 |

## 输入与预处理

### Vision ONNX

- 输入：`pixel_values`，float32，`[1, 3, 256, 256]`
- 输出：`image_embeds`，float32，`[1, 1024]`
- 预处理：BGR 转 RGB，resize 到 `256 x 256`，使用 mean/std `[127.5, 127.5, 127.5]`

### Vision HMM

- 图像输入：`pixel_values`，YUV420SP，缓冲区大小 `[1, 3, 1080, 1920]`
- 动态参数：`resizer_crop_pixel_values`，int32，`[1, 10]`
- 若图片超过 `1080 x 1920`，先等比例缩小，再填充到缓冲区左上角，由 dynamic resizer 完成 crop 和 resize 到 `256 x 256`

### Text ONNX/HMM

- 输入：`input_ids`、`attention_mask`，int64，`[1, 64]`
- 输出：`text_embeds`，float32，`[1, 1024]`
- Prompt：`a photo of a {class}`
- 当前 demo 和评测使用全 1 `attention_mask`

## ImageNet 数据

默认数据目录：

```text
$HOUMO_EXAMPLES_PATH/data/datasets/imagenet
```

未设置 `HOUMO_EXAMPLES_PATH` 时使用当前目录下的 `imagenet/`。目录结构如下：

```text
imagenet/
├── ILSVRC2012_img_val/
├── val.txt
└── synset_1000.txt
```

## 免责声明

您明确了解并同意，以下链接中的软件、数据或模型由第三方提供并负责维护，使用风险由您自行承担，并受对应使用条款、许可协议和隐私政策约束。

- SigLIP2 模型：https://modelscope.cn/models/google/siglip2-large-patch16-256
- ImageNet 数据集：https://image-net.org/challenges/LSVRC
