# Ornith 1.0-35B

本示例展示如何量化、编译并运行 Ornith 1.0-35B 视觉语言模型，通过 `tcim_lite` 部署到后摩 XH2 设备上进行图文理解推理。模型目录复用 Qwen3.5 MoE 的 Merak 量化导出流程和 XH2 编译接口，运行时的 tokenizer、chat template、图片处理、cache 绑定、prefill 和 decode 逻辑由本目录实现。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称：Ornith 1.0-35B
- 来源：[deepreinforce-ai/Ornith-1.0-35B](https://www.modelscope.cn/models/deepreinforce-ai/Ornith-1.0-35B)
- 许可：以模型来源页面及其随附许可文件为准

预训练模型和编译产物不包含在本仓库中。下载、量化和运行模型前，请确认已获得相应模型及数据的使用权限。

本例仅适用于 XH2。当前目录使用 `config.yaml` 作为默认配置的单一真值来源，默认模型配置为：

- `model_name=ornith1.0`
- `model_size=35b`
- `quant_type=w4a8`
- `context_length=256k`
- `prefill_length=256`
- `ncore=2`
- `ndevice=1`
- `batch=1`
- 视觉输入尺寸：`448x448x2`

### 默认配置

`demo.py` 默认读取 `config.yaml`，并根据模型名称和规格推导模型路径。默认图片为 `../../../data/pic/beach.jpeg`，默认问题为“描述这张图片”。Demo 当前仅支持 `batch=1`；当 `ndevice > 1` 时，prefill 和 decode 模型路径使用 `.hmms` 后缀，视觉模型仍使用 `.hmm`。

### 资源要求

- 运行平台：XH2。
- Demo 默认使用 1 张设备，且仅支持 `batch=1`。
- 量化硬件要求：暂无数据，请根据 Merak 工作流和实际量化环境确认。
- 编译时 host 内存要求：暂无数据，请根据实际编译环境确认。
- 设备运行所需显存及其他硬件资源：暂无数据，请根据目标 XH2 环境确认。

### 用于校准的数据集

量化流程使用的校准数据由 `config.yaml` 指向的 Merak workflow 配置决定。本仓库不分发该校准数据集；使用者需要按照对应 workflow 的要求准备数据，并遵守数据集原始许可及使用限制。

## 2 快速开始

以下命令默认在 Linux 环境、仓库根目录执行环境初始化后运行。

### 2.1 环境准备

初始化 iModelzoo 环境，进入 Ornith 模型目录并安装 Demo 依赖：

```bash
source env.sh
cd models/vlm/ornith1.0
pip3 install -r requirements.txt
export HOUMO_TARGET=xh2
```

`requirements.txt` 中包含 Demo 所需的 Python 依赖。量化和编译还需要与当前 Houmo XH2 工具链匹配的运行环境。

### 2.2 获取模型和数据

设置 `HOUMO_TARGET=xh2` 后，使用 `get_model.py` 下载模型。该脚本支持下载原始模型（`raw`）和预编译芯片模型（`hmm`）。

下载原始模型用于量化：

```bash
python3 get_model.py \
    --config config.yaml \
    --type raw \
    --model_name ornith1.0 \
    --model_size 35b
```

默认原始模型目录为 `Ornith-1.0-35B/`。

如果只运行 Demo，可下载预编译芯片模型：

```bash
python3 get_model.py \
    --config config.yaml \
    --type hmm \
    --model_name ornith1.0 \
    --model_size 35b
```

`--type` 的默认值为 `hmm`。默认下载源为 `jfrog`；如需从 ModelScope 下载，可增加 `--source_type modelscope`。下载文件默认保存到当前模型目录，也可以通过 `--download_dir` 和 `--extract_dir` 覆盖下载目录和解压目录。运行 Demo 前必须已有本地 HMM 和 `hmquant` 资源。

默认 Demo 产物路径如下：

- `output/${HOUMO_TARGET}/ornith1.0-35b_prefill.hmm`
- `output/${HOUMO_TARGET}/ornith1.0-35b_decode.hmm`
- `output/${HOUMO_TARGET}/ornith1.0-35b_visual_448x448x2.hmm`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `output/${HOUMO_TARGET}/hmquant/hf_config/`

当使用多设备编译时，prefill 和 decode 产物为对应的 `.hmms` 文件，例如 `ornith1.0-35b_prefill.hmms` 和 `ornith1.0-35b_decode.hmms`。

### 2.3 量化

量化使用 `config.yaml` 中的 Merak workflow 配置，并将量化和导出结果整理到 `output/${HOUMO_TARGET}/hmquant/`。确保原始模型和量化环境已准备完成后，在模型目录执行：

```bash
python3 ptq.py \
    --config config.yaml \
    --model_name ornith1.0 \
    --model_size 35b
```

常用量化选项：

```bash
# 从已有量化模型目录直接导出，跳过量化
python3 ptq.py \
    --config config.yaml \
    --model_name ornith1.0 \
    --model_size 35b \
    --model_dir /path/to/quanted-model \
    --export-from-quanted-model

# 覆盖权重量化 bit 数，并覆盖视觉尺寸
python3 ptq.py \
    --bits 4 \
    --max-size-w 448 \
    --max-size-h 448 \
    --max-size-t 2
```

`--dump-golden` 可在导出后生成 golden 数据，`--quick-test` 可在导出后执行 HMONNX 快速生成测试；这两个选项均不是运行 Demo 的必需步骤。

### 2.4 编译

编译前，确认 `output/${HOUMO_TARGET}/hmquant/` 下已有 `prefill`、`decode` 和视觉模型目录。执行：

```bash
python3 build.py \
    --config config.yaml \
    --model_name ornith1.0 \
    --model_size 35b \
    --ndevice 1
```

默认编译产物为：

- `output/${HOUMO_TARGET}/ornith1.0-35b_prefill.hmm`
- `output/${HOUMO_TARGET}/ornith1.0-35b_decode.hmm`
- `output/${HOUMO_TARGET}/ornith1.0-35b_visual_448x448x2.hmm`

多设备编译示例：

```bash
python3 build.py \
    --config config.yaml \
    --model_name ornith1.0 \
    --model_size 35b \
    --ndevice 2
```

多设备编译后，Demo 会将 prefill 和 decode 的默认路径切换为 `.hmms`；视觉模型仍使用 `.hmm`。`build.py` 当前实际执行编译流程，`--stage test` 不执行额外的余弦相似度校验，不建议将其作为验证步骤使用。

### 2.5 演示

完成环境初始化并进入 `models/vlm/ornith1.0` 后，在本机 XH2 环境中可以直接运行默认图文 Demo：

```bash
source env.sh
cd models/vlm/ornith1.0
export HOUMO_TARGET=xh2
python3 demo.py
```

自定义单图问题：

```bash
python3 demo.py \
    --question "请详细描述图片中的内容" \
    --image_path /path/to/image.jpg
```

输入多张图片：

```bash
python3 demo.py \
    --question "比较这两张图片" \
    --image_path /path/to/first.jpg /path/to/second.jpg
```

如果模型产物不在默认目录，可显式指定路径：

```bash
python3 demo.py \
    --prefill_path /path/to/prefill.hmm \
    --decode_path /path/to/decode.hmm \
    --visual_path /path/to/visual.hmm \
    --embedding_path /path/to/quant_embedding.pt \
    --tokenizer_dir /path/to/hf_config
```

常用 Demo 参数：

- `--question TEXT`：设置问题，默认值为“描述这张图片”。
- `--image_path IMAGE...`：设置一个或多个图片路径。
- `--system_prompt TEXT`：设置 system prompt。
- `--max_new_tokens N`：设置最大生成 token 数，默认值为 `1024`。
- `--temperature FLOAT --top_k N --top_p FLOAT`：设置采样参数。
- `--prefill_path PATH --decode_path PATH --visual_path PATH`：覆盖模型路径。
- `--embedding_path PATH --tokenizer_dir PATH`：覆盖量化 embedding 和 tokenizer 目录。
- `--perf [true|false]`：控制 PerfTracker 性能统计，默认开启；`--perf` 等同于 `--perf true`。

性能统计开启时，Demo 结束后输出初始化、TTFT、E2E，以及 visual、prefill、decode 阶段的统计结果。六段阶段口径统一为阶段总时间、preprocess、set_input、infer、get_output 和 postprocess：visual 按图片采样汇总，prefill 按 chunk 采样汇总，decode 按 token 采样汇总，并记录 `input_tokens`、`output_tokens`、`decode_tokens` 和 `num_images`。可通过以下方式关闭：

```bash
python3 demo.py --perf false
```

## 3 一键评估

`test.sh` 默认执行 `demo` 步骤。脚本会检查 XH2 环境，并按需创建 Demo 虚拟环境；`demo` 阶段直接运行 `python3 demo.py`，不会自动下载预编译 HMM。

使用本地已有量化和编译产物运行 Demo：

```bash
bash test.sh --step demo --skip_download
```

完整执行下载原始模型、量化、编译和 Demo：

```bash
bash test.sh --step all
```

只执行量化或编译：

```bash
bash test.sh --step quant
bash test.sh --step build --skip_download
```

`test.sh` 支持的常用参数包括：

```text
-s, --step              执行阶段，默认 demo，可使用 quant、build、demo、all
-name, --model_name     模型名称，默认 ornith1.0
-size, --model_size     模型规格，默认 35b
-b, --batch             batch 大小；Ornith Demo 仅支持 1
--ndevice               设备数量，默认 1
--context_length        覆盖上下文长度
--prefill_length        覆盖 prefill 长度
--quant_type            覆盖量化类型
--skip_download         跳过下载步骤
--system_prompt         传递给 Demo 的 system prompt
-h, --help              显示帮助信息
```

## 4 参考结果

### 4.1 演示结果

```shell
Q: 描述这张图片
A: 这张图片展现了一个温馨、宁静的海滩场景，充满人与自然和谐共处的美感。

**主体人物与动物：**
- 一位年轻女性坐在沙滩上，侧身面向一只金毛犬（或类似品种的大型犬）。她留着深色长发，身穿蓝白格子衬衫和深色裤子，赤脚踩在沙地上。
- 她面带微笑，正伸出右手与狗狗击掌（“paw shake”），互动亲密而愉快。
- 狗狗站立着，前爪搭在女子手上，佩戴着蓝色花纹的胸背带，表情温顺专注，似乎很享受这个互动。

**环境背景：**
- 场景位于一片开阔的沙滩上，沙粒细腻，留有脚印和自然纹理。
- 背景是平静的海面，远处可见轻柔的海浪正在涌向岸边。
- 天空明亮，阳光从画面右侧斜射进来，形成温暖的逆光效果，为整个画面镀上一层柔和的金色光晕，营造出日落时分或清晨的浪漫氛围。

**整体氛围：**
- 图片传递出一种放松、幸福、陪伴的情感。人与宠物之间的信任与默契通过击掌这一简单动作得以体现。
- 构图平衡，色彩柔和，光影运用出色，具有强烈的治愈感和生活美学气息。

这是一幅充满温情的生活瞬间抓拍，适合用于表达“陪伴”、“自由”、“自然”或“人宠关系”等主题。
```

### 4.2 性能结果

```shell
Performance Summary: llm

Timing
Scope          Count  Total(ms)    Avg(ms)    Min(ms)    Max(ms)            Speed
-------------  -----  ---------  ---------  ---------  ---------  ---------------
init               1  28800.845  28800.845  28800.845  28800.845                -
prefill            1    364.218    364.218    364.218    364.218  584.81 tokens/s
  set_input        1      3.482      3.482      3.482      3.482                -
  infer            1    350.784    350.784    350.784    350.784  607.21 tokens/s
  get_output       1      0.316      0.316      0.316      0.316                -
  postprocess      1      3.782      3.782      3.782      3.782                -
  preprocess       1      5.661      5.661      5.661      5.661                -
decode             1  11297.310  11297.310  11297.310  11297.310   26.91 tokens/s
  set_input      304    189.015      0.622      0.276      2.724                -
  infer          304  10253.837     33.730     33.408     35.579   29.65 tokens/s
  get_output     304     87.714      0.289      0.153      0.577                -
  postprocess    304    702.546      2.311      0.800      2.841                -
  preprocess     304     25.285      0.083      0.035      0.192                -
ttft               1    837.819    837.819    837.819    837.819                -
e2e              301  12124.430     40.280     35.393    838.041                -
visual             1    453.247    453.247    453.247    453.247                -
  set_input        1      4.213      4.213      4.213      4.213                -
  infer            1    396.909    396.909    396.909    396.909                -
  get_output       1      0.510      0.510      0.510      0.510                -
  postprocess      1      5.630      5.630      5.630      5.630                -
  preprocess       1     45.786     45.786     45.786     45.786                -

Overall Performance Metrics
Input Tokens: 213
Output Tokens: 305
TTFT (Time To First Token): 837.82 ms
E2E Latency (End-to-End): 12124.43 ms
E2E TPS (Throughput): 25.16 tokens/s
TPOT (Time Per Output Token): 37.16 ms/token
```

## 5 免责声明

模型权重和校准数据由各自的第三方作者或数据提供方发布，其许可证、使用限制和风险由对应来源定义。使用者应在下载、量化、部署和分发前确认并遵守相关许可、隐私、安全及合规要求。本仓库示例不对第三方模型或数据的准确性、适用性和合规性作保证。
