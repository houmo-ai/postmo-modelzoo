# Emotion2vec+

本示例展示如何把 Emotion2vec+ 语音情感识别模型量化和编译，部署到后摩芯片的设备上，并完成九分类情感识别、句级特征和帧级特征提取。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: emotion2vec+ large
- 来源: [iic/emotion2vec_plus_large](https://www.modelscope.cn/models/iic/emotion2vec_plus_large)
- 许可: Apache License 2.0 ([ModelScope 模型卡](https://www.modelscope.cn/models/iic/emotion2vec_plus_large))

`emotion2vec_plus_large` 是约 300M 参数的语音情感识别模型，官方 checkpoint 支持以下 9 个类别：

- 生气 / angry
- 厌恶 / disgusted
- 恐惧 / fearful
- 开心 / happy
- 中立 / neutral
- 其他 / other
- 难过 / sad
- 吃惊 / surprised
- unknown (`<unk>`)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=emotion2vec`
- `model_size=plus_large`
- `ncore=2`
- `ndevice=1`
- `quant_type=w8a8h1_sefp`
- `workflow_config=workflows/xh2a/audio_models/emotion2vec/emotion2vec_plus_large_xh2a_w8a8_16s.yaml`

量化和编译后的模型使用固定 16 kHz 单声道输入。单次 HMM 输入窗口为 16 秒：

- `waveform`: `FLOAT16 [1, 256000]`
- `valid_frames`: `INT32 [1]`

HMM 输出为：

- `frame_features`: `FLOAT16 [1, 799, 1024]`
- `frame_padding_mask`: `INT8 [1, 799]`
- `utterance_feature`: `FLOAT16 [1, 1024]`
- `probabilities`: `FLOAT16 [1, 9]`

长音频由 `demo.py` 分块推理。`--chunk_size` 的单位为秒，默认值为 16，可配置范围为 `0 < chunk_size <= 16`；每个分块都会补齐到 HMM 的固定输入长度，并通过 `valid_frames` 和 `frame_padding_mask` 去除补齐部分对应的无效帧。

每个 chunk 都会独立生成帧级特征、句级特征、分类概率和分类摘要。整段音频的分类结果由所有 chunk 的句级特征通过 `quant_embedding.pt` 中的官方分类头计算得到。

### 资源要求

**后摩设备运行要求：**

- `emotion2vec plus_large`: 单卡12GB可运行

**量化硬件要求：**

- `emotion2vec plus_large`: 需要 GPU，显存最小要求暂无数据

**编译时 host 内存要求：**

- `emotion2vec plus_large`: 实测峰值约 28.9GB，建议预留至少 32GB 空闲 host 内存

默认量化流程不下载或使用额外的第三方校准数据集。`ptq.py` 只有在显式指定 `--dump-golden` 时才会读取 `--golden-audio` 指定的本地音频，用于导出对比数据。

## 2 快速开始

目前 Emotion2vec+ 模型仅提供 python 脚本量化、编译和演示方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装演示运行所需的 python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

自行量化时还需要安装 PTQ 依赖：

```bash
pip3 install -r requirements_ptq.txt
```

量化脚本依赖 xh2modelzoo，请确保环境中包含：

```bash
export PYTHONPATH=/hmdd/3rdparty/xh2modelzoo:$PYTHONPATH
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过 `--type` 选择下载原始模型或芯片模型。

如果需要自行量化，可以下载完整的 ModelScope 原始模型。默认下载到 `emotion2vec_plus_large` 目录，其中包含 `model.pt`、`tokens.txt`、配置文件和示例音频：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可以下载预编译 HMM：

```bash
python3 get_model.py --type hmm
```

`--type hmm` 会下载并解压以下运行所需产物：

- `output/xh2/emotion2vec-plus_large.hmm`
- `output/xh2/hmquant/quant_embedding.pt`
- ModelScope 模型仓库中的配置、标签和示例文件，忽略 `*.pt` 原始权重

因此仅运行 HMM 演示时不需要完整的 `model.pt`。`quant_embedding.pt` 包含整段音频分类所需的官方分类头权重，标签从 `tokens.txt` 读取。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/asr/emotion2vec` 目录执行量化：

```bash
source env.sh
cd models/asr/emotion2vec
export PYTHONPATH=/hmdd/3rdparty/xh2modelzoo:$PYTHONPATH
python3 ptq.py
```

默认量化类型为 `w8a8h1_sefp`。量化完成的主要产物位于 `output/$HOUMO_TARGET/hmquant`：

```text
output/xh2/hmquant/
|-- emotion2vec_meta.json
|-- emotion2vec_plus_large_xh2a_w8a8_16s.yaml
|-- export_hmonnx.log
|-- hmquant_emotion2vec_plus_large_external_data
|-- hmquant_emotion2vec_plus_large_with_act.onnx
`-- quant_embedding.pt
```

其中 `quant_embedding.pt` 由 `ptq.py` 导出，包含整段音频九分类所需的官方分类头权重。

### 2.4 编译

将量化后的 HMONNX 模型编译为在芯片上运行的 HMM：

```bash
python3 build.py
```

默认输出名称为：

- `output/xh2/emotion2vec-plus_large.hmm`

编译脚本默认从 `output/xh2/hmquant` 自动查找 HMONNX，并从 `config.yaml` 读取 `ncore=2` 和 `ndevice=1`。

### 2.5 演示

Emotion2vec+ 模型使用 python API 进行演示。`demo.py` 默认读取：

- `output/${HOUMO_TARGET}/emotion2vec-plus_large.hmm`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `emotion2vec_plus_large/tokens.txt`
- `../../../data/audio/audio.mp3`

执行：

```bash
python3 demo.py
```

演示提供以下输出：

1. 整段音频和每个 chunk 的九分类语音情感识别结果。
2. Top-K 情感结果，默认 Top-3。
3. 每个 chunk 的句级特征，形状为 `[1024]`，dtype 为 `float16`。
4. 每个 chunk 的帧级特征，形状为 `[T, 1024]`，dtype 为 `float16`。
5. 每个 chunk 的九分类概率，形状为 `[9]`，dtype 为 `float16`。
6. 每个 chunk 的独立 JSON 摘要，包含时间范围、分类结果和各输出路径。
7. 整段音频的分类、所有 chunk 信息和性能数据汇总。

默认输出目录结构如下：

```text
results/
|-- frame/
|   |-- audio.chunk_0001.npy
|   `-- ...
|-- utterance/
|   |-- audio.chunk_0001.npy
|   `-- ...
|-- probabilities/
|   |-- audio.chunk_0001.npy
|   `-- ...
|-- summary/
|   |-- audio.chunk_0001.json
|   `-- ...
`-- Summary.json
```

其中：

- `frame/<audio>.chunk_NNNN.npy`: 对应 chunk 的帧级特征。
- `utterance/<audio>.chunk_NNNN.npy`: 对应 chunk 的句级特征。
- `probabilities/<audio>.chunk_NNNN.npy`: 对应 chunk 的九分类概率。
- `summary/<audio>.chunk_NNNN.json`: 对应 chunk 的起止时间、情感分类和输出元数据。
- `Summary.json`: 所有输入音频的整段分类、chunk 列表、模型加载耗时、HMM 推理耗时、端到端耗时和 RTF。

终端使用 `logger.info` 输出 chunk 处理进度和整段音频分类，使用 `logger.success` 输出性能统计与 `Summary.json` 路径。每个 chunk 的详细分类不在终端重复打印，可从对应 JSON 文件获取。

自定义音频和输出目录：

```bash
python3 demo.py \
    --audio /path/to/audio.wav \
    --output-dir results \
    --top-k 5 \
    --chunk_size 10
```

以上命令按 10 秒切分音频。`--chunk_size` 支持小数，但不能超过 HMM 的 16 秒输入窗口。

常用演示参数如下：

```text
--model_path       HMM 模型路径
--quant-embedding  quant_embedding.pt 分类头权重路径
--tokenizer_dir    包含 tokens.txt 的目录
--audio            一个或多个本地音频文件
--output-dir       模型输出和 JSON 摘要目录，默认 results
--top-k            输出的 Top-K 情感数量，默认 3
--chunk_size       音频分块时长，单位为秒，默认 16
```

可以显式指定模型、分类头权重和标签目录：

```bash
python3 demo.py \
    --model_path output/xh2/emotion2vec-plus_large.hmm \
    --quant-embedding output/xh2/hmquant/quant_embedding.pt \
    --tokenizer_dir emotion2vec_plus_large \
    --audio first.wav second.wav
```

## 3 一键评估

以上步骤可以通过 `test.sh` 执行。默认模型为 `emotion2vec plus_large`，默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译 HMM、quant_embedding.pt 和非权重模型资源
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

`test.sh` 支持以下与本模型流程相关的参数：

```text
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                           Supports comma-separated values or repeated flags.
  -name, --model_name     Model name. Default: emotion2vec.
  -size, --model_size     Model size. Default: plus_large.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载完整原始模型
bash test.sh -s quant

# 仅运行 build，要求 output/xh2/hmquant 中已有量化产物
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型、quant_embedding.pt 和标签文件
bash test.sh -s demo

# 跳过下载，使用当前目录中的已有产物
bash test.sh -s demo --skip_download

# 组合执行量化和编译
bash test.sh -s quant,build
```

量化步骤会检查 GPU；编译步骤要求 x86_64 平台；演示步骤要求连接后摩设备并能够加载 `tcim_lite` 运行时。

## 4 参考结果

### 4.1 演示结果

使用仓库公共音频 `data/audio/audio.mp3` 的实测分类结果如下。当前日志由 `loguru` 输出，时间戳、模块名和行号会随运行环境及代码版本变化：

```text
INFO | Processing Chunk 1/1 (2.432s)
INFO | [audio] ../../../data/audio/audio.mp3
INFO | All labels and scores:
INFO |   生气/angry: 0.000011
INFO |   厌恶/disgusted: 0.000017
INFO |   恐惧/fearful: 0.000003
INFO |   开心/happy: 0.986024
INFO |   中立/neutral: 0.013914
INFO |   其他/other: 0.000001
INFO |   难过/sad: 0.000024
INFO |   吃惊/surprised: 0.000005
INFO |   <unk>: 0.000000
INFO | Top-3 emotions:
INFO |   1. 开心/happy: 0.986024
INFO |   2. 中立/neutral: 0.013914
INFO |   3. 难过/sad: 0.000024
INFO | Utterance embedding: [1024], saved 1 chunks -> results/utterance
INFO | Frame embedding: [121, 1024], saved 1 chunks -> results/frame
SUCCESS | Audio Duration: 2.432 seconds
SUCCESS | Chunks: 1 (configured chunk size: 16.000 seconds)
SUCCESS | Summary: results/Summary.json
```

该音频还会生成：

- `results/frame/audio.chunk_0001.npy`
- `results/utterance/audio.chunk_0001.npy`
- `results/summary/audio.chunk_0001.json`
- `results/Summary.json`

### 4.2 性能结果

同一音频的一次历史实测结果如下，性能会随设备、运行时版本和系统负载变化：

```text
Model Load: 1074.90 ms
HMM Inference: 58.18 ms
E2E Latency: 0.071 seconds
```

当前版本还会通过 `logger.success` 输出平均每 chunk 推理耗时、HMM RTF 和端到端 RTF；对应原始数值同时保存在 `results/Summary.json`。由于历史实测日志未记录这些字段，此处不补充推测值。

编译实测生成的 HMM 大小约 332MB，编译过程峰值 host 内存约 28.9GB。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
