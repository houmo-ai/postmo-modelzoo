# Qwen3-VL-Embedding

本示例展示如何把 Qwen3-VL-Embedding 多模态向量模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-VL-Embedding-8B
- 来源: [https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B)
- 许可: Apache License 2.0 ([https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0))

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-vl-embedding`
- `model_size=8b`
- `context_length=8k`
- `prefill_length=256`
- `ndevice=1`
- `max_size_w=896`、`max_size_h=896`、`max_size_t=2`

量化校准仅使用本仓库内置的示例图片，不依赖任何第三方校准数据集。

## 2 快速开始

目前 Qwen3-VL-Embedding 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型或芯片模型。

如果需要自己量化，可选择下载原始模型，原始模型默认下载到 `Qwen3-VL-Embedding-8B` 目录：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/embedding/qwen3-vl-embedding` 目录，执行 `ptq.py` 脚本。默认的 `input_sequence_length` 和 `context_length` 分别来自 `config.yaml`，当前默认值是 `256` 和 `8k`。

```bash
source env.sh
cd models/embedding/qwen3-vl-embedding
python3 ptq.py
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- visual
|-- prefill
|-- decode
|-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

默认输出名称分别为：

- `qwen3-vl-embedding-8b_visual_896x896x2.hmm`
- `qwen3-vl-embedding-8b_prefill.hmm`
- `qwen3-vl-embedding-8b_decode.hmm`

### 2.5 演示

Qwen3-VL-Embedding 模型使用 python API 进行演示，支持文本、图片以及"文本+图片"的多模态向量提取，并计算 query 与 document 之间的相似度。

1. `demo_hmonnx.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：

  - `output/${HOUMO_TARGET}/hmquant/visual/`
  - `output/${HOUMO_TARGET}/hmquant/prefill/`
  - `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

2. 执行脚本，推理完成将输出相似度结果。

```bash
python3 demo_hmonnx.py
```

## 3 一键评估

以上步骤可以通过 test.sh 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

test.sh 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: qwen3-vl-embedding.
  -size, --model_size     Model size. Default: 8b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别是 `qwen3-vl-embedding` 和 `8b`。

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 显式指定模型名和模型规格（当前默认就是 qwen3-vl-embedding 8b）
bash test.sh -s demo -name qwen3-vl-embedding -size 8b

# 组合执行多个步骤
bash test.sh -s quant,build

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

## 4 参考结果

### 4.1 演示结果

演示对 4 条文本 query 和 3 条 document（纯文本、仅图片、文本+图片）分别提取向量，并计算相似度矩阵。芯片量化模型（`w8a8h1_sefp`）的相似度绝对值相比 HF float 参考略低，但各 query 对 document 的排序与 HF float 参考完全一致。

```bash
====================================================================================================
Query                                              Doc 0 [text]       Doc 1 [image]      Doc 2 [text+image]
====================================================================================================
A woman playing with her dog on a beach at sunse   0.7513            0.6785            0.6352
Pet owner training dog outdoors near water.        0.4409            0.3355            0.3855
Woman surfing on waves during a sunny day.         0.3764            0.2372            0.2819
City skyline view from a high-rise building at n   0.0686            -0.0132           0.0227
====================================================================================================

Best match per query:
  Q0: "A woman playing with her dog on a beach "  ->  Doc 0 [text]  (score: 0.7513)
  Q1: "Pet owner training dog outdoors near wat"  ->  Doc 0 [text]  (score: 0.4409)
  Q2: "Woman surfing on waves during a sunny da"  ->  Doc 0 [text]  (score: 0.3764)
  Q3: "City skyline view from a high-rise build"  ->  Doc 0 [text]  (score: 0.0686)
```

### 4.2 性能结果

Total Input: 1828 tokens, Prefill Cost 1495.100 ms
Prefill Speed: 1222.66 tokens/s
E2E Latency (End-to-End Latency): 4.453 seconds
====================================================================================================
                    Qwen3-VL-Embedding Performance Summary
====================================================================================================
Configuration Details:
  Batch Size:      1
  Total Input Length:   1828 tokens
  Number of Images:      2 images
  Vision Model Load Time: 11224.23 ms
  Prefill Model Load Time: 62832.23 ms
Vision Stage Performance:
  Total Time: 2395.72 ms
  Preprocessing Time:  438.49 ms
  API SetInput Time:   13.40 ms
  API Inference Time: 1775.07 ms | Speed:    1.13 images/s
  API GetOutput Time:  106.42 ms
Prefill Stage Performance:
  Total Time: 2006.52 ms
  Tokenization Time:   27.36 ms
  Embedding Time:  293.72 ms
  API SetInput Time:   61.94 ms
  API Inference Time: 1495.18 ms | Prefill Speed: 1222.60 tokens/s
  API GetOutput Time:  128.33 ms
Overall Performance Metrics:
  E2E Latency (End-to-End): 4453.06 ms
  E2E TPS (Input Throughput):  410.50 tokens/s
 ====================================================================================================


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
