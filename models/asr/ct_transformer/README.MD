# CT-Transformer

本示例展示如何把CT-Transformer标点恢复模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: CT-Transformer (标点恢复/Punctuation Restoration)
- 来源: [https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large)
- 许可: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前CT-Transformer模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```


### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型：

```bash
python3 get_model.py --type hmm 
```

如果需要自行编译模型，可以下载原始模型：

```bash
python3 get_model.py --type raw
```

### 2.3 导出ONNX

将原始PyTorch模型导出为FP32 ONNX格式：

```bash
python3 export_onnx.py
```

产物：
- `work_dirs/export_fp32/onnx/model.onnx` — 原始导出ONNX
- `work_dirs/export_fp32/onnx/model_simplified.onnx` — onnxsim简化后的ONNX（推荐后续使用）

### 2.4 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/asr/ct_transformer`目录，执行ptq.py脚本。默认的量化方式从config.yaml读取。

```bash
source env.sh
cd models/asr/ct_transformer

python3 ptq.py --model_name ct_transformer --model_size large
```

产物：
- `work_dirs/hmonnx/hmquant_ct_transformer.onnx` — 量化后的HMONNX模型

参数说明：
- `--config`: 配置文件路径（默认 `config.yaml`）
- `--model_name`: 模型名称（默认从config读取）
- `--model_size`: 模型大小（默认从config读取）
- `--model-dir`: 原始模型目录（默认根据config自动推导）
- `--input`: 输入文本文件（默认 `ct_transformer/example/punc_example.txt`）
- `--onnx-path`: FP32 ONNX路径
- `--simplified-onnx-path`: 简化后ONNX路径
- `--quant-type`: 量化类型（默认从config读取，如 `w8a8_sefp`、`w8a8h1_sefp`）
- `--device`: 量化设备（默认 `cuda:0`）
- `--save-golden`: 保存golden输出（用于后续build.py测试比对）
- `--no-eval`: 跳过精度评估
- `--reuse-onnx`: 复用已有ONNX模型

### 2.5 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name ct_transformer --model_size large --stage all
```

编译好的模型存放在`output/xh2/`目录：
- `ct_transformer.hmm`

参数说明：
- `--config`: 配置文件路径（默认 `config.yaml`）
- `--model_name`: 模型名称（默认从config读取）
- `--model_size`: 模型大小（默认从config读取）
- `--model_dir`: HMONNX模型目录（默认 `work_dirs/hmonnx`）
- `--ncore`: 核心数（默认从config读取或 `$HOUMO_CORE_NUM`）
- `--stage build`: 仅构建
- `--stage test`: 仅测试（与golden比对余弦相似度）
- `--stage all`: 构建 + 测试
- `--j`: 并行编译任务数
- `--output_dir`: 输出目录（默认 `output/xh2/`）

### 2.6 演示

CT-Transformer模型使用python API进行演示。

```bash
python3 demo.py --model_name ct_transformer --model_size large
```

推理完成将输出性能结果。

参数说明：
- `--config`: 配置文件路径（默认 `config.yaml`）
- `--model_name`: 模型名称（默认从config读取）
- `--model_size`: 模型大小（默认从config读取）
- `--model-dir`: 原始模型目录（用于加载词表/分词器）
- `--input`: 输入文本文件
- `--hmm-dir`: HMM模型所在目录（默认 `output/xh2/`）
- `--model-path`: 直接指定HMM模型路径（替代 `--hmm-dir` + `--model_name`）

### 2.7 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

test.sh支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step         execution step, default is demo, support: all, demo, build, quant.
  -size, --model_size    model size, default is large.
  -name, --model_name    model name, default is ct_transformer.
  --quant_type           quantization type, e.g. w8a8_sefp.
  --skip_download        skip model download steps.
  -h, --help             help information
```

示例：

```bash
# 仅运行demo（使用预编译模型）
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 量化+编译（跳过demo）
bash test.sh -s quant,build

# 指定量化类型
bash test.sh -s all --quant_type w8a8_sefp
```

## 3 config.yaml 配置说明

模型配置通过 `config.yaml` 管理：

```yaml
default_model_name: "ct_transformer"
default_model_size: "large"

model_configs:
  ct_transformer:
    "large":
      modelscope_repo: ["iic/punc_ct-transformer_cn-en-common-vocab471067-large"]
      model_name: "ct_transformer"
      model_size: "large"
      ncore: 2
      ndevice: 1
      quant_type: "w8a8_sefp"
```

| 参数 | 说明 | 支持范围 |
|------|------|----------|
| `ncore` | 核心数 | 1, 2 |
| `ndevice` | 芯片数 | 1 |
| `quant_type` | 量化精度 | `w8a8_sefp`, `w8a8h1_sefp` |

## 4 I/O约定

| 角色 | 名称 | 形状 | 类型 |
|------|------|------|------|
| Input | `text` | `[1, fixed_length]` | `int32` |
| Input | `text_lengths` | `[1]` | `int32` |
| Output | `logits` | `[1, fixed_length, num_puncs]` | `float32` |

其中 `fixed_length` 由输入文本自动计算（最长token数 + 200缓冲），`num_puncs` 为标点类别数。

## 5 资源需求

### 5.1 模型量化资源需求

| 资源类型 | 需求 |
|----------|------|
| Host内存 | 约 8 GB |
| GPU显存 | 约 4 GB（w8a8_sefp 量化精度，单卡 CUDA 12.2） |
| 量化精度 | 支持 `w8a8_sefp`、`w8a8h1_sefp` |

### 5.2 模型编译资源需求

| 资源类型 | 需求 |
|----------|------|
| Host内存 | 约 4 GB |
| 编译产物 | `ct_transformer.hmm`（约 676 MB） |

### 5.3 模型推理资源需求

| 资源类型 | 需求 |
|----------|------|
| Host内存（峰值） | 约 2.5 GB |
| HMM模型大小 | 676 MB |
| 输入长度 | `fixed_length = 220`（自动根据输入文本计算） |

## 6 Demo 结果 & 性能

### 6.1 测试环境

| 项目 | 详情 |
|------|------|
| 镜像 | `harbor.houmo.ai/toolchain/release:v1.4.0-ubuntu24.04-py312-x86.64` |
| HOUMO_TARGET | xh2 |
| Host CPU | 20 核 |
| Host 内存 | 62 GB |

### 6.2 Demo 参考结果

测试输入：`ct_transformer/example/punc_example.txt`（3 段中文文本）

```
[{'text': ' 1跨境河流是养育沿岸人民的生命之源。长期以来，为帮助下游地区防灾减灾，中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难，甚至冒着生命危险向印方提供汛期水文资料，处理紧急事件。中方重视印方在跨境河流问题上的关切，愿意进一步完善双方联合工作机制。凡是中方能做的，我们都会去做，而且会做得更好。我请印度朋友们放心，中国在上游的任何开发利用，都会经过科学规划和论证，兼顾上下游的利益。'}, {'text': ' 2从存储上来说，仅仅是全景图片，它就会是图片的四倍的容量。然后全景的视频会是普通视频八倍的这个存储的容要求，而三d的模型会是图片的十倍。这都对我们今天运行在的云计算的平台，存储的平台提出了更高的要求。'}, {'text': ' 3那今天的会就到这里吧，happy new year,明年见。'}]
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:248 - ====================================================================================================
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:249 -                     Model Inference Performance Summary Report
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:250 - ====================================================================================================
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:251 - Performance Details:
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:252 -   Load Model+Tokenizer   : 15211.20 ms
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:253 -   Total Inference (3  seqs) :   83.65 ms
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:255 -   Average Inference/seq  :   27.88 ms
2026-06-02 04:56:04.236 | SUCCESS  | __main__:main:256 - ====================================================================================================
```

## 7 功能支持说明

CT-Transformer 为标点恢复（Punctuation Restoration）小模型，以下列出各功能的支持状态：

| 功能 | 支持状态 | 说明 |
|------|----------|------|
| 多 batch 推理 | 不支持 | 模型固定 batch=1 |
| 多核 (ncore) | 支持 | 支持 1/2 核编译 |
| 多芯 (ndevice) | 不支持 | 仅支持单芯片 |
| MTP | 不适用 | 非 LLM 模型 |
| prefix_caching | 不适用 | 非 LLM 模型 |
| 交互模式 (it) | 不支持 | 仅支持批量文本文件输入 |
| 多轮对话 (history) | 不适用 | 标点恢复为单轮任务 |

## 8 注意事项

1. 确保 `ct_transformer/` 目录存在且包含 `config.yaml`、词表和模型权重
2. PTQ量化步骤可选，如需直接从ONNX构建HMM可跳过步骤2.4
3. TCIM构建仅支持x86_64平台
4. `demo.py` 中的分词和词表处理依赖原始模型目录中的 `jieba_usr_dict` 和 `tokens.json`
5. 所有脚本均需在 `ct_transformer/` 目录下执行
6. `HOUMO_TARGET` 环境变量默认为 `xh2`，可通过 `export HOUMO_TARGET=xh2` 设置

## 9 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
