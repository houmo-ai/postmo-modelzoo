# Qwen2.5-VL-7B-Instruct

本示例展示如何把Qwen2.5VL模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen2.5-VL-7B-Instruct
- 来源: https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。
同时，原有的预处理代码 `image_processing_qwen2_vl.py` 和 `processing_qwen2_5_vl.py` 已进行专门的改编和修改，并包含在项目提交文件中。特此声明。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen2.5-vl`
- `model_size=7b`
- `ndevice=1`

### 用于校准的数据集

本项目引用 LAION 220k-GPT4Vision-captions-from-LIVIS 数据集仅用于**研究和评估目的**。

- 数据集名称: 220k-GPT4Vision-captions-from-LIVIS
- 数据集许可: Apache License 2.0
- 来源: https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**，由 `ptq.py` 在量化时默认通过 HuggingFace 接口按需下载。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (Apache License 2.0)。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前Qwen2.5VL模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen2.5-vl --model_size 7b
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm --model_name qwen2.5-vl --model_size 7b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/vlm/qwen2.5-vl` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd models/vlm/qwen2.5-vl
python3 ptq.py --model-name qwen2.5-vl --model-size 7b
```

#### 2.3.1 自定义数据集量化

下载自定义数据集到当前目录下
```bash
python3 get_model.py --type raw --model_name qwen2.5-vl --model_size 7b
mkdir -p data/calib_data/LMUData
ln -s ${HOUMO_EXAMPLES_PATH}/data/datasets/LMUData/* data/calib_data/LMUData
```

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen2.5-vl --model-size 7b --data_files ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 256 --calib_dataset vllm_custom_data
```

#### 2.3.2 默认公开数据集量化

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen2.5-vl --model-size 7b
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen2.5-vl --model_size 7b
```

默认输出名称分别为：

- `qwen2.5-vl-7b_prefill.hmm`
- `qwen2.5-vl-7b_decode.hmm`
- `qwen2.5-vl-7b_visual_448x448x2.hmm`

### 2.5 演示

Qwen2.5-vl模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过demo中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

注意：由于ubuntu18.04下qwen_vl_utils无法正常安装，导致无法正常运行。

```bash
# Qwen2.5-VL-7B 推荐惩罚系数是1.1
python3 demo.py --model_name qwen2.5-vl --model_size 7b --image ../../../data/pic/beach.jpeg --repetition_penalty 1.1
```

若未显式传 `tokenizer_dir`、`vit_path`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen2.5-vl`
- `MODEL_SIZE=7b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录；`quant` 和 `llm_perf` 流程都会用到该路径。

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 7b.
  -name, --model_name     Model name. Default: qwen2.5-vl.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

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

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型，链接 `LMUData` 数据集，并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 `prefill`、`decode` 和 `visual` 模型。
- `demo`：下载预编译模型后，执行 Python demo。
- 若环境中存在 `llm_perf`，`demo` 流程末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

```bash
question:
请描述图片内容。
response:
这张图片展示了一位坐在沙滩上的女士和一只狗。女士穿着格子衬衫，正在与狗互动。狗看起来是一只金毛猎犬，正抬起前爪，似乎在请求食物或玩具。背景是广阔的海洋和天空，在夕阳的映照下显得非常温暖和谐。整个场景给人一种宁静、愉快的感觉。
```

### 4.2 性能结果

```bash
====================================================================================================
                                         Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
    Batch Size:      1
    Input Length per Sample:    282 tokens
    Output Length per Sample:     71 tokens
    Number of Images:      1 images
    Prefill Model Load Time: 2033.03ms
    Decode Model Load Time:  162.20ms
    Vision Model Load Time: 1164.99ms
Vision Stage Performance:
    Total Time:  201.65ms | Speed:    4.96 images/s
    Preprocessing Time:  2.52ms | Speed:  396.85 images/s
    API SetInput Time:   3.95ms
    API Inference Time: 192.13ms | Speed:    5.20 images/s
    API GetOutput Time:  2.06ms
Prefill Stage Performance:
    Total Time:  451.60ms | Speed:  624.44 tokens/s
    Tokenization Time: Skipped (No operation)
    Embedding Time:    0.42ms
    API SetInput Time:   4.26ms
    API Inference Time: 437.51ms | Prefill Speed:  644.55 tokens/s
    API GetOutput Time:  1.10ms
Decode Stage Performance:
    Total Time: 3813.21ms | Speed:   18.36 tokens/s
    Tokenization Time:    7.97ms
    Embedding Time:   16.18ms
    API SetInput Time:   2.39ms
    API Inference Time: 44.70ms | Decode Speed:   22.37 tokens/s
    API GetOutput Time:  0.53ms
Overall Performance Metrics:
    TTFT (Time To First Token):  653.25 ms
    TPOT (Time Per Output Token): 54.47 ms/token
    E2E Latency (End-to-End):      4.60 seconds
    E2E TPS (Throughput):         15.44 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
