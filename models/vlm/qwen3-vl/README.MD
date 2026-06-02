# Qwen3-VL-4B-Instruct 和 Qwen3-VL-8B-Instruct

本示例展示如何把 Qwen3-VL 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3-VL-4B-Instruct
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3-VL-8B-Instruct
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。
同时，原有的预处理代码 `image_processing_qwen2_vl.py` 和 `processing_qwen3_vl.py` 已进行专门的改编和修改，并包含在项目提交文件中。特此声明。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3-vl`
- `model_size=8b`（可选 `4b`）
- `context_length=32k`
- `prefill_length=256`
- `ndevice=1`

视觉产物命名统一为 `${model_name}-${model_size}_visual_448x448x2.hmm`。

### 用于校准的数据集

本项目引用 LAION 220k-GPT4Vision-captions-from-LVIS 数据集仅用于**研究和评估目的**。

- 数据集名称: 220k-GPT4Vision-captions-from-LVIS
- 数据集许可: Apache License 2.0
- 来源: https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**，由 `ptq.py` 在量化时默认通过 HuggingFace 接口按需下载。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (Apache License 2.0)。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen3-VL 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3-vl --model_size 8b

# 下载 4b 原始模型用于量化
python3 get_model.py --type raw --model_name qwen3-vl --model_size 4b
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm --model_name qwen3-vl --model_size 8b

# 下载 4b 预编译模型
python3 get_model.py --type hmm --model_name qwen3-vl --model_size 4b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

#### 2.3.1 自定义数据集量化

下载自定义数据集到当前目录下：

```bash
python3 get_model.py --type raw
mkdir -p data/calib_data/LMUData
ln -s ../../../data/datasets/LMUData/* data/calib_data/LMUData
```

量化下载的原始模型。在当前目录下执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen3-vl --model-size 8b --data_files ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 64 --calib_dataset vllm_custom_data
```

#### 2.3.2 默认公开数据集量化

量化下载的原始模型。在当前目录下执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen3-vl --model-size 8b

# 或量化 4b
python3 ptq.py --model-name qwen3-vl --model-size 4b
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3-vl --model_size 8b

# 或编译 4b
python3 build.py --model_name qwen3-vl --model_size 4b
```

默认输出名称分别为：

对于 Qwen3-VL-8B-Instruct：
- `qwen3-vl-8b_prefill.hmm`
- `qwen3-vl-8b_decode.hmm`
- `qwen3-vl-8b_visual_448x448x2.hmm`

对于 Qwen3-VL-4B-Instruct：
- `qwen3-vl-4b_prefill.hmm`
- `qwen3-vl-4b_decode.hmm`
- `qwen3-vl-4b_visual_448x448x2.hmm`

### 2.5 演示

Qwen3-VL 模型使用 Python API 和 C++ 示例进行演示，C++ 模型示例的详细介绍见 `cpp/README.MD`。

1. 将编译好的模型放在当前目录，同时将对应量化模型的 `quant_embedding.pt` 文件放在量化模型目录下。

2. 执行脚本，推理完成将输出性能结果。

注意：由于 Ubuntu 18.04 下 `qwen_vl_utils` 无法正常安装，示例无法正常运行。

`demo.py` 现在支持显式指定输入模式：

- 图片模式：使用 `--image` 输入一张或多张图片。
- 视频模式：使用 `--video` 输入真实视频，或者使用 `--media_type video --image ...` 将一张或多张图片按伪视频方式输入。
- 当未显式指定 `--prompt` 时，图片模式默认提示词为“描述图片内容”，视频模式默认提示词为“描述视频内容”。
- 视频模式下，无论输入真实视频还是图片伪视频，都会按 visual 模型时间维度自动进行抽帧或 padding。
- 多图伪视频的时间语义可通过 `--multi_image_video_fps` 指定，默认值为 `2.0`。

```bash
# Qwen3-VL-4B 推荐惩罚系数是4
# Qwen3-VL-8B 推荐惩罚系数是1.5

# 单图或多图图片模式
python3 demo.py --model_name qwen3-vl --model_size 8b --image ../../../data/pic/beach.jpeg --repetition_penalty 1.5
python3 demo.py --model_name qwen3-vl --model_size 4b --image ../../../data/pic/beach.jpeg --repetition_penalty 4
python3 demo.py --model_name qwen3-vl --model_size 8b --image img1.jpg img2.jpg img3.jpg --repetition_penalty 1.5

# 真实视频模式
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --video demo.mp4 --repetition_penalty 1.5

# 多图伪视频模式
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --image img1.jpg img2.jpg img3.jpg --repetition_penalty 1.5

# 指定多图伪视频的 fps
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --image img1.jpg img2.jpg img3.jpg --multi_image_video_fps 4 --repetition_penalty 1.5
```

若未显式传 `tokenizer_dir`、`vit_path`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3-vl`
- `MODEL_SIZE=8b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录；`quant`、`cpp demo` 和 `llm_perf` 流程都会用到该路径。

当前脚本会把 `model_name`、`model_size` 和 `ndevice` 透传给 `get_model.py`、`ptq.py`、`build.py`、`demo.py`。默认值分别为 `qwen3-vl`、`8b` 和 `1`。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 8b. Optional: 4b, 8b.
  -name, --model_name     Model name. Default: qwen3-vl.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 4b quant
bash test.sh -s quant --model_size 4b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 4b build
bash test.sh -s build --model_size 4b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 运行 4b demo
bash test.sh -s demo --model_size 4b

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型，链接 `LMUData` 数据集，并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 `prefill`、`decode` 和 `visual` 模型。
- `demo`：下载预编译模型后，先执行 Python demo，再执行 C++ demo。
- 若环境中存在 `llm_perf`，`demo` 流程末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

```bash
question:
描述图片内容
response:
这是一张充满温馨与宁静氛围的海滩日落照片。
画面中，一位年轻女子和一只金毛犬正坐在沙滩上。女子身穿蓝白格子衬衫，侧身面向狗狗，面带微笑，似乎正在与它互动。她的一只手轻轻抬起，可能是在与狗狗玩耍或进行训练。金毛犬戴着一个蓝色的项圈，正用前爪轻轻搭在女子的手上，姿态温顺可爱，显得十分乖巧。
他们身处一片柔软的沙滩上，背景是波光粼粼的大海和被夕阳染成温暖橙黄色的天空。阳光从画面右侧洒下，在沙滩上投下柔和的光影，营造出一种温暖、放松的感觉。远处的海浪正轻轻拍打着海岸线。
整张图片构图和谐，色彩温暖，捕捉了人与宠物之间亲密无间的美好瞬间，传达出一种悠闲自在、幸福满足的情感。
```

### 4.2 性能结果

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:    209 tokens
  Output Length per Sample:    159 tokens
  Number of Images:      1 images
  Prefill Model Load Time: 1777.62ms
  Decode Model Load Time:  150.99ms
  Vision Model Load Time:  379.54ms
Vision Stage Performance:
  Total Time:  125.42ms | Speed:    7.97 images/s
  Preprocessing Time: 24.92ms | Speed:   40.13 images/s
  API SetInput Time:   1.17ms
  API Inference Time: 90.97ms | Speed:   10.99 images/s
  API GetOutput Time:  6.97ms
Prefill Stage Performance:
  Total Time:  177.68ms | Speed: 1176.28 tokens/s
  Tokenization Time:   23.31ms
  Embedding Time:    0.01ms
  API SetInput Time:  11.67ms
  API Inference Time: 122.99ms | Prefill Speed: 1699.33 tokens/s
  API GetOutput Time:  0.58ms
Decode Stage Performance:
  Total Time: 9301.73ms | Speed:   16.99 tokens/s
  Tokenization Time:   56.17ms
  Embedding Time:    0.45ms
  API SetInput Time:   2.98ms
  API Inference Time: 47.01ms | Decode Speed:   21.27 tokens/s
  API GetOutput Time:  0.53ms
Overall Performance Metrics:
  TTFT (Time To First Token):  303.10 ms
  TPOT (Time Per Output Token): 58.87 ms/token
  E2E Latency (End-to-End):      9.67 seconds
  E2E TPS (Throughput):         16.45 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
