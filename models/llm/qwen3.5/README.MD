# Qwen3.5 / Qwen3.6

本示例展示如何把 Qwen3.5 和 Qwen3.6 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3.5-0.8B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-2B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-2B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-4B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-4B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-9B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-9B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.6-27B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.6-27B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.6-35B-A3B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=qwen3.6`
- `model_size=35b-a3b`
- `context_length=256k`
- `prefill_length=256`
- `ndevice=1`

支持的模型组合为：

- `qwen3.5`: `0.8b`、`2b`、`4b`、`9b`
- `qwen3.6`: `27b`、`35b-a3b`

标准流程编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`
- `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

MTP 流程额外产物命名为：

- `${model_name}-${model_size}_prefill_mtp.hmm`
- `${model_name}-${model_size}_decode_mtp.hmm`

`demo.py` 与 `demo_mtp.py` 默认从 `config.yaml` 中读取 `modelscope_repo`，并使用其最后一段作为 `tokenizer_dir` 默认值。

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen3.5 / Qwen3.6 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3.5 --model_size 9b
python3 get_model.py --type raw --model_name qwen3.6 --model_size 35b-a3b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3.6 --model_size 35b-a3b
```

若需要 MTP 预编译模型，可增加 `--mtp`。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/qwen3.5
```

推荐先通过 `get_model.py --type raw` 下载原始模型，再执行：

```bash
python3 ptq.py --model ./Qwen3.6-35B-A3B
```

如果当前目录下已经存在与配置匹配的模型目录，也可以直接运行 `python3 ptq.py`，脚本会按 `config.yaml` 推导默认模型目录名。

量化输出位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

标准编译：

```bash
python3 build.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 编译：

```bash
python3 build_mtp.py --model_name qwen3.6 --model_size 35b-a3b
```

默认输出名称分别为：

- `qwen3.6-35b-a3b_prefill.hmm`
- `qwen3.6-35b-a3b_decode.hmm`
- `qwen3.6-35b-a3b_visual_896x896x2.hmm`
- `qwen3.6-35b-a3b_prefill_mtp.hmm`
- `qwen3.6-35b-a3b_decode_mtp.hmm`

### 2.5 演示

标准演示：

```bash
python3 demo.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 演示：

```bash
python3 demo_mtp.py --model_name qwen3.6 --model_size 35b-a3b
```

多设备场景下，demo 脚本会自动把 `prefill` / `decode` 默认模型后缀从 `.hmm` 切换为 `.hmms`，`visual` 仍保持 `.hmm`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3.6`
- `MODEL_SIZE=35b-a3b`
- `NDEVICE=1`
- `MTP=false`

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
  -name, --model_name     Model name. Choices: qwen3.5, qwen3.6.
  -size, --model_size     Model size.
  --ndevice               Number of devices. Default: 1.
  --mtp                   Enable mtp mode.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 指定模型为 Qwen3.5-9B
bash test.sh --model_name qwen3.5 --model_size 9b

# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo --model_name qwen3.6 --model_size 35b-a3b

# 完整流程
bash test.sh -s all --model_name qwen3.6 --model_size 35b-a3b

# MTP 演示或编译
bash test.sh --mtp --model_name qwen3.6 --model_size 35b-a3b

# 跳过下载，直接使用当前目录已有模型
bash test.sh -s demo --skip_download --model_name qwen3.6 --model_size 35b-a3b
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化。
- `build`：默认调用 `build.py`；当 `--mtp` 启用时改为调用 `build_mtp.py`。
- `demo`：下载预编译模型后执行 `demo.py`；当 `--mtp` 启用时执行 `demo_mtp.py`。
- 非 MTP `demo` 流程若环境中存在 `llm_perf`，末尾还会自动执行性能测试。

## 4 参考结果

### 4.1 演示结果

此结果为 qwen3.6-35b-a3b 的演示输出，包含模型回答和性能测试结果。

```bash
prefill model loaded
decode model loaded
vision model loaded
question: 描述这些图片
images: ['/home/workspace/imodelzoo/tests/apis_tests/../../data/pic/beach.jpeg']
processor image_grid_thw: tensor([[ 1, 56, 56]])
processor image token count: 784
Running vision model...
image_embeds shape: torch.Size([784, 2048])
response:
这张图片展现了一幅温馨、宁静又充满活力的海滩场景，主要人物是一位年轻女性和一只金毛犬。
以下是详细描述：
🌅 **环境与氛围**：
- 时间似乎是日落或日出时分，阳光从画面右侧斜射过来，在沙滩和海面上洒下温暖的金色光芒。
- 背景是平静的海洋，海浪轻轻拍打着岸边，远处海天相接，视野开阔。
- 整体色调柔和温暖，带有梦幻般的逆光效果，营造出一种治愈、放松、人与自然和谐共处的氛围。

🐶 **狗狗**：
- 一只体型健壮的金毛寻回犬（或类似品种），毛色呈浅金色，在阳光下显得柔软发亮。
- 它戴着一条彩色图案的胸背带（可能是牵引绳的一部分），正坐着，前爪抬起，与女子击掌（“high five”），表情专注而友好。
- 尾巴自然垂在沙地上，姿态放松但配合度高，显示出良好的训练和亲密关系。

👩 **女子**：
- 年轻女性，长发披肩，面带灿烂笑容，侧脸对着镜头，眼神温柔地注视着狗狗。
- 她穿着格子衬衫（蓝白或灰白相间）和深色裤子，赤脚坐在沙滩上，姿态随意自然。
- 她伸出右手与狗狗击掌，动作轻盈，充满互动乐趣。

🏖️ **细节**：
- 沙滩上有清晰的脚印和凹陷，说明有人活动过。
- 狗狗旁边放着一个红色的小玩具（可能是绳结玩具），暗示他们刚刚玩耍过。
- 光线在人物和动物身上形成柔和的轮廓光，增强了画面的艺术感和情感温度。

✅ **总结**：
这是一张充满生活气息和情感温度的照片，捕捉了人与宠物之间纯真快乐的瞬间。它不仅展现了美丽的自然风光，更传递了陪伴、信任与幸福的主题，适合用于表达“治愈”、“陪伴”、“自由”或“简单快乐”的概念。

——  

====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:    809 tokens
  Output Length per Sample:    433 tokens
  Number of Images:      1 images
  Prefill Model Load Time: 7380.25ms
  Decode Model Load Time:  206.93ms
  Vision Model Load Time: 1118.98ms
Vision Stage Performance:
  Total Time: 1278.11ms | Speed:    0.78 images/s
  Preprocessing Time: 81.88ms | Speed:   12.21 images/s
  API SetInput Time:  11.24ms
  API Inference Time: 1171.61ms | Speed:    0.85 images/s
  API GetOutput Time:  2.23ms
Prefill Stage Performance:
  Total Time: 2588.12ms | Speed:  312.58 tokens/s
  Tokenization Time:   15.26ms
  Embedding Time:   12.26ms
  API SetInput Time:  23.54ms
  API Inference Time: 1245.24ms | Prefill Speed:  649.67 tokens/s
  API GetOutput Time:  0.78ms
Decode Stage Performance:
  Total Time: 20242.60ms | Speed:   21.34 tokens/s
  Tokenization Time: 4578.06ms
  Embedding Time:   78.09ms
  API SetInput Time:   1.83ms
  API Inference Time: 33.72ms | Decode Speed:   29.66 tokens/s
  API GetOutput Time:  0.66ms
Overall Performance Metrics:
  TTFT (Time To First Token): 3866.24 ms
  TPOT (Time Per Output Token): 46.86 ms/token
  E2E Latency (End-to-End):     22.98 seconds
  E2E TPS (Throughput):         18.84 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
