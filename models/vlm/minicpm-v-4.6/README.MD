# MiniCPM-V 4.6

本示例展示如何量化和编译 MiniCPM-V 4.6 模型，并通过 Houmo Python Engine 部署到后摩芯片设备上进行文本和图片理解推理。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: MiniCPM-V 4.6
- 来源: [OpenBMB/MiniCPM-V-4.6](https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4.6)
- 许可: Apache License 2.0 ([LICENSE](https://github.com/OpenBMB/MiniCPM-V/blob/main/LICENSE))

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型配置为：

- `model_name=minicpm`
- `model_size=v-4.6`
- `context_length=256k`
- `prefill_length=256`
- `quant_type=w8a8h1_sefp`
- `ncore=2`
- `ndevice=1`
- `batch=1`

Python Demo 默认使用 `16x` 视觉下采样模式，也支持通过 `--downsample-mode 4x` 切换到 `4x` 模式。

### 资源要求

**后摩设备运行要求：**

- `MiniCPM-V 4.6`: 单卡12GB可运行

**量化硬件要求：**

- `MiniCPM-V 4.6`: 需要 GPU，显存要求≥24GB

**编译时 host 内存要求：**

- `MiniCPM-V 4.6`: host内存≥128GB

## 2 快速开始

MiniCPM-V 4.6 当前提供 Python 脚本量化、编译方式和基于 Houmo Python Engine 的 Python 推理 Demo。芯片模型只能在 Linux 环境下量化和编译。

以下命令默认在仓库根目录执行过环境初始化后，从当前模型目录运行：

```bash
source env.sh
cd models/vlm/minicpm-v-4.6
```

### 2.1 环境准备

安装示例运行所需的 Python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

当前依赖包含 `transformers==5.7.0`。

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型资源。脚本默认从同目录的 `config.yaml` 读取模型名称、规格、设备数、上下文长度和编译配置。

如果需要自行量化，可下载原始模型：

```bash
python3 get_model.py --type raw --model_name minicpm --model_size v-4.6
```

原始模型默认下载到当前目录的 `MiniCPM-V-4.6/`。

如果仅运行演示，可下载预编译芯片模型：

```bash
python3 get_model.py --type hmm --model_name minicpm --model_size v-4.6
```

下载后的资源默认解压到当前目录。Demo 默认从以下位置读取芯片模型和相关资源：

- `output/${HOUMO_TARGET}/minicpm-v-4.6_prefill.hmm`
- `output/${HOUMO_TARGET}/minicpm-v-4.6_decode.hmm`
- `output/${HOUMO_TARGET}/minicpm-v-4.6_visual_16x.hmm`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `MiniCPM-V-4.6/`

当 `--ndevice` 大于 1 时，prefill 和 decode 的默认文件后缀切换为 `.hmms`；视觉模型仍使用 `.hmm`。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前模型目录运行量化脚本：

```bash
source env.sh
cd models/vlm/minicpm-v-4.6
python3 ptq.py --model-name minicpm --model-size v-4.6
```

`ptq.py` 默认读取 `MiniCPM-V-4.6/` 中的原始模型，并使用 `config.yaml` 指定的 Merak workflow 执行量化和 HMONNX 导出。默认输出目录为：

```text
output/${HOUMO_TARGET}/hmquant/
```

主要产物包括：

- `prefill/`
- `decode/`
- `vision_4x/`
- `vision_16x/`
- `quant_embedding.pt`

如需覆盖已有量化输出，可执行：

```bash
python3 ptq.py --model-name minicpm --model-size v-4.6 --overwrite
```

### 2.4 编译

编译前需要准备与 `build.py` 输入约定一致的中间模型，默认位于：

```text
output/${HOUMO_TARGET}/hmquant/
```

该目录需要包含 `prefill`、包含 `decode` 名称的目录，以及以 `vision` 或 `visual` 命名的视觉模型目录。这些中间模型可通过上一节的 `ptq.py` 生成。

执行编译：

```bash
python3 build.py --model_name minicpm --model_size v-4.6 --ndevice 1
```

默认输出目录为 `output/${HOUMO_TARGET}`。prefill 和 decode 输出名称为：

- `minicpm-v-4.6_prefill.hmm`
- `minicpm-v-4.6_decode.hmm`

视觉模型的输出名称由输入视觉目录名决定。默认量化流程会生成 `vision_4x` 和 `vision_16x`，对应输出为：

- `minicpm-v-4.6_visual_4x.hmm`
- `minicpm-v-4.6_visual_16x.hmm`

如需在编译后执行 golden comparison，可使用：

```bash
python3 build.py --model_name minicpm --model_size v-4.6 --stage all
```

### 2.5 演示

`demo.py` 使用 `Demo -> MiniCPMV46Engine -> MiniCPMV46Process / MiniCPMV46Module` 分层框架执行推理，支持文本输入、单图输入和多图输入。

文本推理：

```bash
python3 demo.py --question "介绍下存算一体的技术优势。"
```

单图推理：

```bash
python3 demo.py \
    --image-path ../../../data/pic/beach.jpeg \
    --question "介绍下这个图片?"
```

多图推理：

```bash
python3 demo.py \
    --image-path image1.jpg image2.jpg \
    --question "比较这两张图片。"
```

使用 `4x` 视觉模型时，需要同时指定下采样模式或显式视觉模型路径：

```bash
python3 demo.py \
    --downsample-mode 4x \
    --image-path ../../../data/pic/beach.jpeg
```

采样相关参数：

- 默认使用 greedy decoding。
- 使用 `--no-greedy` 启用随机采样。
- `--temperature` 设置采样温度，默认值为 `0.7`。
- `--seed` 设置随机种子，默认值为 `42`。

常用路径参数：

```bash
python3 demo.py \
    --tokenizer-dir MiniCPM-V-4.6 \
    --prefill-path output/${HOUMO_TARGET}/minicpm-v-4.6_prefill.hmm \
    --decode-path output/${HOUMO_TARGET}/minicpm-v-4.6_decode.hmm \
    --vit-path output/${HOUMO_TARGET}/minicpm-v-4.6_visual_16x.hmm \
    --embedding-path output/${HOUMO_TARGET}/hmquant/quant_embedding.pt
```

默认启用性能报告，可通过以下命令关闭：

```bash
python3 demo.py --perf false
```

## 3 一键评估

`test.sh` 默认执行 `demo` 步骤，并在未指定 `--skip_download` 时先下载预编译模型：

```bash
bash test.sh
```

使用本地已有资源执行 Demo：

```bash
bash test.sh -s demo --skip_download
```

仅执行量化，会先下载原始模型：

```bash
bash test.sh -s quant
```

使用已有原始模型执行量化：

```bash
bash test.sh -s quant --skip_download
```

使用已有量化中间模型执行编译：

```bash
bash test.sh -s build --skip_download
```

当前推荐使用的参数为：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags.
  -name, --model_name     Model name. Default: minicpm.
  -size, --model_size     Model size. Default: v-4.6.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  --system_prompt         System prompt passed to demo.py.
  -h, --help              Show help information.
```

完整流程会依次下载原始模型、量化、编译并运行 Demo：

```bash
bash test.sh -s all
```

使用本地已有原始模型执行完整流程：

```bash
bash test.sh -s all --skip_download
```

当前脚本行为说明：

- `quant`：检查 GPU，按需下载原始模型，并调用 `ptq.py` 生成量化 HMONNX 中间模型。
- `build`：调用 `build.py` 编译 prefill、decode、vision 4x 和 vision 16x 模型。
- `demo`：默认下载预编译模型后调用 `demo.py`；与其他步骤组合执行时直接使用本地构建产物。
- `all`：依次执行 quant、build 和 demo。

## 4 参考结果

### 4.1 演示结果

暂无结果，待后续实测补充。

### 4.2 性能结果

暂无可提供的参考性能数据，待后续实测补充。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
