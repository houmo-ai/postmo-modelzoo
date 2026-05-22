# Qwen3.5 / Qwen3.6

本示例展示如何把Qwen3.5模型量化和编译，部署到后摩芯片的设备上。Qwen3.6-35B-A3B 与当前Qwen3.5 MoE量化导出流程兼容，可通过 `--model_name` 选择。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 来源：
    + Qwen3.5-0.8B: https://modelscope.cn/models/Qwen/Qwen3.5-0.8B
    + Qwen3.5-2B: https://modelscope.cn/models/Qwen/Qwen3.5-2B
    + Qwen3.5-4B: https://modelscope.cn/models/Qwen/Qwen3.5-4B
    + Qwen3.5-9B: https://modelscope.cn/models/Qwen/Qwen3.5-9B
    + Qwen3.5-27B: https://modelscope.cn/models/Qwen/Qwen3.5-27B
    + Qwen3.5-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B
    + Qwen3.6-27B: https://modelscope.cn/models/Qwen/Qwen3.6-27B
    + Qwen3.6-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B
- 许可: Apache License 2.0

预训练模型在运行时下载，工程发布**不包含**该模型。

## 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:
- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：
- Qwen3.5-0.8B: https://modelscope.cn/models/Qwen/Qwen3.5-0.8B
- Qwen3.5-2B: https://modelscope.cn/models/Qwen/Qwen3.5-2B
- Qwen3.5-4B: https://modelscope.cn/models/Qwen/Qwen3.5-4B
- Qwen3.5-9B: https://modelscope.cn/models/Qwen/Qwen3.5-9B
- Qwen3.5-27B: https://modelscope.cn/models/Qwen/Qwen3.5-27B
- Qwen3.5-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B
- Qwen3.6-27B: https://modelscope.cn/models/Qwen/Qwen3.6-27B
- Qwen3.6-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B

本示例为Qwen3.5适配目录，同时兼容Qwen3.6-35B-A3B。Qwen3.6-35B-A3B 复用当前Qwen3.5 MoE量化、导出、编译和演示流程。

## 2 快速开始

目前Qwen3.5 / Qwen3.6模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2.2 获取模型和数据

通过 get_model.py 脚本下载模型，可通过参数选择下载原始模型或芯片模型。

Qwen3.5-0.8B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 0.8b
```

Qwen3.5-2B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 2b
```

Qwen3.5-4B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 4b
```

Qwen3.5-9B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 9b
```

Qwen3.5-27B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 27b
```

Qwen3.5-35B-A3B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 35b-a3b
```

Qwen3.6-35B-A3B 模型：
```bash
python3 get_model.py --type raw --model_name qwen3.6 --model_size 35b-a3b
```

如果仅查看演示结果，可选择下载芯片模型：
```bash
python3 get_model.py --type hmm --model_name qwen3.6 --model_size 35b-a3b
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后 source 环境变量，并进入 `models/llm/qwen3.5` 目录。当前目录中的量化脚本整理中，本节暂记为 TODO。
量化下载的原始模型。在当前目录下，执行脚本：
```bash
pip3 install transformers==5.2.0
python ptq.py  --model /path/to/downloaded/model
```

量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：

```bash
.
|-- decoder
|-- prefill
|-- quant_embedding.pt
`-- weight.npy
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。

```bash
python3 build.py
```

如需指定模型版本，可附加 `--model_size` 参数，具体取值请以脚本帮助信息为准。
Qwen3.6-35B-A3B 使用 `--model_size 3.6-35b-a3b`，test.sh 会自动使用 `qwen3.6` 作为导出和编译模型名前缀。

编译好的模型存放在当前目录，包括2个文件：
- qwen3.5_prefill.hmm
- qwen3.5_decode.hmm

如包含视觉分支，还会生成：
- qwen3.5_visual.hmm

Qwen3.6-35B-A3B 对应输出文件名前缀为 `qwen3.6`。

### 2.5 演示

Qwen3.5 / Qwen3.6模型使用python API进行演示。

```bash
python3 demo.py
```

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

如需指定模型版本，可附加 `-size/--model_size` 参数，具体取值请以脚本帮助信息为准。Qwen3.6-35B-A3B 示例：
```bash
bash test.sh --model_name qwen3.6 --model_size 35b-a3b
```

## 3 参考结果

### 3.1 演示结果

TODO

### 3.2 性能结果

TODO

### 3.3 性能测试工具

- 使用llm_perf性能评测工具测试模型性能。
- llm_perf工具详细使用说明参考${HOUMO_EXAMPLE_PATH}/tools/llm_perf目录的README.MD。
- 如果使用发布的容器，llm_perf工具已经预编译安装，可以直接通过llm_perf --help获取帮助。
- 配置文件与测试样例仍在整理中，本节暂记为 TODO。

## 4 命令行参数说明

### get_model.py
- `--model_name`: 模型名称，支持 `qwen3.5` 或 `qwen3.6`，默认 `qwen3.6`
- `--model_size`: 模型大小，支持  `0.8b`、`2b`、`4b`、`9b`、`27b`、`35b-a3b`，默认 `35b-a3b`
- `--type`: 下载类型，支持 `raw` 或 `hmm`，默认 `hmm`
- `--download_dir`: 下载目录，默认当前目录
- `--extract_dir`: 解压目录，默认 `None`
- `--source_type`: 下载源，支持 `jfrog` 或 `modelscope`，默认 `jfrog`
- `--context_length`: 上下文长度，默认 `8k`
- `--batch`: 批次大小，默认 `1`
- `--ndevice`: 设备数量，默认 `1`

### ptq.py
- TODO

### build.py
- `--model_name`: 模型名称，支持 `qwen3.5` 或 `qwen3.6`，默认 `qwen3.6`
- `--model_size`: 模型大小参数，具体取值请以脚本帮助信息为准
- `--batch`: 批次大小（默认：1）
- `--context_length`: 上下文长度（默认：262144）
- `--flash_attention`: Flash Attention优化（默认：2 1）
- `--device_kernel_split`: 设备内核分割
- `--stage`: 构建阶段，build/test/all（默认：build）

### demo.py
- `--tokenizer_dir`: tokenizer目录（默认：qwen3.6）
- `--embedding_path`: embedding权重路径
- `--prefill_path`: prefill模型路径
- `--decode_path`: decode模型路径
- `--vision_path`: vision模型路径
- `--ndevice`: 设备数量（默认：1）
- `--image_path`: 图片输入路径，支持一次传入多张图片、重复传参或逗号分隔
- `--it`: 交互模式，若提供 `--vision_path`，每轮对话可逐张输入图片路径，留空即开始推理，也可首个输入直接留空走纯文本
- `--history`: 保留对话历史
- `--debug`: 输出调试日志

### test.sh
- `-name, --model_name`: 模型名称，支持 `qwen3.5` 或 `qwen3.6`，默认 `qwen3.6`
- `-size, --model_size`: 模型大小，支持 `0.8b`、`2b`、`4b`、`9b`、`27b`、`35b-a3b` 或 `35b-a3b`，默认 `35b-a3b`
- `-size, --model_size`: 模型大小，支持 `0.8b`、`2b`、`4b`、`9b`、`27b`、`35b-a3b` 或 `35b-a3b`，默认 `35b-a3b`
- `-s, --step`: 执行步骤，demo/build/quant/all（默认：demo）

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
