# Qwen3.5(通义千问3.5)

本示例展示如何把Qwen3.5模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: Qwen3.5-2B / Qwen3.5-4B / Qwen3.5-9B / Qwen3.5-27B / Qwen3.5-35B-A3B
- 来源: https://modelscope.cn/models/Qwen/Qwen3.5-2B https://modelscope.cn/models/Qwen/Qwen3.5-4B https://modelscope.cn/models/Qwen/Qwen3.5-9B https://modelscope.cn/models/Qwen/Qwen3.5-27B https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B
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
- Qwen3.5-2B: https://modelscope.cn/models/Qwen/Qwen3.5-2B
- Qwen3.5-4B: https://modelscope.cn/models/Qwen/Qwen3.5-4B
- Qwen3.5-9B: https://modelscope.cn/models/Qwen/Qwen3.5-9B
- Qwen3.5-27B: https://modelscope.cn/models/Qwen/Qwen3.5-27B
- Qwen3.5-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B

本示例为Qwen3.5适配目录。不同模型版本涉及的参数和流程仍在整理中，具体以脚本实际支持情况为准。

## 2 快速开始

目前Qwen3.5模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2.2 获取模型和数据

通过 get_model.py 脚本下载模型，可通过参数选择下载原始模型或芯片模型。

2B 模型：
```bash
python3 get_model.py --type raw --model_size 2b
```

4B 模型：
```bash
python3 get_model.py --type raw --model_size 4b
```

9B 模型：
```bash
python3 get_model.py --type raw --model_size 9b
```

27B 模型：
```bash
python3 get_model.py --type raw --model_size 27b
```

35B-A3B 模型：
```bash
python3 get_model.py --type raw --model_size 35b-a3b
```

如果仅查看演示结果，可选择下载芯片模型：
```bash
python3 get_model.py --type hmm --model_size 9b
```

如需下载 35B-A3B 的芯片模型，可将 `--model_size` 改为 `35b-a3b`。

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

编译好的模型存放在当前目录，包括2个文件：
- qwen3.5_prefill.hmm
- qwen3.5_decode.hmm

如包含视觉分支，还会生成：
- qwen3.5_visual.hmm

### 2.5 演示

Qwen3.5模型使用python API进行演示。

```bash
python3 demo.py
```

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

如需指定模型版本，可附加 `-m/--model_size` 参数，具体取值请以脚本帮助信息为准。

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
- `--model_size`: 模型大小，支持 `2b`、`4b`、`9b`、`27b` 或 `35b-a3b`，默认 `9b`
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
- `--model_size`: 模型大小参数，具体取值请以脚本帮助信息为准
- `--model_name`: 输出模型名称（默认：qwen3.5）
- `--batch`: 批次大小（默认：1）
- `--context_length`: 上下文长度（默认：2048）
- `--flash_attention`: Flash Attention优化（默认：2）
- `--device_kernel_split`: 设备内核分割
- `--stage`: 构建阶段，build/test/all（默认：build）

### demo.py
- `--tokenizer_dir`: tokenizer目录（默认：qwen3.5）
- `--embedding_path`: embedding权重路径
- `--prefill_path`: prefill模型路径
- `--decode_path`: decode模型路径
- `--vision_path`: vision模型路径
- `--ndevice`: 设备数量（默认：1）
- `--image_path`: 图片输入路径
- `--it`: 交互模式
- `--history`: 保留对话历史
- `--debug`: 输出调试日志

### test.sh
- `-m, --model_size`: 模型大小，支持 `2b`、`4b`、`9b`、`27b` 或 `35b-a3b`，默认 `9b`
- `-s, --step`: 执行步骤，all/demo/build（默认：all）
- `-t, --model_type`: 模型获取方式，precompiled/compile（默认：precompiled）

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
