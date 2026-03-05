# Qwen3 Pipeline 模型编译指南

本指南介绍如何使用构建脚本将 Qwen3 模型编译为适用于后摩鸿途芯片设备的多设备流水线格式。

---

## 1. 功能概述

该构建脚本 (`build_multidevices_pipeline.py`) 实现以下功能：
- 将 Qwen3 模型拆分为多个部分，支持多设备流水线部署
- 提供模型构建和测试流程
- 支持预填充 (prefill) 和解码 (decoder) 阶段的模型处理
- 生成量化后的 ONNX 模型用于推理优化

---

## 2. 环境准备

### 2.1 系统要求

- **操作系统**: Linux
- **GPU 要求**: 支持 CUDA 12.2，显存 >= 40GB
- **内存要求**: 空闲内存 >= 100GB
- **Docker**: 推荐使用后摩鸿途工具链 Docker 镜像

启动 Docker 时映射 GPU 并指定可用的 GPU ID：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86_64 /bin/bash
```

### 2.2 依赖库安装

参考$HOUMO_EXAMPLE_PATH/models/llm/qwen3目录的README.MD安装相关量化依赖。

### 2.3 环境变量设置

参考$HOUMO_EXAMPLE_PATH/models/llm/qwen3目录的README.MD设置环境变量。

### 2.4 预训练模型

本示例使用从 ModelScope 下载的预训练模型：

- **模型名称**: Qwen3-8B
- **来源**: https://modelscope.cn/models/Qwen/Qwen3-8B
- **许可**: Apache License 2.0

预训练模型在运行时下载，工程发布不包含该模型。

---

## 3. 输入模型准备

本 Pipeline 需要使用经过量化的 Qwen3 模型作为输入。量化脚本位于 `$HOUMO_EXAMPLE_PATH/models/llm/qwen3` 目录。

### 3.1 获取预训练模型

通过 `get_model.py` 脚本下载模型，可使用 `--type raw` 参数选择下载原始模型：

```bash
cd $HOUMO_EXAMPLE_PATH/models/llm/qwen3
python3 get_model.py --type raw
```

### 3.2 执行量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.2 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID，显存需要达到 40GB 以上，空闲内存需要达到 100GB 以上。

进入 docker 后 source 环境变量，并进入 `models/llm/qwen3` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd $HOUMO_EXAMPLE_PATH/models/llm/qwen3
python3 ptq.py
```

如需使用自定义校准数据集，请按以下步骤操作：

**步骤 1：准备校准数据**

准备一个 JSON Lines 格式的文件，其中包含您目标应用场景的文本数据。每行格式如下：

```json
{"text": "您的领域相关文本内容..."}
```

**步骤 2：运行量化命令**

```bash
# 使用预设的校准数据集或者换成自己的 jsonl 文件
python3 ptq.py --calib_data /path/to/your/calibration_data.jsonl
```

### 3.3 量化输出

量化完成的模型存放在 `output/$HOUMO_TARGET/hmquant` 目录：

```bash
output/xh2/hmquant/
├── decoder/
│   ├── hmquant_qwen3_with_act.onnx
│   └── hmquant_qwen3_with_act_external_data
└── prefill/
    ├── hmquant_qwen3_with_act.onnx
    └── hmquant_qwen3_with_act_external_data
```

---

## 4. 使用方法

### 4.1 快速开始

**模型分割与构建模型**
将量化完成的模型存放在当前文件夹的 `output/$HOUMO_TARGET/hmquant` 目录：
```bash
python build_multidevices_pipeline.py
```

### 4.2 命令行参数说明

| 参数                    | 描述                         |默认值                            |
| ----------------------- | ---------------------------- | -------------------------------- |
| `--model_dir`           | 模型目录路径                 | `output/[HOUMO_TARGET]/hmquant`  |
| `--model_name`          | 输出模型名称                 | `qwen3`                          |
| `--batch`               | 批次大小                     | `1`                              |
| `--ncore`               | 核心数量                     | `$HOUMO_CORE_NUM` (默认 2)       |
| `--ndevice`             | 设备数量                     | `4`                              |
| `--output_dir`          | 构建输出目录                 | `output/[HOUMO_TARGET]`          |
| `--j`                   | 构建并行任务数               | CPU 核心数                       |
| `--context_length`      | 上下文长度                   | `32768`                          |
| `--prefill_length`      | 预填充长度                   | `256`                            |
| `--flash_attention`     | Flash Attention 优化 (0/1/2) | `2`                              |

---

## 5. 输出结果

构建完成后，输出目录结构如下：

```bash
output/xh2/
├── qwen3_prefill_part0.hmm     # 预填充部分 0 的编译模型
├── qwen3_prefill_part1.hmm     # 预填充部分 1 的编译模型
├── qwen3_prefill_part2.hmm     # 预填充部分 2 的编译模型
├── qwen3_prefill_part3.hmm     # 预填充部分 3 的编译模型
├── qwen3_decode_part0.hmm      # 解码部分 0 的编译模型
├── qwen3_decode_part1.hmm      # 解码部分 1 的编译模型
├── qwen3_decode_part2.hmm      # 解码部分 2 的编译模型
├── qwen3_decode_part3.hmm      # 解码部分 3 的编译模型
└── tcim/                       # 编译中间文件
    ├── qwen3_prefill_part0/
    ├── qwen3_prefill_part1/
    ├── qwen3_prefill_part2/
    ├── qwen3_prefill_part3/
    ├── qwen3_decode_part0/
    ├── qwen3_decode_part1/
    ├── qwen3_decode_part2/
    └── qwen3_decode_part3/
```

---

## 6. 注意事项

### 6.1 硬件限制

- **HOUMO_TARGET**：当前脚本仅支持 `HOUMO_TARGET=xh2`

### 6.2 模型要求

- **块数量**：模型的 Transformer 层数必须能被 `ndevice` 整除，否则无法正确分割
- **量化模型**：必须使用经过量化的 ONNX 模型作为输入

---

## 7. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
