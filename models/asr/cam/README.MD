# CAMPPlus (声纹识别)

本示例展示如何把CAMPPlus声纹识别模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: CAMPPlus (Speaker Verification / 声纹识别)
- 来源: [https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
- 许可: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=campplus`
- `model_size=standard`
- `ncore=2`
- `ndevice=1`

### 模型参数说明

| 参数 | 支持范围 | 说明 |
|------|---------|------|
| batch | 1 | 单batch推理 |
| ncore | 2 | 双核编译 |
| ndevice | 1 | 单芯片 |
| 量化类型 | w8a8_sefp | 权重8bit/激活8bit，SEFP量化 |
| 输入帧数 | 529 | 固定帧数，不足自动padding |
| 输入维度 | [batch, 529, 80] | FBank特征 |

### 模型输入输出

| 名称 | 维度 | 说明 |
|------|------|------|
| feature (输入) | [batch, 529, 80] | FBank声学特征 |
| embedding (输出) | [batch, 192] | 说话人嵌入向量 |

## 2 快速开始

目前CAMPPlus模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv：

```bash
# 推理环境
pip3 install -r requirements.txt

# 量化环境（需要GPU）
pip3 install -r requirements_ptq.txt
```

此外需要安装xhquant、tcim/tcim_lite、hmatc等后摩工具链。

### 2.2 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm --model_name campplus --model_size standard
```

如果需要自行编译模型，可以下载原始模型：

```bash
python3 get_model.py --type raw --model_name campplus --model_size standard
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA兼容的驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/asr/cam`目录，执行ptq.py脚本。默认的量化方式为 `w8a8_sefp`。

```bash
source env.sh
cd models/asr/cam
python3 ptq.py --model-name campplus --model-size standard
```

**量化资源需求：**
- Host内存: ~20GB
- GPU内存: ~8GB

产物：
- `work_dirs/cam/export_fp32/onnx/model.onnx` — 原始导出ONNX
- `work_dirs/cam/export_fp32/onnx/model_simplified.onnx` — onnxsim简化后的ONNX
- `output/xh2/hmquant/prefill/model_XH2a_w8a8_sefp.onnx` — 量化后的HMONNX模型

量化支持以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | campplus | 模型名称 |
| `--model-size` | standard | 模型大小 |
| `--quant-type` | w8a8_sefp | 量化类型 |
| `--device` | cuda:0 | 量化设备 |
| `--fixed-frames` | 529 | 固定帧数 |
| `--save-golden` | False | 保存golden数据用于编译测试 |
| `--no-eval` | False | 跳过精度评估 |

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name campplus --model_size standard
```

**编译资源需求：**
- Host内存: ~16GB

默认输出名称分别为：

- `output/xh2/cam_embedding.hmm`

编译支持以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | campplus | 模型名称 |
| `--model_size` | standard | 模型大小 |
| `--ncore` | 2 | 核心数 |
| `--ndevice` | 1 | 芯片数 |
| `--stage` | build | 阶段: build/test/all |
| `--j` | CPU核数 | 并行编译任务数 |

### 2.5 演示

CAMPPlus模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 读取 `model_name`、`model_size`，并自动推导模型路径。

```bash
python3 demo.py --model_name campplus --model_size standard
```

演示结果示例：
- 相同说话人相似度: ~0.85
- 不同说话人相似度: ~0.12
- 阈值: 0.31（相似度 > 阈值判定为同一人）

### 2.6 一键测试

使用 `test.sh` 一键执行完整流程：

```bash
# 仅演示（默认，下载预编译模型）
bash test.sh

# 仅演示（跳过下载，使用已有模型）
bash test.sh --skip_download

# 仅量化
bash test.sh -s quant

# 仅编译
bash test.sh -s build

# 量化+编译+演示
bash test.sh -s all

# 指定量化类型
bash test.sh -s all --quant_type w8a8_sefp
```

### 2.7 性能结果

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 特征提取 | ~30ms | 双声道各提取一次 |
| HMM推理(单次) | ~2ms | embedding模型推理 |
| 总处理 | ~65ms | 双声道特征提取+推理+比对 |

## 3 不支持的功能

- 多batch推理（仅支持batch=1）
- 多芯片（仅支持单芯片ndevice=1）
- MTP（本模型不适用）
- prefix_caching（本模型不适用）

编译好的模型存放在`output/xh2/`目录：
- `cam_embedding.hmm`

参数说明：
- `--model_dir`: HMONNX模型目录（默认 `work_dirs/cam/hmonnx`）
- `--model_name`: 输出模型名（默认 `cam_embedding`）
- `--ncore`: 核心数（默认 `$HOUMO_CORE_NUM` 或 2）
- `--stage build`: 仅构建
- `--stage test`: 仅测试（与golden比对余弦相似度）
- `--stage all`: 构建 + 测试
- `--j`: 并行编译任务数
- `--output_dir`: 输出目录（默认 `output/xh2`）

### 2.6 演示

CAMPPlus模型使用python API进行演示。

```bash
python3 demo.py
```

推理完成将输出性能结果。

参数说明：
- 默认加载 `output/xh2/cam_embedding.hmm`（可在代码中修改 `DEFAULT_MODEL_PATH`）

### 2.7 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

test.sh支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step         execution step, default is all, support: all, demo, build.
  -t, --model_type   model type, default is precompiled, support: precompiled, compile.
  -q, --quant_type   quantization type, default is w8a8_sefp.
  -d, --device       PTQ device, default is cuda:0.
  -h, --help         help information
```

示例：

```bash
# 仅运行demo（使用预编译模型）
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -t compile
```

## 3 I/O约定

| 角色 | 名称 | 形状 | 类型 |
|------|------|------|------|
| Input | `feature` | `[1, 529, 80]` | `float32` |
| Output | `embedding` | `[1, 192]` | `float32` |

其中 `529` 为固定帧数（不足则尾部补零，超出则截断），`80` 为 fbank 特征维度，`192` 为说话人嵌入向量维度。

## 4 目录结构

```
cam/
├── CAM_plus/                         # 模型目录（含config、权重）
│   ├── configuration.json
│   ├── campplus_cn_common.bin         # 模型权重
│   └── examples/
│       ├── speaker1_a_cn_16k.wav      # 示例音频（说话人1-A）
│       ├── speaker1_b_cn_16k.wav      # 示例音频（说话人1-B）
│       └── speaker2_a_cn_16k.wav      # 示例音频（说话人2-A）
├── requirements.txt                  # Python 依赖
├── get_model.py                      # 获取模型（原始/预编译芯片模型）
├── export_onnx.py                    # 导出FP32 ONNX
├── ptq.py                            # xhquant PTQ量化 + 精度对比评估
├── build.py                          # TCIM构建HMM模型
├── demo.py                           # HMM推理Demo
├── test.sh                           # 一键评估脚本
├── output/xh2/                       # 产物目录
│   └── cam_embedding.hmm             # HMM构建产物
└── work_dirs/cam/                    # 工作目录
    ├── export_fp32/onnx/             # ONNX导出产物
    │   ├── model.onnx
    │   └── model_simplified.onnx
    ├── hmonnx/                       # 量化产物
    │   └── model_XH2a_w8a8_sefp.onnx
    └── tcim/cam_embedding/           # TCIM编译中间文件
```

## 5 注意事项

1. 确保 `CAM_plus/` 目录存在且包含 `configuration.json` 和模型权重
2. PTQ量化步骤可选，如需直接从ONNX构建HMM可跳过步骤2.4
3. TCIM构建仅支持x86_64平台
4. `demo.py` 中的特征提取依赖 `modelscope` pipeline 加载 CAMPPlus 模型
5. 所有脚本均需在 `cam/` 目录下执行
6. `HOUMO_TARGET` 环境变量默认为 `xh2`，可通过 `export HOUMO_TARGET=xh2` 设置

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
