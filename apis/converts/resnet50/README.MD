# Resnet50

本示例展示 `ResNet50` 模型在后摩工具链中的两套转换流程：

- `hmatc/`：基于 `hmatc` 命令行工具完成量化与编译。
- `xh2/`：基于 `xhquant` + `tcim` Python API 完成 PTQ、编译和 golden 对比，面向 `XH2` 平台。

[TOC]

## 1.模型说明

本例使用的 `ResNet50` 实现来自 `torchvision` 项目 `v0.11.1`：
[https://github.com/pytorch/vision](https://github.com/pytorch/vision)

可通过以下方式从原始实现导出 `ONNX` 模型：

```python
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand((1, 3, 224, 224))
torch.onnx.export(model, dummy_input, "resnet50.onnx", export_params=True, opset_version=13)
```

仓库中的下载脚本 `get_model.py` 会将模型下载到 `apis/models/resnet50.onnx`。

## 2.目录与脚本说明

### 2.1 顶层文件

- `get_model.py`：从制品库拉取 `resnet50.onnx`，保存到 `apis/models/`。
- `README.MD`：当前说明文档。

### 2.2 `xh2/` 目录

- `ptq.py`：执行 PTQ，将 `ONNX` 转为量化后的中间模型，并生成 golden 输入输出数据。
- `build.py`：将量化后的模型编译为 `.hmm`，并加载运行时做一次推理和 golden 对比。
- `test.sh`：按阶段串联 `get_model.py`、`ptq.py` 和 `build.py`，支持 `quant`、`build`、`all` 三种模式。

`xh2/test.sh` 的阶段含义：

- `-s quant`：下载模型并执行 `ptq.py`。
- `-s build`：执行 `build.py`，要求量化产物已存在。
- `-s all`：先量化，再编译并测试。

### 2.3 `hmatc/` 目录

- `config.yml`：`hmatc` 量化/编译配置，定义模型路径、输入预处理、校准集路径、量化参数和编译选项。
- `test.sh`：依次执行模型下载、`hmatc quant`、`hmatc build`，用于一键跑通 `hmatc` 流程。

### 2.4 环境依赖

请确保主机环境中已经安装后摩相关工具链: `xhquant`, `tcim`, `tcim_lite`。

## 3.快速开始

### 3.1 `xh2` 一键流程

在 `xh2/` 目录下执行：

```bash
cd apis/converts/resnet50/xh2
export HOUMO_TARGET=xh2
./test.sh
```

分阶段执行时可使用：

```bash
cd apis/converts/resnet50/xh2
export HOUMO_TARGET=xh2
./test.sh -s quant
./test.sh -s build
```

### 3.2 手动执行 `xh2` 流程

如果需要逐步调试，可以分别执行：

```bash
cd apis/converts/resnet50/xh2
export HOUMO_TARGET=xh2
python3 ../get_model.py
python3 ptq.py
python3 build.py
```

### 3.3 `hmatc` 一键流程

在 `hmatc/` 目录下执行：

```bash
cd apis/converts/resnet50/hmatc
./test.sh
```

脚本实际执行的步骤为：

```bash
python3 ../get_model.py
hmatc quant -c config.yml
hmatc build -c config.yml
```

## 4.运行结果示例

以下日志来自 `xh2` 流程。

量化并生成 golden 成功后会输出：

```bash
start save model and generate golden...
xhquant/xhonnxruntime/hmonnx_inference.py:458 => Loading ONNX model from output/xh2/hmquant/resnet50.onnx
xhquant/xhonnxruntime/hmonnx_inference.py:461 => resnet50.onnx loading completed
xhquant/xhonnxruntime/hmonnx_inference.py:467 => parsing ONNX model from output/xh2/hmquant/resnet50.onnx
xhquant/xhonnxruntime/hmonnx_inference.py:554 => resnet50.onnx parsing completed
save model and generate golden completed.
```

编译成功后会输出：

```bash
HMCC-LINK: raw data size set to 30653056
HMCC-LINK: header size: 40, kernel size: 874945
HMCC-LINK: model config:
 - args_num: 4
 - core_num: 2
 - tile_num: 16
 - core_mask: 0
 - is_pic: 1
 - for_bench: 1
 - spec_batch: 1
 - is_pack_inout: 0
HMCC-LINK: dumping model into /houmo-example/apis/converts/resnet50/xh2/output/xh2/resnet50.hmm
HMCC-LINK: dump model complete, size is 31594776
INFO:root:<=== hdpl compile completed.
resnet50 build completed in 19.486 s.
```

编译完成后，`build.py` 会自动加载 `.hmm`、执行一次推理，并和 golden 输出做余弦相似度比对：

```bash
===> resnet50 test start...
resnet50 load completed in 426.869 ms.
input_num: 1
input_info[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
input data path: output/xh2/hmquant/hmquant_resnet50_input.1_input.npy
golden input[input.1] shape = (1, 3, 224, 224), dtype = float16
resnet50 set 1 inputs completed in 0.302 ms.
resnet50 infer completed in 1.860 ms.
output_num: 1
output_info[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = (1, 1000), dtype = float16
[compare] golden output [495] match=True, similarity=0.999887
resnet50 get 1 ouputs completed in 0.021 ms.
<=== resnet50 test success.
```

## 5.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- `ResNet50` `torchvision` 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- `ILSVRC_2012` 验证集链接：[https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
