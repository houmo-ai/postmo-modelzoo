# Resnet50

本示例展示如何把 resnet50 模型量化，编译成可以运行在后摩芯片的 XH2 设备上的模型。

[TOC]

## 1.模型说明

本例使用的 resnet50 实现来自 github 开源项目[https://github.com/pytorch/vision](https://github.com/pytorch/vision)的 v0.11.1 分支。

可以通过以下代码从原始途径获取模型：

```python
import torch
import torchvision
model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand((1, 3, 224, 224))
torch.onnx.export(model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13)
```

## 2.环境依赖

主机 Python 环境中需安装后摩量化及编译工具：hmquant & houmo-tcim。

## 3.快速开始

通过 test 脚本一键完成，执行：

```bash
./test.sh
```

或者可以通过参数分阶段执行，

1. 仅运行量化阶段，执行：

```bash
./test.sh -s quant
```

2. (需先执行量化) 仅运行编译阶段，执行：

```bash
./test.sh -s build
```

其他情况可参考脚本中的命令单独执行。

## 4.运行结果

基于 XH2 平台的后摩设备，展示量化&编译结果示例。

量化成功后打印：

```bash
start save model and generate golden...
xhquant/xhonnxruntime/hmonnx_inference.py:458 => Loading ONNX model from output/xh2/hmquant/resnet50.onnx
xhquant/xhonnxruntime/hmonnx_inference.py:461 => resnet50.onnx loading completed
xhquant/xhonnxruntime/hmonnx_inference.py:467 => parsing ONNX model from output/xh2/hmquant/resnet50.onnx
xhquant/xhonnxruntime/hmonnx_inference.py:554 => resnet50.onnx parsing completed
save model and generate golden completed.
```

编译成功后打印：

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

编译完成后会使用 golden 数据执行一次推理测试，打印：

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

- reset50 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
