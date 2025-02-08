# MobileNetv2

本示例展示如何把mobilenet_v2模型量化，编译，最后部署到后摩后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 mobilenet v2 实现来自 github 开源项目[https://github.com/pytorch/vision](https://github.com/pytorch/vision)的v0.11.1分支。

可以通过以下代码从原始途径获取模型：

```python
import torch
import torchvision
model = torchvision.models.mobilenet_v2(pretrained=True)
dummy_input = torch.rand(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, 'mobilenetv2.onnx', export_params=True, opset_version=13,
)
```

## 2.快速开始

请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- mobilenet v2 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/mobilenetv2.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/mobilenetv2.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
