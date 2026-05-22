# Resnet50

本示例展示如何把resnet50模型量化，编译，最后部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 resnet50 实现来自 github 开源项目[https://github.com/pytorch/vision](https://github.com/pytorch/vision)的v0.11.1分支。

可以通过以下代码从原始途径获取模型：

```python
import torch
import torchvision
model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand((1, 3, 224, 224))
torch.onnx.export(model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13)
```

本模型int8精度稍差，需要混合量化。本示例已使用混合量化配置，所以性能方面有所损失。


## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-examples根目录下README.md文件的快速上手章节。

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh2       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh2       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh2          # 芯片
hmatc eval -c config.yml -t xh2  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh2 -wn 10 -sn 1000 -tn 4
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
hmatc demo -c config.yml -t xh2
```

执行完成后会打印模型的分类结果信息。

## 3.参考结果

### 3.1 精度结果
```bash
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.725800'
# xh2
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.724500'
```

### 3.2 性能结果

```bash
[Latency] Inference  avg:   3.487 ms, max:   3.605 ms, min:   2.028 ms, tp99:   3.551 ms, tp999:   3.605 ms
[Latency] Input      avg:   0.238 ms, max:   0.351 ms, min:   0.212 ms, tp99:   0.328 ms, tp999:   0.351 ms
[Latency] Output     avg:   0.037 ms, max:   0.080 ms, min:   0.032 ms, tp99:   0.076 ms, tp999:   0.080 ms
[Latency] End2end    avg:   3.762 ms, max:   3.902 ms, min:   2.289 ms, tp99:   3.822 ms, tp999:   3.902 ms
[Throughput] total: 949.193 ms, avg: 0.949 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1053.526
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- reset50 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
