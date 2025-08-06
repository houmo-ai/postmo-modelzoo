# Efficientnet

本示例展示如何把efficientnet模型量化，编译，最后部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 efficientnet 实现来自 github 开源项目[https://github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)。

可以通过以下代码从原始途径获取模型：

```python
import torch
efficientnet = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_efficientnet_b0', pretrained=True)
input = torch.randn([1, 3, 224, 224])
torch.onnx.export(efficientnet, input, 'efficientnet_b0_224x224.onnx', opset_version=13)
```

本模型ptq量化效果欠佳，建议使用qat量化。

## 2.快速开始

可以通过hmatc工具一键执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh1          # 芯片
hmatc eval -c config.yml -t xh1  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 8
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
hmatc demo -c config.yml -t xh1
```

执行完成后会打印模型的分类结果信息。

## 3.参考结果

### 3.1 精度结果
```bash
# xh1
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 20, 'top1_acc': '0.500000', 'latency_ms': '6.341147'
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 20, 'top1_acc': '0.800000', 'latency_ms': '7.828081'
```

### 3.2 性能结果

```bash
[latency] Inference     avg:   8.106 ms,        max:   9.581 ms
[latency] Input         avg:   1.617 ms,        max:   3.241 ms
[latency] Output        avg:   0.000 ms,        max:   0.000 ms
[latency] End2End       avg:   9.724 ms,        max:  12.807 ms
[Throughput] total: 1222.208 ms, avg: 1.222 ms
[Throughput] qps: 818.191
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- efficientnet torchvision 模型：[https://github.com/NVIDIA/DeepLearningExamples/blob/96bdb5b35ff2ebaa53e3c9b680e04515435c6296/PyTorch/Classification/ConvNets/image_classification/models/efficientnet.py](https://github.com/NVIDIA/DeepLearningExamples/blob/96bdb5b35ff2ebaa53e3c9b680e04515435c6296/PyTorch/Classification/ConvNets/image_classification/models/efficientnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
