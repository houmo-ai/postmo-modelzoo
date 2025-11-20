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

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

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
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.725840'
# xh1
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.715960'
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:   6.252 ms, max:   7.116 ms, min:   4.969 ms, tp99:   6.959 ms, tp999:   7.116 ms
[Latency] Input      avg:   0.063 ms, max:   0.156 ms, min:   0.041 ms, tp99:   0.104 ms, tp999:   0.156 ms
[Latency] Output     avg:   0.049 ms, max:   0.166 ms, min:   0.031 ms, tp99:   0.087 ms, tp999:   0.166 ms
[Latency] End2end    avg:   6.364 ms, max:   7.227 ms, min:   5.087 ms, tp99:   7.072 ms, tp999:   7.227 ms
[Throughput] total: 803.029 ms, avg: 0.803 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1245.285
```

>**注意：**
>
> 结果均是以芯片核数两倍的线程数来测试，这里分别是8和4线程，仅供参考，具体结果以实际测试为准。
> 

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- reset50 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
