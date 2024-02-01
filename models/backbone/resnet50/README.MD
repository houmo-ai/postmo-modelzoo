# Resnet50

本示例展示如何把resnet50模型量化，编译，最后部署到鸿途H30芯片的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [准备数据集和模型](#32-准备数据集和模型)
  - [量化并编译模型](#33-量化并编译模型)
  - [执行推理](#34-执行推理)
  - [一键运行](#35-一键运行)
  - [性能评测](#36-性能评测)
- [精度调优](#4精度调优)
  - [QAT训练](#41-QAT训练)
- [免责声明](#5免责声明)

## 1.模型说明

本例使用的 resnet50 实现来自 github 开源项目[https://github.com/pytorch/vision](https://github.com/pytorch/vision)的v0.11.1分支。

## 2.前置条件

环境中完成工具链基础包，TCIM包部署以及量化工具包的部署

## 3.快速开始

### 3.1 环境准备

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

设置完成后进入模型目录：

```bash
cd models/backbone/resnet50
```

### 3.2 准备数据集和模型

1. 准备imagenet验证集放到data/datasets/imagenet目录下，仓库中已有少量数据供简单验证，如果需要测试真实精度需要自行下载完整数据集
2. 下载原始模型，如已有模型可跳过此步

```bash
python3 get_model.py
```

或者可以通过以下代码从原始途径获取模型：

```python
import torch
import torchvision
model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand((1, 3, 224, 224))
torch.onnx.export(model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13)
```

### 3.3 量化并编译模型

- 通过hmassist工具量化编译模型

```bash
hmquant.sh
hmbuild.sh
```

- 通过脚本量化编译模型

```bash
python3 ptq.py
python3 build.py
```

### 3.4 执行推理

```bash
hmaccuracy.sh
```

结果：

```bash
Accuracy:  top1: 75.68%, top5: 92.696%
```

### 3.5 一键运行

以上 3.2~3.4 的步骤也可以通过shell脚本来实现一键执行

```bash
bash eval.sh
```

### 3.6 性能评测

```bash
bash perf.sh
```

结果：

```bash
Input input.1: (24, 224, 224, 3), uint8
========= RUN 1000 ROUNDS =======
Inference time cost total = 4720125us
Inference time cost per frame = 4720us
Average Throughput(QPS): 5932.2fps
```

## 4.精度调优

### 4.1 QAT训练

#### 4.1.1 环境准备

QAT训练需要使用带Nvidia GPU的机器，安装好cuda驱动，nvidia-smi确认安装正常。启动Docker时增加 --gpus all选项将GPU映射进docker，同时增加--shm-size 10g选项增大shared memory，实际大小取决于模型大小、训练图像的数量、batch数等，需要自行调整。检查torch和torchvision版本是否与cuda版本匹配。

将imagenet的训练集和验证集数据放到$DATASETS_PATH/imagenet目录下train和val目录中，注意数据是以分类名为文件夹存放的。

#### 4.1.2 开始运行

可通过qat.sh脚本直接运行，可通过脚本中参数修改数据集路径和训练batch数等，目前默认batch数为8。

```bash
bash qat.sh
```

参考结果：

```bash
=> using pre-trained model 'resnet50'
calibrating
0it [00:20, ?it/s]
Epoch: [0][ 0/13]        Time  1.153 ( 1.153)        Data  0.461 ( 0.461)        Loss 3.6372e-01 (3.6372e-01)        Acc@1  75.00 ( 75.00)        Acc@5 100.00 (100.00)
Epoch: [0][ 1/13]        Time  0.130 ( 0.641)        Data  0.001 ( 0.231)        Loss 1.0830e-01 (2.3601e-01)        Acc@1 100.00 ( 87.50)        Acc@5 100.00 (100.00)
Epoch: [0][ 2/13]        Time  0.124 ( 0.469)        Data  0.001 ( 0.155)        Loss 9.9412e-03 (1.6065e-01)        Acc@1 100.00 ( 91.67)        Acc@5 100.00 (100.00)
Epoch: [0][ 3/13]        Time  0.123 ( 0.383)        Data  0.001 ( 0.116)        Loss 1.4792e+00 (4.9029e-01)        Acc@1  87.50 ( 90.62)        Acc@5  87.50 ( 96.88)
...
```

## 5.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- reset50 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
