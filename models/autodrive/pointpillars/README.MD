# Pointpillars

本示例展示如何把pointpillars模型量化，编译，最后部署到鸿途H30芯片的设备上。

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
- [免责声明](#4免责声明)

## 1.模型说明

本例使用的 efficientnet 实现来自 github 开源项目[https://github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)。

## 2.前置条件

环境中完成工具链基础包，TCIM包部署以及量化工具包的部署。

## 3.快速开始

### 3.1 环境准备

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

设置完成后进入模型目录：

```bash
cd models/backbone/efficientnet
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
efficientnet = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_efficientnet_b0', pretrained=True)
input = torch.randn([1, 3, 224, 224])
torch.onnx.export(efficientnet, input, 'efficientnet_b0_224x224.onnx', opset_version=13)
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
Accuracy:  top1: 66.6%, top5: 90.4%
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
Input input.1: (1, 224, 224, 3), uint8
========= RUN 200 ROUNDS =======
Inference time cost total = ......us
Inference time cost per frame = ....us
Average Throughput(QPS): ....fps
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- efficientnet torchvision 模型：[https://github.com/NVIDIA/DeepLearningExamples/blob/96bdb5b35ff2ebaa53e3c9b680e04515435c6296/PyTorch/Classification/ConvNets/image_classification/models/efficientnet.py](https://github.com/NVIDIA/DeepLearningExamples/blob/96bdb5b35ff2ebaa53e3c9b680e04515435c6296/PyTorch/Classification/ConvNets/image_classification/models/efficientnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
