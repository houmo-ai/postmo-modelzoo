# resnet50

本 sample 展示如何把resnet50模型量化，编译，最后部署到hm800的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [免责声明](#6免责声明)

## 1.模型说明

本例使用的 resnet50 实现来自 github 开源项目[https://github.com/pytorch/vision](https://github.com/pytorch/vision)的v0.11.1分支。

## 2.前置条件

环境中完成工具链基础包，TCIM包部署以及量化工具包的部署

## 3.快速开始

### 3.1 环境准备

准备imagenet验证集的下载，并放到imodelzoo/data/datasets/imagenet目录下

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd imodelzoo/cv/resnet50/
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd prepare_model
bash run.sh
```

### 3.4 量化并编译模型

```bash
cd compile_model
bash run.sh
```

### 3.5 执行推理

```bash
cd inferece_model
bash run.sh
```

结果：

```bash
Accuracy:  top1: 75.434%, top5: 92.696%
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行cd imodelzoo/cv/resnet50/ && bash run.sh 来实现一键执行

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- reset50 torchvision 模型：[https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/v0.11.1/torchvision/models/resnet.py)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
