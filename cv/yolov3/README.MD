# YOLOv3

本 sample 展示如何把量化后的YOLOv3模型编译，部署到hm800的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [裁剪并编译模型](#34-裁剪并编译模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
  - [性能评测](#37-性能评测)
- [免责声明](#6免责声明)

## 1.模型说明

本例使用的 YOLOv3 实现来源 github 开源项目[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)的v9.0 tag。模型代码导出后，再去除掉yolo相关的节点，细节见量化工具文档。

## 2.前置条件

环境中完成工具链基础包，TCIM包部署以及量化工具包的部署

## 3.快速开始

#### 3.1 环境准备

准备COCO验证集的下载，并放到imodelzoo/data/datasets/COCO目录下

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd imodelzoo/cv/yolov3/
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

### 3.4 量化，裁剪并编译模型

```bash
cd compile_model
bash run.sh --batch 1
```

### 3.5 执行推理

```bash
cd inference_model
bash run.sh
```

结果：

```bash
Accuracy: mAP: 0.4959523458638628
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行cd imodelzoo/cv/yolov3/ && bash eval.sh 来实现一键执行

### 3.7 性能评测

```bash
bash perf.sh
```

结果：

```bash
Input data: (4, 416, 416, 3), int8
========= RUN 1000 ROUNDS =======

Inference time cost total = 4816406us
Inference time cost per frame = 4816.4us
Average Throughput(QPS): 830.49fps
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv3 模型：[https://github.com/ultralytics/yolov3/tree/v9.6.0](https://github.com/ultralytics/yolov3/tree/v9.0)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
