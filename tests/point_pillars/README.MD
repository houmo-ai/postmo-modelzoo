# PointPillar

本示例展示如何把PointPillar量化模型编译，部署到鸿途H30芯片的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [准备模型](#32-准备模型)
  - [编译模型](#33-编译模型)
  - [执行推理](#34-执行推理)
  - [一键运行](#35-一键运行)
  - [性能评测](#36-性能评测)
- [免责声明](#4免责声明)

## 1.模型说明

本例使用的 pointpillar 实现基于 github 开源项目[https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/pointpillar.yaml](https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/pointpillar.yaml)的master分支, 255db8f02a8bd07211d2c91f54602d63c4c93356 commit。 把原始模型拆分成多段进行部署

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
cd cv/point_pillars/
```

### 3.2 准备模型

下载量化模型，如已有模型可跳过此步

```bash
cd prepare_model
bash run.sh
```

### 3.2 编译模型

```bash
cd compile_model
bash run.sh
```

### 3.4 执行推理

```bash
cd inference_model
bash run.sh
```

结果：

```bash
...
Cyclist 1.64279 0.84 1.33764 -11.7036 8.20979 -1.90914 1.6888 0.342373
Car 4.21616 1.9293 1.59828 23.3306 -28.6628 -1.20162 -0.751731 0.328884
Cyclist 1.81 0.892421 1.44216 -15.9207 18.04 -1.99377 -0.322036 0.306472
============point_pillars result ok===========
```

### 3.5 一键运行

以上 3.2~3.4 的步骤也可以通过运行cd cv/point_pillars/ && bash run_perf.sh 来实现一键执行

### 3.6 性能评测

```bash
bash run_perf.sh
```

结果：

```bash
point_pillars test count 222 cost 38.085306ms each time stream_num 4 fps 104
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- reset50 torchvision 模型：[https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
