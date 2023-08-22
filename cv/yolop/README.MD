# yolop

本 sample 展示如何把yolop模型量化，编译，最后部署到hm800的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [一键运行](#33-性能评测)
- [免责声明](#6免责声明)

## 1.模型说明

本例使用的 yolop 实现来源 github 开源项目[https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)的master分支。模型代码做一定调整后导出，细节见量化工具文档。

## 2.前置条件

环境中完成工具链基础包，TCIM包部署以及量化工具包的部署

## 3.快速开始

### 3.1 环境准备

无

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd imodelzoo
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

设置完成后进入模型目录：

```bash
cd cv/yolop/
```

### 3.3 性能评测

```bash
bash perf.sh
```

结果：

```bash
Input images: (1, 192, 320, 3), uint8
Inference time cost total = 115161us
Inference time cost per frame = 115161us
Average Throughput(QPS): 8.6835fps
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- yolop 模型：[https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx](https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
