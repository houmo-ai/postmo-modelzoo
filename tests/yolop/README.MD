# YoloP

本示例展示如何把yolop模型量化，编译，最后部署到鸿途H30芯片的设备上。

## 目录

- [模型说明](#1模型说明)
- [前置条件](#2前置条件)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [准备数据集和模型](#32-准备数据集和模型)
  - [一键运行](#33-性能评测)
- [精度调优](#4精度调优)
  - [QAT训练](#41-QAT训练)
- [免责声明](#5免责声明)

## 1.模型说明

本例使用的 yolop 实现来源 github 开源项目[https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)的master分支。模型代码做一定调整后导出，细节见量化工具文档。

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
cd cv/yolop/
```

### 3.2 准备数据集和模型

1. 准备bdd100k数据集放到data/datasets/bdd100k目录下，仓库中未提供数据，采用随机数进行PTQ量化，如需测试精度请下面的参考QAT训练

2. 下载原始模型，如已有模型可跳过此步

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

## 4.精度调优

### 4.1 QAT训练

#### 4.1.1 环境准备

QAT训练需要使用带Nvidia GPU的机器，安装好cuda驱动，nvidia-smi确认安装正常。启动Docker时增加 --gpus all选项将GPU映射进docker，同时增加--shm-size 10g选项增大shared memory，实际大小取决于模型大小、训练图像的数量、batch数等，需要自行调整。检查torch和torchvision版本是否与cuda版本匹配。

将bdd100k的数据集数据放到$DATASETS_PATH/bdd100k目录下，包含bdd100k/images/100k，det_annotations，da_seg_annotations，ll_seg_annotations这几个子目录。

#### 4.1.2 开始运行

可通过qat.sh脚本直接运行，可通过修改train_yolop.py脚本中的cfg.TRAIN.BATCH_SIZE_PER_GPU参数修改batch等参数，目前默认值BATCH_SIZE_PER_GPU=8。

```bash
cd cv/yolop/compile_model
./qat.sh
```

注：如果遇到报错RuntimeError: result type Float can't be cast to the desired output type long int
修改/hmodel/thirdparty/YOLOP/lib/core/postprocess.py第30行加上.long()，修改后为：

```bash
gain = torch.ones(7, device=targets.device).long()
```

参考结果：

```bash
...
=> loaded checkpoint '/root/.cache/torch/hub/checkpoints/End-to-end.pth' 
building database...
...
...
Results saved to runs/BddDataset/_2023-09-06-20-33/visualization
Epoch: [10]    Loss(0.497)
Driving area Segment: Acc(0.974)    IOU (0.859)    mIOU(0.914)
Lane line Segment: Acc(0.643)    IOU (0.281)  mIOU(0.633)
Detect: P(0.084)  R(0.893)  mAP@0.5(0.767)  mAP@0.5:0.95(0.438)
```

## 5.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- yolop 模型：[https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx](https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx)
- BDD100K 验证集链接: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
