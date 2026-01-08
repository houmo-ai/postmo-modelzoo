# Resnet50 Pipeline Example

## 目录

[TOC]

## 概述

本示例以 resnet50 模型为例展示流水线（pipeline）推理的使用方式。流水线方式可以并行处理数据的拷入、拷出和推理过程，提升单线程下的推理吞吐。

此种方式只支持 c++。

resnet50.cc 展示 c++下 pipeline 推理过程，包括：

-   h2d 线程将 host 侧输入数据拷贝到 device 侧
-   infer 线程设置 device 侧输入和输出，然后执行推理
-   d2h 线程将 device 侧输出数据拷贝到 host 侧

## 软件依赖

除 TCIM Examples 公用依赖外，本示例没有其他依赖。

## 快速上手

### 获取芯片模型

可以通过以下两种方式获取芯片模型：

1. 自行编译：通过原始模型或量化模型，经过 tcim 工具链编译生成，具体可参考 houmo-modelzoo
2. 直接下载：通过 get_model.py 脚本下载预编译的模型

```bash
python3 get_model.py
```

### 模型 c++推理

使用 runtime c++ API 进行模型推理，并输出结果。执行时可通过参数修改配置：

-   -s 10：测试样本数，默认为 10

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50_pipeline
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> resnet50_pipeline c++ example start...
model: ./resnet50_xh2_b1_1core.hmm, samples: 10
model ./resnet50_xh2_b1_1core.hmm loaded.
Count of Input: 1
Input[input.1] info: TensorInfo: shape: [1,3,224,224], stride: [301056,100352,448,2], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Count of Output: 1
Output[495] info: TensorInfo: shape: [1,1000], stride: [2048,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2048
sample queue size is 10
thread H2D start...
thread infer start...
thread D2H start...
run sample 0 end.
run sample 1 end.
run sample 2 end.
run sample 3 end.
run sample 4 end.
run sample 5 end.
run sample 6 end.
run sample 7 end.
run sample 8 end.
run sample 9 end.
infer thread completed. 10 sampels tested.
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
top1: Index=65 Conf=16, Label=[sea snake]
<=== resnet50_pipeline c++ example completed.
```

### 一键执行

可以使用 run 脚本一键执行查看结果

linux 下执行：

```bash
bash run.sh
```

windows 下执行：

```bash
bash run.bat
```
