# Resnet50 Pipeline Example

## 目录

[TOC]

## 概述

本示例以resnet50模型为例展示流水线（pipeline）推理的使用方式。流水线方式可以并行处理数据的拷入、拷出和推理过程，提升单线程下的推理吞吐。

此种方式只支持c++。

resnet50.cc 展示c++下pipeline推理过程，包括：

- h2d线程将host侧输入数据拷贝到device侧
- infer线程设置device侧输入和输出，然后执行推理
- d2h线程将device侧输出数据拷贝到host侧


## 软件依赖

除TCIM Examples公用依赖外，本示例没有其他依赖。

## 快速上手

### 获取芯片模型

可以通过以下两种方式获取芯片模型：

1. 自行编译：通过原始模型或量化模型，经过tcim工具链编译生成，具体可参考houmo-modelzoo
2. 直接下载：通过get_model.py脚本下载预编译的模型

```bash
python3 get_model.py
```

### 模型c++推理

使用runtime c++ API进行模型推理，并输出结果。执行时可通过参数修改配置：

- -s 10：测试样本数，默认为10

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_resnet50_pipeline
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
sample 0 top1: Index=65 Conf=120, Label=[sea snake]
sample 1 top1: Index=65 Conf=120, Label=[sea snake]
sample 2 top1: Index=65 Conf=120, Label=[sea snake]
sample 3 top1: Index=65 Conf=120, Label=[sea snake]
...
```

### 一键执行

可以使用run脚本一键执行查看结果

linux下执行：

```bash
bash run.sh
```

windows下执行：

```bash
bash run.bat
```