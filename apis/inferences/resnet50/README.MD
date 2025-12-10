# Resnet50 Example

## 目录

[TOC]

## 概述

TCIM Examples 以resnet50模型为例展示TCIM API推理的基本使用方式。

1. resnet50.py 展示python下单线程推理过程，包括：
    - python单线程推理，使用默认stream（无需指定）
2. resnet50.cc 展示c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认stream（无需指定）


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

### 模型python推理

使用runtime python API进行模型推理，并输出结果：

```bash
python3 resnet50.py
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> resnet50 python example start...
input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = [1, 1000], dtype = <class 'numpy.float32'>, format = ND
top1: predict cls = 65, prob = 0.651222
top2: predict cls = 58, prob = 0.157115
top3: predict cls = 67, prob = 0.054723
top4: predict cls = 54, prob = 0.048293
top5: predict cls = 62, prob = 0.036454
<=== resnet50 python example completed.
```

### 模型c++推理

使用runtime c++ API进行模型推理，并输出结果：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> resnet50 c++ example start...
model ./resnet50_xh2_b1_1core.hmm loaded.
Count of Input: 1
Input[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Input f32[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT32, format: ND, size: 602112, memsize: 602112
img_bytes: 602112
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
top1: Index=65 Conf=16, Label=[sea snake]
top2: Index=58 Conf=14.5781, Label=[water snake]
top3: Index=67 Conf=13.5234, Label=[diamondback, diamondback rattlesnake, Crotalus adamanteus]
top4: Index=54 Conf=13.3984, Label=[hognose snake, puff adder, sand viper]
top5: Index=62 Conf=13.1172, Label=[rock python, rock snake, Python sebae]
<=== resnet50 c++ example completed.
```

**说明：由于c++和python前处理有差异，且c++后处理没有做softmax，部分结果可能存在少量差异。**

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