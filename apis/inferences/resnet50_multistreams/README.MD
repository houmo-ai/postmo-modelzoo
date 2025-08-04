# Resnet50 Multistreams Example

## 目录

[TOC]

## 概述

Resnet50 Multistreams Example 以resnet50模型为例展示TCIM API在多设备、多线程以及多stream下的使用方式。

1. resnet50_multistreams.py 展示python下多线程多stream推理过程
2. resnet50_multistreams.cc 展示c++下多设备多线程以及多stream推理过程
3. 使用create_weight_manager接口共享模型内存
4. 支持配置多线程，每个线程使用1个stream
5. 支持配置多设备，每个设备使用相同数量的线程

注：本示例仅支持后摩Target为XH1。

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

使用runtime python API进行模型推理，并输出结果。执行时可通过参数修改配置：

- -n 1：设备数，默认为1
- -t 4：线程数，默认为4
- -s 10：测试样本数，默认为10

以2个设备为例（有两张卡device id分别为0和1）：

```bash
python3 resnet50_multistreams.py -n 2
```

测试程序将测试样本存入输入队列，由每个线程的推理模块竞争推理，并将推理后的结果存入输出队列。以下打印表示每个线程推理完成后的数量统计（注：模拟器下仅支持单线程）：

```bash
thread 0 on device 0 completed. 3 sampels tested.
thread 1 on device 0 completed. 3 sampels tested.
thread 2 on device 0 completed. 2 sampels tested.
thread 3 on device 0 completed. 2 sampels tested.
thread 4 on device 1 completed. 3 sampels tested.
thread 5 on device 1 completed. 3 sampels tested.
thread 6 on device 1 completed. 2 sampels tested.
thread 7 on device 1 completed. 2 sampels tested.
```

最后对输出队列的每个结果进行后处理和检查

```bash
sample 0 top1: predict cls = 65, prob = 0.385072
sample 1 top1: predict cls = 65, prob = 0.385072
sample 2 top1: predict cls = 65, prob = 0.385072
sample 3 top1: predict cls = 65, prob = 0.385072
...
```

### 模型c++推理

使用runtime c++ API进行模型推理，并输出结果。执行时可通过参数修改配置：

- -n 1：设备数，默认为1
- -t 4：线程数，默认为4
- -s 10：测试样本数，默认为10

以2个设备为例（有两张卡device id分别为0和1）：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_resnet50_multistreams -n 2
```

测试程序将测试样本存入输入队列，由每个线程的推理模块竞争推理，并将推理后的结果存入输出队列。以下打印表示每个线程推理完成后的数量统计，同时显示每个线程工作在哪个设备上（注：模拟器下仅支持单线程单设备）：

```bash
thread 0 on device 0 completed. 3 sampels tested.
thread 1 on device 0 completed. 3 sampels tested.
thread 2 on device 0 completed. 2 sampels tested.
thread 3 on device 0 completed. 2 sampels tested.
thread 4 on device 1 completed. 3 sampels tested.
thread 5 on device 1 completed. 3 sampels tested.
thread 6 on device 1 completed. 2 sampels tested.
thread 7 on device 1 completed. 2 sampels tested.
```

最后对输出队列的每个结果进行后处理和检查

```bash
sample 0 top1: Index=65 Conf=119, Label=[sea snake]
sample 1 top1: Index=65 Conf=119, Label=[sea snake]
sample 2 top1: Index=65 Conf=119, Label=[sea snake]
sample 3 top1: Index=65 Conf=119, Label=[sea snake]
...
```

**说明：由于c++和python前处理有差异，且c++后处理没有做softmax，部分结果可能存在少量差异。**

### 一键执行

可以使用run脚本一键执行查看结果, 以2个设备为例（有两张卡device id分别为0和1）

linux下执行：

```bash
bash run.sh -n 2
```

windows下执行（windows需要在代码中修改参数）：

```bash
bash run.bat
```
