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
===> resnet50_multistreams python example start...
devices: 1
threads: 4
samples: 10
model: ./resnet50_xh2_b1_1core.hmm
thread 0 on device 0 load model ./resnet50_xh2_b1_1core.hmm loaded.
input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
thread 1 on device 0 load model ./resnet50_xh2_b1_1core.hmm loaded.
input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
thread 2 on device 0 load model ./resnet50_xh2_b1_1core.hmm loaded.
thread 3 on device 0 load model ./resnet50_xh2_b1_1core.hmm loaded.
input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
thread 3 on device 0 run sample 0 end.
thread 2 on device 0 run sample 1 end.
thread 1 on device 0 run sample 3 end.
thread 3 on device 0 run sample 4 end.
thread 0 on device 0 run sample 2 end.
thread 2 on device 0 run sample 5 end.
thread 1 on device 0 run sample 6 end.
thread 0 on device 0 run sample 8 end.
thread 2 on device 0 run sample 9 end.
thread 3 on device 0 run sample 7 end.
thread 1 on device 0 completed. 2 sampels tested.
thread 0 on device 0 completed. 2 sampels tested.
thread 2 on device 0 completed. 3 sampels tested.
thread 3 on device 0 completed. 3 sampels tested.
sample 0 top1: predict cls = 65, prob = 0.651222
sample 1 top1: predict cls = 65, prob = 0.651222
sample 2 top1: predict cls = 65, prob = 0.651222
sample 3 top1: predict cls = 65, prob = 0.651222
sample 4 top1: predict cls = 65, prob = 0.651222
sample 5 top1: predict cls = 65, prob = 0.651222
sample 6 top1: predict cls = 65, prob = 0.651222
sample 7 top1: predict cls = 65, prob = 0.651222
sample 8 top1: predict cls = 65, prob = 0.651222
sample 9 top1: predict cls = 65, prob = 0.651222
<=== resnet50_multistreams python example completed.
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
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50_multistreams -n 2
```

测试程序将测试样本存入输入队列，由每个线程的推理模块竞争推理，并将推理后的结果存入输出队列。以下打印表示每个线程推理完成后的数量统计，同时显示每个线程工作在哪个设备上（注：模拟器下仅支持单线程单设备）：

```bash
===> resnet50_multistreams c++ example start...
model: ./resnet50_xh2_b1_1core.hmm
devices: 1
threads: 4
samples: 10
sample queue size is 10
thread 1 on device 0 model ./resnet50_xh2_b1_1core.hmm loaded.
Input[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Output[495] TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
thread 2 on device 0 model ./resnet50_xh2_b1_1core.hmm loaded.
Input[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Output[495] TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
thread 3 on device 0 model ./resnet50_xh2_b1_1core.hmm loaded.
Input[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Output[495] TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
thread 0 on device 0 model ./resnet50_xh2_b1_1core.hmm loaded.
Input[input.1] TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
Output[495] TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
thread 2 on device 0 infer start...
thread 0 on device 0 infer start...
thread 1 on device 0 infer start...
thread 3 on device 0 infer start...
thread 1 on device 0 run sample 2 end.
thread 3 on device 0 run sample 3 end.
thread 2 on device 0 run sample 0 end.
thread 0 on device 0 run sample 1 end.
thread 1 on device 0 run sample 4 end.
thread 3 on device 0 run sample 5 end.
thread 0 on device 0 run sample 7 end.
thread 2 on device 0 run sample 6 end.
thread 0 on device 0 completed. 2 sampels tested.
thread 2 on device 0 completed. 2 sampels tested.
thread 1 on device 0 run sample 8 end.
thread 3 on device 0 run sample 9 end.
thread 1 on device 0 completed. 3 sampels tested.
thread 3 on device 0 completed. 3 sampels tested.
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
<=== resnet50_multistreams c++ example completed.
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
