# Yolov5s Example

## 目录

[TOC]

## 概述

TCIM Examples 以yolov5s模型为例展示TCIM API推理的基本使用方式。

1. build.py 展示模型编译过程，包括：
    - 量化后模型加载和编译
    - 芯片模型加载和golden数据验证
2. yolov5s.py 展示python下单线程推理过程，包括：
    - python单线程推理，使用默认stream（无需指定）
3. yolov5s.cc 展示c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认stream（无需指定）

## 软件依赖

本示例依赖opencv库进行图像预处理和可视化，依赖ultralytics进行数据可视化。

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
python3 yolov5s.py
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
input[images] shape = [1, 640, 640, 3], dtype = <class 'numpy.int8'>, format = YUV420SP
output[340] shape = [1, 3, 80, 80, 85], dtype = <class 'numpy.float32'>, format = ND
output[378] shape = [1, 3, 40, 40, 85], dtype = <class 'numpy.float32'>, format = ND
output[416] shape = [1, 3, 20, 20, 85], dtype = <class 'numpy.float32'>, format = ND
box num = 18
x1:0, y1:164, x2:161, y2:264, conf:0.901221, cls:62
x1:288, y1:219, x2:357, y2:319, conf:0.857656, cls:56
x1:561, y1:210, x2:638, y2:280, conf:0.777500, cls:62
...
```
检测效果图片保存在demo_results/python目录下。

### 模型c++推理

使用runtime c++ API进行模型推理，并输出结果：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_yolov5s
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
Count of Input: 1
Input[images] TensorInfo: shape: [1,640,640,3], dtype: INT8, format: YUV420SP, size: 1228800, memsize: 1228800
Count of Output: 3
Output[340] TensorInfo: shape: [1,3,80,80,85], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
Output[378] TensorInfo: shape: [1,3,40,40,85], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
Output[416] TensorInfo: shape: [1,3,20,20,85], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000
detect num: 15
box[0, 164, 160, 264], conf:0.893384, cls:62
box[294, 220, 350, 319], conf:0.835627, cls:56
box[560, 201, 638, 288], conf:0.800825, cls:62
box[446, 120, 461, 141], conf:0.695586, cls:74
```
检测效果图片保存在demo_results/cpp目录下。

*注：由于c++与python版本的前后处理不完全一致，算法结果可能有少量偏差。*

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