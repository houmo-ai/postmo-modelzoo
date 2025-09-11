# Yolov5s Example

## 目录

[TOC]

## 概述

该示例以yolov5s为例将带有dynamic_resizer的模型推理数据处理过程隐藏简化用户工作。


## 软件依赖

本示例依赖opencv库进行图像预处理和可视化。

## 快速开始

1. 获取模型文件，执行：

```bash
python3 get_model.py
```
```

3. 编译并执行c++演示程序，执行命令：

```bash
# 后处理实现为方法1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ -DCMAKE_BUILD_TYPE=Release ..
make
make install
cd ..
./example_yolov5s_dynamic
```

## 参考结果

```bash
Device Num: 2
Backend Name: Xh1HdiBackend
Model Version: 20250102
Model CoreNum: 1
Model InputNum: 2
Model OutputNum: 2
Input[0] name: images, shape: [1, 3, 1080, 1920], dtype: UINT8, fmt: YUV420SP, memSize: 3110400
Input[1] name: resizer_crop_images, shape: [1, 10], dtype: INT32, fmt: ND, memSize: 40
Output[0] name: 340, shape: [1, 3, 80, 80, 85], dtype: INT8, fmt: ND, memSize: 2457600
Output[1] name: 378, shape: [1, 3, 40, 40, 85], dtype: INT8, fmt: ND, memSize: 614400
Output[2] name: 416, shape: [1, 3, 20, 20, 85], dtype: INT8, fmt: ND, memSize: 153600
RESIZER_INPUT_H: 1080, RESIZER_INPUT_W: 1920
MODEL_INPUT_H: 640, MODEL_INPUT_W: 640
IMAGE_H: 426, IMAGE_W: 640
detection size: 17
```
检测效果图片保存在当前目录下`result.png`。

### 一键执行

可以使用run脚本一键执行查看结果

```bash
bash run.sh
```