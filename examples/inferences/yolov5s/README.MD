# Yolov5s Example

## 目录

[TOC]

## 概述

TCIM Examples 以yolov5s模型为例展示TCIM API推理的基本使用方式。

1. yolov5s.py 展示python下单线程推理过程，包括：
    - python单线程推理，使用默认stream（无需指定）
2. yolov5s.cc 展示c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认stream（无需指定）

对于yolov5s模型中的后处理部分，本示例提供两种实现方法：
1. （默认）基于代码实现后处理逻辑。
2. 将yolov5s后处理部分独立为一个onnx模型，使用onnxruntime推理得到结果。

其中方法2在本示例中仅提供Linux系统运行方法，如需在windows系统中运行可自行安装配置onnxruntime环境，则可使用相同步骤运行方法2示例。

## 软件依赖

本示例依赖opencv库进行图像预处理和可视化，依赖ultralytics进行数据可视化，依赖onnxruntime库推理yolov5s后处理模型。

## 快速开始

1. 获取模型文件，执行：

```bash
# 后处理实现为方法1
python3 get_model.py
# 后处理实现为方法2 (仅支持Linux)
python3 get_model.py --enable_ort
```

2. 执行python演示程序，执行命令：

```bash
# 后处理实现为方法1
python3 yolov5s.py
# 后处理实现为方法2 (仅支持Linux)
python3 yolov5s.py --enable_ort
```

3. 编译并执行c++演示程序，执行命令：

```bash
# 后处理实现为方法1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_yolov5s
```

```bash
# 后处理实现为方法2 (仅支持Linux)
mkdir build
cd build
cmake -DENABLE_ORT=ON -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_yolov5s --enable_ort
```

## 参考结果

### Python示例结果

Python演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
input[images] shape = [1, 640, 640, 3], dtype = <class 'numpy.int8'>, format = YUV420SP
output[340] shape = [1, 3, 80, 80, 85], dtype = <class 'numpy.float32'>, format = ND
output[378] shape = [1, 3, 40, 40, 85], dtype = <class 'numpy.float32'>, format = ND
output[416] shape = [1, 3, 20, 20, 85], dtype = <class 'numpy.float32'>, format = ND
box num = 20
x1:9, y1:166, x2:149, y2:264, conf:0.915732, cls:62
x1:291, y1:218, x2:356, y2:317, conf:0.871175, cls:56
x1:414, y1:156, x2:466, y2:298, conf:0.789320, cls:0
x1:556, y1:211, x2:642, y2:287, conf:0.767701, cls:62
...
x1:383, y1:170, x2:407, y2:216, conf:0.265686, cls:58
demo results saved to demo_results/python/000000000139.jpg
<=== yolov5s python example completed.
```
检测效果图片保存在demo_results/python目录下。

### C++示例结果

C++演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
Count of Input: 1
Input[images] TensorInfo: shape: [1,640,640,3], stride: [614400,640,1,614400,640,2,1,409600], dtype: UINT8, format: YUV420SP, size: 614400, memsize: 614400
Count of Output: 3
Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000
detect num: 17
box[10, 167, 150, 265], conf:0.923558, cls:62
box[294, 218, 352, 317], conf:0.856569, cls:56
box[417, 155, 464, 299], conf:0.790732, cls:0
box[558, 210, 639, 286], conf:0.706285, cls:62
...
box[240, 194, 253, 212], conf:0.258923, cls:75
demo results saved to demo_results/cpp/000000000139.jpg
<=== yolov5s c++ example completed.
```
检测效果图片保存在demo_results/cpp目录下。

*注：由于c++与python版本的前后处理不完全一致，算法结果可能有少量偏差。*

### 一键执行

可以使用run脚本一键执行查看结果

linux下执行：

```bash
# 后处理实现为方法1
bash run.sh
# 后处理实现为方法2
bash run.sh enable_ort
```

windows下执行：

```bash
bash run.bat
```