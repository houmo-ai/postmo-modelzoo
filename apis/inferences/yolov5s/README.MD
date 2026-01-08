# Yolov5s Example

## 目录

[TOC]

## 概述

TCIM Examples 以 yolov5s 模型为例展示 TCIM API 推理的基本使用方式。

1. yolov5s.py 展示 python 下单线程推理过程，包括：
    - python 单线程推理，使用默认 stream（无需指定）
2. yolov5s.cc 展示 c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认 stream（无需指定）

对于 yolov5s 模型中的后处理部分，本示例提供两种实现方法：

1. （默认）基于代码实现后处理逻辑。
2. 将 yolov5s 后处理部分独立为一个 onnx 模型，使用 onnxruntime 推理得到结果。

其中方法 2 在本示例中仅提供 Linux 系统运行方法，如需在 windows 系统中运行可自行安装配置 onnxruntime 环境，则可使用相同步骤运行方法 2 示例。

## 软件依赖

本示例依赖 opencv 库进行图像预处理和可视化，依赖 onnxruntime 库推理 yolov5s 后处理模型。

## 快速开始

1. 获取模型文件，执行：

```bash
# 后处理实现为方法1
python3 get_model.py
# 后处理实现为方法2 (仅支持Linux)，其中将安装执行C++示例所需的onnxruntime环境
python3 get_model.py --enable_ort
```

2. 执行 python 演示程序，执行命令：

```bash
# 后处理实现为方法1
python3 yolov5s.py
# 后处理实现为方法2 (仅支持Linux)
python3 yolov5s.py --enable_ort
```

3. 编译并执行 c++演示程序，执行命令：

```bash
# 后处理实现为方法1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
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
# 将onnxruntime的动态库路径加入LD_LIBRARY_PATH，执行C++用例时需要依赖
WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$WORK_PATH/../../models/3rdparty/onnxruntime/lib:$LD_LIBRARY_PATH
# 执行C++示例
./example_yolov5s --enable_ort
```

## 参考结果

### Python 示例结果

Python 演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> yolov5s python example start...
input[images] shape = [1, 3, 640, 640], dtype = <class 'numpy.float16'>, format = ND
output[340] shape = [1, 3, 80, 80, 85], dtype = <class 'numpy.float32'>, format = ND
output[378] shape = [1, 3, 40, 40, 85], dtype = <class 'numpy.float32'>, format = ND
output[416] shape = [1, 3, 20, 20, 85], dtype = <class 'numpy.float32'>, format = ND
box num = 19
x1:7, y1:168, x2:153, y2:261, conf:0.903683, cls:62
x1:293, y1:217, x2:353, y2:317, conf:0.844447, cls:56
x1:413, y1:156, x2:465, y2:298, conf:0.811396, cls:0
x1:557, y1:211, x2:641, y2:284, conf:0.747585, cls:62
x1:449, y1:168, x2:513, y2:287, conf:0.660653, cls:72
x1:551, y1:299, x2:586, y2:402, conf:0.637132, cls:39
x1:168, y1:233, x2:184, y2:267, conf:0.625747, cls:75
x1:549, y1:301, x2:586, y2:401, conf:0.598986, cls:75
x1:360, y1:221, x2:426, y2:315, conf:0.595127, cls:56
x1:480, y1:360, x2:637, y2:426, conf:0.575638, cls:60
x1:404, y1:221, x2:442, y2:307, conf:0.542933, cls:56
x1:447, y1:121, x2:462, y2:141, conf:0.535311, cls:74
x1:231, y1:172, x2:267, y2:213, conf:0.490232, cls:58
x1:332, y1:178, x2:368, y2:224, conf:0.480379, cls:58
x1:411, y1:157, x2:475, y2:293, conf:0.443573, cls:72
x1:490, y1:174, x2:513, y2:283, conf:0.384144, cls:72
x1:349, y1:211, x2:362, y2:231, conf:0.356749, cls:75
x1:357, y1:215, x2:374, y2:231, conf:0.356060, cls:75
x1:355, y1:217, x2:422, y2:318, conf:0.310120, cls:60
demo results saved to demo_results/python/000000000139.jpg
<=== yolov5s python example completed.
```

检测效果图片保存在 demo_results/python 目录下。

### C++示例结果

C++演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> yolov5s c++ example start...
Load yolov5s model from file ./yolov5s_clip_xh2_b1_1core.hmm.
Model ./yolov5s_clip_xh2_b1_1core.hmm loaded.
Count of Input: 1
Input[images] info: TensorInfo: shape: [1,3,640,640], stride: [], dtype: FLOAT16, format: ND, size: 2457600, memsize: 2457600
Input[images] float32 info: TensorInfo: shape: [1,3,640,640], stride: [], dtype: FLOAT32, format: ND, size: 4915200, memsize: 4915200
image bytes: 4915200.
Count of Output: 3
Output[340] info: TensorInfo: shape: [1,3,80,80,85], stride: [], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
Output[378] info: TensorInfo: shape: [1,3,40,40,85], stride: [], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
Output[416] info: TensorInfo: shape: [1,3,20,20,85], stride: [], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000
detect num: 16
box[6, 168, 152, 261], conf:0.903683, cls:62
box[292, 217, 352, 317], conf:0.844447, cls:56
box[412, 155, 464, 298], conf:0.811396, cls:0
box[557, 211, 639, 284], conf:0.747585, cls:62
box[448, 168, 512, 286], conf:0.660653, cls:72
box[550, 298, 585, 401], conf:0.637132, cls:39
box[167, 233, 184, 267], conf:0.625747, cls:75
box[359, 220, 425, 315], conf:0.595127, cls:56
box[479, 360, 636, 425], conf:0.575638, cls:60
box[404, 220, 442, 307], conf:0.542933, cls:56
box[447, 120, 461, 141], conf:0.535311, cls:74
box[230, 171, 267, 213], conf:0.490232, cls:58
box[331, 177, 368, 224], conf:0.480379, cls:58
box[490, 174, 513, 282], conf:0.384144, cls:72
box[348, 211, 362, 230], conf:0.356749, cls:75
box[356, 214, 373, 231], conf:0.356060, cls:75
demo results saved to demo_results/cpp/000000000139.jpg.
<=== yolov5s c++ example completed.
```

检测效果图片保存在 demo_results/cpp 目录下。

_注：由于 c++与 python 版本的前后处理不完全一致，算法结果可能有少量偏差。_

### 一键执行

可以使用 run 脚本一键执行查看结果

linux 下执行：

```bash
# 后处理实现为方法1
bash run.sh
# 后处理实现为方法2
bash run.sh enable_ort
```

windows 下执行：

```bash
bash run.bat
```
