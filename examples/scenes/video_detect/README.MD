# VideoDetect Example

## 目录

[TOC]

## 概述

本示例以后摩视频解码+yolov5s检测模型推理+resnet50分类模型推理为例展示视频检测推理Pipeline的基本使用方式。

本示例仅支持c++。

## 软件依赖

本示例依赖opencv库进行图像预处理和可视化。

## 快速上手

### 获取芯片模型

可以通过以下两种方式获取芯片模型：

1. 自行编译：通过原始模型或量化模型，经过tcim工具链编译生成，需要开启动态resize并设置输入尺寸（匹配解码输出尺寸），具体可参考houmo-modelzoo
```bash
python3 ptq.py --input_shape=1,3,1080,1920 --dynamic_resize
python3 build.py --input_shape=1,3,1080,1920 --dynamic_resize
```

2. 直接下载：通过get_model.py脚本下载预编译的模型

```bash
python3 get_model.py
```

### 模型c++推理

可通过修改video_detect.cc中main函数的变量指定运行参数:
- det_model_path：检测模型路径
- cls_model_path：分类模型路径
- stream_path：视频流路径，目前支持h264/h265文件
- det_thread_num：检测线程数
- cls_thread_num：分类线程数
- frame_limit：取视频流的前n帧，0表示所有（ISIM下推理性能低，默认设置为2）

使用runtime c++ API进行模型推理，并输出结果：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
./example_video_detect
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
Count of Input: 2
Input[images] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 3
Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [2457600,819200,10240,128,1], dtype: INT8, format: ND, size: 2457557, memsize: 2457600, quant_info: size=1 [0, 0.00780907]
Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [614400,204800,5120,128,1], dtype: INT8, format: ND, size: 614357, memsize: 614400, quant_info: size=1 [0, 0.00783619]
Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [153600,51200,2560,128,1], dtype: INT8, format: ND, size: 153557, memsize: 153600, quant_info: size=1 [0, 0.00785193]
[info] thread 0 model loaded: yolov5s.hmm
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
[info] thread 0 model loaded: resnet50.hmm
[info] classify thread 0 infer start...
[info] thread 1 model loaded: resnet50.hmm
[info] detect thread 0 infer start...
[info] classify thread 1 infer start...
frame size is 1920 x 1080
[info] <=== PushStream thread exit. 2 frames received.
[info] ===> get frame thread EOS received.
[info] <=== GetFrame thread exit. 2 frames received.
[debug] detect thread 0 run sample 0 end. cost 12.066000 ms.
detect num: 11
demo results saved to demo_results/0.jpg
[debug] classify thread 0 run sample 0 obj 0 end. cost 2.918000 ms.
sample 0 box[314, 596, 638, 842], det[conf:0.917086, cls:2], cls[id:656, conf:12.269067, lable:[minivan]]
[debug] classify thread 0 run sample 0 obj 1 end. cost 2.581000 ms.
sample 0 box[729, 302, 891, 427], det[conf:0.813372, cls:2], cls[id:713, conf:10.568404, lable:[photocopier]]
[debug] classify thread 0 run sample 0 obj 2 end. cost 2.608000 ms.
sample 0 box[330, 966, 762, 1067], det[conf:0.801487, cls:2], cls[id:545, conf:10.932832, lable:[electric fan, blower]]
[debug] classify thread 0 run sample 0 obj 3 end. cost 2.706000 ms.
sample 0 box[427, 358, 639, 510], det[conf:0.797969, cls:2], cls[id:817, conf:11.054308, lable:[sports car, sport car]]
[debug] detect thread 0 run sample 1 end. cost 12.081000 ms.
[debug] classify thread 0 run sample 0 obj 4 end. cost 2.562000 ms.
sample 0 box[806, 378, 1039, 600], det[conf:0.692485, cls:7], cls[id:656, conf:9.961025, lable:[minivan]]
...
[info] ===> detect thread EOS received.
[info] <=== detect thread 0 completed. 2 sampels tested.
[info] ===> classify thread EOS received.
[info] <=== classify thread 0 completed. 2 sampels tested.
```

检测效果图片保存在demo_results目录下。

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