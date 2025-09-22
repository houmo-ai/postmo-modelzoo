# VideoDetect Example

## 目录

[TOC]

## 概述

本示例展示了两种编解码场景：
- 视频解码+yolov5s检测模型推理+resnet50分类模型推理，展示视频检测推理Pipeline的基本使用方式。
- JPEG图像编解码+yolov5s检测模型推理+resnet50分类模型推理，展示JPEG编解码+推理Pipeline的基本使用方式。

对于视频解码，本示例提供两种实现方案：
- （默认）后摩视频解码。在本方案中，提供Resize视频解码输出功能: 默认开启，将后摩视频解码输出resize为yolov5s输入大小 1080x1920，支持的解码输出最大尺寸为3840x2160。
- Rockchip视频解码（后文均简称RK视频解码）。

本示例默认执行场景：后摩视频解码并Resize解码输出，Resize后图像用于yolov5s检测模型推理+resnet50分类模型推理。如需运行其他场景，可通过一键执行脚本的参数控制，或修改编译参数控制。

本示例仅支持c++。由于后摩视频解码暂不支持windows，因此本示例不支持windows。

## 软件依赖

本示例依赖opencv库进行图像预处理和可视化，依赖ffmpeg库进行图像取流。

如视频解码实现方案使用RK解码，则需依赖mpp库进行视频解码。mpp库介绍及安装方法见：https://github.com/rockchip-linux/mpp 。
成功安装mpp库后，将在`/usr/local/lib`下产生mpp相关的so库，mpp库提供的头文件将在`/usr/local/include/rockchip/`中。如果上述路径不存在mpp相关资源，则需从mpp安装路径拷贝相关资源至`/usr/local/lib`和`/usr/local/include/rockchip/`, 用于C++演示程序的编译。

## 快速上手

### 1. 获取芯片模型

通过get_model.py脚本下载预编译的模型，含yolov5s和resnet50两个模型。

```bash
python3 get_model.py
```

### 2. 模型推理

#### 2.1 演示程序可修改变量说明

可通过修改 `video_detect.cc` 中 `main` 函数的变量指定运行参数:
- `det_model_path`：检测模型路径
- `cls_model_path`：分类模型路径
- `stream_path`：视频流路径，目前支持h264/h265文件
- `det_thread_num`：检测线程数
- `cls_thread_num`：分类线程数
- `frame_limit`：取视频流的前n帧，0表示所有（ISIM下推理性能低，默认设置为2）

可通过控制编译提供的-D参数`ENABLE_RESIZER`来开关Resize视频解码输出功能:
- `-DENABLE_RESIZER=ON`: 开启Resize视频解码输出功能
- `-DENABLE_RESIZER=OFF`: 关闭Resize视频解码输出功能

可通过控制编译提供的-D参数`ENABLE_RK_DECODER`来开关RK视频解码:
- `-DENABLE_RK_DECODER=ON`: 开启RK视频解码（同时会关闭后摩视频解码）
- `-DENABLE_RK_DECODER=OFF`: 关闭RK视频解码（同时会开启后摩视频解码）

可通过控制编译提供的-D参数`ENABLE_JPEG`来开关RK解码:
- `-DENABLE_JPEG=ON`: 开启JPEG编解码（同时会关闭后摩解码）
- `-DENABLE_JPEG=OFF`: 开启JPEG编解码（同时会开启后摩解码）

*注：无需修改或配置上述变量，程序默认也可运行。*

#### 2.2 编译并执行c++演示程序

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ -DENABLE_RESIZER=ON -DENABLE_RK_DECODER=OFF -DENABLE_JPEG=OFF ..
make -j
make install
cd ..
./example_video_detect
```

如果使用RK解码方案，则需在执行C++演示程序前，设置环境变量：
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`

#### 2.3 参考结果

模型推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
# Linux x86 环境，使用后摩解码并开启Resize解码输出功能
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
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
[info] thread 1 model loaded: resnet50.hmm
[info] classify thread, module 1 infer start...[info] classify thread, module 0 infer start...
[info] detect thread, moudle 0 infer start...
[info] open stream ../../data/1080P_traffic_4s.h264 completed.
[info] frame size is 1920 x 1080
[info] ===> decoder heigth:1080, width:1920
[debug] get frame 1, stamp 41622, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] -->> resized yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 0, queue size 0
[debug] get frame 2, stamp 45144, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] -->> resized yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 1, queue size 0
[debug] get frame 3, stamp 48465, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] -->> resized yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 2, queue size 1
[debug] get frame 4, stamp 51800, planes 2, len 0
......
[debug] detect thread 0 run sample 99 end. cost 15.214 ms.
[info] detect num:14
[debug] classify thread 1 run sample 98 obj 7 end. cost 2.502 ms.
sample 98 box[1182, 153, 1231, 203], det[conf:0.611767, cls:2], cls[id:482, conf:10.932832, lable:[cassette player]]
[debug] classify thread 1 run sample 98 obj 8 end. cost 2.527 ms.
sample 98 box[0, 313, 117, 380], det[conf:0.599321, cls:2], cls[id:817, conf:13.119398, lable:[sports car, sport car]]
[debug] demo results saved to demo_results/99.jpg
[debug] detect push task req_id 99, obj num 14, queue size 0
[info] ===> detect thread EOS received.
[info] <=== detect thread 0 completed. 100 sampels tested.
[debug] classify thread 1 run sample 98 obj 9 end. cost 2.555 ms.
[debug] classify thread 1 run sample 98 obj 15 end. cost 2.783 ms.
sample 98 box[1073, 134, 1109, 165], det[conf:0.253683, cls:2], cls[id:481, conf:11.661688, lable:[cassette]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 1 completed. 48 sampels tested.
[debug] classify thread 0 run sample 99 obj 11 end. cost 2.446 ms.
sample 99 box[1110, 140, 1157, 176], det[conf:0.448824, cls:2], cls[id:531, conf:10.082500, lable:[digital watch]]
[debug] classify thread 0 run sample 99 obj 12 end. cost 2.45 ms.
sample 99 box[887, 138, 943, 183], det[conf:0.422602, cls:2], cls[id:675, conf:10.446928, lable:[moving van]]
[debug] classify thread 0 run sample 99 obj 13 end. cost 2.468 ms.
sample 99 box[1424, 128, 1489, 185], det[conf:0.325398, cls:2], cls[id:766, conf:9.110693, lable:[rotisserie]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 0 completed. 52 sampels tested.
```

```bash
# Linux x86 环境，使用后摩解码并关闭Resize解码输出功能
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
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
[info] thread 1 model loaded: resnet50.hmm
[info] classify thread, module 0 infer start...
[info] detect thread, moudle 0 infer start...
[info] classify thread, module 1 infer start...
[info] open stream ../../data/1080P_traffic_4s.h264 completed.
[info] frame size is 1920 x 1080
[info] ===> decoder heigth:1080, width:1920
[debug] get frame 1, stamp 33944, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 0, queue size 0
[debug] get frame 2, stamp 40931, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 1, queue size 0
[debug] get frame 3, stamp 44345, planes 2, len 0
[info] -->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[debug] decoder pull data req_id 2, queue size 1
[debug] get frame 4, stamp 48120, planes 2, len 0
......
[debug] detect thread 0 run sample 99 end. cost 15.503 ms.
[info] detect num:14
[debug] detect push task req_id 99, obj num 14, queue size 0
[info] ===> detect thread EOS received.
[info] <=== detect thread 0 completed. 100 sampels tested.
[debug] classify thread 0 run sample 99 obj 1 end. cost 2.849 ms.
sample 99 box[1002, 579, 1236, 903], det[conf:0.822449, cls:7], cls[id:662, conf:8.503314, lable:[modem]]
[debug] classify thread 1 run sample 98 obj 15 end. cost 2.645 ms.
sample 98 box[1073, 134, 1109, 165], det[conf:0.253683, cls:2], cls[id:481, conf:11.661688, lable:[cassette]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 1 completed. 50 sampels tested.
[debug] classify thread 0 run sample 99 obj 11 end. cost 2.607 ms.
sample 99 box[1110, 140, 1157, 176], det[conf:0.448824, cls:2], cls[id:531, conf:10.082500, lable:[digital watch]]
[debug] classify thread 0 run sample 99 obj 12 end. cost 2.521 ms.
sample 99 box[887, 138, 943, 183], det[conf:0.422602, cls:2], cls[id:675, conf:10.446928, lable:[moving van]]
[debug] classify thread 0 run sample 99 obj 13 end. cost 2.507 ms.
sample 99 box[1424, 128, 1489, 185], det[conf:0.325398, cls:2], cls[id:766, conf:9.110693, lable:[rotisserie]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 0 completed. 50 sampels tested.
```

```bash
# Linux aarch64 环境，使用RK解码
[info] Count of Input: 2
[info] Input[images] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
[info] Count of Output: 3
[info] Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [2457600,819200,10240,128,1], dtype: INT8, format: ND, size: 2457557, memsize: 2457600, quant_info: size=1 [0, 0.00780907]
[info] Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [614400,204800,5120,128,1], dtype: INT8, format: ND, size: 614357, memsize: 614400, quant_info: size=1 [0, 0.00783619]
[info] Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [153600,51200,2560,128,1], dtype: INT8, format: ND, size: 153557, memsize: 153600, quant_info: size=1 [0, 0.00785193]
[info] thread 0 model loaded: yolov5s.hmm
[info] Count of Input: 2
[info] Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
[info] Count of Output: 1
[info] Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
[info] thread 0 model loaded: resnet50.hmm
[info] Count of Input: 2
[info] Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
[info] Count of Output: 1
[info] Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
[info] thread 1 model loaded: resnet50.hmm
[info] classify thread, module 0 infer start...
[info] detect thread, moudle 0 infer start...
[info] classify thread, module 1 infer start...
[info] [RK Decoder] mpp ctx 0x7f94005bc0 start to decode stream...
[info] [RK Decoder] frame info change: width = 1920, height = 1080, hor_stride = 1920, ver_stride = 1088, buf_size = 4177920
[debug] detect thread 0 run sample 0 end. cost 19.725 ms.
[info] detect num:11
[debug] demo results saved to demo_results/0.jpg
[debug] detect push task req_id 0, obj num 11, queue size 0
[debug] classify thread 1 run sample 0 obj 0 end. cost 5.652 ms.
sample 0 box[314, 596, 638, 842], det[conf:0.917086, cls:2], cls[id:656, conf:12.269067, lable:[minivan]]
[debug] classify thread 1 run sample 0 obj 1 end. cost 5.851 ms.
sample 0 box[729, 302, 891, 427], det[conf:0.813372, cls:2], cls[id:713, conf:10.568404, lable:[photocopier]]
[debug] classify thread 1 run sample 0 obj 2 end. cost 5.208 ms.
sample 0 box[330, 966, 762, 1067], det[conf:0.801487, cls:2], cls[id:545, conf:10.932832, lable:[electric fan, blower]]
......
[debug] detect thread 0 run sample 97 end. cost 21.547 ms.
[info] detect num:15
[debug] demo results saved to demo_results/97.jpg
[debug] detect push task req_id 97, obj num 15, queue size 0
[info] ===> detect thread EOS received.
[info] <=== detect thread 0 completed. 98 sampels tested.
[debug] classify thread 0 run sample 97 obj 4 end. cost 6.363 ms.
sample 97 box[1180, 153, 1231, 204], det[conf:0.717143, cls:2], cls[id:482, conf:11.661688, lable:[cassette player]]
[debug] classify thread 1 run sample 96 obj 14 end. cost 7.514 ms.
sample 96 box[1069, 137, 1109, 165], det[conf:0.261245, cls:2], cls[id:465, conf:11.904639, lable:[bulletproof vest]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 1 completed. 48 sampels tested.
[debug] classify thread 0 run sample 97 obj 12 end. cost 6.775 ms.
sample 97 box[1110, 140, 1157, 179], det[conf:0.424432, cls:2], cls[id:482, conf:12.512019, lable:[cassette player]]
[debug] classify thread 0 run sample 97 obj 13 end. cost 6.529 ms.
sample 97 box[799, 133, 855, 167], det[conf:0.406686, cls:2], cls[id:507, conf:10.325453, lable:[combination lock]]
[debug] classify thread 0 run sample 97 obj 14 end. cost 6.575 ms.
sample 97 box[1424, 129, 1490, 184], det[conf:0.284296, cls:2], cls[id:766, conf:10.325453, lable:[rotisserie]]
[info] ===> classify thread EOS received.
[info] <=== classify thread 0 completed. 50 sampels tested.
```

```bash
# Linux aarch64 环境，使用JPEG图像编解码
Count of Input: 2
Input[images] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 3
Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [2457600,819200,10240,128,1], dtype: INT8, format: ND, size: 2457557, memsize: 2457600, quant_info: size=1 [0, 0.00780907]
Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [614400,204800,5120,128,1], dtype: INT8, format: ND, size: 614357, memsize: 614400, quant_info: size=1 [0, 0.00783619]
Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [153600,51200,2560,128,1], dtype: INT8, format: ND, size: 153557, memsize: 153600, quant_info: size=1 [0, 0.00785193]
I20250922 15:57:03.958438 40282 jpeg_detect.cc:43] thread 0 model loaded: yolov5s.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20250922 15:57:04.181274 40282 jpeg_detect.cc:56] thread 0 model loaded: resnet50.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20250922 15:57:04.247911 40282 jpeg_detect.cc:56] thread 1 model loaded: resnet50.hmm.
I20250922 15:57:04.258720 40297 detect_frames.cc:214] classify thread, module 0 infer start...
I20250922 15:57:04.258725 40290 detect_frames.cc:20] detect thread, moudle 0 infer start...
I20250922 15:57:04.258728 40298 detect_frames.cc:214] classify thread, module 1 infer start...
I20250922 15:57:04.314933 40309 jpeg_codec.cc:5] ===> EncodeImage thread start, ready to push 20 images.
I20250922 15:57:04.355386 40309 jpeg_codec.h:91] get encoded data 1, stamp 916, planes 1, length 39794.
I20250922 15:57:04.355801 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 0, queue size 0.
I20250922 15:57:04.355920 40310 jpeg_codec.h:176] Jpeg decoder 368376140688 push data, size: 39794.
I20250922 15:57:04.358020 40309 jpeg_codec.h:91] get encoded data 2, stamp 3549, planes 1, length 39794.
I20250922 15:57:04.358870 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 1, queue size 0.
I20250922 15:57:04.361510 40309 jpeg_codec.h:91] get encoded data 3, stamp 7040, planes 1, length 39794.
I20250922 15:57:04.362420 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 2, queue size 1.
I20250922 15:57:04.365075 40309 jpeg_codec.h:91] get encoded data 4, stamp 10604, planes 1, length 39794.
I20250922 15:57:04.366015 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 3, queue size 2.
I20250922 15:57:04.367726 40310 jpeg_codec.h:182] Jpeg decoder 368376140688 pull data.
I20250922 15:57:04.368690 40309 jpeg_codec.h:91] get encoded data 5, stamp 14219, planes 1, length 39794.
I20250922 15:57:04.369858 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 4, queue size 3.
I20250922 15:57:04.371160 40310 jpeg_codec.h:217] get frame 1, stamp 3111, planes 2, total length 410880.
-->> decoded yuv info:TensorInfo: shape: [3,426,640], stride: [640,1,640,2,1,272640], dtype: UINT8, format: YUV420SP, size: 408960, memsize: 408960
I20250922 15:57:04.372662 40309 jpeg_codec.h:91] get encoded data 6, stamp 18191, planes 1, length 39794.
I20250922 15:57:04.373657 40309 jpeg_codec.cc:62] Jpeg encoder push encoded data 5, queue size 4.
I20250922 15:57:04.374217 40310 jpeg_codec.cc:174] -->> resized yuv info: height=1080, width=1920.
......
I20250922 15:56:07.653195 39610 detect_frames.cc:292] classify thread 0 run sample 19 obj 12 end. cost 3.043 ms.
I20250922 15:56:07.653901 39610 detect_frames.cc:300] sample 19 box[1033, 444, 1104, 563], det[conf:0.3794272, cls:58], cls[id:741, conf:6.3167477, lable:[prayer rug, prayer mat]]
I20250922 15:56:07.657508 39610 detect_frames.cc:292] classify thread 0 run sample 19 obj 13 end. cost 2.956 ms.
I20250922 15:56:07.657695 39610 detect_frames.cc:300] sample 19 box[1369, 924, 1919, 1071], det[conf:0.28132197, cls:60], cls[id:626, conf:7.410031, lable:[lighter, light, igniter, ignitor]]
I20250922 15:56:07.660861 39610 detect_frames.cc:292] classify thread 0 run sample 19 obj 14 end. cost 3.093 ms.
I20250922 15:56:07.661239 39610 detect_frames.cc:300] sample 19 box[999, 467, 1050, 552], det[conf:0.2585618, cls:58], cls[id:885, conf:5.34494, lable:[velvet]]
I20250922 15:56:07.661484 39610 detect_frames.cc:225] ===> classify thread EOS received.
I20250922 15:56:07.661882 39610 detect_frames.cc:309] <=== classify thread 0 completed. 10 sampels tested.
```

检测效果图片保存在demo_results目录下。

### 3. 一键执行

可以使用run脚本一键执行并查看结果，run脚本中不提供依赖环境安装。

linux下执行：

```bash
# (默认) 使用后摩视频解码+开启Resize视频解码输出功能
bash run.sh
# 使用后摩视频解码+关闭Resize视频解码输出功能
bash run.sh --disable_resizer
# 使用RK视频解码（主机上需已安装mpp库）
bash run.sh --enable_rk_decoder
# 运行JPEG图像编解码示例
bash run.sh --enable_jpeg
```
