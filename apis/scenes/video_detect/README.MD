# VideoDetect Example

## 目录

[TOC]

## 概述

本示例展示了两种编解码场景：

- 视频编解码 + yolov5s 检测模型推理 + resnet50 分类模型推理，展示视频检测推理 Pipeline 的基本使用方式。
- JPEG 图像编解码 + yolov5s 检测模型推理 + resnet50 分类模型推理，展示 JPEG 编解码+推理 Pipeline 的基本使用方式。

对于视频解码，本示例提供两种实现方案：

- （默认）后摩视频编解码。在本方案中，提供 Resize 视频解码输出功能: 默认开启，将后摩视频解码输出 resize 为 yolov5s 输入大小 1080x1920，支持的解码输出最大尺寸为 3840x2160。
- Rockchip 视频解码（后文均简称 RK 视频解码）。

本示例默认执行场景：后摩视频解码并 Resize 解码输出，Resize 后图像用于 yolov5s 检测模型推理+resnet50 分类模型推理，并将 yolov5s 检测结果编码为 h264 文件 。如需运行其他场景，可通过一键执行脚本的参数控制，或修改编译参数控制。

本示例仅支持 c++。由于后摩视频解码暂不支持 windows，因此本示例不支持 windows。

## 软件依赖

本示例依赖 opencv 库进行图像预处理和可视化，依赖 ffmpeg 库进行图像取流，需自行安装 ffmpeg 环境。

如视频解码实现方案使用 RK 解码，则需依赖 mpp 库进行视频解码。mpp 库介绍及安装方法见：https://github.com/rockchip-linux/mpp 。
成功安装 mpp 库后，将在`/usr/local/lib`下产生 mpp 相关的 so 库，mpp 库提供的头文件将在`/usr/local/include/rockchip/`中。如果上述路径不存在 mpp 相关资源，则需从 mpp 安装路径拷贝相关资源至`/usr/local/lib`和`/usr/local/include/rockchip/`, 用于 C++演示程序的编译。

## 快速上手

### 1. 获取芯片模型

通过 get_model.py 脚本下载预编译的模型，含 yolov5s 和 resnet50 两个模型。

```bash
python3 get_model.py
```

### 2. 模型推理

#### 2.1 演示程序可修改变量说明

可通过修改 `video_detect.cc` 中 `main` 函数的变量指定运行参数:

- `det_model_path`：检测模型路径
- `cls_model_path`：分类模型路径
- `stream_path`：视频流路径，目前支持 h264/h265 文件
- `codec_thread_num`：编解码器线程数
- `frame_limit`：取视频流的前 n 帧，0 表示所有（ISIM 下推理性能低，默认设置为 2）

可通过控制编译提供的-D 参数`ENABLE_RESIZER`来开关 Resize 视频解码输出功能:

- `-DENABLE_RESIZER=ON`: 开启 Resize 视频解码输出功能
- `-DENABLE_RESIZER=OFF`: 关闭 Resize 视频解码输出功能

可通过控制编译提供的-D 参数`ENABLE_RK_DECODER`来开关 RK 视频解码:

- `-DENABLE_RK_DECODER=ON`: 开启 RK 视频解码（同时会关闭后摩视频解码）
- `-DENABLE_RK_DECODER=OFF`: 关闭 RK 视频解码（同时会开启后摩视频解码）

可通过控制编译提供的-D 参数`ENABLE_JPEG`来开关 JPEG 编解码:

- `-DENABLE_JPEG=ON`: 开启 JPEG 编解码（同时会关闭后摩解码）
- `-DENABLE_JPEG=OFF`: 关闭 JPEG 编解码（同时会开启后摩解码）

可通过控制编译提供的-D 参数`ENABLE_GEN_IMGS`来开关 Demo 结果生成:

- `-DENABLE_GEN_IMGS=ON`: 开启 Demo 结果生成
- `-DENABLE_GEN_IMGS=OFF`: 关闭 Demo 结果生成（同时会关闭后摩编码）

可通过控制编译提供的-D 参数`ENABLE_HM_ENCODER`来开关后摩编码:

- `-DENABLE_HM_ENCODER=ON`: 开启后摩编码 （同时会开启 Demo 结果生成）
- `-DENABLE_HM_ENCODER=OFF`: 关闭后摩编码

可通过控制编译提供的-D 参数`ENABLE_DECODER_ONLY`来开关仅解码:

- `-DENABLE_DECODER_ONLY=ON`: 开启仅解码
- `-DENABLE_DECODER_ONLY=OFF`: 关闭仅解码

_注：无需修改或配置上述变量，程序默认也可运行。_

#### 2.2 编译并执行 c++演示程序

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ -DENABLE_RESIZER=ON -DENABLE_RK_DECODER=OFF -DENABLE_GEN_IMGS=ON -DENABLE_HM_ENCODER=ON -DENABLE_DECODER_ONLY=OFF -DENABLE_JPEG=OFF -DCMAKE_BUILD_TYPE=Release ..
make -j
make install
cd ..
./example_video_detect
```

如果使用 RK 解码方案，则需在执行 C++演示程序前，设置环境变量：
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
I20251020 17:16:11.066434 173269 video_detect.cc:105] thread 0 detection model loaded: yolov5s.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20251020 17:16:11.278972 173269 video_detect.cc:122] thread 0 classify model loaded: resnet50.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20251020 17:16:11.342782 173269 video_detect.cc:122] thread 1 classify model loaded: resnet50.hmm.
I20251020 17:16:11.353951 173284 detect_frames.cc:241] classify thread, module 0 infer start...
I20251020 17:16:11.353952 173285 detect_frames.cc:241] classify thread, module 1 infer start...
I20251020 17:16:11.354134 173277 detect_frames.cc:20] detect thread, moudle 0 infer start...
I20251020 17:16:11.373563 173288 video_codec.cc:257] open stream ../../data/1080P_traffic_4s.h264 completed.
I20251020 17:16:11.381962 173294 video_detect.cc:21] ==> Start to monitor encoding/decoding speed, codec num is 1.
I20251020 17:16:11.382004 173294 video_detect.cc:43] Codec Stats: 0x5562a72910 decoding speed: 0 fps, encoding speed: 0 fps.
I20251020 17:16:11.385533 173292 video_codec.cc:523] ===> PushEncodeStream thread start, size 1920x1080, ysize=2073600.
I20251020 17:16:11.479029 173293 video_codec.cc:617] ===> GetEncodeStream thread start, ffmpeg ctx 0x7f50000b60, frame size is 1920x1080, fps is 30.
I20251020 17:16:11.484971 173288 video_codec.cc:282] frame size is 1920 x 1080.
I20251020 17:16:11.515329 173289 video_codec.cc:397] ===> decoder heigth: 1080, width: 1920.
I20251020 17:16:11.515340 173289 video_codec.h:209] HmCodecDecGetAvailBuf, vir_fd_: 366730476992
I20251020 17:16:11.516169 173293 video_codec.h:70] HmCodecEncGetAvailBuf, vir_fd_: 366730499088
I20251020 17:16:11.517367 173288 video_codec.h:202] HmCodecDecPushStreamData, vir_fd_: 366730476992, size: 251265, last_flag: 0
I20251020 17:16:11.569538 173288 video_codec.h:202] HmCodecDecPushStreamData, vir_fd_: 366730476992, size: 14107, last_flag: 0
I20251020 17:16:11.604949 173288 video_codec.h:202] HmCodecDecPushStreamData, vir_fd_: 366730476992, size: 782, last_flag: 0
I20251020 17:16:11.612495 173289 video_codec.h:253] get frame 1, stamp 97149, planes 2, total len 3110400.
-->> decoded yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
-->> resized yuv info:TensorInfo: shape: [3,1080,1920], stride: [1920,1,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
I20251020 17:16:11.618239 173289 video_codec.cc:495] decoder pull data req_id 0, queue size 0.
......
I20251020 17:16:23.071904 173277 detect_frames.cc:155] detect num: 15
I20251020 17:16:23.075141 173293 video_codec.h:107] get encoded data 80, stamp 11557790, planes 1, length 1191505, ptr 0x7f50227230.
I20251020 17:16:23.108135 173277 detect_frames.cc:191] detect push enc task, req_id 80, queue size 0.
I20251020 17:16:23.108173 173277 detect_frames.cc:210] detect push task req_id 80 obj num 15, queue size 0.
I20251020 17:16:23.108195 173292 video_codec.h:56] HmCodecEncPushYuvData, vir_fd_: 366730499088, last_flag: 0
......
W20251020 17:16:24.489738 173293 video_codec.h:81] Video encoder received EOS frame.
I20251020 17:16:24.489780 173293 video_codec.cc:629] ===> get encoded frame thread EOS received.
I20251020 17:16:24.492479 173285 detect_frames.cc:319] classify thread 1 run sample 99 obj 10 end. cost 3.302 ms.
I20251020 17:16:24.492614 173285 detect_frames.cc:327] sample 99 box[1100, 362, 1252, 519], det[conf:0.52612543, cls:2], cls[id:817, conf:11.054308, lable:[sports car, sport car]]
I20251020 17:16:24.495924 173285 detect_frames.cc:319] classify thread 1 run sample 99 obj 11 end. cost 3.227 ms.
I20251020 17:16:24.496075 173285 detect_frames.cc:327] sample 99 box[1110, 140, 1157, 176], det[conf:0.44882423, cls:2], cls[id:531, conf:10.0825, lable:[digital watch]]
I20251020 17:16:24.499570 173285 detect_frames.cc:319] classify thread 1 run sample 99 obj 12 end. cost 3.426 ms.
I20251020 17:16:24.499730 173285 detect_frames.cc:327] sample 99 box[887, 138, 943, 183], det[conf:0.42260215, cls:2], cls[id:675, conf:10.446928, lable:[moving van]]
I20251020 17:16:24.503567 173285 detect_frames.cc:319] classify thread 1 run sample 99 obj 13 end. cost 3.77 ms.
I20251020 17:16:24.503708 173285 detect_frames.cc:327] sample 99 box[1424, 128, 1489, 185], det[conf:0.32539758, cls:2], cls[id:766, conf:9.110693, lable:[rotisserie]]
I20251020 17:16:24.503764 173285 detect_frames.cc:252] ===> classify thread EOS received.
I20251020 17:16:24.503788 173285 detect_frames.cc:336] <=== classify thread 1 completed. 58 sampels tested.
I20251020 17:16:24.794116 173293 video_codec.h:45] HmCodecEncClose, vir_fd_: 366730499088
I20251020 17:16:24.799573 173293 video_codec.cc:669] All task done, stop stats thread.
I20251020 17:16:24.799629 173293 video_codec.cc:672] <=== GetEncodeStream thread exit. 101 images encoded.
I20251020 17:16:25.383602 173294 video_detect.cc:49] <== End to monitor encoding/decoding speed.
```

```bash
# Linux aarch64 环境，使用RK解码
Count of Input: 2
Input[images] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 3
Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [2457600,819200,10240,128,1], dtype: INT8, format: ND, size: 2457557, memsize: 2457600, quant_info: size=1 [0, 0.00780907]
Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [614400,204800,5120,128,1], dtype: INT8, format: ND, size: 614357, memsize: 614400, quant_info: size=1 [0, 0.00783619]
Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [153600,51200,2560,128,1], dtype: INT8, format: ND, size: 153557, memsize: 153600, quant_info: size=1 [0, 0.00785193]
I20251020 17:22:30.728044 177124 video_detect.cc:105] thread 0 detection model loaded: yolov5s.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20251020 17:22:30.896620 177124 video_detect.cc:122] thread 0 classify model loaded: resnet50.hmm.
Count of Input: 2
Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [2073600,1920,1,1036800,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[dyn_info] TensorInfo: shape: [10], stride: [4], dtype: INT32, format: ND, size: 40, memsize: 40
Count of Output: 1
Output[495] TensorInfo: shape: [1,1000], stride: [1024,1], dtype: INT8, format: ND, size: 1000, memsize: 1024, quant_info: size=1 [0, 0.121476]
I20251020 17:22:30.945210 177124 video_detect.cc:122] thread 1 classify model loaded: resnet50.hmm.
I20251020 17:22:30.950472 177132 detect_frames.cc:20] detect thread, moudle 0 infer start...
I20251020 17:22:30.950495 177133 detect_frames.cc:241] classify thread, module 0 infer start...
I20251020 17:22:30.950511 177134 detect_frames.cc:241] classify thread, module 1 infer start...
I20251020 17:22:30.963369 177138 video_codec.cc:523] ===> PushEncodeStream thread start, size 1920x1080, ysize=2073600.
I20251020 17:22:30.963812 177140 video_detect.cc:21] ==> Start to monitor encoding/decoding speed, codec num is 1.
I20251020 17:22:30.963846 177140 video_detect.cc:43] Codec Stats: 0x55b79516e0 decoding speed: 0 fps, encoding speed: 0 fps.
I20251020 17:22:31.037617 177139 video_codec.cc:617] ===> GetEncodeStream thread start, ffmpeg ctx 0x7f88000b60, frame size is 1920x1080, fps is 30.
......
I20251020 17:22:41.163661 177133 detect_frames.cc:319] classify thread 0 run sample 65 obj 12 end. cost 7.673 ms.
I20251020 17:22:41.164760 177133 detect_frames.cc:327] sample 65 box[1043, 146, 1089, 186], det[conf:0.4502268, cls:2], cls[id:465, conf:11.540212, lable:[bulletproof vest]]
I20251020 17:22:41.233027 177132 detect_frames.cc:191] detect push enc task, req_id 66, queue size 0.
I20251020 17:22:41.233173 177138 video_codec.h:56] HmCodecEncPushYuvData, vir_fd_: 368155355280, last_flag: 0
I20251020 17:22:41.234962 177132 detect_frames.cc:210] detect push task req_id 66 obj num 13, queue size 0.
I20251020 17:22:41.235152 177134 detect_frames.cc:259] classify thread receives task 66, obj num 13.
I20251020 17:22:41.241842 177134 detect_frames.cc:319] classify thread 1 run sample 66 obj 0 end. cost 6.362 ms.
I20251020 17:22:41.242229 177134 detect_frames.cc:327] sample 66 box[153, 642, 582, 966], det[conf:0.9101195, cls:2], cls[id:466, conf:11.904639, lable:[bullet train, bullet]]
......
I20251020 17:22:43.890947 177134 detect_frames.cc:319] classify thread 1 run sample 97 obj 14 end. cost 7.205 ms.
I20251020 17:22:43.891099 177134 detect_frames.cc:327] sample 97 box[1424, 129, 1490, 184], det[conf:0.284296, cls:2], cls[id:766, conf:10.325453, lable:[rotisserie]]
I20251020 17:22:43.903406 177134 detect_frames.cc:252] ===> classify thread EOS received.
I20251020 17:22:43.903438 177134 detect_frames.cc:336] <=== classify thread 1 completed. 48 sampels tested.
I20251020 17:22:43.965303 177140 video_detect.cc:43] Codec Stats: 0x55b79516e0 decoding speed: 0 fps, encoding speed: 12.998544 fps.
I20251020 17:22:44.160198 177139 video_codec.h:45] HmCodecEncClose, vir_fd_: 368155355280
I20251020 17:22:44.164432 177139 video_codec.cc:669] All task done, stop stats thread.
I20251020 17:22:44.164478 177139 video_codec.cc:672] <=== GetEncodeStream thread exit. 99 images encoded.
I20251020 17:22:44.965698 177140 video_detect.cc:49] <== End to monitor encoding/decoding speed.
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

检测效果图片保存在 demo_results 目录下。

### 3. 一键执行

可以使用 run 脚本一键执行并查看结果，run 脚本中不提供依赖环境安装。

linux 下执行：

```bash
# (默认) 使用后摩视频解码+开启Resize视频解码输出功能
bash run.sh
# 使用后摩视频解码+关闭Resize视频解码输出功能
bash run.sh --disable_resizer
# 使用RK视频解码（主机上需已安装mpp库）
bash run.sh --enable_rk_decoder
# 使用RK视频解码（主机上需已安装mpp库）+ 不生成demo结果(关闭后摩编码)
bash run.sh --enable_rk_decoder --disable_gen_imgs
# 运行JPEG图像编解码示例
bash run.sh --enable_jpeg
```
