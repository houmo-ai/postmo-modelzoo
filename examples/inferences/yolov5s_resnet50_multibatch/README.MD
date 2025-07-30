# MultiBatch Example

## 目录

[TOC]

## 概述

MultiBatch Example以yolov5s模型和resnet50模型为例，展示TCIM API多batch动态resizer的两种使用方式：

1. yolov5s模型展示了N-N多batch动态resizer使用方式，即模型的图像输入batch数量为N，对应的resizer_crop输入batch数量也为N。

2. resnet50模型展示了1-N多batch动态resizer使用方式，即模型的图像输入batch数量为1，对应的resizer_crop输入batch数量为N。以resnet50为例说明，其resizer_crop输入batch为4表示一次推理可配置4组resizer_crop信息，实现对4个ROI分类的效果。
    - 基于resnet50模型，提供两种resize图片的示例方法：等比例缩放ROI，非等比例缩放ROI。示例程序中默认使用等比例缩放，示例程序支持通过入参设置resize图片类型:
        - `0`: (默认)代表等比例缩放ROI。
        - `1`: 代表非等比例缩放ROI。

当前示例仅支持Linux系统。

## 软件依赖

本示例依赖opencv库进行图像预处理和可视化。

## 快速开始

1. 获取模型文件及示例输入数据，执行：

```bash
python3 get_model.py
```

2. 编译并执行c++演示程序，执行命令：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
# 等比例缩放ROI
./multibatch_example
# 非等比例缩放ROI
./multibatch_example 1
```

## 参考结果

### C++示例结果

C++演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
[info] ===> multibatch c++ example start, tcim version: 2.4.2 Jul 20 2025
[info] ===> multibatch yolov5s example start...
[info] Load yolov5s hmm model from file yolov5s_xh1_b4_1core_O2.hmm
[info] Loaded yolov5s model yolov5s_xh1_b4_1core_O2.hmm
[info] Count of Input: 2
[info] Input[images] TensorInfo: shape: [4,3,1080,1920], stride: [3110400,1920,1,3110400,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 12441600, memsize: 12441600
[info] Input[resizer_crop_images] TensorInfo: shape: [4,10], stride: [], dtype: INT32, format: ND, size: 160, memsize: 160, quant_info: size=1 [0, 1]
[info] Count of Output: 3
[info] Output[340] TensorInfo: shape: [4,3,80,80,85], stride: [], dtype: FLOAT32, format: ND, size: 26112000, memsize: 26112000
[info] Output[378] TensorInfo: shape: [4,3,40,40,85], stride: [], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
[info] Output[416] TensorInfo: shape: [4,3,20,20,85], stride: [], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
[info] detection results, batch 0, detections num: 16
box[8, 426, 481, 672], conf:0.891797, cls:62
box[875, 554, 1063, 808], conf:0.820147, cls:56
box[1238, 397, 1395, 758], conf:0.794059, cls:0
box[1674, 528, 1919, 740], conf:0.777099, cls:62
box[1321, 420, 1544, 744], conf:0.628656, cls:72
box[1082, 552, 1270, 806], conf:0.623837, cls:56
box[1651, 756, 1760, 1015], conf:0.620479, cls:75
box[1225, 556, 1329, 768], conf:0.601326, cls:56
box[694, 448, 804, 538], conf:0.578198, cls:58
box[1344, 303, 1386, 358], conf:0.533498, cls:74
box[1151, 435, 1211, 543], conf:0.489449, cls:0
box[1471, 442, 1541, 717], conf:0.462559, cls:72
box[997, 449, 1097, 561], conf:0.448309, cls:58
box[1372, 914, 1919, 1079], conf:0.417810, cls:60
box[1008, 529, 1044, 584], conf:0.280216, cls:75
box[999, 463, 1053, 554], conf:0.253529, cls:58
[info] detection results, batch 0 saved to ./demo_results/yolov5s_output_batch0.jpg
[info] detection results, batch 1, detections num: 1
box[47, 84, 1856, 1071], conf:0.687419, cls:21
[info] detection results, batch 1 saved to ./demo_results/yolov5s_output_batch1.jpg
[info] detection results, batch 2, detections num: 9
box[0, 622, 1217, 1079], conf:0.853730, cls:59
box[1008, 482, 1287, 781], conf:0.807163, cls:58
box[556, 276, 722, 516], conf:0.797923, cls:58
box[762, 525, 1008, 703], conf:0.386822, cls:56
box[289, 433, 341, 512], conf:0.380541, cls:39
box[1572, 325, 1657, 404], conf:0.358054, cls:73
box[1461, 336, 1605, 395], conf:0.305474, cls:73
box[1422, 334, 1474, 404], conf:0.295784, cls:73
box[1491, 551, 1566, 648], conf:0.266749, cls:73
[info] detection results, batch 2 saved to ./demo_results/yolov5s_output_batch2.jpg
[info] detection results, batch 3, detections num: 1
box[611, 143, 1306, 508], conf:0.894014, cls:11
[info] detection results, batch 3 saved to ./demo_results/yolov5s_output_batch3.jpg
[info] <=== multibatch yolov5s example end.
[info] ===> multibatch resnet50 example start...
[info] Load resnet50 hmm model from file resnet50_roi4_b1_xh1_O2.hmm
[info] Loaded resnet50 model resnet50_roi4_b1_xh1_O2.hmm
[info] Count of Input: 2
[info] Input[input.1] TensorInfo: shape: [1,3,1080,1920], stride: [3110400,1920,1,3110400,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
[info] Input[resizer_crop_input.1] TensorInfo: shape: [4,10], stride: [], dtype: INT32, format: ND, size: 160, memsize: 160, quant_info: size=1 [0, 1]
[info] Count of Output: 1
[info] Output[495] TensorInfo: shape: [4,1000], stride: [], dtype: FLOAT32, format: ND, size: 16000, memsize: 16000
[info] batch 0, detections size:16
batch[0] detection[0] box[8, 426, 481, 672], det[conf:0.891797, cls:62], cls[id:598], conf:7.894640, lable:[home theater, home theatre]]
batch[0] detection[1] box[875, 554, 1063, 808], det[conf:0.820147, cls:56], cls[id:532], conf:7.031164, lable:[dining table, board]]
batch[0] detection[2] box[1238, 397, 1395, 758], det[conf:0.794059, cls:0], cls[id:421], conf:6.537749, lable:[bannister, banister, balustrade, balusters, handrail]]
batch[0] detection[3] box[1674, 528, 1919, 740], det[conf:0.777099, cls:62], cls[id:766], conf:8.264702, lable:[rotisserie]]
batch[0] detection[4] box[1321, 420, 1544, 744], det[conf:0.628656, cls:72], cls[id:799], conf:7.647933, lable:[sliding door]]
batch[0] detection[5] box[1082, 552, 1270, 806], det[conf:0.623837, cls:56], cls[id:532], conf:10.608423, lable:[dining table, board]]
batch[0] detection[6] box[1651, 756, 1760, 1015], det[conf:0.620479, cls:75], cls[id:443], conf:9.374886, lable:[bib]]
batch[0] detection[7] box[1225, 556, 1329, 768], det[conf:0.601326, cls:56], cls[id:905], conf:7.401226, lable:[window shade]]
batch[0] detection[8] box[694, 448, 804, 538], det[conf:0.578198, cls:58], cls[id:921], conf:7.771286, lable:[book jacket, dust cover, dust jacket, dust wrapper]]
batch[0] detection[9] box[1344, 303, 1386, 358], det[conf:0.533498, cls:74], cls[id:126], conf:7.524579, lable:[isopod]]
batch[0] detection[10] box[1151, 435, 1211, 543], det[conf:0.489449, cls:0], cls[id:921], conf:4.440735, lable:[book jacket, dust cover, dust jacket, dust wrapper]]
batch[0] detection[11] box[1471, 442, 1541, 717], det[conf:0.462559, cls:72], cls[id:760], conf:9.621593, lable:[refrigerator, icebox]]
batch[0] detection[12] box[997, 449, 1097, 561], det[conf:0.448309, cls:58], cls[id:602], conf:6.784457, lable:[horizontal bar, high bar]]
batch[0] detection[13] box[1372, 914, 1919, 1079], det[conf:0.417810, cls:60], cls[id:711], conf:8.634763, lable:[perfume, essence]]
batch[0] detection[14] box[1008, 529, 1044, 584], det[conf:0.280216, cls:75], cls[id:729], conf:5.427565, lable:[plate rack]]
batch[0] detection[15] box[999, 463, 1053, 554], det[conf:0.253529, cls:58], cls[id:912], conf:3.700613, lable:[worm fence, snake fence, snake-rail fence, Virginia fence]]
[info] batch 1, detections size:1
batch[1] detection[0] box[47, 84, 1856, 1071], det[conf:0.687419, cls:21], cls[id:294], conf:15.665927, lable:[brown bear, bruin, Ursus arctos]]
[info] batch 2, detections size:9
batch[2] detection[0] box[0, 622, 1217, 1079], det[conf:0.853730, cls:59], cls[id:797], conf:14.802451, lable:[sleeping bag]]
batch[2] detection[1] box[1008, 482, 1287, 781], det[conf:0.807163, cls:58], cls[id:799], conf:7.524579, lable:[sliding door]]
batch[2] detection[2] box[556, 276, 722, 516], det[conf:0.797923, cls:58], cls[id:738], conf:6.044334, lable:[pot, flowerpot]]
batch[2] detection[3] box[762, 525, 1008, 703], det[conf:0.386822, cls:56], cls[id:465], conf:7.401226, lable:[bulletproof vest]]
batch[2] detection[4] box[289, 433, 341, 512], det[conf:0.380541, cls:39], cls[id:480], conf:6.167688, lable:[cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM]]
batch[2] detection[5] box[1572, 325, 1657, 404], det[conf:0.358054, cls:73], cls[id:421], conf:7.771286, lable:[bannister, banister, balustrade, balusters, handrail]]
batch[2] detection[6] box[1461, 336, 1605, 395], det[conf:0.305474, cls:73], cls[id:872], conf:6.414395, lable:[tripod]]
batch[2] detection[7] box[1422, 334, 1474, 404], det[conf:0.295784, cls:73], cls[id:593], conf:8.017994, lable:[harmonica, mouth organ, harp, mouth harp]]
batch[2] detection[8] box[1491, 551, 1566, 648], det[conf:0.266749, cls:73], cls[id:906], conf:7.401226, lable:[Windsor tie]]
[info] batch 3, detections size:1
batch[3] detection[0] box[611, 143, 1306, 508], det[conf:0.894014, cls:11], cls[id:919], conf:13.692267, lable:[street sign]]
[info] <=== multibatch resnet50 example end.
[info] <=== multibatch c++ example end.
```

yolov5s的多Batch检测效果图片保存在./demo_results/目录下。


### 一键执行

可以使用run脚本一键执行查看结果

Linux下执行：

```bash
# 等比例缩放ROI
bash run.sh
# 非等比例缩放ROI
bash run.sh 1
```
