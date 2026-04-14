# Resnet50 Example

## 目录

[TOC]

## 概述

TCIM Examples 以 resnet50 模型为例展示 TCIM API 推理的基本使用方式。

1. resnet50.py 展示 python 下单线程推理过程，包括：
    - python 单线程推理，使用默认 stream（无需指定）
2. resnet50.cc 展示 c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认 stream（无需指定）

## 软件依赖

除 TCIM Examples 公用依赖外，本示例没有其他依赖。

## 快速上手

### 获取芯片模型

可以通过以下两种方式获取芯片模型：

1. 自行编译：通过原始模型或量化模型，经过 tcim 工具链编译生成，具体可参考 houmo-examples
2. 直接下载：通过 get_model.py 脚本下载预编译的模型

```bash
python3 get_model.py
```

### 模型 python 推理

使用 runtime python API 进行模型推理，并输出结果：

```bash
python3 resnet50.py
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
2026-04-14 06:49:19.504 | INFO     | __main__:<module>:41 - ===> resnet50 python example start...
2026-04-14 06:49:19.507 | INFO     | __main__:<module>:58 - input image shape: (375, 500, 3)
2026-04-14 06:49:19.507 | INFO     | __main__:<module>:62 - Found model file: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
2026-04-14 06:49:19.944 | INFO     | __main__:<module>:71 - Model input info:
2026-04-14 06:49:19.945 | INFO     | __main__:<module>:76 -   Input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, mem_size = 393216, stride = [393216, 512, 1, 393216, 512, 2, 1, 262144], format = YUV420SP
2026-04-14 06:49:19.945 | INFO     | __main__:<module>:76 -   Input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, mem_size = 40, stride = [40, 4], format = ND
2026-04-14 06:49:19.945 | INFO     | __main__:<module>:100 - pad input image to (512, 512, 3) height = 512, width = 512
2026-04-14 06:49:19.957 | INFO     | __main__:<module>:144 - input_data shape: (1, 3, 512, 512), dtype: uint8, dyn_info: [[  0   0 374 500 224 224   0   0   0   0]]
2026-04-14 06:49:19.959 | INFO     | __main__:<module>:168 - output[495] shape = [1, 1000], dtype = <class 'numpy.float32'>, format = ND
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:181 - top1: predict cls = 65, prob = 0.753624
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:181 - top2: predict cls = 58, prob = 0.118313
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:181 - top3: predict cls = 54, prob = 0.043525
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:181 - top4: predict cls = 62, prob = 0.029681
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:181 - top5: predict cls = 67, prob = 0.015641
2026-04-14 06:49:19.960 | INFO     | __main__:<module>:187 - <=== resnet50 python example completed.
```

### 模型 c++推理

使用 runtime c++ API 进行模型推理，并输出结果：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
I20260414 08:32:55.240370 462441 resnet50.cc:63] ===> resnet50 c++ example start...
I20260414 08:32:55.241035 462441 resnet50.cc:82] Found .hmm file: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 08:32:55.244415 462441 resnet50.cc:104] input image shape: [375 x 500 x 3]
I20260414 08:32:55.244419 462441 resnet50.cc:108] Load resnet50 model from file ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 08:32:55.685303 462441 resnet50.cc:114] Model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm loaded.
I20260414 08:32:55.685339 462441 resnet50.cc:125] Model input info:
I20260414 08:32:55.685402 462441 resnet50.cc:130]   Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:32:55.685428 462441 resnet50.cc:130]   Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:32:55.685637 462441 resnet50.cc:154] pad input image to [512 x 512 x 3], height=512, width=512
I20260414 08:32:55.693682 462441 resnet50.cc:244] dyn_info: [0, 0, 374, 500, 224, 224, 0, 0, 0, 0]
I20260414 08:32:55.695903 462441 resnet50.cc:260] output[495] info: TensorInfo: shape: [1,1000], stride: [2000,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
I20260414 08:32:55.696690 462441 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 08:32:55.696707 462441 utils.hpp:259] top2: Index=58 Conf=0.19954717, Label=[water snake]
I20260414 08:32:55.696725 462441 utils.hpp:259] top3: Index=54 Conf=0.02808156, Label=[hognose snake, puff adder, sand viper]
I20260414 08:32:55.696742 462441 utils.hpp:259] top4: Index=62 Conf=0.015752329, Label=[rock python, rock snake, Python sebae]
I20260414 08:32:55.696761 462441 utils.hpp:259] top5: Index=67 Conf=0.012267925, Label=[diamondback, diamondback rattlesnake, Crotalus adamanteus]
I20260414 08:32:55.696787 462441 resnet50.cc:281] <=== resnet50 c++ example completed.
```

**说明：由于 c++和 python 前处理有差异，部分结果可能存在少量差异。**

### 一键执行

可以使用 run 脚本一键执行查看结果

linux 下执行：

```bash
bash run.sh
```

windows 下执行：

```bash
bash run.bat
```
