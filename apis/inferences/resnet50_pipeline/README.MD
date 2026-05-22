# Resnet50 Pipeline Example

## 目录

[TOC]

## 概述

本示例以 resnet50 模型为例展示流水线（pipeline）推理的使用方式。流水线方式可以并行处理数据的拷入、拷出和推理过程，提升单线程下的推理吞吐。

此种方式只支持 c++。

resnet50.cc 展示 c++下 pipeline 推理过程，包括：

-   h2d 线程将 host 侧输入数据拷贝到 device 侧
-   infer 线程设置 device 侧输入和输出，然后执行推理
-   d2h 线程将 device 侧输出数据拷贝到 host 侧

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

### 模型 c++推理

使用 runtime c++ API 进行模型推理，并输出结果。执行时可通过参数修改配置：

-   -s 10：测试样本数，默认为 10

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50_pipeline
```

推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
I20260414 09:09:26.819265 499071 resnet50_pipeline.cc:152] ===> resnet50_pipeline c++ example start...
I20260414 09:09:26.820042 499071 resnet50_pipeline.cc:172] Found .hmm file: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 09:09:26.820061 499071 resnet50_pipeline.cc:181] Find resnet50 model file ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 09:09:26.820063 499071 resnet50_pipeline.cc:187] model: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm, samples: 10
I20260414 09:09:27.262700 499071 resnet50_pipeline.cc:203] model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm loaded.
I20260414 09:09:27.262731 499071 resnet50_pipeline.cc:216] Count of Input: 2
I20260414 09:09:27.262782 499071 resnet50_pipeline.cc:221] Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [262144,512,1,131072,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 09:09:27.262812 499071 resnet50_pipeline.cc:221] Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 09:09:27.262825 499071 resnet50_pipeline.cc:241] Count of Output: 1
I20260414 09:09:27.262837 499071 resnet50_pipeline.cc:245] Output[495] info: TensorInfo: shape: [1,1000], stride: [2048,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2048
I20260414 09:09:27.271314 499071 resnet50_pipeline.cc:268] input image shape: [375 x 500 x 3]
I20260414 09:09:27.271510 499071 resnet50_pipeline.cc:281] pad input image to [512 x 512 x 3], height=512, width=512
I20260414 09:09:27.271517 499071 resnet50_pipeline.cc:316] dyn_info: [0, 0, 374, 500, 224, 224, 0, 0, 0, 0]
I20260414 09:09:27.280380 499071 resnet50_pipeline.cc:355] sample queue size is 10.
I20260414 09:09:27.280535 499100 resnet50_pipeline.cc:364] thread H2D start...
I20260414 09:09:27.280554 499101 resnet50_pipeline.cc:459] thread infer start...
I20260414 09:09:27.280574 499102 resnet50_pipeline.cc:417] thread D2H start...
I20260414 09:09:27.282814 499101 resnet50_pipeline.cc:506] run sample 0 end.
I20260414 09:09:27.284649 499101 resnet50_pipeline.cc:506] run sample 1 end.
I20260414 09:09:27.286468 499101 resnet50_pipeline.cc:506] run sample 2 end.
I20260414 09:09:27.288312 499101 resnet50_pipeline.cc:506] run sample 3 end.
I20260414 09:09:27.290164 499101 resnet50_pipeline.cc:506] run sample 4 end.
I20260414 09:09:27.292052 499101 resnet50_pipeline.cc:506] run sample 5 end.
I20260414 09:09:27.293938 499101 resnet50_pipeline.cc:506] run sample 6 end.
I20260414 09:09:27.295879 499101 resnet50_pipeline.cc:506] run sample 7 end.
I20260414 09:09:27.297785 499101 resnet50_pipeline.cc:506] run sample 8 end.
I20260414 09:09:27.299697 499101 resnet50_pipeline.cc:506] run sample 9 end.
I20260414 09:09:27.299716 499101 resnet50_pipeline.cc:509] infer thread completed, 10 samples tested.
I20260414 09:09:27.300420 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.300656 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.300806 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.300935 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301064 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301171 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301261 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301344 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301426 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301503 499071 utils.hpp:259] top1: Index=65 Conf=0.7130048, Label=[sea snake]
I20260414 09:09:27.301514 499071 resnet50_pipeline.cc:556] <=== resnet50_pipeline c++ example completed.
<=== resnet50_pipeline c++ example completed.
```

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
