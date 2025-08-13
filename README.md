# houmo-examples

## 目录

[TOC]

## 概述

houmo-examples是为用户快速将模型和应用移植到后摩芯片上而提供的示例库，为用户提供量化、编译、精度和性能评估、应用部署等一整套代码和工具，降低用户的学习和开发成本。

模型示例仅支持linux平台，API示例支持linux和windows平台，支持情况还取决于每个示例依赖的功能在各平台的支持情况，具体请参考每个示例的readme文件。

houmo-examples目录结构如下，其中README.md为本说明文件：

```bash
.
├── README.md
├── apis
    ├── common
    ├── converts
    ├── inferences
    └── scenes
├── models
    ├── asr
    ├── autodrive
    ├── backbone
    ├── detection
    ├── diffusion
    └── llm
├── data
├── hmodel
├── env.sh
└── requirements.txt
```

主要目录和文件说明如下：

| 目录             | 说明                                        |
| ---------------- | ------------------------------------------- |
| models           | 模型示例，展示模型的转换和评估过程          |
| apis             | API示例，展示hal和runtime接口的调用过程     |
| data             | 评估使用的数据文件，如数据集等              |
| hmodel           | 量化模型配置和工具，主要用于大模型和QAT训练 |
| env.sh           | 环境配置脚本                                |
| requirements.txt | python三方依赖                              |


## 软件依赖

示例中使用了一些第三方库实现程序编译、图像和数据处理、结果显示等功能，需要安装第三方软件，请自行安装。
以下为公共依赖，每个示例可能有其他依赖，请参考各示例的README.md文件。

- Python（3.9以上版本，需要和houmo-tcim-runtime支持的版本一致）
  - windows下将python可执行程序目录设置为环境变量PYTHON_DIR
- CMake（建议3.16.3以上版本），主要用于c++示例编译
  - linux下可通过apt等包管理工具直接安装
  - windows下可下载安装包安装，将bin目录设置为环境变量CMAKE_DIR
- OpenCV库（4.x版本），主要用于c++示例图像读取和处理，结果渲染显示
  - linux下可通过apt等包管理工具直接安装
  - windows下可下载安装包解压，将安装目录设置为环境变量OPENCV_DIR（目录下存在OpenCVConfig.cmake）并将dll目录加入PATH

python依赖可通过requirements.txt安装：

```bash
pip install -r requirements.txt
```

此外，示例运行需要依赖houmo-tcim-runtime，参考后摩大道软件平台快速入门配置runtime环境。


## 模型示例

模型示例主要依赖hmatc工具完成评估功能，可通过每个模型示例下的test.sh脚本一键执行，也可参考脚本中的命令分步执行，相关参数在config.yml配置。

模型示例列表如下，type列为模型类型，target列为支持的芯片平台，quant表示提供量化示例，build表示提供编译示例，demo表示提供python端到端demo，eval表示提供精度评估。

量化和编译功能仅支持在量化工具和编译器支持的平台上运行，其中大模型量化需要使用GPU。涉及到模型推理相关的功能（如perf/demo/eval等）最好使用后摩芯片平台运行，运行时需要关注其他限制，如固件类型（如大模型只能在非VPU固件上运行），硬件规格（如2核芯片只能运行2核以下编译的模型）。如果没有安装后摩芯片可以通过`export HDPL_PLATFORM=ISIM`指定模拟器运行，速度较慢。

| models               | path        | target  | quant | build | demo | eval |
| -------------------- | ----------- | ------- | ----- | ----- | ---- | ---- |
| resnet50             | backbone    | xh1     | yes   | yes   | yes  | yes  |
| mobilenetv2          | backbone    | xh1     | yes   | yes   | yes  | yes  |
| efficientnet         | backbone    | xh1     | yes   | yes   | yes  | yes  |
| yolov3               | detection   | xh1     | yes   | yes   | yes  | yes  |
| yolov5s              | detection   | xh1     | yes   | yes   | yes  | yes  |
| yolov8m              | detection   | xh1     | yes   | yes   | yes  | yes  |
| yolop                | autodrive   | xh1     | yes   | yes   | yes  | x    |
| wenet                | asr         | xh1     | x     | yes   | yes  | x    |
| qwen2.5              | llm         | xh1/xh2 | yes   | yes   | yes  | x    |
| qwen3                | llm         | xh1/xh2 | yes   | yes   | yes  | x    |
| qwen3-14b            | llm         | xh2     | yes   | yes   | yes  | x    |
| deepseek             | llm         | xh1     | x     | yes   | yes  | x    |
| deepseek-r1-qwen3-8b | llm         | xh2     | yes   | yes   | yes  | x    |
| sdxl                 | diffusion   | xh1     | x     | yes   | yes  | x    |
| qwen2.5-vl           | vllm        | xh1     | x     | yes   | yes  | x    |


## API示例

API示例在apis目录下，如下表所示，type列为示例类型，其中convert表示模型转换，inference表示模型推理，scenes表示应用场景。language列为支持的编程语言，target列为支持的芯片平台。

量化和编译示例仅支持在量化工具和编译器支持的平台上运行，部署示例支持情况还受到其他交付件的限制，如固件类型（如芯片解码的示例只能在VPU固件上运行，大模型只能在非VPU固件上运行）、硬件规格（如2核芯片只能运行2核以下编译的模型），具体请查看示例内readme文件。

| examples                     | path         | description                     | language   | target  | platform    |
| ---------------------------- | ------------ | ------------------------------- | ---------- | ------- | ----------- |
| resnet50                     | converts     | resnet50 量化编译示例           | python     | xh1/xh2 | x64         |
| resnet50                     | inferences   | resnet50 单线程推理示例         | python/c++ | xh1/xh2 | x64/aarch64 |
| yolov5s                      | inferences   | yolov5s 单线程推理示例          | python/c++ | xh1/xh2 | x64/aarch64 |
| qwen3                        | inferences   | qwen3 大语言模型推理示例        | python     | xh1/xh2 | x64/aarch64 |
| resnet50_multistreams        | inferences   | resnet50 多线程多stream推理示例 | c++        | xh1     | x64/aarch64 |
| resnet50_pipeline            | inferences   | resnet50 流水推理示例           | c++        | xh1     | x64/aarch64 |
| yolov5s_resnet50_multibatch  | inferences   | yolov5s_resnet50多batch推理示例 | c++        | xh1     | x64/aarch64 |
| video_detect                 | scenes       | 视频流目标检测分析业务示例      | c++        | xh1     | x64/aarch64 |

**注：由于xh2暂不支持resizer，以上所有c++推理示例暂不支持xh2平台**


## 快速上手

参考各示例的README.md文件。
