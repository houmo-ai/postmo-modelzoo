# houmo-examples

## 目录

[TOC]

## 概述

houmo-examples是为用户快速将模型和应用移植到后摩芯片上而提供的示例库，为用户提供量化、编译、精度和性能评估、应用部署等一整套代码和工具，降低用户的学习和开发成本。

目前仅支持linux平台。

houmo-examples目录结果如下，其中README.md为本说明文件：

```bash
|-- README.md
|-- apis
|-- models
|-- data
|-- hmodel
`-- env.sh
```

主要目录和文件说明如下：

| 目录     | 说明                                        |
| -------- | ------------------------------------------- |
| apis     | API示例，接口按主要用途分文件夹存放         |
| models   | 模型示例，模型按照主要用途分文件夹存放      |
| data     | 评估使用的数据文件，如数据集等              |
| hmodel   | 量化模型配置和工具，主要用于大模型和QAT训练 |
| env.sh   | 环境配置脚本                                |


## 软件依赖

示例中使用了一些第三方库实现程序编译、图像和数据处理、结果显示等功能，需要安装第三方软件，请自行安装。
以下为公共依赖，每个示例可能有其他依赖，请参考各示例的README.md文件。

- CMake（建议3.16.3以上版本），主要用于tcim_perf工具编译
  - linux下可通过apt等包管理工具直接安装

python依赖可通过requirements.txt安装：

```bash
pip install -r requirements.txt
```

## API示例

API示例列表如下，type列为示例类型，其中convert表示模型转换，inference表示模型推理，scenes表示应用场景。language列为支持的编程语言，target列为支持的芯片平台。

量化和编译示例仅支持在量化工具和编译器支持的平台上运行，部署示例支持情况还受到其他交付件的限制，如固件类型（如芯片解码的示例只能在VPU固件上运行，大模型只能在非VPU固件上运行）、硬件规格（如2核芯片只能运行2核以下编译的模型），具体请查看示例内readme文件。

| example name                 | path                                        | type      | language   | target  | platform    |
| ---------------------------- | ------------------------------------------- | --------- | ---------- | ------- | ----------- |
| resnet50量化编译             | apis/converts/resnet50                      | convert   | python     | xh1/xh2 | x64         |
| resnet50单线程推理           | apis/inferences/resnet50                    | inference | python/c++ | xh1/xh2 | x64/aarch64 |
| yolov5s单线程推理            | apis/inferences/yolov5s                     | inference | python/c++ | xh1/xh2 | x64/aarch64 |
| resnet50多线程多stream推理   | apis/inferences/resnet50_multistreams       | inference | c++        | xh1     | x64/aarch64 |
| resnet50流水推理             | apis/inferences/resnet50_pipeline           | inference | c++        | xh1     | x64/aarch64 |
| yolov5s_resnet50多batch推理  | apis/inferences/yolov5s_resnet50_multibatch | inference | c++        | xh1     | x64/aarch64 |
| 视频流目标检测分析           | apis/scenes/video_detect                    | scenes    | c++        | xh1     | x64/aarch64 |

**注：由于xh2暂不支持resizer，以上所有c++推理示例暂不支持xh2平台**

## 模型示例

模型示例主要依赖hmatc工具完成评估功能，可通过每个模型示例下的test.sh脚本一键执行，也可参考脚本中的命令分步执行，相关参数在config.yml配置。

模型示例列表如下，type列为模型类型，target列为支持的芯片平台，quant表示提供量化示例，build表示提供编译示例，demo表示提供python端到端demo，eval表示提供精度评估。

量化和编译功能仅支持在量化工具和编译器支持的平台上运行，其中大模型量化需要使用GPU。涉及到模型推理相关的功能（如perf/demo/eval等）最好使用后摩芯片平台运行，运行时需要关注其他限制，如固件类型（如大模型只能在非VPU固件上运行），硬件规格（如2核芯片只能运行2核以下编译的模型）。如果没有安装后摩芯片可以通过`export HDPL_PLATFORM=ISIM`指定模拟器运行，速度较慢。

| models               | path                            | type      | target  | quant | build | demo | eval |
| -------------------- | ------------------------------- | --------- | ------- | ----- | ----- | ---- | ---- |
| resnet50             | models/backbone/resnet50        | backbone  | xh1/xh2 | yes   | yes   | yes  | yes  |
| mobilenetv2          | models/backbone/mobilenet_v2    | backbone  | xh1/xh2 | yes   | yes   | yes  | yes  |
| efficientnet         | models/backbone/efficientnet    | backbone  | xh1     | yes   | yes   | yes  | yes  |
| yolov8m              | models/detection/yolov8m        | detection | xh1/xh2 | yes   | yes   | yes  | yes  |
| yolov5s              | models/detection/yolov5s        | detection | xh1/xh2 | yes   | yes   | yes  | yes  |
| yolov3               | models/detection/yolov3         | detection | xh1     | yes   | yes   | yes  | yes  |
| yolop                | models/autodrive/yolop          | autodrive | xh1     | yes   | yes   | yes  | x    |
| wenet                | models/asr/wenet                | asr       | xh1     | x     | yes   | yes  | x    |
| qwen2.5              | models/llm/qwen2.5              | llm       | xh1/xh2 | yes   | yes   | yes  | x    |
| qwen3-8b             | models/llm/qwen3                | llm       | xh1/xh2 | yes   | yes   | yes  | x    |
| qwen3-14b            | models/llm/qwen3-14b            | llm       | xh2     | yes   | yes   | yes  | x    |
| deepseek             | models/llm/deepseek             | llm       | xh1     | x     | yes   | yes  | x    |
| deepseek-r1-0528     | models/llm/deepseek-r1-qwen3-8b | llm       | xh2     | yes   | yes   | yes  | x    |
| sdxl                 | models/diffusion/sdxl           | diffusion | xh1     | x     | yes   | yes  | x    |
| sd3                  | models/diffusion/sd3            | diffusion | xh2     | x     | yes   | yes  | x    |
| qwen2.5-vl           | models/vllm/qwen2.5-vl          | vllm      | xh1     | x     | yes   | yes  | x    |
