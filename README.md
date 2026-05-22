# houmo-examples

## 目录

[TOC]

## 概述

houmo-examples 是为用户快速将模型和应用移植到后摩芯片上而提供的示例库，为用户提供量化、编译、精度和性能评估、应用部署等一整套代码和工具，降低用户的学习和开发成本。

模型示例仅支持 linux 平台，API 示例支持 linux 和 windows 平台，支持情况还取决于每个示例依赖的功能在各平台的支持情况，具体请参考每个示例的 readme 文件。

houmo-examples 目录结构如下，其中 README.md 为本说明文件：

```bash
.
├── apis
    ├── common
    ├── converts
    ├── inferences
├── data
├── hmatc
├── hmodel
├── models
    ├── asr
    ├── autodrive
    ├── backbone
    ├── detection
    ├── embedding
    ├── estimation
    ├── llm
    ├── ocr
    ├── omni
    ├── segmentation
    ├── tts
    ├── vlm
    └── benchmark.yml
├── tools
├── env.sh
├── env.bat
├── README.md
└── requirements.txt
```

主要目录和文件说明如下：

| 目录             | 说明                                          |
| ---------------- | --------------------------------------------- |
| apis             | API 示例，展示 hal 和 runtime 接口的调用过程  |
| data             | 评估使用的数据文件，如数据集等                |
| hmatc            | hmatc 工具源码，通过配置文件进行一键评估      |
| hmodel           | 量化模型配置和工具，主要用于大模型和 QAT 训练 |
| models           | 模型示例，展示模型的转换和评估过程            |
| tools            | 应用层工具源码，如算力测试工具等              |
| env.sh/env.bat   | 环境配置脚本                                  |
| requirements.txt | python 三方依赖                               |

## 软件依赖

示例中使用了一些第三方库实现程序编译、图像和数据处理、结果显示等功能，需要安装第三方软件，请自行安装。
以下为公共依赖，每个示例可能有其他依赖，请参考各示例的 README.md 文件。

-   Python（3.9 以上版本，需要和 houmo-tcim-runtime 支持的版本一致）
    -   windows 下将 python 可执行程序目录设置为环境变量 PYTHON_DIR
-   CMake（建议 3.16.3 以上版本），主要用于 c++示例编译
    -   linux 下可通过 apt 等包管理工具直接安装
    -   windows 下可下载安装包安装，将 bin 目录设置为环境变量 CMAKE_DIR
-   OpenCV 库（4.x 版本），主要用于 c++示例图像读取和处理，结果渲染显示
    -   linux 下可通过 apt 等包管理工具直接安装
    -   windows 下可下载安装包解压，将安装目录设置为环境变量 OPENCV_DIR（目录下存在 OpenCVConfig.cmake）并将 dll 目录加入 PATH

python 依赖可通过 requirements.txt 安装：

```bash
pip install -r requirements.txt
```

此外，示例运行需要依赖`houmo-tcim-runtime`以及`houmo-drv`中的推理库，参考后摩大道软件平台快速入门安装推理所需的软件包。
linux环境下运行env.sh配置环境变量，windows环境下请参照 tools/win_envs 目录的 README.MD 进行环境变量的设置。

示例可能用到的环境变量如下：
- HOUMO_TARGET：芯片平台
- HOUMO_VERSION：后摩大道版本，可通过这个环境变量设置下载预编译模型的版本号
- HOUMO_PATH：后摩大道工具链软件安装目录
- HOUMO_SDK_PATH：后摩大道驱动软件安装目录
- TCIM_RUNTIME_PATH：后摩大道runtime软件安装目录
- HOUMO_EXAMPLES_PATH：后摩大道示例目录
- HOUMO_DATASETS_PATH：后摩大道示例所用的数据目录
- HOUMO_MODEL_PATH：后摩大道示例所用的模型目录
- HDPL_PLATFORM：runtime运行平台，可通过这个环境变量设置运行在仿真平台或芯片上
- HF_ENDPOINT：指定 Hugging Face 资源的访问镜像地址，可自行决定设置
- HF_TOKEN：设置 Hugging Face 的身份验证令牌（Token），可自行决定设置

## 应用工具

为方便用户评测，示例仓库提供了一些工具源码供用户使用，如下表所示。具体使用方式请参考工具内的 readme 文件。

| tools          | path  | description            | language | target | arch        | os                  |
| -------------- | ----- | ---------------------- | -------- | ------ | ----------- | ------------------- |
| bandwidth_perf | tools | 带宽测试工具           | python   | xh2    | x64/aarch64 | linux               |
| computing_perf | tools | 算力测试工具           | python   | xh2    | x64/aarch64 | linux               |
| hm_check       | tools | 硬件环境检测工具       | c++      | xh2    | x64/aarch64 | linux/android       |
| hmeval         | tools | 大模型精度测试工具     | python   | xh2    | x64/aarch64 | linux               |
| llm_perf       | tools | 大语言模型性能测试工具 | c++      | xh2    | x64/aarch64 | linux/android/win11 |
| tcim_perf      | tools | 模型测试工具           | c++      | xh2    | x64/aarch64 | linux/android       |
| win_envs       | tools | win 环境设置工具       | python   | xh2    | x64         | win11               |

## 模型示例

模型示例主要依赖 hmatc 工具完成评估功能，可通过每个模型示例下的 test.sh 脚本一键执行，也可参考脚本中的命令分步执行，相关参数在 config.yml 配置。

模型示例列表如下，type 列为模型类型，target 列为支持的芯片平台，quant 表示提供量化示例，build 表示提供编译示例，perf 表示提供性能评估，demo 表示提供 python 端到端 demo，eval 表示提供精度评估。

量化和编译功能仅支持在量化工具和编译器支持的平台上运行，其中大模型量化需要使用 GPU。涉及到模型推理相关的功能（如 perf/demo/eval 等）最好使用后摩芯片平台运行，运行时需要关注其他限制，如固件类型（如大模型只能在非 VPU 固件上运行），硬件规格（如 2 核芯片只能运行 2 核以下编译的模型）。如果没有安装后摩芯片可以通过`export HDPL_PLATFORM=ISIM`指定模拟器运行，速度较慢。

| models                  | path         | target | quant | build | perf | demo | eval |
| ----------------------- | ------------ | ------ | ----- | ----- | ---- | ---- | ---- |
| qwen3-asr               | asr          | xh2    | yes   | yes   | yes  | yes  | x    |
| glm-asr-nano-2512       | asr          | xh2    | yes   | yes   | yes  | yes  | x    |
| sensevoice              | asr          | xh2    | yes   | yes   | yes  | yes  | x    |
| whisper                 | asr          | xh2    | yes   | yes   | yes  | yes  | x    |
| whisper-turbo           | asr          | xh2    | yes   | yes   | yes  | yes  | x    |
| yolop                   | autodrive    | xh2    | yes   | yes   | yes  | yes  | x    |
| efficientnet            | backbone     | xh2    | yes   | yes   | yes  | yes  | yes  |
| mobilenetv2             | backbone     | xh2    | yes   | yes   | yes  | yes  | yes  |
| resnet50                | backbone     | xh2    | yes   | yes   | yes  | yes  | yes  |
| ViT-B-16                | backbone     | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov8m-cls             | backbone     | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolo11m                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolo12m                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolo26m                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov3                  | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov5m_face            | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov5s                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov5s_feature         | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov7                  | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov8m                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov9m                 | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolov10m                | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| yolox                   | detection    | xh2    | yes   | yes   | yes  | yes  | yes  |
| bge                     | embedding    | xh2    | yes   | yes   | yes  | yes  | x    |
| gte                     | embedding    | xh2    | x     | x     | yes  | yes  | x    |
| qwen3-embedding         | embedding    | xh2    | yes   | yes   | yes  | yes  | x    |
| yolov8m-pose            | estimation   | xh2    | yes   | yes   | yes  | yes  | yes  |
| deepseek-r1-qwen3-8b    | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| gpt-oss                 | llm          | xh2    | x     | yes   | yes  | yes  | x    |
| qwen2.5                 | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| qwen3                   | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| qwen3-30b-a3b           | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| qwen3.5                 | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| CoPaw-Flash-9B          | llm          | xh2    | yes   | yes   | yes  | yes  | x    |
| glm-ocr                 | ocr          | xh2    | yes   | yes   | yes  | yes  | x    |
| lprnet                  | ocr          | xh2    | yes   | yes   | yes  | yes  | yes  |
| PPOCRv3                 | ocr          | xh2    | yes   | yes   | yes  | yes  | yes  |
| glm-ocr                 | ocr          | xh2    | yes   | yes   | yes  | yes  | x    |
| minicpmo                | omni         | xh2    | x     | yes   | yes  | yes  | x    |
| qwen3-reranker          | reranker     | xh2    | yes   | yes   | yes  | yes  | x    |
| yolov8m-seg             | segmentation | xh2    | yes   | yes   | yes  | yes  | yes  |
| cosyvoice3              | tts          | xh2    | yes   | yes   | yes  | yes  | x    |
| qwen2.5-vl              | vlm          | xh2    | yes   | yes   | yes  | yes  | x    |
| qwen3-vl                | vlm          | xh2    | yes   | yes   | yes  | yes  | x    |
| gemma4                  | vlm          | xh2    | yes   | yes   | yes  | yes  | x    |

## API 示例

API 示例在 apis 目录下，如下表所示，type 列为示例类型，其中 convert 表示模型转换，inference 表示模型推理。language 列为支持的编程语言，target 列为支持的芯片平台。

量化和编译示例仅支持在量化工具和编译器支持的平台上运行，部署示例支持情况还受到其他交付件的限制，如固件类型（如芯片解码的示例只能在 VPU 固件上运行，大模型只能在非 VPU 固件上运行）、硬件规格（如 2 核芯片只能运行 2 核以下编译的模型），具体请查看示例内 readme 文件。

windows 的示例运行前请参照 tools/win_envs 目录的 README.MD 进行环境变量的设置。

| examples              | path       | description                       | language   | target | arch        | os        |
| --------------------- | ---------- | --------------------------------- | ---------- | ------ | ----------- | --------- |
| qwen3_pipeline        | converts   | qwen3 流水并行模型分隔编译示例    | python     | xh2    | x64         | linux     |
| qwen3_speculative     | converts   | qwen3 投机解码模型编译示例        | python     | xh2    | x64         | linux     |
| resnet50              | converts   | resnet50 量化编译示例             | python     | xh2    | x64         | linux     |
| qwen3                 | inferences | qwen3 大语言模型推理示例          | python     | xh2    | x64/aarch64 | win/linux |
| qwen3_pipeline        | inferences | qwen3 流水并行示例                | python     | xh2    | x64/aarch64 | linux     |
| qwen3_speculative     | inferences | qwen3 投机解码示例                | python     | xh2    | x64/aarch64 | linux     |
| resnet50              | inferences | resnet50 单线程推理示例           | python/c++ | xh2    | x64/aarch64 | win/linux |
| resnet50_multistreams | inferences | resnet50 多线程多 stream 推理示例 | c++        | xh2    | x64/aarch64 | win/linux |
| resnet50_pipeline     | inferences | resnet50 流水推理示例             | c++        | xh2    | x64/aarch64 | win/linux |
| yolov5s               | inferences | yolov5s 单线程推理示例            | python/c++ | xh2    | x64/aarch64 | win/linux |

## 快速上手

参考各示例的 README.md 文件。
