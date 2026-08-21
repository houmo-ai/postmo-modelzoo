# houmo-examples

## 目录

[TOC]

> **数据集版权提示：除少量 COCO 2017 采样（CC BY 4.0，已署名）用于演示外，本项目不随代码分发其他第三方数据集。**
> 示例代码仅提供数据加载和评估逻辑。除内置的 COCO 采样外，`data/` 目录仅保留结构占位。用户需自行从官方渠道获取其他数据，并严格遵守其许可协议和商用限制。学术非商用数据集（如 BDD100K、nuScenes、WIDER FACE）禁止企业商用和二次分发，本项目不提供自动下载或镜像。详见 [DATASET_NOTICE.md](DATASET_NOTICE.md)。

## 概述

houmo-examples 是为用户快速将模型和应用移植到后摩芯片上而提供的示例库，为用户提供量化、编译、精度和性能评估、应用部署等一整套代码和工具，降低用户的学习和开发成本。

模型示例仅支持 linux 平台，API 示例支持 linux 和 windows 平台，支持情况还取决于每个示例依赖的功能在各平台的支持情况，具体请参考每个示例的 readme 文件。

houmo-examples 目录结构如下，其中 README.md 为本说明文件：

```bash
.
├── apis
│   ├── converts
│   ├── data
│   ├── inferences
│   └── models
├── cmake
├── data
├── 3rdparty
├── hmatc
├── hmodel
├── licenses
├── models
│   ├── asr
│   ├── autodrive
│   ├── backbone
│   ├── detection
│   ├── diffusion
│   ├── embedding
│   ├── estimation
│   ├── llm
│   ├── ocr
│   ├── omni
│   ├── reranker
│   ├── segmentation
│   ├── tts
│   ├── vlm
│   └── benchmark.yml
├── tools
├── utils
├── env.sh
├── env.bat
├── LICENSE
├── NOTICE
├── README.md
└── requirements.txt
```

主要目录和文件说明如下：

| 目录或文件                          | 说明                                                    |
| ----------------------------------- | ------------------------------------------------------- |
| 3rdparty                            | 仓库内使用的 C/C++ 第三方源码                           |
| apis                                | API 示例，展示 HAL 和 Runtime 接口的调用过程            |
| cmake                               | C/C++ 示例共用的 CMake 配置                             |
| data                                | 数据目录，内置少量 COCO 采样，其他数据集用户自行获取，详见 DATASET_NOTICE.md |
| hmatc                               | HMATC 工具源码，通过配置文件进行一键评估                |
| hmodel                              | 量化模型配置和工具，主要用于大模型和 QAT 训练           |
| licenses                            | 仓库资产和示例引用的许可证文本                          |
| models                              | 模型示例，展示模型的转换、量化、编译、推理和评估流程    |
| tools                               | 应用层工具源码，如算力测试和性能评估工具                |
| utils                               | 模型和 API 共用的 C++、Python 代码及运行库              |
| env.sh/env.bat                      | Linux 和 Windows 环境配置脚本                           |
| LICENSE/NOTICE                      | 项目许可证和第三方组件摘要                              |
| requirements.txt                    | Python 第三方依赖                                       |

本项目采用 Apache License 2.0。项目许可证见 [LICENSE](LICENSE)，项目和第三方组件摘要见 [NOTICE](NOTICE)。第三方组件不受本项目 Apache License 2.0 的统一许可约束，使用和分发时应遵守各组件自身的许可证条款。

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

如果需要对大模型进行量化，同时是在github等开源平台下载的代码，需要自行下载以下几个仓库：
- [houmo-xh2modelzoo](https://github.com/houmo-ai/houmo-xh2modelzoo)，安装或者重命名为hmodel/xh2目录
- [houmo-gptqmodel](https://github.com/houmo-ai/houmo-gptqmodel)，安装或者重命名为hmodel/gptqmodel目录

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

模型示例列表如下，type 列为模型类型，target 列为支持的芯片平台，quant 表示提供量化示例，build 表示提供编译示例，perf 表示提供性能评估，demo 表示提供 python 端到端 demo，eval 表示提供精度评估，support 表示当前版本是否支持，如果该项是版本号表示支持的最后一个版本。

量化和编译功能仅支持在量化工具和编译器支持的平台上运行，其中大模型量化需要使用 GPU。涉及到模型推理相关的功能（如 perf/demo/eval 等）需要使用后摩芯片平台运行，运行时需要关注其他限制，如固件类型（如大模型只能在非 VPU 固件上运行），硬件规格（如 2 核芯片只能运行 2 核以下编译的模型）。


| models                  | path         | target | quant | build | perf | demo | eval | support |
| ----------------------- | ------------ | ------ | ----- | ----- | ---- | ---- | ---- | ------- |
| emotion2vec             | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3-asr               | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| glm-asr-nano-2512       | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| sensevoice              | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| whisper                 | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| whisper-turbo           | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| cam                     | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| ct_transformer          | asr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| yolop                   | autodrive    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| efficientnet            | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| mobilenetv2             | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| resnet50                | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| ViT-B-16                | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov8m-cls             | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| dinov3-base             | backbone     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolo11m                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolo12m                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolo26m                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov3                  | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov5m_face            | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov5s                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov5s_feature         | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov7                  | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov8m                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov9m                 | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov10m                | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolox                   | detection    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| z-image                 | diffusion    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| bge                     | embedding    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| gte                     | embedding    | xh2    | ❌️   | ❌️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3-embedding         | embedding    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3-vl-embedding      | embedding    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| siglip2                 | embedding    | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| yolov8m-pose            | estimation   | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| funaudiochat            | lalm         | xh2    | ❌️   | ❌️    | ✅️  | ✅️   | ❌️  | ✅️      |
| deepseek-r1-qwen3-8b    | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| gpt-oss                 | llm          | xh2    | ❌️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen2.5                 | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| qwen3                   | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| qwen3-30b-a3b           | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| qwen3-next              | llm          | xh2    | ❌️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3.5                 | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| CoPaw-Flash-9B          | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| glm-ocr                 | ocr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| lprnet                  | ocr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| mineru2.5               | ocr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| PPOCRv3                 | ocr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| paddleocr-vl            | ocr          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| minicpmo                | omni         | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| qwen3-omni              | omni         | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3-reranker          | reranker     | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| sam2                    | segmentation | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| sam3                    | segmentation | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| yolov8m-seg             | segmentation | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ✅️  | ✅️      |
| cosyvoice3              | tts          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen3-tts               | tts          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen2.5-vl              | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| qwen3-vl                | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | v1.3.0  |
| gemma4                  | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| ornith1.0               | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| minicpm-v-4.5           | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| minicpm-v-4.6           | vlm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |
| qwen-agentworld         | llm          | xh2    | ✅️   | ✅️    | ✅️  | ✅️   | ❌️  | ✅️      |



## API 示例

API 示例在 apis 目录下，如下表所示，type 列为示例类型，其中 convert 表示模型转换，inference 表示模型推理。language 列为支持的编程语言，target 列为支持的芯片平台。

量化和编译示例仅支持在量化工具和编译器支持的平台上运行，部署示例支持情况还受到其他交付件的限制，如固件类型（如芯片解码的示例只能在 VPU 固件上运行，大模型只能在非 VPU 固件上运行）、硬件规格（如 2 核芯片只能运行 2 核以下编译的模型），具体请查看示例内 readme 文件。

windows 的示例运行前请参照 tools/win_envs 目录的 README.MD 进行环境变量的设置。

| examples              | path       | description                       | language   | target | arch        | os        |
| --------------------- | ---------- | --------------------------------- | ---------- | ------ | ----------- | --------- |
| qwen3_pipeline        | converts   | qwen3 流水并行模型分隔编译示例    | python     | xh2    | x64         | linux     |
| qwen3_speculative     | converts   | qwen3 投机解码模型编译示例        | python     | xh2    | x64         | linux     |
| resnet50              | converts   | resnet50 量化编译示例             | python     | xh2    | x64         | linux     |
| yolov5s               | converts   | yolov5s 模型转换编译示例          | python     | xh2    | x64         | linux     |
| qwen3                 | inferences | qwen3 大语言模型推理示例          | python     | xh2    | x64/aarch64 | win/linux |
| qwen3_multibatch      | inferences | qwen3 多 batch 推理示例           | python     | xh2    | x64/aarch64 | linux     |
| qwen3_pipeline        | inferences | qwen3 流水并行示例                | python     | xh2    | x64/aarch64 | linux     |
| qwen3_speculative     | inferences | qwen3 投机解码示例                | python     | xh2    | x64/aarch64 | linux     |
| resnet50              | inferences | resnet50 单线程推理示例           | python/c++ | xh2    | x64/aarch64 | win/linux |
| resnet50_multistreams | inferences | resnet50 多线程多 stream 推理示例 | c++        | xh2    | x64/aarch64 | win/linux |
| resnet50_pipeline     | inferences | resnet50 流水推理示例             | c++        | xh2    | x64/aarch64 | win/linux |
| yolov5s               | inferences | yolov5s 单线程推理示例            | python/c++ | xh2    | x64/aarch64 | win/linux |

## 快速上手

参考各示例的 README.md 文件。
