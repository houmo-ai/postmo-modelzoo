# Qwen3-VL C++ 推理示例

本示例展示如何使用C++在后摩鸿途系列芯片（XH2）的设备上运行Qwen3-VL视觉语言模型推理。当前支持Linux和Android Native环境。

## 目录

- [Qwen3-VL C++ 推理示例](#qwen3-vl-c-推理示例)
  - [目录](#目录)
  - [1.工具说明](#1工具说明)
    - [1.1 文件及功能说明](#11-文件及功能说明)
    - [1.2 流程说明](#12-流程说明)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 用法说明](#14-用法说明)
  - [2.快速开始](#2快速开始)
    - [2.1 环境准备](#21-环境准备)
    - [2.2 编译](#22-编译)
    - [2.3 运行](#23-运行)
  - [3.参考结果](#3参考结果)
  - [4.免责声明](#4免责声明)

## 1.工具说明

### 1.1 文件及功能说明

文件目录如下：

| 文件名称              | 说明                                              |
| --------------------- | ------------------------------------------------- |
| demo.cc               | 主入口，拉起Qwen3-VL推理流程                      |
| HmQwenVLInfer.cc      | 主文件，实现Qwen3-VL完整推理流程                  |
| HmQwenVLInfer.h       | HmQwenVLInfer类头文件                             |
| HmQwenVLTokenizer.cc  | Tokenizer类实现，封装tokenizers-cpp              |
| HmQwenVLTokenizer.h   | Tokenizer类头文件                                 |
| HmImageProcessor.cc   | 图像预处理类实现                                  |
| HmImageProcessor.h    | 图像预处理类头文件                                |
| utils.h               | 工具函数和结构定义                                |
| CMakeLists.txt        | CMake编译配置文件                                 |
| build.sh              | Linux平台编译脚本                                 |
| build_ndk.sh          | Android平台编译脚本                               |
| build_win.bat         | Windows11平台编译脚本                             |
| tcim_runtime.cmake    | TCIM Runtime库查找配置                            |

功能支持度说明：
- 支持的芯片型号为：xh2。
- 支持的大道软件版本要求：xh2≥v1.0.0。
- 支持多图像输入。
- 支持文本-only模式（不指定图像）。
- 支持采样参数配置（temperature、top-k、top-p、repetition_penalty）。

### 1.2 流程说明

程序按以下流程进行：

1. 解析命令行参数，加载模型配置
2. 初始化Tokenizer和图像处理器
3. 加载Visual、Prefill、Decode三个模型
4. 处理输入图像（如有），运行Visual模型获取视觉特征
5. 将prompt文本tokenize后与视觉特征拼接
6. 运行Prefill模型处理输入序列
7. 循环运行Decode模型生成输出token，直到生成结束token
8. 对输出token进行detokenize，返回最终响应文本

性能指标：
| 性能指标                                | 说明                                           |
| --------------------------------------- | ---------------------------------------------- |
| Vision Cost                             | Visual模型推理耗时，单位（ms）                 |
| Prefill Speed                           | Prefill阶段速度，单位（tokens/s）              |
| Decode Speed                            | Decode阶段速度，单位（tokens/s）               |
| TTFT (Time to First Token)              | 输出第一个token的耗时，单位（ms）              |
| TPOT (Time Per Output Token)            | Decode阶段每token平均耗时，单位（ms/token）    |
| E2E Latency (End-to-End Latency)        | 端到端总时间，单位（s）                        |
| E2E TPS (End-to-End Tokens Per Second)  | 端到端每秒输出的token数，单位（tokens/s）      |

### 1.3 参数说明

命令行参数说明：

| 参数                    | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| --image <path>          | 输入图像路径，可多次指定以输入多张图像                       |
| --prompt <text>         | 用户输入文本，默认："请分析输入内容并简洁作答。"             |
| --visual <path>         | Visual模型路径，默认：output/xh2/qwen3-vl_visual.hmm         |
| --prefill <path>        | Prefill模型路径，默认：output/xh2/qwen3-vl_prefill.hmm       |
| --decode <path>         | Decode模型路径，默认：output/xh2/qwen3-vl_decode.hmm         |
| --tokenizer <path>      | tokenizer.json路径，默认：qwen3-vl/tokenizer.json            |
| --embedding <path>      | Embedding权重路径，默认：output/xh2/hmquant/quant_embedding.bin |
| --repetition-penalty <f>| 重复惩罚系数，默认：1.5（Qwen3-VL-8B推荐值）                 |
| --temperature <f>       | 采样温度，默认：1.0                                          |
| --top-k <n>             | Top-k采样，默认：-1（禁用）                                  |
| --top-p <f>             | Top-p采样，默认：1.0                                         |
| -h, --help              | 打印帮助信息                                                 |

### 1.4 用法说明

基本用法示例：

```bash
# 单图像推理
./example_cxx_qwen3_vl --image demo.jpg --prompt "请描述图片内容。"

# 多图像推理
./example_cxx_qwen3_vl --image image1.jpg --image image2.jpg --prompt "比较这两张图片。"

# 纯文本推理（无图像）
./example_cxx_qwen3_vl --prompt "你好，请介绍一下自己。"

# 自定义采样参数
./example_cxx_qwen3_vl --image demo.jpg --prompt "描述图片" --temperature 0.7 --top-p 0.9

# 指定模型路径
./example_cxx_qwen3_vl --image demo.jpg --visual my_visual.hmm --prefill my_prefill.hmm --decode my_decode.hmm
```

## 2.快速开始

### 2.1 环境准备

进入houmo-examples根目录，检查并设置环境变量：

```bash
source env.sh
```

需要设置的环境变量：
- `HOUMO_TARGET`: 目标芯片类型，设置为 `xh2`
- `HOUMO_PATH`: Houmo SDK路径，包含include和lib目录
- `TCIM_RUNTIME_PATH`: TCIM Runtime库路径（可选，默认使用相对路径）

对于Android平台交叉编译，还需设置：
- `NDK_PATH`: Android NDK路径
- `OPENCV_INCLUDE_DIRS`: OpenCV头文件目录
- `OPENCV_LIB_DIR`: OpenCV库文件目录

### 2.2 编译

**本示例依赖opencv库进行图像预处理，请提前安装或者从官网下载对应平台的Release版本。**
``` bash
#Linux   : sudo apt install -y libopencv-dev 
#Android : 1. 下载源码，自行使用ndk交叉编译，安装到3rdparty/opencv/Android_xh2目录下，
#          2. 或者使用get_3rdparty.py下载预编译的Android版opencv。
#Win11   : 下载windows11的Release版本，参考${HOUMO_EXAMPLES_PATH}/tools/win_envs/README.MD完成环境变量设置。
```

**Linux平台编译：**

```bash
cd /hmdd/imodelzoo/models/vlm/qwen3-vl/cpp
./build.sh
```

编译生成的可执行文件在 `../bin` 目录下。

**Android平台编译：**

```bash
# 设置NDK路径
export NDK_PATH=/path/to/android-ndk-r28c
export HOUMO_SDK_PATH=/path/to/houmo-drv-xh2
export TCIM_RUNTIME_PATH=/path/to/houmo-tcim-runtime-xh2

./build_ndk.sh
```

编译生成的可执行文件在 `../android` 目录下。

**Windows11平台编译：**

```bash
# 下载win11平台opencv的Release版本
# 参考${HOUMO_EXAMPLES_PATH}/tools/win_envs目录下的README.MD设置win11环境变量。

./build_win.bat
```

编译生成的可执行文件在 `../bin` 目录下。

### 2.3 运行

确保以下文件已准备好：
- Visual模型：qwen3-vl_visual.hmm
- Prefill模型：qwen3-vl_prefill.hmm
- Decode模型：qwen3-vl_decode.hmm
- tokenizer.json
- quant_embedding.bin : pt转bin的方式，参考tools/llm_perf文件夹README.MD中convert_embed.py脚本的相关使用说明。

```bash
# 设置环境变量
export HOUMO_TARGET=xh2
export TCIM_BACKEND=Xh2HalBackend

# 运行推理
./example_cxx_qwen3_vl --image demo.jpg --prompt "请描述图片内容。"
```

## 3.参考结果

```bash
Backend: xh2
TCIM version: v1.1.0
- Git Commit: clean
- Build Time: 2026-03-07 02:09:38 Mar  7 2026
Initializing Qwen3-VL inference engine...
Visual model loaded
Prefill model loaded
Decode model loaded
VIT input shape: 1 3 2 448 448
Image dimensions: 448x448
Question: 请分析输入内容并简洁作答。
Images: beach.jpeg
Response: 图片中，一位女士坐在沙滩上，与一只戴着蓝色背带的金毛犬互动。两人在夕阳下相视而笑，氛围温馨。背景是海浪和金色的落日，画面充满宁静与幸福。

[SUCCESS] Total Input: 215 tokens, Output 50 tokens, Vision Cost 112.419 ms, Prefill Cost 124.458 ms, Decode Cost 2635.608 ms
[SUCCESS] Prefill Speed: 1727.50 tokens/s; Decode Speed: 18.97 tokens/s
[SUCCESS] TTFT: 236.877 ms
[SUCCESS] TPOT: 52.712 ms/token
[SUCCESS] E2E Latency: 2.874 seconds
[SUCCESS] E2E TPS: 17.40 tokens/s
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。