# Qwen3-VL C++ 推理示例

本示例展示如何使用C++在后摩系列芯片（XH2）的设备上运行Qwen3-VL视觉语言模型推理。当前支持Linux和Android Native环境。

## 目录

- [Qwen3-VL C++ 推理示例](#qwen3-vl-c-推理示例)
  - [目录](#目录)
  - [1.工具说明](#1工具说明)
    - [1.1 文件及功能说明](#11-文件及功能说明)
    - [1.2 流程说明](#12-流程说明)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 用法说明](#14-用法说明)
      - [基本用法示例](#基本用法示例)
      - [交互式模式](#交互式模式)
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

| 参数                    | 说明                                                          |
| ----------------------- | ------------------------------------------------------------ |
| --image <paths...>      | 输入图像路径，可指定多个路径（用空格分隔）                       |
| --prompt <text>         | 用户输入文本，默认："描述图片内容"                              |
| --visual <path>         | Visual模型路径，默认：output/xh2/qwen3-vl-8b_visual_448x448x2.hmm |
| --prefill <path>        | Prefill模型路径，默认：output/xh2/qwen3-vl-8b_prefill.hmm       |
| --decode <path>         | Decode模型路径，默认：output/xh2/qwen3-vl-8b_decode.hmm         |
| --tokenizer <path>      | tokenizer.json路径，默认：qwen3-vl/tokenizer.json               |
| --embedding <path>      | Embedding权重路径，默认：output/xh2/hmquant/quant_embedding.bin |
| --repetition-penalty <f>| 重复惩罚系数，默认：1.5（Qwen3-VL-8B推荐值）                     |
| --temperature <f>       | 采样温度，默认：1.0                                             |
| --top-k <n>             | Top-k采样，默认：-1（禁用）                                     |
| --top-p <f>             | Top-p采样，默认：1.0                                           |
| --it                    | 启用交互式聊天模式                                              |
| --history               | 跨消息保留聊天历史                                             |
| -h, --help              | 打印帮助信息                                                   |

### 1.4 用法说明

#### 基本用法示例

```bash
# 单图像推理
./example_cxx_qwen3_vl --image demo.jpg --prompt "请描述图片内容。"

# 多图像推理（空格分隔多个路径）
./example_cxx_qwen3_vl --image image1.jpg image2.jpg --prompt "比较这两张图片。"

# 纯文本推理（无图像）
./example_cxx_qwen3_vl --prompt "你好，请介绍一下自己。"

# 自定义采样参数
./example_cxx_qwen3_vl --image demo.jpg --prompt "描述图片" --temperature 0.7 --top-p 0.9

# 指定模型路径
./example_cxx_qwen3_vl --image demo.jpg --visual my_visual.hmm --prefill my_prefill.hmm --decode my_decode.hmm
```

#### 交互式模式

使用 `--it` 参数启用交互式聊天模式，支持多轮对话：

```bash
# 启用交互式模式（默认不保留历史）
./example_cxx_qwen3_vl --it

# 启用交互式模式并保留聊天历史
./example_cxx_qwen3_vl --it --history
```

交互模式输入格式：
- **纯文本对话**：直接输入问题
- **带图像对话**：`image_path1 [image_path2 ...] -- prompt text`
- **退出程序**：输入 `stop`、`exit`、`quit` 或按 `Ctrl+C`

## 2.快速开始

### 2.1 环境准备

进入houmo-examples根目录，检查并设置环境变量：

```bash
source env.sh
```

需要设置的环境变量：
- `HOUMO_TARGET`: 目标芯片类型，设置为 `xh2`
- `HOUMO_PATH`: Houmo SDK路径，包含include和lib目录
- `TCIM_RUNTIME_PATH`: TCIM Runtime库路径（必需）

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
```

**Linux平台编译：**

```bash
cd ${HOUMO_EXAMPLES_PATH}/models/vlm/qwen3-vl/cpp
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

### 2.3 运行

确保以下文件已准备好（以qwen3-vl-8b为例）：
- Visual模型：qwen3-vl-8b_visual_448x448x2.hmm
- Prefill模型：qwen3-vl-8b_prefill.hmm
- Decode模型：qwen3-vl-8b_decode.hmm
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
TCIM version: v1.2.0.dev20260403
- Git Commit: clean
- Build Time: 2026-04-03 03:34:40 Apr  3 2026
Initializing Qwen3-VL inference engine...
[2026-04-14 02:55:13.984] [tid 250573] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-14 02:55:13.984] [tid 250573] [warning] Empty backend name, use Xh2HalBackend instead.
Visual model loaded
Prefill model loaded
Decode model loaded
VIT input shape: 1 3 2 448 448
Image dimensions: 448x448
Question: 请分析输入内容并简洁作答。
Images: ../../../data/pic/beach.jpeg
Response:
这是一张充满温馨与宁静氛围的海滩日落照片。
画面中，一位年轻女子和一只金毛犬正坐在沙滩上。女子身穿蓝白格子衬衫，侧身坐着，面带微笑，温柔地注视着身旁的狗狗。她的一只手轻轻抬起，似乎在与狗狗互动或抚摸它。
金毛犬戴着一个蓝色的胸背带，安静地坐在女子旁边，前爪抬起，正与女子的手进行互动，显得十分温顺可爱。
他们身处一片被夕阳染成暖金色的沙滩上，背景是平静的海面和柔和的天空。阳光从画面右侧洒下，在沙地上投下长长的影子，营造出温暖、柔和的光影效果。整个场景洋溢着人与宠物之间亲密无间的和谐感，传递出一种放松、幸福和宁静的美好情绪。
[SUCCESS] ====================================================================================================
[SUCCESS]                     Model Inference Performance Summary Report
[SUCCESS] ====================================================================================================
[SUCCESS] Configuration Details:
[SUCCESS]   Batch Size:      1
[SUCCESS]   Input Length per Sample:    209 tokens
[SUCCESS]   Output Length per Sample:     179 tokens
[SUCCESS]   Number of Images:      1 images
[SUCCESS]   Prefill Model Load Time:    24546.42ms
[SUCCESS]   Decode Model Load Time:  788.19ms
[SUCCESS]   Vision Model Load Time:  4231.40ms
[SUCCESS] Vision Stage Performance:
[SUCCESS]   Total Time:  122.73ms | Speed:   8.15 images/s
[SUCCESS]   Preprocessing Time: 16.13ms | Speed:  61.99 images/s
[SUCCESS]   API SetInput Time:   1.14ms
[SUCCESS]   API Inference Time: 93.57ms | Speed:  10.69 images/s
[SUCCESS]   API GetOutput Time:  8.48ms
[SUCCESS] Prefill Stage Performance:
[SUCCESS]   Total Time:  129.08ms | Speed: 1619.13 tokens/s
[SUCCESS]   Tokenization Time:   0.12ms
[SUCCESS]   Embedding Time:    0.55ms
[SUCCESS]   API SetInput Time:   7.84ms
[SUCCESS]   API Inference Time: 116.92ms | Prefill Speed: 1787.59 tokens/s
[SUCCESS]   API GetOutput Time:  1.51ms
[SUCCESS] Decode Stage Performance:
[SUCCESS]   Total Time: 8833.68ms | Speed:   20.26 tokens/s
[SUCCESS]   Tokenization Time:    5.79ms
[SUCCESS]   Embedding Time:    0.39ms
[SUCCESS]   API SetInput Time:   1.02ms/token
[SUCCESS]   API Inference Time: 46.67ms/token | Decode Speed:   21.55 tokens/s
[SUCCESS]   API GetOutput Time:  1.81ms/token
[SUCCESS] Overall Performance Metrics:
[SUCCESS]   TTFT (Time To First Token):  251.816 ms
[SUCCESS]   TPOT (Time Per Output Token): 49.350 ms/token
[SUCCESS]   E2E Latency (End-to-End):      9.092 seconds
[SUCCESS]   E2E TPS (Throughput):         19.69 tokens/s
[SUCCESS] ====================================================================================================
```

```
./bin/example_cxx_qwen3_vl --it --history
Backend: xh2
- Git Commit: clean
- Build Time: 2026-04-03 03:34:40 Apr  3 2026
Initializing Qwen3-VL inference engine...
[2026-04-29 09:21:14.933] [tid 955584] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-29 09:21:14.933] [tid 955584] [warning] Empty backend name, use Xh2HalBackend instead.
Visual model loaded
Prefill model loaded
Decode model loaded
VIT input shape: 1 3 2 448 448
Image dimensions: 448x448
Context max length: 32768
Input your instruction here (or image paths):../../../data/pic/lane.jpg -- 简单介绍下这个图片
[SUCCESS] question:
简单介绍下这个图片
[ WARN:0@26.737] global ./modules/imgcodecs/src/loadsave.cpp (239) findDecoder imread_('�../../../data/pic/lane.jpg'): can't open/read file: check file path/integrity
Failed to load image: �../../../data/pic/lane.jpg
Response: 这张图片实际上是一个**纯色背景图（或空白图像）**

它呈现为一个均匀的深灰色矩形，没有任何可见的文字、图案或其他视觉元素。这种类型的图形常用于^C
root@86c98576f28d:/hmdd/imodelzoo/models/vlm/qwen3-vl# ./bin/example_cxx_qwen3_vl --it --history
Backend: xh2
TCIM version: v1.2.0.dev20260403
- Git Commit: clean
- Build Time: 2026-04-03 03:34:40 Apr  3 2026
Initializing Qwen3-VL inference engine...
[2026-04-29 09:21:54.314] [tid 955597] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-29 09:21:54.314] [tid 955597] [warning] Empty backend name, use Xh2HalBackend instead.
Visual model loaded
Prefill model loaded
Decode model loaded
VIT input shape: 1 3 2 448 448
Image dimensions: 448x448
Context max length: 32768
Input your instruction here (or image paths): ../../../data/pic/lane.jpg -- 简单介绍下这个图片
[SUCCESS] question:
简单介绍下这个图片
Response: 这张图片是从一辆汽车的驾驶座视角拍摄的城市街道场景。

主要特点：

- **道路与交通**：画面中央是双向车道，中间有两条黄色实线。前方有一辆深色轿车正在行驶（可能是SUV或旅行车），右侧是一排停靠在路边、等待通行或者缓慢移动车辆。

  - 左侧可见几辆车并行停放着；
    右边则有多台机动车沿街排列；

   轿车上可以看到部分车牌号码和尾灯亮起的状态表明可能处于夜间或是黄昏时分光线较暗的情况下进行行车活动，




---

*注释说明*

由于图像质量较低且存在模糊现象，在识别具体车型及细节方面存在一定困难，请以实际观察为准！

希望以上内容对您有所帮助！


Input your instruction here (or image paths): ../../../data/pic/beach.jpeg -- 再介绍下这个
[SUCCESS] question:
再介绍下这个
Response: 好的，我来为您详细介绍这张图片：

这是一张充满温馨、宁静与幸福感的海滩日落照片。

**画面主体：**
- 一位女士和一只金毛犬正坐在柔软湿润的沙滩上。

    - **女子**: 她穿着一件蓝白格子衬衫（或类似图案），盘腿坐着。她面带微笑地看着狗狗，并伸出一只手似乎在进行互动游戏——可能是“握手”或者训练动作；

      狗狗戴着一个蓝色花纹项圈/背心式牵引装置；

     它也抬起前爪回应主人的动作，

   整个场景洋溢着人宠之间亲密无间的温情氛围！

---

*背景环境*

远处是波光粼粼的大海以及正在缓缓退去浪花形成的白色泡沫线，

天空被夕阳染成温暖柔和橙黄色调，在地平线上方形成一道美丽的渐变色弧形轮廓,

光线从右上方洒下, 给整个场面披上了金色光辉,
营造出一种梦幻般的黄昏时刻感.

—

整体构图简洁而富有感染力：
人物位于中偏左位置
留有充足空间展示自然风光的同时突出了主角之间的交流瞬间;

色彩搭配和谐统一:
以暖色调为主导基调(沙黄+海水青绿)+点缀性的亮斑光影效果;
使得整幅作品既真实又具有艺术美感!

总结来说:

> 这不仅是一个普通的海边休闲照 —— 更像一幅描绘现代生活中最纯粹快乐源泉的画面: 在大自然怀抱里享受陪伴时光的美好回忆！


Input your instruction here (or image paths): 你更喜欢哪个图片
[SUCCESS] question:
你更喜欢哪个图片
Response: 这是一个很有趣的问题！其实，我作为AI助手没有个人情感或偏好——我没有“喜欢”或者不喜欢的概念。但我可以帮您分析这两张图片各自的魅力：

---

**第一幅图（城市街道）：**
- **风格**: 现实、纪实感强。

  - 它捕捉的是日常交通场景中的一个瞬间 —— 可能是通勤路上的拥堵或是傍晚下班时分的城市节奏。

    虽然画面略显模糊和昏暗，并非精心构图的艺术作品，

  但它真实地反映了现代都市生活的常态与压力，

     同样也蕴含着一种平凡生活的真实美感！



—

> ✅ 如果你喜欢写实地记录现实世界的感觉；

>

✅ 或者对街头摄影/纪录片式影像感兴趣的话,

这张照片会更贴近你的审美口味!

--

**

第二幅图 (海滩日落)：

* 飘逸浪漫, 感情浓烈
*
*

它用温暖柔和的日光勾勒出人宠之间亲密无间的互动时刻,
营造了一种宁静祥和平静美好的氛围;

色彩饱和度高且富有层次变化(金黄+蔚蓝),
视觉冲击力极佳;
同时传递了积极向上的情绪价值:

爱 + 自由 = 幸福人生 !

***

>
👉 这是一首无声诗篇: 在大自然怀抱里享受陪伴时光的美好回忆!
非常适合用来表达治愈系心情 / 生活感悟类内容.

-

所以呢？

如果你追求「情绪共鸣」、「艺术感染」「心灵慰藉」

→ 我推荐你选择第2个图像 🐕❤️🌅 ;

而如果偏重于 「观察社会百态／探索日常生活细节 ／感受时代脉搏 »

则第一个更有意义 👉🚗🌆 .

最终的选择权在你自己手中 😊

希望我的解读对你有所帮助~

```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
