# Qwen3-VL(通义千问3-VL)

本示例展示如何把Qwen3VL模型量化和编译，部署到后摩芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Instruct
- 来源:
  - https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct
  - https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
- 许可: Apache License 2.0 （https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md）

预训练模型在运行时下载，工程发布**不包含**该模型。
同时，原有的预处理代码 image_processing_qwen2_vl.py 和 processing_qwen3_vl.py 已进行专门的改编和修改，并包含在项目提交文件中。特此声明。

## 数据集使用声明

本项目使用了由LAION提供的"220k-GPT4Vision-captions-from-LIVIS"数据集。该数据集包含22万条通过GPT-4 Vision生成的图像描述文本，源自LIVIS数据集。
- 数据集获取方式：

    数据集可通过Hugging Face平台公开获取：

    https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS

- 许可信息： 本数据集采用Apache License 2.0开源协议授权。根据该许可条款，使用者可以自由地：

    使用、复制和分发数据集

    修改和创建衍生作品

    将数据集用于商业用途

    同时需遵守协议中规定的署名要求和责任限制条款。
- 合规性说明：

  本项目的使用方式完全符合Apache License 2.0的规定要求，特此声明。

[TOC]

## 1 模型说明

本例支持以下模型实现来源：

- [https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- [https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)

当前脚本默认值统一由同目录下的 `config.yaml` 提供，默认模型为 `qwen3-vl` + `8b`，同时支持 `qwen3-vl` + `4b`；视觉产物命名统一为 `${model_name}-${model_size}_visual_448x448x2.hmm`。

## 2 快速开始

目前Qwen3VL模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3-vl --model_size 8b

# 下载 4b 原始模型用于量化
python3 get_model.py --type raw --model_name qwen3-vl --model_size 4b
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm --model_name qwen3-vl --model_size 8b

# 下载 4b 预编译模型
python3 get_model.py --type hmm --model_name qwen3-vl --model_size 4b
```

### 2.3 量化

#### 2.3.1 自定义数据集量化

下载自定义数据集到当前目录下
```bash
python3 get_model.py --type raw
mkdir -p data/calib_data/LMUData
ln -s ../../../data/datasets/LMUData/* data/calib_data/LMUData
```

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen3-vl --model-size 8b --data_files ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 64 --calib_dataset vllm_custom_data
```

#### 2.3.2 默认公开数据集数据集量化

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python3 ptq.py --model-name qwen3-vl --model-size 8b

# 或量化 4b
python3 ptq.py --model-name qwen3-vl --model-size 4b
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py --model_name qwen3-vl --model_size 8b

# 或编译 4b
python3 build.py --model_name qwen3-vl --model_size 4b
```

编译输出默认位于：

- `output/${HOUMO_TARGET}/qwen3-vl-8b_prefill.hmm`
- `output/${HOUMO_TARGET}/qwen3-vl-8b_decode.hmm`
- `output/${HOUMO_TARGET}/qwen3-vl-8b_visual_448x448x2.hmm`

### 2.5 演示

Qwen3-vl模型使用python API和c++示例进行演示，c++模型示例的详细介绍见cpp/README.MD文件。

1. 将编译好的模型放在当前目录，同时将对应量化模型的 `quant_embedding.pt` 文件放在量化模型目录下

2. 执行脚本，推理完成将输出性能结果。

注意：由于ubuntu18.04下qwen_vl_utils无法正常安装，导致无法正常运行。

`demo.py` 现在支持显式指定输入模式：

- 图片模式：使用 `--image` 输入一张或多张图片。
- 视频模式：使用 `--video` 输入真实视频，或者使用 `--media_type video --image ...` 将一张或多张图片按伪视频方式输入。
- 当未显式指定 `--prompt` 时，图片模式默认提示词为“描述图片内容”，视频模式默认提示词为“描述视频内容”。
- 视频模式下，无论输入真实视频还是图片伪视频，都会按 visual 模型时间维度自动进行抽帧或 padding。
- 多图伪视频的时间语义可通过 `--multi_image_video_fps` 指定，默认值为 `2.0`。

```bash
# Qwen3-VL-4B 推荐惩罚系数是4
# Qwen3-VL-8B 推荐惩罚系数是1.5

# 单图或多图图片模式
python3 demo.py --model_name qwen3-vl --model_size 8b --image ../../../data/pic/beach.jpeg --repetition_penalty 1.5
python3 demo.py --model_name qwen3-vl --model_size 4b --image ../../../data/pic/beach.jpeg --repetition_penalty 4
python3 demo.py --model_name qwen3-vl --model_size 8b --image img1.jpg img2.jpg img3.jpg --repetition_penalty 1.5

# 真实视频模式
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --video demo.mp4 --repetition_penalty 1.5

# 多图伪视频模式
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --image img1.jpg img2.jpg img3.jpg --repetition_penalty 1.5

# 指定多图伪视频的 fps
python3 demo.py --model_name qwen3-vl --model_size 8b --media_type video --image img1.jpg img2.jpg img3.jpg --multi_image_video_fps 4 --repetition_penalty 1.5
```

若未显式传 `tokenizer_dir`、`vit_path`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导默认值；当 `ndevice > 1` 时，`prefill` / `decode` 默认后缀会切换为 `.hmms`。

### 2.6 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3-vl`
- `MODEL_SIZE=8b`
- `NDEVICE=1`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录；`quant`、`cpp demo` 和 `llm_perf` 流程都会用到该路径。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size. Default: 8b. Optional: 4b, 8b.
  -name, --model_name     Model name. Default: qwen3-vl.
  --ndevice               Number of devices. Default: 1.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 4b quant
bash test.sh -s quant --model_size 4b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 4b build
bash test.sh -s build --model_size 4b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 运行 4b demo
bash test.sh -s demo --model_size 4b

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

当前脚本行为说明：

- `quant`：下载原始模型，链接 `LMUData` 数据集，并调用 `ptq.py` 执行量化。
- `build`：调用 `build.py` 编译 `prefill`、`decode` 和 `visual` 模型。
- `demo`：下载预编译模型后，先执行 Python demo，再执行 C++ demo。
- 若环境中存在 `llm_perf`，`demo` 流程末尾还会自动执行性能测试。

## 3 参考结果

### 3.1 演示结果

```bash
2026-04-13 04:31:19.416] [tid 200602] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-13 04:31:19.416] [tid 200602] [warning] Empty backend name, use Xh2HalBackend instead.
2026-04-13 04:31:20.181 | INFO     | __main__:__init__:338 - vit model loaded
2026-04-13 04:31:23.454 | INFO     | __main__:__init__:342 - prefill model loaded
2026-04-13 04:31:23.626 | INFO     | __main__:__init__:352 - decode model loaded
2026-04-13 04:31:25.937 | SUCCESS  | __main__:<module>:1327 - question:
请分析输入内容并简洁作答。
2026-04-13 04:31:26.238 | SUCCESS  | __main__:chat_vit_prefill:1181 - response:
图片中，一位女士坐在沙滩上，与一只戴着蓝色牵引带的金毛犬互动。背景是日落时分的海面，光线温暖柔和，营造出宁静温馨的氛围。
```

### 3.2 性能结果
#### M50 2core
```bash
2026-04-13 04:31:28.931 | SUCCESS  | perf_infomations:show_summary:316 - ====================================================================================================
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:317 -                     Model Inference Performance Summary Report
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:318 - ====================================================================================================
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:321 - Configuration Details:
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:322 -   Batch Size:      1
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:323 -   Input Length per Sample:    215 tokens
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:324 -   Output Length per Sample:     44 tokens
2026-04-13 04:31:28.932 | SUCCESS  | perf_infomations:show_summary:326 -   Number of Images:      1 images
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:329 -   Prefill Model Load Time: 3272.80ms
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:331 -   Decode Model Load Time:  171.68ms
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:333 -   Vision Model Load Time:  765.13ms
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:338 - Vision Stage Performance:
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:339 -   Total Time:  128.28ms | Speed:    7.80 images/s
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:340 -   Preprocessing Time: 27.25ms | Speed:   36.70 images/s
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:341 -   API SetInput Time:   1.30ms
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:342 -   API Inference Time: 90.72ms | Speed:   11.02 images/s
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:343 -   API GetOutput Time:  6.99ms
2026-04-13 04:31:28.933 | SUCCESS  | perf_infomations:show_summary:346 - Prefill Stage Performance:
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:347 -   Total Time:  150.32ms | Speed: 1430.27 tokens/s
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:349 -   Tokenization Time:   22.49ms
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:352 -   Embedding Time:    0.00ms
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:353 -   API SetInput Time:   4.56ms
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:354 -   API Inference Time: 121.20ms | Prefill Speed: 1773.90 tokens/s
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:355 -   API GetOutput Time:  0.49ms
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:358 - Decode Stage Performance:
2026-04-13 04:31:28.934 | SUCCESS  | perf_infomations:show_summary:359 -   Total Time: 2683.29ms | Speed:   16.03 tokens/s
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:361 -   Tokenization Time:    9.42ms
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:364 -   Embedding Time:    0.15ms
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:365 -   API SetInput Time:   2.63ms
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:366 -   API Inference Time: 51.37ms | Decode Speed:   19.47 tokens/s
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:367 -   API GetOutput Time:  0.44ms
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:370 - Overall Performance Metrics:
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:371 -   TTFT (Time To First Token):  278.60 ms
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 62.40 ms/token
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:373 -   E2E Latency (End-to-End):      2.99 seconds
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:374 -   E2E TPS (Throughput):         14.69 tokens/s
2026-04-13 04:31:28.935 | SUCCESS  | perf_infomations:show_summary:376 - ====================================================================================================
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
