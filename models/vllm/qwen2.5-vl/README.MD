# Qwen2.5-VL(通义千问2.5-VL)

本示例展示如何把Qwen2.5VL模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)。
本例适用于xh1和xh2，量化仅支持xh2。

## 2 快速开始

目前Qwen2.5VL模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
python3 get_model.py --type raw
pip3 install transformers==4.56.0
python3 get_model.py --type raw
python3 ptq.py
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
# 3b模型
python3 get_model.py --type hmm
# 7b模型
python3 get_model.py --type hmm --model_size 7b
```

也可以直接下载以上所有模型和数据集（注意相同模型会覆盖，请自行保存）：

```bash
# 3b模型
python3 get_model.py
# 7b模型
python3 get_model.py --model_size 7b
```

### 2.2 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
# 3b模型
python3 build.py
# 7b模型
python3 build.py --model_size 7b
```

编译好的模型存放在当前目录，包括2个文件：
qwen2.5-vl_prefill.hmm
qwen2.5-vl_decoder.hmm
qwen2.5-vl_visual.hmm

### 2.3 演示

Qwen2.5-vl模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过demo中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

3. 注意：由于ubuntu18.04下qwen_vl_utils无法正常安装，导致无法正常运行。

```bash
pip3 install transformers==4.51.0
# 3b模型
python3 demo.py
# 7b模型
python3 demo.py --model_size 7b
```

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question:
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
请描述图片内容。
response:
这张图片展示了一位坐在沙滩上的女士和一只狗。女士穿着格子衬衫，坐在沙滩上，与狗互动。狗坐在女士的前面，伸出前爪与女士的手相触。背景是海浪和天空，阳光从右上方照射下来，给整个场景增添了一种温暖的氛围。整个画面给人一种温馨和谐的感觉。
```

### 3.2 性能结果
#### M30 4core
```bash
Total Images: 1, Total Input: 602 tokens, Output 74 tokens, Vision Cost 675.809 ms, Prefill Cost 833.611 ms, Decode Cost 6201.947 ms
Vision Cost 675.809 ms/image
Prefill Speed: 722.16 tokens/s; Decode Speed: 11.77 tokens/s
TTFT (Time to First Token): 1651.820 ms
TPOT (Time Per Output Token): 84.958 ms/token
E2E Latency (End-to-End Latency): 7.854 seconds
E2E TPS (End-to-End Tokens Per Second): 9.42 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
