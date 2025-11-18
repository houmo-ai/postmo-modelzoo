# Bge embedder和reranker

本示例展示如何把MiniCPMO中的各个模型量化和编译，部署到后摩M50芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 MiniCPMO 各模块模型实现来源于[https://huggingface.co/openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)。

## 2 快速开始

MiniCPMO支持Omni、vllm、llm、voice clone和only tts等图像、本文、语音功能。

### 2.1 获取模型和数据

通过get_model脚本下载模型，支持量化后模型编译和直接运行编译后芯片模型

```bash
# 直接获取可直接运行的芯片模型
python3 get_model.py --type hmm

# 量化后模型，需要编译器编译后才能运行
python3 get_model.py --type quant
```

### 2.2 量化

由于MiniCPMO模块较多，部分模块需要手动拆分和修改，故暂未提供模型量化示例。

### 2.3 编译

```bash
python3 build.py
```

### 2.3 演示

MiniCPMO模型使用python API进行演示

1. 将编译好的模型放在当前目录

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
#默认为omni模式，如需体验llm
python3 demo.py --example_idx 1
#如需体验vllm
python3 demo.py --example_idx 2
#如需体验多张图的vllm
python3 demo.py --example_idx 3
#如需语音克隆
python3 demo.py --example_idx 4
#如需仅跑tts文本转语音，且声音为女声
python3 demo.py --example_idx 5
#如需仅跑tts文本转语音，且声音为男声
python3 demo.py --example_idx 6
#如遇到长文本，需要拆分多段来运行tts
python3 demo.py --example_idx 7
```

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
Audio saved to output_omni.wav
画面中有一个穿着红色夹克和黑色裤子的人，他正在一个被雪覆盖的山地地形上滑雪。他正沿着一条清晰的滑雪道滑行，周围是高耸的雪山和多云的天空。
```

### 3.2 性能结果

```bash
Total Cost 22600.990 ms
Input Tokens: 1399, Output tokens: 50
Vision Cost: 6056.815 ms
Audio Cost: 468.451 ms
LLM Prefill Speed: 885.53 tokens/s
TTFT (Time to First Token): 8115.782 ms
TPOT (Time Per Output Token): 14.48 tokens/s
ViT+Wishper+LLM TPS (Tokens Per Second): 4.35 tokens/s
TTS Prefill Mean Time: 13.415 ms x 30 times
TTS Decoder Mean Time: 338.789 ms x 31 groups
TTS Dvae Cost: 42.395 ms
TTS Vocos Cost: 41.258 ms
Total Audio Duration Generated: 16.45 s
TTS Total Cost: 10996.603 ms
TTS Real-Time Factor(RTF): 0.668568
TTS Generate Speed: 1.50 x real-time
E2E Latency (End-to-End Latency): 22.601 seconds
```

### 3.3 备注

```bash
由于tts对长文本容易产生异常（原算法即存在的现象），我们提供了一种文本拆分的方法。
只需将chat()或chat_otts()调用时给定text_split=True，即可触发。
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
