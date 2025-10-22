# Bge embedder和reranker

本示例展示如何把MiniCPMO中的各个模型量化和编译，部署到后摩M50芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 MiniCPMO 各模块模型实现来源于[https://huggingface.co/openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)。

## 2 快速开始

MiniCPMO暂时只支持vllm或llm模式，其它功能模式(语音相关)待支持。

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
#默认为vllm模式，如需体验llm
python3 demo.py --example_idx 1
#如需体验多张图的vllm
python3 demo.py --example_idx 3
```

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
[{'role': 'user', 'content': [<PIL.Image.Image image mode=RGB size=1920x1440 at 0x7FC2E770B7D0>, '图中是哪家航空公司的飞机？']}]
图中的飞机属于中国南方航空。
```

### 3.2 性能结果

```bash
Total cost 9383.719 ms
Input tokens: 700, Output tokens: 9
Vision Cost 7646.908 ms
LLM Prefill Speed: 616.23 tokens/s
TTFT (Time to First Token): 8787.575 ms
TPOT (Time Per Output Token): 16.46 tokens/s
E2E Latency (End-to-End Latency): 9.384 seconds
TPS (Tokens Per Second): 0.96 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
