# GLM-OCR

本示例展示如何把GLM-OCR模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: GLM-OCR
- 来源: https://www.modelscope.cn/models/zhipuAI/GLM-OCR
- 许可: MIT License

预训练模型在运行时下载，工程发布**不包含**该模型。


## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/zhipuAI/GLM-OCR](https://www.modelscope.cn/models/zhipuAI/GLM-OCR)。

## 2 快速开始

目前GLM-OCR模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.2 量化

下载模型到当前目录下
```bash
python3 get_model.py --type raw
```

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==51.0
pip3 install transformers==5.1.0
python ptq.py
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括3个文件：
glm-ocr_prefill.hmm
glm-ocr_decode.hmm
glm-ocr_visual.hmm1

### 2.4 演示

GLM-OCR模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.5 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question:
Text Recognition:
response:
后摩智能
M50芯片
```

### 3.2 性能结果
#### M50 2core
```bash
Output 156 tokens, Prefill Cost 36.223 ms
Prefill Speed: 4306.66 tokens/s
Output 8 tokens, Decode Cost 83.455 ms
Decode Speed: 95.86 tokens/s
TTFT (Time to First Token): 158.199 ms
TPOT (Time Per Output Token): 10.432 ms/token
E2E Latency (End-to-End Latency): 0.242 seconds
E2E TPS (End-to-End Tokens Per Second): 33.11 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
