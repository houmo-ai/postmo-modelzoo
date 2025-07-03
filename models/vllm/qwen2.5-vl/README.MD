# Qwen2.5-VL(通义千问2.5-VL)

本示例展示如何把Qwen2.5VL模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)。
本例只适用于xh1。

## 2 快速开始

目前Qwen2.5VL模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载量化模型或芯片模型。

目前暂时不支持用户自己量化。

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

也可以直接下载以上所有模型和数据集（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py
```

### 2.2 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括2个文件：
prefill.hmm
decoder.hmm
visual.hmm

### 2.3 演示

Qwen2模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下，也可通过demo中的`EMBEDDING_PATH`变量指定位置

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
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
这张图片展示了一位女士和她的狗在海滩上玩耍的场景。女士坐在沙滩上，狗坐在她对面，似乎在进行某种游戏。女士穿着格子衬衫和黑色裤子，狗戴着项圈。背景是海滩，海水和天空。
```

### 3.2 性能结果

```bash
total: 55 tokens, cost 5.036 s
visual + prefill time: 2397.724 ms, 0.42 tokens/s
decode average time: 48.854 ms, 20.47 tokens/s
end2end average time: 91.561 ms, 10.92 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
