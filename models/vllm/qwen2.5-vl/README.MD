# Qwen2.5-VL(通义千问2.5-VL)

本示例展示如何把Qwen3模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)。

## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型和wikitext2数据集，下载后的模型存放在qwen3-8b目录：

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
图片中展示了一位女士和她的狗在海滩上互动的场景。女士坐在沙滩上，狗坐在她旁边，两者都面带微笑。图片的背景是日出或日落，天空呈现出淡蓝色和橙色的渐变，沙滩和海水的交界处也呈现出橙色的渐变。图片的构图和构图都显得和谐，给人一种宁静和愉悦的感觉。
```

### 3.2 性能结果

```bash
total: 86 tokens, cost 6.641 s
visual + prefill time: 2539.945 ms, 0.39 tokens/s
decode average time: 48.243 ms, 20.73 tokens/s
end2end average time: 77.216 ms, 12.95 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
