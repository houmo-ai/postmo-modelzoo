# wenet

本示例展示如何把wenet模型编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 wenet 实现来源于[https://github.com/wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)。

本例中使用的wenet库在公版源码的基础上做了裁剪，删除了部分不需要的代码以支持python3.9（原始版本只支持python3.10以上）

## 2 快速开始

目前wenet模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

如果仅编译芯片模型，可选择下载量化模型，下载后的模型存放在`output/$HOUMO_TARGET/hmquant`目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type quant
```

也可以直接下载以上所有模型（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py
```

### 2.2 量化

该模型目前不支持用户自行量化，如有需要，请联系我们。

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在`output/$HOUMO_TARGET`目录：
wenet_encoder.hmm

### 2.3 演示

wenet模型使用python API进行演示

1. 执行脚本，推理完成将输出性能结果。

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
audio input file: /hmdd/data/audio/4s.wav
大学生利用漏洞免费吃肯德基获刑
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
