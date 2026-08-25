# DINOv3-base

本示例展示如何把量化后的`DINOv3-base`模型量化、编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

### 1.1 基础说明

本例使用的`DINOv3`模型地址：[https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)

下载模型后可使用`transformers`库导出为`onnx`模型，示例使用DINOv3-vit base patch16预训练模型作为骨干网络。

建议导出用onnxsim工具简化后使用。

### 1.2 进阶说明

dinov3系列模型为视觉语言基座大模型，主要提供特征提取功能，然后接上例如图像分类、目标检测、语义分割等后任务head实现具体的视觉AI任务逻辑，所以这些后任务的head需要客户自行在特定数据集进行监督、半监督或无监督训练，并在部署时串接到dinov3的backbone上即可。训练时可直接冻结dinov3骨干的权重更新。
本例提供了一个在imagenet上训练的图像分类示例供参考。我们只使用了imagenet1000k数据集的验证集做特征提取然后喂给分类head linear进行多个epoch训练后得到jit trace模型，并在示例的后处理中进行加载，且使用了cpu推理（实际部署时可使用量化工具加载或者直接接到dinov3的onnx后重新量化和编译即可）。训练脚本可参考train_dinov3_classifier_head.py，分为dinov3提取特征，然后训练linear head。

## 2.快速开始

通过hmatc工具执行性能测试和精度测试。
校准使用的 ImageNet 2012 数据集需自行下载，下载方式请参考[数据集说明](../../../data/datasets/README.md)。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh2       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh2       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh2          # 芯片
hmatc eval -c config.yml -t xh2  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh2 -wn 10 -sn 1000 -tn 4
```

执行完成后会打印模型推理延迟、吞吐量等信息。

> **注意：** 
> 
> 性能测试的线程数建议是芯片核数的两倍，核数可通过环境变量HOUMO_CORE_NUM查看
>


### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh2
hmatc demo -c config.yml -t xh2
# onnx
hmatc demo -c config.yml -t xh2 --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh2/vis_onnx目录。

## 3.参考结果

### 3.1 精度结果

```bash
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 10000, 'top1_acc': '0.957900'
# xh2
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 10000, 'top1_acc': '0.978900'
```
可使用完整imagenet验证集验证。

### 3.2 性能结果

```bash
[Latency] Inference  avg:   9.461 ms, max:  12.550 ms, min:   8.948 ms, tp99:   9.759 ms, tp999:  12.550 ms
[Latency] Input      avg:   0.331 ms, max:   0.677 ms, min:   0.069 ms, tp99:   0.601 ms, tp999:   0.677 ms
[Latency] Output     avg:   0.488 ms, max:   0.828 ms, min:   0.176 ms, tp99:   0.727 ms, tp999:   0.828 ms
[Latency] End2end    avg:  10.281 ms, max:  12.953 ms, min:   9.252 ms, tp99:  11.031 ms, tp999:  12.953 ms
[Throughput] total: 10282.537 ms, avg: 10.283 ms, repeat: 1000, rounds: 1
[Throughput] qps: 97.252
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- DINOv3-base 模型：[https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
