# PPOCRv3 det

本示例展示如何把量化后的PPOCRv3 rec模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 PPOCRv3 实现来源 PaddleOCR 开源项目[https://www.paddleocr.ai/v2.10.0/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.html](https://www.paddleocr.ai/v2.10.0/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.html)。

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-examples根目录下README.md文件的快速上手章节。
本模型是根据官方说明使用CCPD2020数据集重新微调训练得到，官方模型评估结果可能与我们报告中结果不一致。

### 2.1 精度评估

```bash
source ~/houmo-examples-xh2/env.sh     # 初始化环境
```

### 2.1 量化

```bash
hmatc quant -c config.yml
```

### 2.2 编译

```bash
hmatc build -c config.yml
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml
hmatc eval -c config.yml --onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -wn 10 -sn 1000 -tn 4
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh2
hmatc demo -c config.yml
# onnx
hmatc demo -c config.yml --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh2/vis_onnx目录。

## 3.参考结果

### 3.1 性能结果

```bash
# 时延
[Latency] Inference  avg:   2.586 ms, max:   2.925 ms, min:   2.355 ms, tp99:   2.895 ms, tp999:   2.925 ms
[Latency] Input      avg:   0.072 ms, max:   0.210 ms, min:   0.032 ms, tp99:   0.095 ms, tp999:   0.210 ms
[Latency] Output     avg:   0.214 ms, max:   0.361 ms, min:   0.111 ms, tp99:   0.252 ms, tp999:   0.361 ms
[Latency] End2end    avg:   2.872 ms, max:   3.345 ms, min:   2.500 ms, tp99:   3.215 ms, tp999:   3.345 ms
[Throughput] total: 2873.640 ms, avg: 2.874 ms, repeat: 1000, rounds: 1
[Throughput] qps: 347.991

# 吞吐
[Latency] Inference  avg:   4.336 ms, max:   4.537 ms, min:   2.467 ms, tp99:   4.513 ms, tp999:   4.537 ms
[Latency] Input      avg:   0.073 ms, max:   0.191 ms, min:   0.032 ms, tp99:   0.120 ms, tp999:   0.191 ms
[Latency] Output     avg:   0.226 ms, max:   0.395 ms, min:   0.112 ms, tp99:   0.356 ms, tp999:   0.395 ms
[Latency] End2end    avg:   4.635 ms, max:   4.935 ms, min:   2.752 ms, tp99:   4.736 ms, tp999:   4.935 ms
[Throughput] total: 1175.978 ms, avg: 1.176 ms, repeat: 1000, rounds: 1
[Throughput] qps: 850.356
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- PPOCRDet 模型：[https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.10](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.10)
- CCPD2020 数据集: [https://aistudio.baidu.com/aistudio/datasetdetail/101595](https://aistudio.baidu.com/aistudio/datasetdetail/101595)
