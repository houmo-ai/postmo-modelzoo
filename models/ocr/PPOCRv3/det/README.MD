# PPOCRv3 det

本示例展示如何把量化后的PPOCRv3 det模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 PPOCRv3 实现来源 PaddleOCR 开源项目[https://www.paddleocr.ai/v2.10.0/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.html](https://www.paddleocr.ai/v2.10.0/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.html)。

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。
本模型是根据官方说明使用CCPD2020数据集重新微调训练得到，官方模型评估结果可能与我们报告中结果不一致。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1       # 芯片, M30系列为xh1，M50系列为xh2
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1       # 芯片, M30系列为xh1，M50系列为xh2
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh1          # 芯片, M30系列为xh1，M50系列为xh2
hmatc eval -c config.yml -t xh1  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 4   # 芯片, M30系列为xh1，M50系列为xh2
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh2
hmatc demo -c config.yml -t xh1             # 芯片, M30系列为xh1，M50系列为xh2
# onnx
hmatc demo -c config.yml -t xh1 --onnx      # 芯片, M30系列为xh1，M50系列为xh2
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh1/vis_onnx目录。

## 3.参考结果

### 3.1 精度结果

基于CCPD2020数据的结果

```bash
# xh1
{'input_size': [1, 3, 1184, 736], 'dataset': 'CCPD2020Val', 'num': 1001, 'hmean': '0.998501', 'latency': '60.359270'}
# xh2
{'input_size': [1, 3, 1184, 736], 'dataset': 'CCPD2020Val', 'num': 1001, 'hmean': '0.998501', 'latency': '73.519865'}
# onnx
{'input_size': [1, 3, 1184, 736], 'dataset': 'CCPD2020Val', 'num': 1001, 'hmean': '0.998501', 'latency': '50.945235'}
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:  57.204 ms, max:  64.370 ms, min:  57.109 ms, tp99:  57.586 ms, tp999:  64.370 ms
[Latency] Input      avg:   0.416 ms, max:   4.295 ms, min:   0.229 ms, tp99:   0.629 ms, tp999:   4.295 ms
[Latency] Output     avg:   0.574 ms, max:   7.673 ms, min:   0.361 ms, tp99:   0.927 ms, tp999:   7.673 ms
[Latency] End2end    avg:  58.193 ms, max:  66.461 ms, min:  57.737 ms, tp99:  60.068 ms, tp999:  66.461 ms
[Throughput] total: 58195.113 ms, avg: 58.195 ms, repeat: 1000, rounds: 1
[Throughput] qps: 17.184
# xh2
[Latency] Inference  avg:  62.217 ms, max:  62.605 ms, min:  61.571 ms, tp99:  62.484 ms, tp999:  62.605 ms
[Latency] Input      avg:   2.518 ms, max:   4.715 ms, min:   1.229 ms, tp99:   2.784 ms, tp999:   4.715 ms
[Latency] Output     avg:   1.012 ms, max:   2.259 ms, min:   0.456 ms, tp99:   1.222 ms, tp999:   2.259 ms
[Latency] End2end    avg:  65.748 ms, max:  67.996 ms, min:  63.942 ms, tp99:  66.189 ms, tp999:  67.996 ms
[Throughput] total: 65749.914 ms, avg: 65.750 ms, repeat: 1000, rounds: 1
[Throughput] qps: 15.209 
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- PPOCRDet 模型：[https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.10](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.10)
- CCPD2020 数据集: [https://aistudio.baidu.com/aistudio/datasetdetail/101595](https://aistudio.baidu.com/aistudio/datasetdetail/101595)
