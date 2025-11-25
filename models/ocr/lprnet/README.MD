# LPRNet

本示例展示如何把量化后的LPRNet模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 LPRNet 实现来源 github 开源项目[https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition)。

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。
评估时，我们根据官网和官方博客链接说明，对CCPD2019进行数据预处理，拆分了base中的19万张图像，作为训练、测试和推理数据，由于官方提供的脚本具有随机性，所以客户自行实现数据拆分时可能会与我司提供的不一致。故，我们提供了我们已经拆分好的测试和验证数据集，详细的下载链接可以参考get_model.py脚本。

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

基于CCPD2019SubDataSet数据的结果（非全量数据集）

```bash
# xh1
{'input_size': [1, 3, 24, 94], 'dataset': 'CCPD2019Sub', 'num': 19139, 'acc': '0.805005', 'latency': '1.345849'}
# xh2
{'input_size': [1, 3, 24, 94], 'dataset': 'CCPD2019Sub', 'num': 19139, 'acc': '0.797011', 'latency': '1.508958'}
# onnx
{'input_size': [1, 3, 24, 94], 'dataset': 'CCPD2019Sub', 'num': 19139, 'acc': '0.817075', 'latency': '2.401659'}
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:   0.993 ms, max:   7.995 ms, min:   0.923 ms, tp99:   2.738 ms, tp999:   7.995 ms
[Latency] Input      avg:   0.061 ms, max:   3.993 ms, min:   0.030 ms, tp99:   0.042 ms, tp999:   3.993 ms
[Latency] Output     avg:   0.040 ms, max:   7.192 ms, min:   0.030 ms, tp99:   0.044 ms, tp999:   7.192 ms
[Latency] End2end    avg:   1.094 ms, max:  12.035 ms, min:   0.988 ms, tp99:   4.232 ms, tp999:  12.035 ms
[Throughput] total: 1095.847 ms, avg: 1.096 ms, repeat: 1000, rounds: 1
[Throughput] qps: 912.536
# xh2
[Latency] Inference  avg:   0.739 ms, max:   1.027 ms, min:   0.668 ms, tp99:   0.856 ms, tp999:   1.027 ms
[Latency] Input      avg:   0.039 ms, max:   0.048 ms, min:   0.020 ms, tp99:   0.043 ms, tp999:   0.048 ms
[Latency] Output     avg:   0.023 ms, max:   0.081 ms, min:   0.011 ms, tp99:   0.028 ms, tp999:   0.081 ms
[Latency] End2end    avg:   0.801 ms, max:   1.091 ms, min:   0.700 ms, tp99:   0.914 ms, tp999:   1.091 ms
[Throughput] total: 802.670 ms, avg: 0.803 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1245.842
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
