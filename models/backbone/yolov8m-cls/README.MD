# YOLOv8m-Cls

本示例展示如何把量化后的YOLOv8m模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv8m-Cls 实现来源 github 开源项目[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt)。

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：
```bash
pip3 install ultralytics
yolo export model=yolov8m-cls.pt format=onnx imgsz=224,224 opset=13 simplify=True
```

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-examples根目录下README.md文件的快速上手章节。
校准使用的 ImageNet 2012 数据集需自行从官方渠道下载，下载方式请参考[数据集说明](../../../data/datasets/README.md)。

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
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.744860'
# xh2
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.743420'
```

### 3.2 性能结果

```bash
[Latency] Inference  avg:   2.746 ms, max:   2.861 ms, min:   1.745 ms, tp99:   2.826 ms, tp999:   2.861 ms
[Latency] Input      avg:   0.238 ms, max:   0.362 ms, min:   0.170 ms, tp99:   0.325 ms, tp999:   0.362 ms
[Latency] Output     avg:   0.039 ms, max:   0.078 ms, min:   0.023 ms, tp99:   0.069 ms, tp999:   0.078 ms
[Latency] End2end    avg:   3.023 ms, max:   3.146 ms, min:   2.049 ms, tp99:   3.080 ms, tp999:   3.146 ms
[Throughput] total: 761.950 ms, avg: 0.762 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1312.422
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv8m-Cls 模型：[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
