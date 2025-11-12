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

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh1          # 芯片
hmatc eval -c config.yml -t xh1  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 8
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh1
hmatc demo -c config.yml -t xh1
# onnx
hmatc demo -c config.yml -t xh1 --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh1/vis_onnx目录。

## 3.参考结果

### 3.1 精度结果
```bash
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.744860'
# xh1
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.742500'
# xh2
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 50000, 'top1_acc': '0.743420'
```

### 3.2 性能结果

```bash
# xh1 performance
[Latency] Inference  avg:   4.476 ms, max:   5.507 ms, min:   2.696 ms, tp99:   4.969 ms, tp999:   5.507 ms
[Latency] Input      avg:   1.402 ms, max:   2.440 ms, min:   0.737 ms, tp99:   1.942 ms, tp999:   2.440 ms
[Latency] Output     avg:   0.068 ms, max:   0.160 ms, min:   0.039 ms, tp99:   0.111 ms, tp999:   0.160 ms
[Latency] End2end    avg:   5.945 ms, max:   6.778 ms, min:   3.964 ms, tp99:   6.542 ms, tp999:   6.778 ms
[Throughput] total: 790.440 ms, avg: 0.790 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1265.118
# xh2 performance
[Latency] Inference  avg:   3.297 ms, max:   4.305 ms, min:   1.824 ms, tp99:   3.396 ms, tp999:   4.305 ms
[Latency] Input      avg:   0.123 ms, max:   0.619 ms, min:   0.073 ms, tp99:   0.146 ms, tp999:   0.619 ms
[Latency] Output     avg:   0.022 ms, max:   0.043 ms, min:   0.010 ms, tp99:   0.032 ms, tp999:   0.043 ms
[Latency] End2end    avg:   3.442 ms, max:   4.497 ms, min:   1.967 ms, tp99:   3.544 ms, tp999:   4.497 ms
[Throughput] total: 869.322 ms, avg: 0.869 ms, repeat: 1000, rounds: 1
[Throughput] qps: 1150.322
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv8m-Cls 模型：[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
