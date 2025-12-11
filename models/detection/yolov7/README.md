# YOLOv7

本示例展示如何把量化后的YOLOv7模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv7 实现来源 github 开源项目[https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)。

下载模型后可使用官方代码导出为onnx模型，执行命令：
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python export.py --weights yolov7.pt --grid --simplify
```

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

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
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.491601', 'map50': '0.684097'
# xh2
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.491300', 'map50': '0.683800'
```

### 3.2 性能结果

```bash
[Latency] Inference  avg:  16.003 ms, max:  18.085 ms, min:  14.759 ms, tp99:  17.988 ms, tp999:  18.085 ms
[Latency] Input      avg:   1.925 ms, max:   2.261 ms, min:   1.096 ms, tp99:   2.201 ms, tp999:   2.261 ms
[Latency] Output     avg:   3.151 ms, max:   3.980 ms, min:   1.970 ms, tp99:   3.824 ms, tp999:   3.980 ms
[Latency] End2end    avg:  21.078 ms, max:  22.202 ms, min:  19.809 ms, tp99:  22.000 ms, tp999:  22.202 ms
[Throughput] total: 5283.884 ms, avg: 5.284 ms, repeat: 1000, rounds: 1
[Throughput] qps: 189.255
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv7 模型：[https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
