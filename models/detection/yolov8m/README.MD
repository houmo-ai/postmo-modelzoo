# YOLOv8m

本示例展示如何把量化后的YOLOv8m模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv8m 实现来源 github 开源项目[https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)。

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：
```bash
pip3 install ultralytics
yolo export model=yolov8m.pt format=onnx imgsz=640,640 opset=13 simplify=True
```

本模型int8精度稍差，需要混合量化。本示例已使用混合量化配置，所以性能方面有所损失。

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
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.478298', 'map50': '0.655562'
# xh1
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.458586', 'map50': '0.646231'
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:  26.222 ms, max:  29.308 ms, min:  12.530 ms, tp99:  27.006 ms, tp999:  29.308 ms
[Latency] Input      avg:   0.277 ms, max:   0.482 ms, min:   0.128 ms, tp99:   0.355 ms, tp999:   0.482 ms
[Latency] Output     avg:   0.338 ms, max:   0.662 ms, min:   0.168 ms, tp99:   0.487 ms, tp999:   0.662 ms
[Latency] End2end    avg:  26.837 ms, max:  29.950 ms, min:  13.116 ms, tp99:  27.627 ms, tp999:  29.950 ms
[Throughput] total: 3416.459 ms, avg: 3.416 ms, repeat: 1000, rounds: 1
[Throughput] qps: 292.701
```

>**注意：**
>
> 结果均是以芯片核数两倍的线程数来测试，这里是8线程，仅供参考，具体结果以实际测试为准。


## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv8m 模型：[https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
