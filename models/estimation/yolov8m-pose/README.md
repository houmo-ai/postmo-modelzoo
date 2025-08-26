# YOLOv8m-Pose

本示例展示如何把量化后的`YOLOv8m-Pose`模型编译，部署到后摩芯片的设备上。

## 1.模型说明

本例使用的模型来自开源项目`Ultralytics`，链接为：[**https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt**](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt)

下载模型后可使用`ultralytics`工具导出为`onnx`模型，执行命令：

```bash
pip3 install ultralytics
yolo export model=yolov8m-pose.pt format=onnx imgsz=640,640 opset=11 simplify=True
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
# xh1
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 32, 'map50_95': '0.511982', 'map50': '0.830081', 'latency': '12.723356'
# onnx
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 32, 'map50_95': '0.584225', 'map50': '0.819675', 'latency': '91.764905'
```

### 3.2 性能结果

```bash
[latency] Inference     avg:  18.076 ms,        max:  19.792 ms
[latency] Input         avg:   1.341 ms,        max:   2.898 ms
[latency] Output        avg:   0.000 ms,        max:   0.000 ms
[latency] End2End       avg:  19.418 ms,        max:  22.394 ms
[Throughput] total: 4864.899 ms, avg: 2.432 ms
[Throughput] qps: 411.108
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv8m-Pose 模型：[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)