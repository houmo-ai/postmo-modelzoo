# YOLOv10m

本示例展示如何把量化后的YOLOv10m模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv10m 实现来源 github 开源项目[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt)。

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：
```bash
pip3 install ultralytics
yolo export model=yolov10m.pt format=onnx imgsz=640,640 opset=13 simplify=True
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

> **注意：** 
> 
> 性能测试的线程数建议是芯片核数的两倍，核数可通过环境变量HOUMO_CORE_NUM查看
>


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
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.489048', 'map50': '0.667118'
# xh1
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.482865', 'map50': '0.657273'
# xh2
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.487788', 'map50': '0.665831'
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:  30.035 ms, max:  31.338 ms, min:  24.975 ms, tp99:  30.997 ms, tp999:  31.338 ms
[Latency] Input      avg:   0.324 ms, max:   0.513 ms, min:   0.158 ms, tp99:   0.419 ms, tp999:   0.513 ms
[Latency] Output     avg:   0.967 ms, max:   1.615 ms, min:   0.505 ms, tp99:   1.292 ms, tp999:   1.615 ms
[Latency] End2end    avg:  31.326 ms, max:  32.867 ms, min:  26.305 ms, tp99:  32.411 ms, tp999:  32.867 ms
[Throughput] total: 3975.190 ms, avg: 3.975 ms, repeat: 1000, rounds: 1
[Throughput] qps: 251.560

# xh2
[Latency] Inference  avg:  23.414 ms, max:  23.939 ms, min:  14.295 ms, tp99:  23.769 ms, tp999:  23.939 ms
[Latency] Input      avg:   0.911 ms, max:   1.876 ms, min:   0.839 ms, tp99:   1.627 ms, tp999:   1.876 ms
[Latency] Output     avg:   1.345 ms, max:   2.509 ms, min:   0.931 ms, tp99:   2.190 ms, tp999:   2.509 ms
[Latency] End2end    avg:  25.670 ms, max:  27.112 ms, min:  16.356 ms, tp99:  26.170 ms, tp999:  27.112 ms
[Throughput] total: 6445.712 ms, avg: 6.446 ms, repeat: 1000, rounds: 1
[Throughput] qps: 155.142
```

>**注意：**
>
> 结果均是以芯片核数两倍的线程数来测试，这里分别是8和4线程，仅供参考，具体结果以实际测试为准。

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv10m 模型：[https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
