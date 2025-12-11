# YOLOv5s

本示例展示如何不使用`Resizer`将`YOLOv5s`模型编译，部署到后摩芯片上。

[TOC]

## 1.模型说明

本例使用的`YOLOv5s` 实现来源`github`开源链接[https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python export.py --weights ./yolov5s.pt --simplify --include onnx --opset 13
```

## 2.快速开始

可以通过`hmatc`工具执行性能测试和精度测试，详细介绍请参考`houmo-modelzoo`根目录下`README.md`文件的快速上手章节。

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
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.361782', 'map50': '0.557085'
# xh2
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 5000, 'map50_95': '0.355800', 'map50': '0.555100'
```

### 3.2 性能结果

```bash
[Latency] Inference  avg:   5.310 ms, max:   7.520 ms, min:   4.541 ms, tp99:   6.633 ms, tp999:   7.520 ms
[Latency] Input      avg:   2.087 ms, max:   7.427 ms, min:   1.096 ms, tp99:   2.760 ms, tp999:   7.427 ms
[Latency] Output     avg:   3.719 ms, max:   8.450 ms, min:   2.167 ms, tp99:   4.868 ms, tp999:   8.450 ms
[Latency] End2end    avg:  11.117 ms, max:  15.629 ms, min:   8.689 ms, tp99:  12.859 ms, tp999:  15.629 ms
[Throughput] total: 2803.001 ms, avg: 2.803 ms, repeat: 1000, rounds: 1
[Throughput] qps: 356.760
```

性能结果相对使用`static_resizer`性能相差较大，非必要不建议将输出作为`feature`。

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv5s 模型：[https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
