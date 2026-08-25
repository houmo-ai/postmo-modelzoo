# YOLOv5m-Face

本示例展示如何把量化后的YOLOv5m-face模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv5m-face 实现来源 github 开源项目[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)。

下载模型文件(yolov5m-face.pt)后，可使用开源项目中的export.py脚本导出为onnx模型，执行命令：
```bash
python3 export.py --weights <yolov5m-face.pt路径>
```

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-examples根目录下README.md文件的快速上手章节。
校准使用的 WIDER FACE 数据集需自行下载，下载方式请参考[数据集说明](../../../data/datasets/README.md)。

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

基于data/datasets/widerface数据的结果（非全量数据集）

```bash
# onnx
{'input_size': [1, 3, 640, 640], 'dataset': 'widerface', 'num': 41, 'ap_easy': '0.907834', 'ap_medium': '0.895021', 'ap_hard': '0.829818'}
# xh2
{'input_size': [1, 3, 640, 640], 'dataset': 'widerface', 'num': 41, 'ap_easy': '0.891293', 'ap_medium': '0.891500', 'ap_hard': '0.842538'}
```

### 3.2 性能结果

```bash
[Latency] Inference  avg:  13.191 ms, max:  14.118 ms, min:   8.711 ms, tp99:  13.956 ms, tp999:  14.118 ms
[Latency] Input      avg:   0.694 ms, max:   0.865 ms, min:   0.283 ms, tp99:   0.800 ms, tp999:   0.865 ms
[Latency] Output     avg:   0.627 ms, max:   0.769 ms, min:   0.315 ms, tp99:   0.713 ms, tp999:   0.769 ms
[Latency] End2end    avg:  14.512 ms, max:  15.099 ms, min:  10.050 ms, tp99:  14.925 ms, tp999:  15.099 ms
[Throughput] total: 3644.023 ms, avg: 3.644 ms, repeat: 1000, rounds: 1
[Throughput] qps: 274.422
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv5-face 模型：[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- Widerface 验证集链接: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
