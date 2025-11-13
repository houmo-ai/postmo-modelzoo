# YOLOXm

本示例展示如何把量化后的YOLOXm模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOX 实现来源 github 开源项目[https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)。

下载模型后可使用tools目录下的export_onnx.py工具导出为onnx模型，执行命令：
```bash
cd YOLOX_HOME
python3 tools/export_onnx.py --output-name yolox_m.onnx -n yolox-m -c yolox_m.pth
```

YOLOX导出的模型，前面会有多个Slice算子，可以将其统一利用hmatc的图优化功能替换为Conv算子。可以通过在config.yaml配置的model下添加如下配置开启优化。
``` yaml
  app_onnx_opt: 
    optimizer: true
    log_level: 0
```

## 2.快速开始

可以通过hmatc工具执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1/xh2       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1/xh2       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh1/xh2          # 芯片
hmatc eval -c config.yml -t xh1/xh2  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh1/xh2 -wn 10 -sn 1000 -tn 8
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh1
hmatc demo -c config.yml -t xh1/xh2
# onnx
hmatc demo -c config.yml -t xh1/xh2 --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh1/vis_onnx目录。

## 3.参考结果

### 3.1 精度结果
```bash
# xh1
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 32, 'map50_95': '0.605732', 'map50': '0.819257', 'latency': '12.069121'
# onnx
'input_size': [1, 3, 640, 640], 'dataset': 'coco_2017Val', 'num': 32, 'map50_95': '0.642724', 'map50': '0.831999', 'latency': '96.470609'
```

### 3.2 性能结果

```bash
[latency] Inference     avg:  19.496 ms,        max:  21.348 ms
[latency] Input         avg:   1.395 ms,        max:   2.892 ms
[latency] Output        avg:   0.000 ms,        max:   0.000 ms
[latency] End2End       avg:  20.892 ms,        max:  24.227 ms
[Throughput] total: 2620.077 ms, avg: 2.620 ms
[Throughput] qps: 381.668
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOXm 模型：[https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
