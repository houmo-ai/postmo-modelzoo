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

`YOLOX`导出的模型，前面会有多个`Slice`算子，可以将其统一利用hmatc的图优化功能替换为Conv算子。可以通过在config.yaml配置的model下添加如下配置开启优化。

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

### 3.1 性能结果

```bash
[Latency] Inference  avg:  21.441 ms, max:  23.336 ms, min:  13.189 ms, tp99:  22.870 ms, tp999:  23.336 ms
[Latency] Input      avg:   2.138 ms, max:  10.335 ms, min:   1.121 ms, tp99:   2.570 ms, tp999:  10.335 ms
[Latency] Output     avg:   1.153 ms, max:   5.390 ms, min:   0.657 ms, tp99:   1.393 ms, tp999:   5.390 ms
[Latency] End2end    avg:  24.732 ms, max:  28.898 ms, min:  20.409 ms, tp99:  25.279 ms, tp999:  28.898 ms
[Throughput] total: 6201.183 ms, avg: 6.201 ms, repeat: 1000, rounds: 1
[Throughput] qps: 161.260
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOXm 模型：[https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
