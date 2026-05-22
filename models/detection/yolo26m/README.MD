# YOLO26m

本示例展示如何将量化后的 `YOLO26m` 模型编译并部署到后摩芯片设备上。

## 目录

- [1. 模型说明](#1-模型说明)
- [2. 快速开始](#2-快速开始)
- [3. 参考结果](#3-参考结果)
- [4. 免责声明](#4-免责声明)

## 1. 模型说明

本例使用的模型来自开源项目 [Ultralytics](https://github.com/ultralytics/ultralytics)，模型下载链接：

- **YOLO26m**: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt

下载模型后可使用 `ultralytics` 工具导出为 `onnx` 模型：

```bash
pip3 install ultralytics
yolo export model=yolo26m.pt format=onnx imgsz=640,640 opset=13 simplify=True
```

## 2. 快速开始

可以通过 `hmatc` 工具执行性能测试和精度测试，详细介绍请参考 houmo-modelzoo 根目录下 README.md 文件的快速上手章节。

### 2.1 量化

```bash
hmatc quant -c config.yml
```

### 2.2 编译

```bash
hmatc build -c config.yml
```

### 2.3 精度评估

```bash
# 芯片
hmatc eval -c config.yml
# onnx
hmatc eval -c config.yml --onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -wn 10 -sn 1000 -tn 4
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.5 结果演示

在编译完成的基础上执行：

```bash
# 芯片
hmatc demo -c config.yml
# onnx
hmatc demo -c config.yml --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在 `vis_xh2/` 或 `vis_onnx/` 目录。

## 3. 参考结果

### 3.1 精度结果

| 平台 | 输入尺寸 | 数据集 | 样本数 | mAP50-95 | mAP50 | 延迟 (ms) |
|------|----------|--------|--------|----------|-------|-----------|
| ONNX | 1×3×640×640 | COCO 2017 Val | 5000 | 0.504706 | 0.686292 | 168.62 |
| XH2  | 1×3×640×640 | COCO 2017 Val | 5000 | 0.500551 | 0.680349 | 21.24 |

### 3.2 性能结果

```
[Latency] Inference  avg:  33.105 ms, max:  33.659 ms, min:  32.337 ms, tp99:  33.547 ms, tp999:  33.659 ms
[Latency] Input      avg:   0.697 ms, max:   0.821 ms, min:   0.435 ms, tp99:   0.777 ms, tp999:   0.821 ms
[Latency] Output     avg:   0.157 ms, max:   0.316 ms, min:   0.047 ms, tp99:   0.196 ms, tp999:   0.316 ms
[Latency] End2end    avg:  33.960 ms, max:  34.393 ms, min:  33.158 ms, tp99:  34.269 ms, tp999:  34.393 ms
[Throughput] total: 8517.324 ms, avg: 8.517 ms, repeat: 1000, rounds: 1
[Throughput] qps: 117.408
```

## 4. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLO26m 模型: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt
- COCO 验证集: https://cocodataset.org/#download