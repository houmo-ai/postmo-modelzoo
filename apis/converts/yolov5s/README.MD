# YOLOv5s

本示例展示如何通过`hmatc`工具量化和编译`YOLOv5s`模型。

[TOC]

## 1.模型说明

本例使用的`YOLOv5s` 实现来源`github`开源项目[https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python export.py --weights ./yolov5s.pt --simplify --include onnx --opset 13
```

## 2.快速开始

### 2.1 量化

```bash
hmatc quant -c config.yml
```

### 2.2 编译

```bash
hmatc build -c config.yml
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv5s 模型：[https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
