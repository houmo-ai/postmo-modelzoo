# YOLOv5s

本示例展示如何把量化后的YOLOv5s模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv5s 实现来源 github 开源项目[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)的v9.0 tag。

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：
```bash
pip3 install ultralytics
yolo export model=yolov5s.pt format=onnx imgsz=640,640 opset=11
```

由于最后yolo检测部分以及3个尺度的输出融合会显著降低ptq量化的精度，所以建议截取3个分支yolo前的模型进行量化，相关截取操作在get_model.py中。
如果想继续提升精度，还可以使用混合精度的方法，具体可参考量化相关文档。

## 2.快速开始

可以通过perf.sh和eval.sh脚本一键执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 精度评估

```bash
bash eval.sh
```

执行完成后会打印模型的数据集精度信息。

### 2.2 性能评估

```bash
bash perf.sh
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.3 结果演示

本示例依赖hmassist演示结果，在编译完成的基础上执行：

```bash
hmdemo.sh
```

执行完成后会打印模型的检测结果信息，图片结果将保存在demo_results目录。

## 3.参考结果

### 3.1 精度结果

```bash
{'input_size': '1x3x640x640', 'dataset': 'coco2017', 'num': 5000, 'map': '0.469', 'map50': '0.674', 'latency': '6.958'}
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.674
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
```

### 3.2 性能结果

```bash
[latency] Inference 	avg:   3.128 ms,	max:   3.594 ms
[latency] Input 	avg:   0.380 ms,	max:   0.713 ms
[latency] Output 	avg:   3.425 ms,	max:   5.555 ms
[latency] End2End 	avg:   6.934 ms,	max:   9.826 ms
[Throughput] total: 6937.970 ms, avg: 6.938 ms
[Throughput] qps: 144.134
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv5s 模型：[https://github.com/ultralytics/yolov3/tree/v9.6.0](https://github.com/ultralytics/yolov3/tree/v9.0)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
