# YOLOv8m

本示例展示如何把量化后的YOLOv8m模型编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv8m 实现来源 github 开源项目[https://github.com/ultralytics/ultralytics/tree/v8.2.0](https://github.com/ultralytics/ultralytics/tree/v8.2.0)。模型代码导出后，再去除掉yolo相关的节点，细节见量化工具文档。

下载模型后可使用ultralytics工具导出为onnx模型，执行命令：
```bash
pip3 install ultralytics
yolo export model=yolov8m.pt format=onnx imgsz=640,640 opset=11
```

## 2.快速开始

可以通过perf.sh和eval.sh脚本一键执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 性能评估

```bash
bash perf.sh
```

执行完成后会打印模型推理延迟、吞吐量等信息。

#### 2.2 精度评估

```bash
bash eval.sh
```

执行完成后会打印模型的数据集精度信息。

## 3.参考结果

### 3.1精度结果
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.621
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.821
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.670
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.735
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.778
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.776
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.839
```

### 3.2性能结果

```bash
[latency] Inference 	avg:  11.748 ms,	max:  12.349 ms
[latency] Input 	avg:   0.676 ms,	max:   1.155 ms
[latency] Output 	avg:   0.846 ms,	max:   1.438 ms
[latency] End2End 	avg:  13.272 ms,	max:  14.933 ms
[Throughput] total: 13277.102 ms, avg: 13.277 ms
[Throughput] qps: 75.318
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv8m 模型：[https://github.com/ultralytics/ultralytics/tree/v8.2.0](https://github.com/ultralytics/ultralytics/tree/v8.2.0
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
