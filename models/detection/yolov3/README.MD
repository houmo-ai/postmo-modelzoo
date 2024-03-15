# YOLOv3

本示例展示如何把量化后的YOLOv3模型量化，编译，部署到鸿途H30芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 YOLOv3 实现来源 github 开源项目[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)的v9.0 tag。模型代码导出后，再去除掉yolo相关的节点，细节见量化工具文档。

## 2.快速开始

请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- YOLOv3 模型：[https://github.com/ultralytics/yolov3/tree/v9.6.0](https://github.com/ultralytics/yolov3/tree/v9.0)
- COCO 验证集链接: [https://cocodataset.org/#download](https://cocodataset.org/#download)
