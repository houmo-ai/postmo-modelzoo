# houmo-modelzoo

## 概述

houmo-modelzoo是为用户快速将模型移植到后摩鸿途H30芯片产品而开发的模型库，为用户提供量化、编译、精度和性能评估等一整套代码和工具，降低用户的学习和开发成本。

## 目录


| 目录     | 说明                 | 备注         |
| -------- | -------------------- | ------------ |
| cv       | 计算机视觉相关模型   |              |
| data     | 评估使用的模型和数据 |              |
| hmodel   | 量化模型配置和工具   | QAT量化使用  |
| hmassist | 参数化评估辅助工具   | 一键完成评估 |
| utils    | C++评估工具和源码    |              |

## 模型支持列表

### ComputerVision：


| MODELS                          | ptq | qat | py-demo | cpp-demo | accuracy |
| ------------------------------- | --- | --- | ------- | -------- | -------- |
| [resnet50](cv/resnet50)         | yes | yes | yes     | yes      | yes      |
| [yolov5s](cv/yolov5s)           | yes | x   | yes     | x        | x        |
| [yolov3](cv/yolov3)             | yes | x   | x       | yes      | x        |
| [mobilenetv2](cv/mobilenet_v2)  | yes | x   | yes     | x        | yes      |
| [efficientnet](cv/efficientnet) | yes | x   | x       | x        | x        |
| [vit](cv/vit)                   | yes | x   | x       | x        | x        |

### AutoDrive:


| MODELS                          | ptq | qat | py-demo | cpp-demo | accuracy |
| ------------------------------- | --- | --- | ------- | -------- | -------- |
| [yolop](cv/yolop)               | yes | yes | yes     | x        | x        |
| [pointpillars](cv/pointpillars) | yes | x   | x       | yes      | yes      |

## C++评估工具


| PROJECT                                                                         | 说明                 |
| ------------------------------------------------------------------------------- | -------------------- |
| [classification](uilts/aotlassification)                                        | imagenet分类精度测试 |
| [tcimexec](uilts/aottcimexec)                                                   | 单线程性能测试       |
| [threadtcimexec](uilts/aottcimexec)                                             |                      |
| [multi_thread_stream_tcim_exec](uilts/multi_thread_stream_tcim_execaottcimexec) | 多线程性能测试       |
