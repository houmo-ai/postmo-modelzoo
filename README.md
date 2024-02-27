# houmo-modelzoo

## 概述

houmo-modelzoo是为用户快速将模型移植到后摩鸿途系列芯片产品而开发的模型库，为用户提供量化、编译、精度和性能评估等一整套代码和工具，降低用户的学习和开发成本。

## 目录

houmo-modelzoo目录结果如下，其中README.md为本说明文件：

```bash
|-- README.md
|-- models
|-- data
|-- hmodel
|-- hmassist
|-- utils
|-- benchmark.yml
|-- release.cmake
|-- requirements.txt
`-- env.sh

```

主要目录说明如下：

| 目录     | 说明                                     |
| -------- | --------------------------------------- |
| models   | 模型示例，模型按照主要用途分文件夹存放      |
| data     | 评估使用的数据文件，如数据集等             |
| hmodel   | 量化模型配置和工具，目前主要是QAT训练使用   |
| hmassist | 参数化评估辅助工具，通过配置文件实现快速评估 |
| utils    | C++测试工具和源码                         |

其他文件说明如下：

| 目录             | 说明                     |
| ---------------- | ----------------------- |
| benchmark.yml    | 模型基准测试配置文件      |
| release.cmake    | c++环境cmake配置文件     |
| requirements.txt | python环境依赖           |
| env.sh           | modelzoo环境配置脚本     |

## 模型示例列表

modelzoo提供的模型示例如下：

| MODELS                                       | ptq | qat | pydemo | cppdemo | eval |
| -------------------------------------------- | --- | --- | ------ | ------- | ---- |
| [resnet50](models/backbone/resnet50)         | yes | yes | yes    | yes     | yes  |
| [mobilenetv2](models/backbone/mobilenet_v2)  | yes | x   | yes    | x       | yes  |
| [efficientnet](models/backbone/efficientnet) | yes | x   | x      | x       | x    |
| [vit](models/backbone/vit)                   | yes | x   | x      | x       | x    |
| [yolov5s](models/detection/yolov5s)          | yes | x   | yes    | x       | x    |
| [yolov3](models/detection/yolov3)            | yes | x   | x      | yes     | x    |
| [yolop](models/autodrive/yolop)              | yes | yes | yes    | x       | x    |
| [pointpillars](models/detection/pointpillars)| yes | x   | x      | yes     | x    |

## C++评估工具列表

| PROJECT                                                                         | 说明                 |
| ------------------------------------------------------------------------------- | -------------------- |
| [classification](uilts/aotlassification)                                        | imagenet分类精度测试  |
| [tcim_perf](uilts/tcim_perf)                                                    | 性能基准测试工具      |

## 使用说明

|  ModelName   |      Shape       |  Dataset   | Batch | CoreNum | Accuracy(onnx) | Accuracy(H30)  | AccRelError | Latency(ms) |   Qps   |
| ------------ | ---------------- | ---------- | ----- | ------- | -------------- | -------------- | ----------- | ----------- | ------- |
|   resnet50   | [1, 224, 224, 3] | ILSVRC2012 |   1   |    1    | top1:0.750000  | top1:0.000000  |  top1:1.000 |    0.752    | 1330.47 |
|              |                  |            |       |         | top5:1.000000  | top5:0.000000  |  top5:1.000 |             |         |
| mobilenetv2  | [1, 224, 224, 3] | ILSVRC2012 |   1   |    1    |   top1:0.707   | top1:0.700000  |  top1:0.067 |    4.528    |  220.83 |
|              |                  |            |       |         |   top5:0.897   | top5:1.000000  |  top5:0.000 |             |         |
| efficientnet | [1, 224, 224, 3] | ILSVRC2012 |   1   |    1    | top1:0.800000  | top1:0.600000  |  top1:0.250 |    12.336   |  81.07  |
|              |                  |            |       |         | top5:1.000000  | top5:0.800000  |  top5:0.200 |             |         |
|     vit      | [1, 224, 224, 3] | ILSVRC2012 |   1   |    1    | top1:0.000000  | top1:0.000000  |             |    14.831   |  67.43  |
|              |                  |            |       |         | top5:0.050000  | top5:0.050000  |  top5:0.000 |             |         |
|   yolov5s    | [1, 640, 640, 3] |  coco2017  |   1   |    1    |  map:0.362095  |  map:0.333328  |  map:0.079  |    4.415    |  226.50 |
|              |                  |            |       |         | map50:0.556590 | map50:0.541854 | map50:0.026 |             |         |
|    yolov3    | [1, 640, 640, 3] |  coco2017  |   1   |    1    |  map:0.411684  | map:0.372781   |  map:0.094  |    27.496   |  36.37  |
|              |                  |            |       |         | map50:0.615220 | map50:0.606683 | map50:0.014 |             |         |
|    yolop     | [1, 384, 640, 3] |  NotTest   |   1   |    1    |    NotTest     |    NotTest     |   NotTest   |    43.674   |  22.90  |