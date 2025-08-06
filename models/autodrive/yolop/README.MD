# YoloP

本示例展示如何把yolop模型量化，编译，最后部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 yolop 实现来源 github 开源项目[https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)的master分支。模型代码做一定调整后导出，细节见量化工具文档。

本例模型的训练数据集为bdd100k

## 2.快速开始

可以通过hmatc工具一键执行性能测试和精度测试，详细介绍请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1       # 芯片
```

执行完成后会打印模型的数据集精度信息。

### 2.3 性能评估

```bash
hmatc perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 8
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.4 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh1
hmatc demo -c config.yml -t xh1
# onnx
hmatc demo -c config.yml -t xh1 --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh1/vis_onnx目录。

## 3.参考结果

### 3.1 性能结果

```bash
[latency] Inference     avg:  16.789 ms,        max:  18.061 ms
[latency] Input         avg:   0.738 ms,        max:   2.871 ms
[latency] Output        avg:   0.000 ms,        max:   0.008 ms
[latency] End2End       avg:  17.528 ms,        max:  20.895 ms
[Throughput] total: 2200.308 ms, avg: 2.200 ms
[Throughput] qps: 454.482
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- yolop 模型：[https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx](https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx)
- BDD100K 验证集链接: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
