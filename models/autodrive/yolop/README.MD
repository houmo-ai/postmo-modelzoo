# YoloP

本示例展示如何把`yolop`模型量化，编译，最后部署到后摩芯片的设备上。

## 1.预训练模型

本项目使用从`github`下载的预训练模型：

- 模型名称: YOLOP
- 来源: [https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)
- 许可: [MIT License](https://github.com/hustvl/YOLOP/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例模型的训练数据集为bdd100k
校准使用的 BDD100K 数据集需自行从官方渠道下载，下载方式请参考[数据集说明](../../../data/datasets/README.md)。

## 2.快速开始

可以通过hmatc工具一键执行性能测试和精度测试，详细介绍请参考houmo-examples根目录下README.md文件的快速上手章节。

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh2       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh2       # 芯片
```

执行完成后会打印模型的数据集精度信息。

### 2.3 性能评估

```bash
hmatc perf -c config.yml -t xh2 -wn 10 -sn 1000 -tn 4
```

执行完成后会打印模型推理延迟、吞吐量等信息。

### 2.4 结果演示

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
[Latency] Inference  avg:  38.491 ms, max:  41.180 ms, min:  26.063 ms, tp99:  40.679 ms, tp999:  41.180 ms
[Latency] Input      avg:   2.108 ms, max:   2.641 ms, min:   1.104 ms, tp99:   2.511 ms, tp999:   2.641 ms
[Latency] Output     avg:   2.962 ms, max:   8.351 ms, min:   1.584 ms, tp99:   3.656 ms, tp999:   8.351 ms
[Latency] End2end    avg:  43.562 ms, max:  49.776 ms, min:  30.699 ms, tp99:  44.774 ms, tp999:  49.776 ms
[Throughput] total: 10935.865 ms, avg: 10.936 ms, repeat: 1000, rounds: 1
[Throughput] qps: 91.442
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- yolop 模型：[https://github.com/hustvl/YOLOP/blob/main/weights/yolop-640-640.onnx](https://github.com/hustvl/YOLOP/blob/main/weights/yolop-640-640.onnx)
- BDD100K 验证集链接: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
