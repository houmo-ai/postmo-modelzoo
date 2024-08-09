# HmAssist

## 概述

HmAssist是后摩AI芯片的开发辅助工具，基于houmo-tcim接口之上封装，主要使用python开发，利用配置文件实现快速模型转换、推理、精度分析、结果展示、精度和性能测试等芯片评估功能，目前支持ONNX、Caffe、PyTorch、TensorFlow、MxNet等框架。本文档主要介绍如何使用TyAssist工具进行芯片快速评估。


## 工具包介绍

```
hmassist
  ├── README.md （说明文档）
  ├── utils（公用模块）
  ├── datasets（常见数据集）
  ├── base（模型和自定义预处理基类）
  ├── src （工具主体代码）
  ├── version （版本说明）
  └── hmassist.py (工具入口)
```

## 环境依赖

- 后摩大道软件平台

## 使用说明

HmAssist支持5种基础功能，分别是量化、编译、推理测试、结果展示、性能测试、精度测试。

```
hmassist.py [-h] --config CONFIG [--target {houmo}]
                   [--backend {chip,onnx}] [--log_dir LOG_DIR]
                   {quant,build,test,demo,perf,eval}
其中log_dir默认为logs，可缺省
```

## 示例
以下使用后摩大道的环境镜像和resnet50模型作为示例。

以resnet50为例，在发布环境下进入houmo-modelzoo，按以下步骤准备环境：
1. 执行source env.sh
2. 进入utils/tcimexec或utils/aottcimexec目录，执行./build.sh
3. 进入cv/resnet50/prepare_model目录，执行./run.sh
4. 进入cv/resnet50目录

### 量化
执行hmquant.sh，执行成功打印如下

```shell
################  ptq quantize finished  ######################
...
                                        量化误差分析表，按余弦相似度从低到高排序                                        
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ node output        ┃ op_type                    ┃ cos_similarity     ┃ snr                   ┃ relative_error       ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ resizer_out_Conv_0 │ HMResizer                  │ 1.0000001192092896 │ 0.0                   │ 0.0                  │
│ onnx::MaxPool_323  │ BaseCIMDConv2d             │ 0.9995769262313843 │ 0.0008601178415119648 │ 0.03329573571681976  │
│ input.8            │ BaseMaxPool2d              │ 0.9997471570968628 │ 0.0005176496342755854 │ 0.022571083158254623 │
...
│ onnx::Flatten_493  │ BaseQuantAdaptiveAvgPool2d │ 0.9918368458747864 │ 0.017477640882134438  │ 0.13558992743492126  │
│ onnx::Gemm_494     │ Reshape                    │ 0.9918368458747864 │ 0.017477640882134438  │ 0.13558992743492126  │
│ 495                │ BaseCIMDConv2d             │ 0.9863150119781494 │ 0.02771996706724167   │ 0.16815221309661865  │
└────────────────────┴────────────────────────────┴────────────────────┴───────────────────────┴──────────────────────┘
```

### 编译
执行hmbuild.sh，执行成功打印如下

JIT结果：

```shell
################  build finished  ######################
resizer input shape is :  (1, 3, 224, 224)
tcim: store model as one fusedop  resnet50
resnet50  saved as one fusedop model
```

AOT结果：

```shell
################  build finished  ######################
resizer input shape is :  (1, 3, 224, 224)
Compiling the model to tcim_resnet50.so ...
```

### 推理测试

执行hminfer.sh，执行成功打印如下，目前golden数据比对结果还有点问题

```shell
[compare] houmo vs quant output [495] similarity=0.572020
success

input[input.1] shape = (1, 224, 224, 3)
model input num =  1
(1, 224, 224, 3)
model output num =  1
output[495] shape = (1, 1000)
```

### 结果展示
执行hmdemo.sh，执行成功打印如下

```shell
predict cls = 809, prob = 1.000000
[end2end] average cost: 2118.850470ms
success

model input num =  1
(1, 224, 224, 3)
model output num =  1
output[495] shape = (1, 1000)
```

### 性能测试
执行hmperf.sh，执行成功打印如下

```shell
Inference time cost total = 9448262us
Inference time cost per frame = 944826.2us
Average Throughput(QPS): 1.06fps
success
```

### 精度测试
执行hmaccuracy.sh，执行成功打印如下

```shell
{'input_size': '1x3x224x224', 'dataset': 'ILSVRC_2012Val', 'num': 20, 'top1': '0.650000', 'top5': '0.950000', 'latency': '1299.612999'}
success
```