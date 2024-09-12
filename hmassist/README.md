# HmAssist

## 概述

HmAssist是后摩AI芯片的集成化开发辅助工具，基于后摩量化工具hmquant和后摩工具链tcim接口之上封装，主要使用python开发，利用yml配置文件实现一键式芯片模型评估功能，包括：
1. 量化
  1.1 ptq量化
  1.2 ptq逐层精度分析
2. 编译
  2.1 直接编译
  2.2 交叉编译
3. 测试
  3.1 golden数据结果比对
  3.2 onnx原模型结果比对
  3.3 算法结果展示(支持onnx原模型)
  3.4 数据集精度评估(支持onnx原模型)
  3.5 性能基准测试

本文档主要介绍如何使用HmAssist工具进行芯片快速评估。


## 工具包介绍

```bash
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

HmAssist支持7种命令模式，分别是量化、编译、推理测试、结果展示、性能测试、精度测试、benchmark测试。

```bash
python3 hmassist.py [-h]
                    {quant,build,test,demo,perf,eval,benchmark}
                    --target {houmo,onnx}
                    [--config config]
                    [--thread_num thread_num]
```
其中：
- config_path为yml配置文件路径，默认为config.yml
- thread_num为性能测试的线程数，默认为1

## 示例
以下使用后摩大道的docker镜像和resnet50模型作为示例。

进入后摩大道docker，进入houmo-modelzoo目录，按以下步骤准备环境：
1. 设置环境变量
2. 进入utils/tcim_perf目录，编译
3. 进入models/backbone/resnet50目录
4. 获取resnet50原始模型

```bash
cd /usr/local/src/houmo-modelzoo
source env.sh
cd utils/tcim_perf && ./build.sh
cd -
cd models/backbone/resnet50
python3 get_model.py --type raw
```

### 量化

对原始模型进行量化，保存量化模型和golden数据在output/$HOUMO_TARGET/result目录：

```bash
hmquant.sh
```

### 编译

对量化模型进行编译，生成芯片模型在配置文件中指定的目录，同时使用golden数据进行推理和验证：

```bash
hmbuild.sh
```

如果使用交叉编译工具链编译aarch64平台的模型，可以在编译之前定义环境变量TCIM_CROSS_COMPILE，编译完成后将模型拷贝到aarch64平台推理。

```bash
export TCIM_CROSS_COMPILE=1
```

### 推理测试

使用配置文件中的指定数据对芯片模型进行推理测试，若先指定target为onnx，则将结果与onnx结果进行比对：

```bash
hmtest.sh --target onnx
hmtest.sh
```

### 结果展示

使用配置文件中的指定数据对芯片模型进行推理，并使用hm_model.py中定义的demo方式对结果进行展示，也可以指定target为onnx作为对比：

```bash
hmdemo.sh --target onnx
hmdemo.sh
```


### 性能测试

使用配置文件中的指定参数对芯片模型进行性能测试，输出延迟、吞吐等结果：

```bash
hmperf.sh
```


### 精度测试

使用配置文件中的指定参数对芯片模型进行推理，并使用hm_dataset.py中定义的评估方式对结果进行评估，也可以指定target为onnx作为对比：

```bash
hmeval.sh --target onnx
hmeval.sh
```

### 批量基准测试

根据benchmark.yml文件中指定的模型和参数，对模型进行一键批量评估：

```bash
hmbenchmark.sh
```
