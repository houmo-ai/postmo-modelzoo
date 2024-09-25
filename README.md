# houmo-modelzoo

## 目录

[TOC]

## 概述

houmo-modelzoo是为用户快速将模型移植到后摩鸿途系列芯片产品而开发的模型库，为用户提供量化、编译、精度和性能评估等一整套代码和工具，降低用户的学习和开发成本。

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

houmo-modelzoo提供的模型示例如下，编译示例每个都提供，其他支持情况见下表，其中raw表示提供原始模型，quant表示提供量化后模型，ptq表示提供ptq量化示例，qat表示提供qat训练示例，pydemo表示提供python端到端demo，c++demo表示提供c++端到端demo，eval表示提供精度评估

| models                                       | type      | raw | quant | ptq | qat | pydemo | c++demo | eval |
| -------------------------------------------- | --------- | --- | ----- | --- | --- | ------ | ------- | ---- |
| [resnet50](models/backbone/resnet50)         | backbone  | yes | yes   | yes | yes | yes    | x       | yes  |
| [mobilenetv2](models/backbone/mobilenet_v2)  | backbone  | yes | yes   | yes | x   | yes    | x       | yes  |
| [efficientnet](models/backbone/efficientnet) | backbone  | yes | yes   | yes | x   | yes    | x       | yes  |
| [yolov5s](models/detection/yolov5s)          | detection | yes | yes   | yes | x   | yes    | x       | yes  |
| [yolov3](models/detection/yolov3)            | detection | yes | yes   | yes | x   | yes    | x       | yes  |
| [yolop](models/autodrive/yolop)              | autodrive | yes | yes   | yes | yes | yes    | x       | x    |
| [pointpillars](models/autodrive/pointpillars)| autodrive | x   | yes   | x   | x   | x      | yes     | x    |
| [petr](models/autodrive/petr)                | autodrive | x   | yes   | x   | x   | yes    | x       | x    |
| [detr3d](models/autodrive/petr)              | autodrive | x   | yes   | x   | x   | yes    | x       | x    |
| [qwen](models/llm/qwen)                      | llm       | x   | yes   | x   | x   | yes    | x       | x    |
| [qwen_mix](models/llm/qwen_mix)              | llm       | x   | yes   | x   | x   | yes    | x       | x    |
| [qwen_mix_multibatches](models/llm/qwen_mix_multibatches) | llm | x | yes | x | x  | yes    | x       | x    |
| [qwen2](models/llm/qwen2)                    | llm       | x   | yes   | x   | x   | yes    | x       | x    |

## C++评估工具列表

| PROJECT                                                                         | 说明                 |
| ------------------------------------------------------------------------------- | -------------------- |
| [tcim_perf](uilts/tcim_perf)                                                    | 性能基准测试工具      |

## 快速上手

### 概述

houmo-modelzoo中的模型提供两种方式作为示例，用户可以根据需求和习惯选用：

1. 接口方式

通过直接调用tcim API接口方式，可以对量化、编译和运行细节做更定制化的修改。目前提供的示例脚本主要有：
- ptq.py 对模型进行PTQ量化，保存量化后模型和golden数据
- qat.py 对模型进行QAT训练，保存量化后模型
- build.py 对量化模型进行编译，生成可在后摩鸿途系列芯片上运行的模型，然后使用该模型推理并与golden数据进行比对
- demo.py 对模型进行芯片上推理，并展示结果

2. 工具方式

通过模型评估辅助工具hmassist，可以对一些常见公版模型做一键评估，了解模型的支持程度、性能、精度等基本信息。对于符合条件的自定义模型，也只需要做少量修改即可适配，可大幅提升模型的评估效率。

### 准备环境

houmo-modelzoo依赖后摩大道其他组件运行，包括量化工具quantool、编译运行工具tcim、houmo-toolchain、芯片驱动等，需要先加载安装相关软件，初次使用推荐采用后摩大道提供的docker镜像。

软件安装好后，需要配置开发运行环境。先检查根目录 `env.sh` 里的环境变量，根据实际情况修改，例如如果要更换数据集路径则修改`DATASETS_PATH`变量。修改完成后执行以下命令：

```bash
source env.sh
```

运行完成后会打印主要的环境变量，请再次检查环境变量是否与预期一致。注意`HDPL_PLATFORM`指示当前运行环境，如果为`ASIC`表示在芯片上运行，如果为`ISIM`表示在模拟器上运行。初始时脚本会自动检测是否是芯片环境，可以通过手动修改该环境变量来切换。

设置完成后进入模型目录，以resnet50为例：

```bash
cd models/backbone/resnet50
```

### 获取模型和数据

1. 准备量化和评估使用的数据集，以resnet50需要的imagenet数据集为例，将imagenet验证集放到data/datasets/imagenet目录下，仓库中已有少量数据供简单验证，如果需要测试真实精度需要自行下载完整数据集
2. 如果需要自己量化，需要下载原始模型，对于有些存在不支持算子的模型，可能需要修改或者裁剪
3. 如果仅评估模型精度和性能，可以直接下载提供的量化模型

可以通过get_model.py脚本获取准备好的模型并进行修改，如果下载地址无法访问，请与我们联系。

```bash
python3 get_model.py
```

可以通过type参数控制下载模型的类型，raw为原始模型，quant为量化模型，all为全部下载，默认为all。下载的模型默认放在MODEL_PATH环境变量定义的目录，量化模型将解压到output/houmo/result目录（注：重新量化后会覆盖该量化模型，可重新执行获取脚本恢复）。如：

```bash
python3 get_model.py --type raw
```

### 一键模型评估

提供一键评估脚本方便用户进行模型评估，一般分为性能和精度两个评估脚本。大部分模型使用hmassist工具进行评估，少量hmassist模型使用API接口的方式，部分模型还不支持数据集精度评估，支持情况请查看`模型示例列表`章节。

如果需要按步骤评估，可以参考后面的`使用TCIM API接口评估`和`使用hmassist工具评估`章节。

#### 性能评估

可以通过每个模型下的perf.sh脚本一键执行性能测试：

```bash
bash perf.sh
```

执行完成后会打印模型推理延迟、吞吐量等信息。

#### 精度评估

可以通过每个模型下的eval.sh脚本一键执行精度测试：

```bash
bash eval.sh
```

执行完成后会打印模型的数据集精度信息。

### 使用TCIM API接口评估

直接使用TCIM API接口进行模型转换和评估，API接口参考《TCIM API手册》。以resnet50为例，hmassist的使用过程如下：

#### PTQ量化

（Post-Training Quantization）量化是一种在神经网络模型训练完成后进行的量化方法。将浮点权重映射到较低比特宽度的定点表示，例如8位或4位整数。量化过程涉及将原始浮点权重（如float32）映射到int8区间[-128,127]，并计算每个通道的缩放因子（scale）和偏移量（zero_point）。通过ptq.py脚本执行：

```bash
python3 ptq.py
```

量化后的模型和golden数据以onnx模型和npy数据的形式默认放在output/houmo/result目录下。量化完成后会进行profile，以表格的形式打印逐层相似度，重点关注输出层的余弦相似度是否符合预期。如果余弦相似度较低，可考虑该结果是否适合使用余弦相似度进行评价，进一步测试模型实际精度。如果确实是量化精度降低较多，可考虑更换量化数据和参数，以及混合量化，QAT训练等方式进一步提升。

#### 编译

将量化模型编译为在芯片上运行的模型。可通过`--batch`参数配置batch数，通过build.py脚本执行：

```bash
python3 build.py
```

#### 性能测试

使用性能基准测试工具进行性能测试，以评估模型在多batch、多线程等不同方式下的推理性能。进入utils/tcim_perf目录，修改run.sh脚本中的参数，包括模型路径、线程数、测试数等，然后执行：

```bash
cd utils/tcim_perf
./build.sh
./run.sh
```

### 使用hmassist工具评估

hmassist工具基于TCIM API接口通过yaml配置文件和python脚本定义模型参数和处理方式，包括三种文件：

1. 模型配置文件[必选]，文件名默认config.yml, 定义模型参数和各过程处理参数，以及自定义处理模块的名称。具体配置请参考hmassist/default_config.yml
2. 模型处理文件[可选]，文件名必须hm_model.py, 里面定义模型处理类，包括前处理、编译参数、后处理、demo和精度处理等方法，可继承hmassist/models下基类
3. 数据集处理文件[可选]，文件名必须hm_dataset.py，里面定义数据集处理类，包括从数据集获取样本、标注和精度评估等方法，可继承hmassist/datasets下基类
4. 量化参数文件[可选]，文件路径由模型配置文件quant字段ptq_cfg_path参数配置，里面定义量化配置

以resnet50为例，hmassist的使用过程如下：

#### 量化

将原始浮点模型量化为定点，以便在芯片上部署。目前hmassist工具仅支持ptq量化，在模型配置文件中quant字段配置量化参数，然后执行hmquant.sh脚本：

```bash
hmquant.sh
```

量化后的模型和golden数据以onnx模型和npy数据的形式默认放在output/houmo/result目录下。量化完成后会进行profile，以表格的形式打印逐层相似度，重点关注输出层的余弦相似度是否符合预期。如果余弦相似度较低，可考虑该结果是否适合使用余弦相似度进行评价，进一步测试模型实际精度。如果确实是量化精度降低较多，可考虑更换量化数据和参数，以及混合量化，QAT训练等方式进一步提升。

#### 编译

将量化模型编译为在芯片上运行的模型。在模型配置文件中build字段配置编译参数，可通过`--batch`参数配置batch数，然后执行hmbuild.sh脚本：

```bash
hmbuild.sh
```

编译后的模型放在当前目录下。编译完成后会使用量化产生的golden输入进行推理，然后与量化产生的golden输出进行比对，如果余弦相似度高于0.999一般认为编译过程结果正确，模型可以使用。

#### 测试

使用指定的测试输入对模型推理结果进行测试。支持指定目标为onnx，用于比较结果是否与onnx推理结果一致。如果先指定目标为onnx，会调用onnx runtime进行推理并保存结果。然后指定目标为后摩芯片，使用相同的输入调用后摩芯片进行推理，将结果与onnx runtime的结果进行余弦相似度进行比较，以确定模型转换的正确性和精度。通过执行hmtest.sh脚本实现：

```bash
hmtest.sh --target onnx
hmtest.sh
```

#### 结果展示

使用指定的测试输入进行模型推理，并加上前后处理和可视化，使用户可以直观的观察推理结果。支持指定目标为onnx，用于比较推理效果是否与onnx一致。展示方法由用户自己实现，需要在hm_model.py文件中定义模型处理类并实现前后处理、demo等接口，可以从已有的实现类中继承。通过执行hmdemo.sh脚本实现：

```bash
hmdemo.sh --target onnx
hmdemo.sh
```

#### 性能测试

使用指定的方式进行性能测试，以评估模型在不用使用方式下的推理性能, 可通过`--thread_num`参数配置线程数。执行前需要先执行utils/tcim_perf下的build.sh

```bash
cd utils/tcim_perf
./build.sh
```

执行hmperf.sh：

```bash
hmperf.sh
```

#### 精度测试

使用指定的数据集进行精度测试，以评估模型在数据集下的推理精度。支持指定目标为onnx，用于比较与原始onnx模型的精度差异。精度测试方法由用户自己实现，需要在hm_model.py文件中定义模型处理类并实现前后处理接口，同时在hm_dataset.py文件中定义数据库处理类，可以从已有的实现类中继承。通过执行hmeval.sh脚本实现：

```bash
hmeval.sh --target onnx
hmeval.sh
```

#### 批量基准测试

使用指定的参数对一组模型进行性能和精度测试，生成测试报告。测试的模型和参数定义在benchmark.yml文件中，内容如下：

```yml
models: {
  resnet50: {location: models/backbone/resnet50, batch: 1, core_num: 1},
  mobilenetv2: {location: models/backbone/mobilenetv2, batch: 1, core_num: 1},
  ...
}
```

通过hmbenchmark.sh脚本执行：

```bash
hmbenchmark.sh
```

执行完成会在log中打印报告，同时在reports目录中保存csv文件。部分模型执行结果如下（模型性能结果根据平台不同会有差异）：

|  ModelName   |      Shape       |  Dataset   | Batch | CoreNum | Accuracy(onnx)        | Accuracy(houmo)       | AccRelError             | Latency(ms) |   Qps   |
| ------------ | ---------------- | ---------- | ----- | ------- | --------------------- | --------------------- | ----------------------- | ----------- | ------- |
|   resnet50   | [1, 224, 224, 3] | ILSVRC2012 |   1   |    1    | top1:0.753 top5:0.925 | top1:0.719 top5:0.923 | top1:-0.046 top5:-0.002 |    0.752    | 1330.47 |
|   yolov5s    | [1, 640, 640, 3] |  coco2017  |   1   |    1    | map:0.362 map50:0.557 | map:0.333 map50:0.542 | map:-0.079 map50:-0.026 |    4.415    |  226.50 |
|    yolov3    | [1, 640, 640, 3] |  coco2017  |   1   |    1    | map:0.412 map50:0.615 | map:0.373 map50:0.607 | map:-0.094 map50:-0.014 |    27.496   |  36.37  |
|    yolop     | [1, 384, 640, 3] |  NotTest   |   1   |    1    | NotTest               | NotTest               | NotTest                 |    43.674   |  22.90  |