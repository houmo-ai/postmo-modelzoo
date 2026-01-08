# Resnet50 Multistreams Example

## 目录

[TOC]

## 概述

Resnet50 Multistreams Example 以 resnet50 模型为例展示 TCIM API 在多设备、多线程以及多 stream 下的使用方式。

1. resnet50_multistreams.py 展示 python 下多线程多 stream 推理过程
2. resnet50_multistreams.cc 展示 c++下多设备多线程以及多 stream 推理过程
3. 使用 create_weight_manager 接口共享模型内存
4. 支持配置多线程，每个线程使用 1 个 stream
5. 支持配置多设备，每个设备使用相同数量的线程

## 软件依赖

除 TCIM Examples 公用依赖外，本示例没有其他依赖。

## 快速上手

### 获取芯片模型

可以通过以下两种方式获取芯片模型：

1. 自行编译：通过原始模型或量化模型，经过 tcim 工具链编译生成，具体可参考 houmo-modelzoo
2. 直接下载：通过 get_model.py 脚本下载预编译的模型

```bash
python3 get_model.py
```

### 模型 python 推理

使用 runtime python API 进行模型推理，并输出结果。执行时可通过参数修改配置：

-   -n 1：设备数，默认为 1
-   -t 4：线程数，默认为 4
-   -s 10：测试样本数，默认为 10

以 2 个设备为例（有两张卡 device id 分别为 0 和 1）：

```bash
python3 resnet50_multistreams.py -n 2
```

测试程序将测试样本存入输入队列，由每个线程的推理模块竞争推理，并将推理后的结果存入输出队列，每个线程推理完成后将打印推理的数量。

注：模拟器下仅支持单线程。

执行成功输出以下打印：

```bash
2025-12-22 15:01:52.164 | INFO     | __main__:<module>:46 - ===> resnet50_multistreams python example start...
2025-12-22 15:01:52.165 | INFO     | __main__:<module>:59 - devices: 1
2025-12-22 15:01:52.165 | INFO     | __main__:<module>:60 - threads: 4
2025-12-22 15:01:52.165 | INFO     | __main__:<module>:61 - samples: 10
2025-12-22 15:01:52.165 | INFO     | __main__:<module>:62 - model: ./resnet50_xh2_b1_1core.hmm
2025-12-22 15:01:54.232 | INFO     | __main__:<module>:171 - thread 0 on device 0 load model ./resnet50_xh2_b1_1core.hmm.
2025-12-22 15:01:54.237 | INFO     | __main__:<module>:171 - thread 1 on device 0 load model ./resnet50_xh2_b1_1core.hmm.
2025-12-22 15:01:54.241 | INFO     | __main__:<module>:171 - thread 2 on device 0 load model ./resnet50_xh2_b1_1core.hmm.
2025-12-22 15:01:54.244 | INFO     | __main__:<module>:171 - thread 3 on device 0 load model ./resnet50_xh2_b1_1core.hmm.
2025-12-22 15:01:54.245 | INFO     | __main__:thread_func:104 - input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.246 | INFO     | __main__:thread_func:104 - input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.246 | INFO     | __main__:thread_func:119 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.246 | INFO     | __main__:thread_func:119 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.247 | INFO     | __main__:thread_func:104 - input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.248 | INFO     | __main__:thread_func:104 - input[input.1] shape = [1, 3, 224, 224], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.248 | INFO     | __main__:thread_func:119 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.248 | INFO     | __main__:thread_func:119 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2025-12-22 15:01:54.253 | INFO     | __main__:thread_func:153 - thread 3 on device 0 run sample 0 end.
2025-12-22 15:01:54.256 | INFO     | __main__:thread_func:153 - thread 2 on device 0 run sample 1 end.
2025-12-22 15:01:54.259 | INFO     | __main__:thread_func:153 - thread 0 on device 0 run sample 2 end.
2025-12-22 15:01:54.262 | INFO     | __main__:thread_func:153 - thread 1 on device 0 run sample 3 end.
2025-12-22 15:01:54.265 | INFO     | __main__:thread_func:153 - thread 3 on device 0 run sample 4 end.
2025-12-22 15:01:54.268 | INFO     | __main__:thread_func:153 - thread 2 on device 0 run sample 5 end.
2025-12-22 15:01:54.271 | INFO     | __main__:thread_func:153 - thread 0 on device 0 run sample 6 end.
2025-12-22 15:01:54.274 | INFO     | __main__:thread_func:153 - thread 1 on device 0 run sample 7 end.
2025-12-22 15:01:54.277 | INFO     | __main__:thread_func:153 - thread 3 on device 0 run sample 8 end.
2025-12-22 15:01:54.279 | INFO     | __main__:thread_func:153 - thread 2 on device 0 run sample 9 end.
2025-12-22 15:01:54.280 | INFO     | __main__:thread_func:156 - thread 0 on device 0 completed. 2 sampels tested.
2025-12-22 15:01:54.280 | INFO     | __main__:thread_func:156 - thread 1 on device 0 completed. 2 sampels tested.
2025-12-22 15:01:54.280 | INFO     | __main__:thread_func:156 - thread 3 on device 0 completed. 3 sampels tested.
2025-12-22 15:01:54.280 | INFO     | __main__:thread_func:156 - thread 2 on device 0 completed. 3 sampels tested.
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 0 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 1 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 2 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 3 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 4 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 5 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 6 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 7 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 8 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:205 - sample 9 top1: predict cls = 65, prob = 0.651222
2025-12-22 15:01:55.935 | INFO     | __main__:<module>:213 - <=== resnet50_multistreams python example completed.
```

### 模型 c++推理

使用 runtime c++ API 进行模型推理，并输出结果。执行时可通过参数修改配置：

-   -n 1：设备数，默认为 1
-   -t 4：线程数，默认为 4
-   -s 10：测试样本数，默认为 10

以 2 个设备为例（有两张卡 device id 分别为 0 和 1）：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_resnet50_multistreams -n 2
```

测试程序将测试样本存入输入队列，由每个线程的推理模块竞争推理，并将推理后的结果存入输出队列。每个线程推理完成后将打印推理的数量，同时显示每个线程工作在哪个设备上。

注：模拟器下仅支持单线程。

执行成功输出以下打印：

```bash
I20251222 15:44:29.958922 2701239 resnet50_multistreams.cc:115] ===> resnet50_multistreams c++ example start...
I20251222 15:44:29.959512 2701239 resnet50_multistreams.cc:141] model: ./resnet50_xh2_b1_1core.hmm, devices: 1, threads: 4, samples: 10
I20251222 15:44:29.966454 2701239 resnet50_multistreams.cc:196] sample queue size is 10
[2025-12-22 15:44:32.039] [tid 2701239] [info] thread 0 on device 0 model ./resnet50_xh2_b1_1core.hmm.
[2025-12-22 15:44:32.044] [tid 2701239] [info] thread 1 on device 0 model ./resnet50_xh2_b1_1core.hmm.
[2025-12-22 15:44:32.048] [tid 2701239] [info] thread 2 on device 0 model ./resnet50_xh2_b1_1core.hmm.
[2025-12-22 15:44:32.051] [tid 2701239] [info] thread 3 on device 0 model ./resnet50_xh2_b1_1core.hmm.
[2025-12-22 15:44:32.052] [tid 2701309] [info] Count of Input: 1
[2025-12-22 15:44:32.052] [tid 2701310] [info] Count of Input: 1
[2025-12-22 15:44:32.052] [tid 2701309] [info] Input[input.1] info: TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
[2025-12-22 15:44:32.052] [tid 2701310] [info] Input[input.1] info: TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
[2025-12-22 15:44:32.052] [tid 2701309] [info] Count of Output: 1
[2025-12-22 15:44:32.052] [tid 2701310] [info] Count of Output: 1
[2025-12-22 15:44:32.052] [tid 2701309] [info] Output[495] info: TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
[2025-12-22 15:44:32.052] [tid 2701310] [info] Output[495] info: TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
[2025-12-22 15:44:32.052] [tid 2701311] [info] Count of Input: 1
[2025-12-22 15:44:32.052] [tid 2701311] [info] Input[input.1] info: TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
[2025-12-22 15:44:32.052] [tid 2701311] [info] Count of Output: 1
[2025-12-22 15:44:32.052] [tid 2701311] [info] Output[495] info: TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
[2025-12-22 15:44:32.052] [tid 2701312] [info] Count of Input: 1
[2025-12-22 15:44:32.052] [tid 2701312] [info] Input[input.1] info: TensorInfo: shape: [1,3,224,224], stride: [], dtype: FLOAT16, format: ND, size: 301056, memsize: 301056
[2025-12-22 15:44:32.052] [tid 2701312] [info] Count of Output: 1
[2025-12-22 15:44:32.052] [tid 2701312] [info] Output[495] info: TensorInfo: shape: [1,1000], stride: [], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
[2025-12-22 15:44:32.057] [tid 2701310] [info] thread 1 on device 0 infer start...
[2025-12-22 15:44:32.057] [tid 2701312] [info] thread 3 on device 0 infer start...
[2025-12-22 15:44:32.057] [tid 2701311] [info] thread 2 on device 0 infer start...
[2025-12-22 15:44:32.057] [tid 2701309] [info] thread 0 on device 0 infer start...
[2025-12-22 15:44:32.061] [tid 2701312] [info] thread 3 on device 0 run sample 1 end.
[2025-12-22 15:44:32.061] [tid 2701310] [info] thread 1 on device 0 run sample 0 end.
[2025-12-22 15:44:32.063] [tid 2701309] [info] thread 0 on device 0 run sample 3 end.
[2025-12-22 15:44:32.063] [tid 2701311] [info] thread 2 on device 0 run sample 2 end.
[2025-12-22 15:44:32.065] [tid 2701312] [info] thread 3 on device 0 run sample 4 end.
[2025-12-22 15:44:32.065] [tid 2701310] [info] thread 1 on device 0 run sample 5 end.
[2025-12-22 15:44:32.067] [tid 2701311] [info] thread 2 on device 0 run sample 7 end.
[2025-12-22 15:44:32.067] [tid 2701311] [info] thread 2 on device 0 completed. 2 sampels tested.
[2025-12-22 15:44:32.067] [tid 2701309] [info] thread 0 on device 0 run sample 6 end.
[2025-12-22 15:44:32.067] [tid 2701309] [info] thread 0 on device 0 completed. 2 sampels tested.
[2025-12-22 15:44:32.069] [tid 2701312] [info] thread 3 on device 0 run sample 8 end.
[2025-12-22 15:44:32.069] [tid 2701312] [info] thread 3 on device 0 completed. 3 sampels tested.
[2025-12-22 15:44:32.069] [tid 2701310] [info] thread 1 on device 0 run sample 9 end.
[2025-12-22 15:44:32.069] [tid 2701310] [info] thread 1 on device 0 completed. 3 sampels tested.
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] top1: Index=65, Conf=16, Label=[sea snake]
[2025-12-22 15:44:32.070] [tid 2701239] [info] <=== resnet50_multistreams c++ example completed.
```

**说明：由于 c++和 python 前处理有差异，且 c++后处理没有做 softmax，部分结果可能存在少量差异。**

### 一键执行

可以使用 run 脚本一键执行查看结果, 以 2 个设备为例（有两张卡 device id 分别为 0 和 1）

linux 下执行：

```bash
bash run.sh -n 2
```

windows 下执行（windows 需要在代码中修改参数）：

```bash
bash run.bat
```
