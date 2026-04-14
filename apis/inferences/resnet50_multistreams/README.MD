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

1. 自行编译：通过原始模型或量化模型，经过 tcim 工具链编译生成，具体可参考 houmo-examples
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
2026-04-14 07:38:37.533 | INFO     | __main__:<module>:77 - ===> resnet50_multistreams python example start...
2026-04-14 07:38:37.534 | INFO     | __main__:<module>:92 - Found model file: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
2026-04-14 07:38:37.534 | INFO     | __main__:<module>:103 - devices: 1
2026-04-14 07:38:37.534 | INFO     | __main__:<module>:104 - threads: 4
2026-04-14 07:38:37.534 | INFO     | __main__:<module>:105 - samples: 10
2026-04-14 07:38:37.534 | INFO     | __main__:<module>:106 - model: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
[2026-04-14 07:38:37.534] [tid 407783] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-14 07:38:37.534] [tid 407783] [warning] Empty backend name, use Xh2HalBackend instead.
2026-04-14 07:38:38.245 | INFO     | __main__:<module>:223 - thread 0 on device 0 load model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
2026-04-14 07:38:38.247 | INFO     | __main__:<module>:223 - thread 1 on device 0 load model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
2026-04-14 07:38:38.248 | INFO     | __main__:<module>:223 - thread 2 on device 0 load model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
2026-04-14 07:38:38.248 | INFO     | __main__:<module>:223 - thread 3 on device 0 load model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
2026-04-14 07:38:38.248 | INFO     | __main__:<module>:234 - Reference model input info:
2026-04-14 07:38:38.248 | INFO     | __main__:<module>:239 -   Input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, mem_size = 393216, stride = [393216, 512, 1, 393216, 512, 2, 1, 262144], format = YUV420SP
2026-04-14 07:38:38.248 | INFO     | __main__:<module>:239 -   Input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, mem_size = 40, stride = [40, 4], format = ND
2026-04-14 07:38:38.251 | INFO     | __main__:<module>:247 - input image shape: (375, 500, 3)
2026-04-14 07:38:38.251 | INFO     | __main__:<module>:264 - pad input image to (512, 512, 3) height = 512, width = 512
2026-04-14 07:38:38.260 | INFO     | __main__:<module>:302 - input_data shape: (1, 3, 512, 512), dtype: uint8, dyn_info: [[  0   0 374 500 224 224   0   0   0   0]]
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, format = YUV420SP
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, format = YUV420SP
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:158 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, format = YUV420SP
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[input.1] shape = [1, 3, 512, 512], dtype = <class 'numpy.uint8'>, format = YUV420SP
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:158 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:142 - input[resizer_crop_input.1] shape = [1, 10], dtype = <class 'numpy.int32'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:158 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2026-04-14 07:38:38.261 | INFO     | __main__:thread_func:158 - output[495] shape = [1, 1000], dtype = <class 'numpy.float16'>, format = ND
2026-04-14 07:38:38.264 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 0 end.
2026-04-14 07:38:38.266 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 1 end.
2026-04-14 07:38:38.268 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 2 end.
2026-04-14 07:38:38.270 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 3 end.
2026-04-14 07:38:38.272 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 4 end.
2026-04-14 07:38:38.274 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 5 end.
2026-04-14 07:38:38.277 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 6 end.
2026-04-14 07:38:38.279 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 7 end.
2026-04-14 07:38:38.281 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 8 end.
2026-04-14 07:38:38.283 | INFO     | __main__:thread_func:203 - thread 3 on device 0 run sample 9 end.
2026-04-14 07:38:38.283 | INFO     | __main__:thread_func:206 - thread 3 on device 0 completed. 10 sampels tested.
2026-04-14 07:38:38.361 | INFO     | __main__:thread_func:206 - thread 0 on device 0 completed. 0 sampels tested.
2026-04-14 07:38:38.362 | INFO     | __main__:thread_func:206 - thread 2 on device 0 completed. 0 sampels tested.
2026-04-14 07:38:38.362 | INFO     | __main__:thread_func:206 - thread 1 on device 0 completed. 0 sampels tested.
2026-04-14 07:38:38.364 | INFO     | __main__:<module>:354 - sample 0 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.364 | INFO     | __main__:<module>:354 - sample 1 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.364 | INFO     | __main__:<module>:354 - sample 2 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.365 | INFO     | __main__:<module>:354 - sample 3 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.365 | INFO     | __main__:<module>:354 - sample 4 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.365 | INFO     | __main__:<module>:354 - sample 5 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.365 | INFO     | __main__:<module>:354 - sample 6 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.366 | INFO     | __main__:<module>:354 - sample 7 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.366 | INFO     | __main__:<module>:354 - sample 8 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.366 | INFO     | __main__:<module>:354 - sample 9 top1: predict cls = 65, prob = 0.753624
2026-04-14 07:38:38.366 | INFO     | __main__:<module>:362 - <=== resnet50_multistreams python example completed.
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
I20260414 08:09:22.643092 435374 resnet50_multistreams.cc:229] ===> resnet50_multistreams c++ example start...
I20260414 08:09:22.643729 435374 resnet50_multistreams.cc:239] houmo target: xh2, tcim version: v1.2.0
- Git Commit: clean
- Build Time: 2026-03-31 16:53:07 Mar 31 2026
I20260414 08:09:22.651982 435374 resnet50_multistreams.cc:250] Found .hmm file: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 08:09:22.651996 435374 resnet50_multistreams.cc:259] Find resnet50 model file ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm
I20260414 08:09:22.652001 435374 resnet50_multistreams.cc:274] model: ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm, devices: 1, threads: 4, samples: 10
[2026-04-14 08:09:22.652] [tid 435374] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-14 08:09:22.652] [tid 435374] [warning] Empty backend name, use Xh2HalBackend instead.
I20260414 08:09:23.087505 435374 resnet50_multistreams.cc:389] thread 0 on device 0 model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
I20260414 08:09:23.089695 435374 resnet50_multistreams.cc:389] thread 1 on device 0 model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
I20260414 08:09:23.090968 435374 resnet50_multistreams.cc:389] thread 2 on device 0 model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
I20260414 08:09:23.092092 435374 resnet50_multistreams.cc:389] thread 3 on device 0 model ./resnet50_xh2_b1_1roi_1core_O2_dynamic.hmm.
I20260414 08:09:23.092104 435374 resnet50_multistreams.cc:403] Reference model input info:
I20260414 08:09:23.092141 435374 resnet50_multistreams.cc:409]   Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:09:23.092155 435374 resnet50_multistreams.cc:409]   Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:09:23.096909 435374 resnet50_multistreams.cc:440] input image shape: [375 x 500 x 3]
I20260414 08:09:23.096963 435374 resnet50_multistreams.cc:452] pad input image to [512 x 512 x 3], height=512, width=512
I20260414 08:09:23.099234 435374 resnet50_multistreams.cc:535] sample queue size is 10
I20260414 08:09:23.099335 435409 resnet50_multistreams.cc:295] Count of Input: 2
I20260414 08:09:23.099352 435410 resnet50_multistreams.cc:295] Count of Input: 2
I20260414 08:09:23.099375 435409 resnet50_multistreams.cc:299] Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:09:23.099382 435409 resnet50_multistreams.cc:299] Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:09:23.099385 435411 resnet50_multistreams.cc:295] Count of Input: 2
I20260414 08:09:23.099397 435412 resnet50_multistreams.cc:295] Count of Input: 2
I20260414 08:09:23.099383 435410 resnet50_multistreams.cc:299] Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:09:23.099386 435409 resnet50_multistreams.cc:306] Count of Output: 1
I20260414 08:09:23.099421 435412 resnet50_multistreams.cc:299] Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:09:23.099426 435411 resnet50_multistreams.cc:299] Input[input.1] info: TensorInfo: shape: [1,3,512,512], stride: [393216,512,1,393216,512,2,1,262144], dtype: UINT8, format: YUV420SP, size: 393216, memsize: 393216
I20260414 08:09:23.099416 435410 resnet50_multistreams.cc:299] Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:09:23.099438 435410 resnet50_multistreams.cc:306] Count of Output: 1
I20260414 08:09:23.099446 435410 resnet50_multistreams.cc:310] Output[495] info: TensorInfo: shape: [1,1000], stride: [2000,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
I20260414 08:09:23.099429 435409 resnet50_multistreams.cc:310] Output[495] info: TensorInfo: shape: [1,1000], stride: [2000,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
I20260414 08:09:23.099435 435411 resnet50_multistreams.cc:299] Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:09:23.099470 435411 resnet50_multistreams.cc:306] Count of Output: 1
I20260414 08:09:23.099427 435412 resnet50_multistreams.cc:299] Input[resizer_crop_input.1] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
I20260414 08:09:23.099475 435411 resnet50_multistreams.cc:310] Output[495] info: TensorInfo: shape: [1,1000], stride: [2000,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
I20260414 08:09:23.099492 435412 resnet50_multistreams.cc:306] Count of Output: 1
I20260414 08:09:23.099540 435412 resnet50_multistreams.cc:310] Output[495] info: TensorInfo: shape: [1,1000], stride: [2000,2], dtype: FLOAT16, format: ND, size: 2000, memsize: 2000
I20260414 08:09:23.104459 435410 resnet50_multistreams.cc:316] thread 1 on device 0 infer start...
I20260414 08:09:23.104459 435412 resnet50_multistreams.cc:316] thread 3 on device 0 infer start...
I20260414 08:09:23.104460 435409 resnet50_multistreams.cc:316] thread 0 on device 0 infer start...
I20260414 08:09:23.104462 435411 resnet50_multistreams.cc:316] thread 2 on device 0 infer start...
I20260414 08:09:23.106803 435412 resnet50_multistreams.cc:367] thread 3 on device 0 run sample 1 end.
I20260414 08:09:23.106856 435410 resnet50_multistreams.cc:367] thread 1 on device 0 run sample 0 end.
I20260414 08:09:23.108567 435411 resnet50_multistreams.cc:367] thread 2 on device 0 run sample 3 end.
I20260414 08:09:23.108612 435409 resnet50_multistreams.cc:367] thread 0 on device 0 run sample 2 end.
I20260414 08:09:23.110297 435412 resnet50_multistreams.cc:367] thread 3 on device 0 run sample 4 end.
I20260414 08:09:23.110347 435410 resnet50_multistreams.cc:367] thread 1 on device 0 run sample 5 end.
I20260414 08:09:23.112094 435411 resnet50_multistreams.cc:367] thread 2 on device 0 run sample 6 end.
I20260414 08:09:23.112118 435411 resnet50_multistreams.cc:371] thread 2 on device 0 completed. 2 sampels tested.
I20260414 08:09:23.112119 435409 resnet50_multistreams.cc:367] thread 0 on device 0 run sample 7 end.
I20260414 08:09:23.112163 435409 resnet50_multistreams.cc:371] thread 0 on device 0 completed. 2 sampels tested.
I20260414 08:09:23.113835 435412 resnet50_multistreams.cc:367] thread 3 on device 0 run sample 8 end.
I20260414 08:09:23.113850 435412 resnet50_multistreams.cc:371] thread 3 on device 0 completed. 3 sampels tested.
I20260414 08:09:23.113874 435410 resnet50_multistreams.cc:367] thread 1 on device 0 run sample 9 end.
I20260414 08:09:23.113889 435410 resnet50_multistreams.cc:371] thread 1 on device 0 completed. 3 sampels tested.
I20260414 08:09:23.114555 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.114765 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.114898 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115010 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115106 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115194 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115298 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115400 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115490 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115573 435374 resnet50_multistreams.cc:220] top1: Index=65, Conf=0.7130048, Label=[sea snake]
I20260414 08:09:23.115590 435374 resnet50_multistreams.cc:582] <=== resnet50_multistreams c++ example completed.
```

**说明：由于 c++和 python 前处理有差异，部分结果可能存在少量差异。**

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
