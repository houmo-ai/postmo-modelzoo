# Computing Perf

本工具用于测试后摩芯片进行模型推理的实际算力。由于模型推理很难发挥芯片的全部理论算力，因此测试数据将低于芯片的理论算力。

[TOC]

## 1.测试说明

本测试支持Linux x86_64及Linux aarch64平台：
- Linux x86_64平台：本测试将生成一个包含若干层卷积的onnx模型，编译该模型并在芯片上推理，统计模型运行时间，最终计算得到芯片运行该模型的算力。
- Linux aarch64平台：本测试将下载一个包含若干层卷积的预编译模型，在芯片上推理该模型，统计模型运行时间，最终计算得到芯片运行该模型的算力。

*注：测试程序中会自动识别Linux平台类型，无需手动指定。*

## 2.测试方法

进入tools/computing_perf目录，执行：

```bash
python3 computing_perf.py
```

## 3.测试结果

基于Linux x86_64平台的执行结果

```bash
#########################################
##  AI core compute performance test   ##
#########################################

=========================================
Generating onnx model
Model generated successfully, saving to ./xh1_perf_test/xh1_conv.onnx.
=========================================
Building model with tcim in output dir: ./xh1_perf_test/output
Model built successfully.
=========================================
Running model ./xh1_perf_test/output/xh1_conv.hmm
thread 2 on device 0 load model done.
thread 0 on device 0 load model done.
thread 1 on device 0 load model done.
thread 3 on device 0 load model done.
Model run successfully on device 0, elapsed time: 5.70 seconds.
=========================================
Performance is 222.19 TOPS.
```

基于Linux aarch64平台的执行结果

```bash
#########################################
##  AI core compute performance test   ##
#########################################

=========================================
Skipping model generation since the current platform is aarch64.
download from http://139.224.0.199:8082/artifactory/houmo/release/models/computing_perf/hmm_xh1_conv_4cores_20250613.zip to /usr/local/src/houmo-examples/models/hmm_xh1_conv_4cores_20250613.zip
download finished.
extract to ./
=========================================
Running model ./xh1_perf_test/output/xh1_conv.hmm
thread 2 on device 0 load model done.
thread 1 on device 0 load model done.
thread 0 on device 0 load model done.
thread 3 on device 0 load model done.
Model run successfully on device 0, elapsed time: 15.48 seconds.
=========================================
Performance is 81.84 TOPS.
```