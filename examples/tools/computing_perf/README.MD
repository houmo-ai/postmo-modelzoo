# Computing Perf

本工具用于测试后摩芯片进行模型推理的实际算力。由于模型推理很难发挥芯片的全部理论算力，因此测试数据将低于芯片的理论算力。

[TOC]

## 1.测试说明
本测试将生成一个包含若干层卷积的onnx模型，编译该模型并在芯片上推理，统计模型运行时间，最终计算得到芯片运行该模型的算力。
## 2.测试方法

进入tools/computing_perf目录，执行：

```bash
python3 computing_perf.py
```

## 3.测试结果

```bash
#########################################
##  AI core compute performance test   ##
#########################################

=========================================
Generating onnx model
Model generated successfully, saving to /tmp/perf_test/xh1_conv.onnx.
=========================================
Building model with tcim in output dir: /tmp/perf_test/output
Model built successfully.
=========================================
Running model /tmp/perf_test/output/xh1_conv.hmm
thread 0 on device 0 load model done.
thread 1 on device 0 load model done.
thread 2 on device 0 load model done.
thread 3 on device 0 load model done.
Model run successfully on device 0, elapsed time: 6.05 seconds.
=========================================
Performance is 222.55 TOPS.
```
