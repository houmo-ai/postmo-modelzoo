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

## 3. XH1 测试结果

基于Linux x86_64平台的执行结果

```bash
Model run successfully on device 0, elapsed time: 11.86 seconds.

============================================================
           Performance Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Number of Conv2d Layers   |              64
Batch Size                |               1
Input Channels            |             256
Output Channels           |             256
Feature Map Size          |      [256, 256]
Kernel Shape              |          [3, 3]
------------------------------------------------------------
Computing amount          |          4.9478 TOPs/sample
Test time                 |         11.8627 seconds
PERFORMANCE               |          213.55 TOPS
============================================================
```

基于Linux aarch64平台的执行结果

```bash
Model run successfully on device 0, elapsed time: 31.93 seconds.

============================================================
           Performance Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Number of Conv2d Layers   |              64
Batch Size                |               1
Input Channels            |             256
Output Channels           |             256
Feature Map Size          |      [256, 256]
Kernel Shape              |          [3, 3]
------------------------------------------------------------
Computing amount          |          4.9478 TOPs/sample
Test time                 |         31.9305 seconds
PERFORMANCE               |           79.34 TOPS
============================================================

```