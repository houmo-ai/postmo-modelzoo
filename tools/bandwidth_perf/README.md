# Bandwidth Perf

本工具用于测试后摩芯片进行模型推理的实际带宽。由于模型推理很难发挥芯片的全部理论带宽，因此测试数据将低于芯片的理论带宽。

[TOC]

## 1.测试说明

本测试支持Linux x86_64及Linux aarch64平台：
- Linux x86_64平台：本测试将生成一个包含若干层OP的onnx模型，编译该模型并在芯片上推理，统计模型运行时间，最终计算得到芯片运行该模型的带宽。
- Linux aarch64平台：本测试将下载一个包含若干层OP的预编译模型，在芯片上推理该模型，统计模型运行时间，最终计算得到芯片运行该模型的带宽。

*注：测试程序中会自动识别Linux平台类型，无需手动指定。*

## 2.测试方法

进入tools/bandwidth_perf目录，执行：

```bash
# 写带宽
python3 bandwidth_perf.py --type w
# 读带宽
python3 bandwidth_perf.py --type r
```

## 3. XH1 测试结果

基于Linux x86_64平台的执行结果

```bash
# 写带宽
Model run successfully on device 0, elapsed time: 7.81 seconds.

============================================================
      Write Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          7.8059 seconds
Write Bandwidth           |           80.06 GiB/s
============================================================

# 读带宽
Model run successfully on device 0, elapsed time: 7.24 seconds.

============================================================
      READ Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          7.2433 seconds
READ BANDWIDTH            |           86.28 GiB/s
============================================================
```

基于Linux aarch64平台的执行结果

```bash
# 写带宽
Model run successfully on device 0, elapsed time: 8.07 seconds.

============================================================
      Write Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          8.0657 seconds
Write Bandwidth           |           77.48 GiB/s
============================================================

# 读带宽
Model run successfully on device 0, elapsed time: 8.09 seconds.

============================================================
      Read Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          8.0867 seconds
Read Bandwidth            |           77.28 GiB/s
============================================================
```

## 3. XH2 测试结果

基于Linux x86_64平台的执行结果

```bash
# 写带宽
Model run successfully on device 0, elapsed time: 5.06 seconds.

============================================================
      Write Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          5.0634 seconds
Write Bandwidth           |          123.43 GiB/s
============================================================

# 读带宽
Model run successfully on device 0, elapsed time: 4.92 seconds.

============================================================
      Read Bandwidth Test Summary
============================================================
Model Parameter           |           Value
------------------------------------------------------------
Data size                 |            8.00 MiB
Round number              |           80000
Test time                 |          4.9157 seconds
Read Bandwidth            |          127.14 GiB/s
============================================================
```