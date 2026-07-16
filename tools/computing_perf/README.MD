# Computing Perf

本工具用于测试后摩芯片进行模型推理的实际算力。由于模型推理很难发挥芯片的全部理论算力，因此测试数据将低于芯片的理论算力。

[TOC]

## 1.测试说明

本测试支持`Linux x86_64`平台：
- `Linux x86_64`平台：本测试将生成一个包含若干层卷积的`onnx`模型，编译该模型并在芯片上推理，统计模型运行时间，最终计算得到芯片运行该模型的算力。

> 注意:
> - 测试程序中会自动识别Linux平台类型，无需手动指定。
> - **`INT8`算力测试需要使用`V2`芯片**

## 2.测试方法

进入`tools/computing_perf`目录，执行：

```bash
# bFP16
python3 computing_perf.py
# INT8
python3 computing_perf.py --compute-mode int8
```

如需测试不包含 load/store 开销的时间，可以添加 `--no-load-store` 参数。该参数在模型编译时仅生成 `xh2.te` 算子列表，用于构建不包含 load/store 的测试模型；脚本会在输出文件名和临时目录中追加 `no_load_store` 后缀。

```bash
# bFP16，不包含 load/store 开销
python3 computing_perf.py --no-load-store
# INT8，不包含 load/store 开销
python3 computing_perf.py --compute-mode int8 --no-load-store
```

## 3. XH2 测试结果

- 基于`Linux x86_64` `1.4G` M50 平台的`bFP16`算力测试执行结果

    ```bash
    Model run successfully on device 0, elapsed time: 1.99 seconds.

    ============================================================
               Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |        [64, 64]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          0.3092 TFLOPs/sample
    Test time                 |          1.9889 seconds
    PERFORMANCE               |           79.61 TFLOPS@bFP16
    ============================================================
    ```

- 基于`Linux x86_64` `1.4G` M50 平台的`INT8`算力测试执行结果

    ```shell
    Model run successfully on device 0, elapsed time: 6.67 seconds.
    
    ============================================================
               Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |      [128, 128]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          1.2370 TOPs/sample
    Test time                 |          6.6712 seconds
    PERFORMANCE               |           94.93 TOPS@INT8
    ============================================================
    ```

- 基于`Linux x86_64` `1.3G` M50平台的`bFP16`、`no-load-store`算力测试执行结果

    ```shell
    ============================================================
           Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |        [64, 64]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          0.3092 TFLOPs/sample
    Test time                 |          1.7482 seconds
    PERFORMANCE               |           90.57 TFLOPS@bFP16
    ============================================================
    ```

- 基于`Linux x86_64` `1.3G` M50平台的`INT8`、`no-load-store`算力测试执行结果

    ```shell
    ============================================================
               Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |      [128, 128]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          1.2370 TOPs/sample
    Test time                 |          6.4797 seconds
    PERFORMANCE               |           97.74 TOPS@INT8
    ============================================================
    ```


- 基于`Linux x86_64` `1.7G` M50平台的`bFP16`、`no-load-store`算力测试执行结果

    ```shell
    ============================================================
               Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |        [64, 64]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          0.3092 TFLOPs/sample
    Test time                 |          1.3238 seconds
    PERFORMANCE               |          119.60 TFLOPS@bFP16
    ============================================================
    ```

- 基于`Linux x86_64` `1.7G` M50平台的`INT8`、`no-load-store`算力测试执行结果

    ```shell
    ============================================================
               Performance Test Summary
    ============================================================
    Model Parameter           |           Value
    ------------------------------------------------------------
    Number of Conv2d Layers   |              64
    Batch Size                |               1
    Input Channels            |             256
    Output Channels           |             256
    Feature Map Size          |      [128, 128]
    Kernel Shape              |          [3, 3]
    ------------------------------------------------------------
    Computing amount          |          1.2370 TOPs/sample
    Test time                 |          4.9401 seconds
    PERFORMANCE               |          128.20 TOPS@INT8
    ============================================================
    ```