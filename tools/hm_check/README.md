# 环境检测工具

检测当前环境是否满足运行要求。

## 检测项：

- 设备固件版本
- 驱动`SDK`版本
- `Runtime`库版本
- 芯片主频
- 算力
- DDR读写带宽
- PCIE传输带宽

## 运行方法：

- 容器环境：

容器环境已预编译安装环境检测工具，可直接运行：

```bash
hm-check
```

- 自行编译：

自行编译需要依赖`tcim_runtime_lite`、`driver_sdk_lib`、`gcc>=9.4`，需要自行准备环境

```bash
cd ${HOUMO_MODELZOO_PATH}/tools/hm_check
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## 检查说明：

    - `PASS`：表示通过
    - `WARN`：表示警告，并且会有相关提示，可能不影响运行
    - `FAIL`：表示失败，并且会有相关提示，会影响运行

```bash
=== System Check Report ===
  Driver Version                 [PASS]   v0.6.0.dev20251211    
  SDK Version                    [PASS]   v0.6.0                
  Runtime Version                [PASS]   v0.6.0.dev20251203    
  Device0 Firmware Version       [PASS]   v0.6.0                
  Device1 Firmware Version       [PASS]   v0.6.0                
  Version Consistency            [WARN]   Mismatch              Maybe cause unknown problems.
  Device0 Cur IPU Freq           [PASS]   1400.00 MHz           
  Device1 Cur IPU Freq           [PASS]   1400.00 MHz           
  Device0 Measured Compute       [PASS]   81.15 TOPS            
  Device1 Measured Compute       [PASS]   81.13 TOPS            
  Device0 Measured DDR Read      [PASS]   125.41 GB/s           
  Device1 Measured DDR Read      [PASS]   126.25 GB/s           
  Device0 Measured DDR Write     [PASS]   121.70 GB/s           
  Device1 Measured DDR Write     [PASS]   118.82 GB/s           
  Device0 Measured PCIe H2D      [PASS]   3.29 GB/s             
  Device0 Measured PCIe D2H      [PASS]   3.32 GB/s             
  Device1 Measured PCIe H2D      [WARN]   6.23 GB/s             The PCIE transfer bandwidth should not be less than 7.88 GB/s * 0.80
  Device1 Measured PCIe D2H      [PASS]   6.59 GB/s             
===========================
✔ All checks passed.
```