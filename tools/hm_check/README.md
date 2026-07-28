# 环境检测工具

检测当前环境是否满足运行要求。当前支持 Linux、Android 和 OpenHarmony (OHOS) 平台。

## 检测项：

- 设备固件版本
- 驱动`SDK`版本
- `Runtime`库版本
- 芯片主频
- 算力
- DDR读写带宽
- PCIE传输带宽

## 运行方法：


- Linux 平台（Bash）编译：

  仓库包含 `build_linux.sh`，在 Linux 环境下使用 `cmake` 进行配置和构建。

  示例：

  ```bash
  # 在 tools/hm_check 目录下
  # 默认脚本会读取环境变量 `TCIM_RUNTIME_PATH` 与 `HOUMO_SDK_PATH`（可通过 `--tcim`/`--houmo` 覆盖）。
  # 也可以先在 shell 中设置环境变量：
  export TCIM_RUNTIME_PATH=/opt/tcim
  export HOUMO_SDK_PATH=/opt/houmo_sdk
  ./build_linux.sh -b build -c Release -j 8 --install
  ```

  > **注意**：
  > - 给脚本添加可执行权限：`chmod +x build_linux.sh`；
  > - 默认会优先使用 `TCIM_RUNTIME_PATH` 与 `HOUMO_SDK_PATH` 环境变量；如果 未设置，脚本会尝试在常见路径（例如 `/opt/tcim`、`/usr/local/tcim` 等）自动 检测；若检测失败，则会要求你以参数或环境变量形式提供路径；
  > - 脚本会检查 `cmake` 是否在 `PATH`，并要求 `TCIM_RUNTIME_PATH` 与   `HOUMO_SDK_PATH` 已设置（或通过 `--tcim`/`--houmo` 提供）。

- Windows 平台（MSVC）编译：

  脚本 `build_windows.ps1` 已加入仓库，支持在 Windows+Visual Studio 环境下使用 `cmake` 进行配置和构建。

  示例（PowerShell）：

  ```powershell
  # 在 tools/hm_check 目录下
  # 指定 TCIM_RUNTIME_PATH / HOUMO_SDK_PATH（也可以先在系统环境中设置）
  .\build_windows.ps1 -BuildDir "build" -Configuration Release -Generator "Visual Studio 17 2022" -Platform x64 -TCIM "C:\tcim" -HOUMO  "C:\houmo_sdk" -Install
  ```

  也可以使用批处理封装：

  ```bat
  build_windows.bat -BuildDir build -Configuration Release -Generator "Visual Studio 17 2022" -Platform x64 -TCIM C:\tcim -HOUMO C:\houmo_sdk
  ```

  > 脚本要点：
  > - 需要 `cmake` 在 PATH 中；
  > - 需要设置 `TCIM_RUNTIME_PATH` 和 `HOUMO_SDK_PATH`（脚本参数或系统环境变量）；
  > - 脚本会调用 `cmake --build --config <Configuration>` 完成并行构建；
  > - 使用 `-Install` 会把可执行安装到 `build\install`。

- Android 平台（NDK / cross build）编译：

  仓库包含 `build_android.sh`，用于交叉编译 `hm-check` 到 Android ABI（例如 `arm64-v8a`）。脚本会优先使用环境变量 `NDK_PATH`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH`；也可以通过 `--ndk`/`--tcim`/`--houmo` 参数覆盖。

  示例（在 tools/hm_check 目录下）：

  ```bash
  # 推荐先在 shell 中设置：
  export NDK_PATH=/path/to/android-ndk
  export TCIM_RUNTIME_PATH=/opt/tcim
  export HOUMO_SDK_PATH=/opt/houmo_sdk
  ./build_android.sh --abi arm64-v8a --platform android-21 -b build_android -c Release --install
  ```

  > 注意：
  > - 脚本会尝试自动查找 NDK（`NDK_PATH`、`$HOME/Android/Sdk/ndk-bundle` 等），但推荐显式提供 `--ndk` 或 `NDK_PATH`；
  > - 确保 `TCIM_RUNTIME_PATH` 和 `HOUMO_SDK_PATH` 指向为目标 ABI（Android）编译好的头文件与库；
  > - 若 CMakeLists 中使用到平台特定工具（例如 `objcopy`、`ld`），交叉编译环境下可能需要调整这些工具的路径或用法。

- OpenHarmony (OHOS) 平台交叉编译：

  仓库包含 `build_OHOS.sh`，用于交叉编译 `hm-check` 到 OHOS ABI（aarch64-linux-ohos）。脚本需要设置环境变量 `OHOS_SDK`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH`。

  示例（在 tools/hm_check 目录下）：

  ```bash
  # 推荐先在 shell 中设置：
  export OHOS_SDK=/path/to/OpenHarmony/release/6.0-Release/linux/native
  export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
  export HOUMO_SDK_PATH=/path/to/houmo-sdk
  ./build_OHOS.sh
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

===== Check Summary =====
  PASS : 16
  WARN : 2
  FAIL : 0
=========================
```