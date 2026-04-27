# TCIM性能测试工具

本代码用于在后摩系列芯片的设备上测试TCIM模型推理性能。当前支持linux和android native环境（使用adb运行）。

## 目录

- [1. 工具说明](#1工具说明)
  - [1.1 文件说明](#11-文件说明)
  - [1.2 流程说明](#12-流程说明)
  - [1.3 参数说明](#13-参数说明)
  - [1.4 用法说明](#14-用法说明)
- [2. 快速开始](#2快速开始)
  - [2.1 环境准备](#21-环境准备)
  - [2.2 一键运行](#22-一键运行)
- [3. 免责声明](#3免责声明)

## 1.工具说明

### 1.1 文件说明

| 文件名       | 说明                                     |
| ------------ | --------------------------------------- |
| tcim_perf.cc | 主文件，实现TCIM模型推理流程              |

说明：

- `#include <getopt.h>`：Windows 平台使用的 getopt 实现，来源于 [mingw-w64](https://mingw-w64.org/) 运行时包，包含 Todd C. Miller 和 NetBSD 基金会的 BSD 许可证声明。


### 1.2 流程说明

程序按以下流程进行：

1. 解析输入参数
2. 创建推理Module
3. 准备输入数据和输入输出内存
4. warm up
5. 创建线程，每个线程绑定一个推理Module
6. 根据设定的次数执行输入数据->推理->获取结果循环，统计每次推理总时延和最大时延
7. 获取执行总时间，计算平均执行时间和吞吐量

### 1.3 参数说明

- `--model`, `-m`: (必选) 指定模型路径
- `--input`, `-i`: 指定测试数据路径，若不指定则使用随机数
- `--warm_up`, `-w`: 指定warm up次数，默认为1
- `--batch`, `-b`: 指定batch数，默认为1
- `--threads`, `-t`: 指定线程数，默认为1
- `--loops`, `-l`: 指定内循环次数(仅内部测试使用)，默认为1
- `--devices`, `-d`: 指定测试设备数，默认为1
- `--samples`, `-s`: 指定测试样本数，默认为1
- `--output`, `-o`: 指定输出文件路径，默认为当前路径
- `--infer_only`, `-y`: 指示是否仅进行推理，不进行输入输出数据传输（同时关闭结果检查）
- `--name`, `-n`: 模型名称，用于检索指定测试数据路径下的模型输入&golden数据
- `--streams`, `-e`: tcim stream数量，默认为4
- `--module_pool`, `-p`: 使用模型池进行推理，默认关闭
- `--modules`, `-c`: 允许实际加载的最大模型数，默认为core数量，本参数仅在使用模型池推理场景下生效
- `--interval`, `-v`: 按照指定间隔(ms)构造输入数据推送至推理任务队列中，默认不开启（在推理前构造所有输入数据放入推理任务队列中）
- `--queue_length`, `-q`: 推理任务队列最大长度，如队列中的推理任务超过最大长度，则中止程序，本参数仅在配置`--interval`参数后生效
- `--help`, `-h`: 参数说明信息

### 1.4 用法说明

#### 1.4.1 时延测试

测试模型的推理时延（delay），适合评估对时延极为敏感的业务场景。一般使用单batch单线程，同时将模型编译成多核将显著改善推理时延。

#### 1.4.2 吞吐测试

测试模型的推理吞吐（throughput），适合评估对时延不太敏感但对吞吐量要求高的业务场景。一般使用`--batch`和`--threads`配置多batch多线程，同时将模型编译成单核可能改善推理吞吐。

#### 1.4.3 吞吐&时延测试

测试模型的吞吐和时延的整体表现，适合评估对时延有限制下追求最大吞吐量的业务场景。可使用`--batch`和`--threads`配置batch和线程数，以及在编译时选择核数，绘制ROC曲线并找到合适的组合。

#### 1.4.4 仅推理测试

由于测试中包含输入和输出数据的时间，而在实际业务中不一定需要（如在芯片侧解码后直接输入，以及输出数据直接给下个模型使用等），导致吞吐测试的结果偏低。可通过`--infer_only`配置为仅推理，获取最纯净的推理吞吐。

## 2.快速开始

### 2.1 环境准备

### 2.1.1 Linux 环境

1. 在后摩智能资源中心获取:

- 后摩大道 linux x86_64 docker 镜像
- 示例代码压缩包，解压示例代码压缩包，解压后文件夹名为：houmo-examples-xh2

2. 将 houmo-examples-xh2 文件夹挂载至后摩 docker 镜像并创建&启动容器，进入docker容器

3. 进入 houmo-examples-xh2 文件夹，先检查 `env.sh` 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 2.1.2 Windows 环境（仅支持Windows11）

1. 在后摩智能资源中心获取:

- 后摩大道 windows系统的固件驱动、RUNTIME SDK并安装说明文档进行安装
- 示例代码压缩包，解压示例代码压缩包，解压后文件夹名为：houmo-examples-xh2

2. 参考houmo-examples-xh2文件夹下的tools/win_envs/README.MD和社区用户手册，配置windows相关开发环境，并检查必要的环境变量。

### 2.1.3 Android 环境

如果需要使用其他平台编译链进行交叉编译需要自行下载交叉编译链并设置环境变量 `TCIM_RUNTIME_PATH` 到目标平台 runtime 库目录, 以 android 平台为例：

1. 在后摩智能资源中心获取:

- 后摩大道 linux x86_64 docker 镜像
- 示例代码压缩包，解压示例代码压缩包，解压后文件夹名为：houmo-examples-xh2
- 后摩大道 android aarch64 的 Runtime SDK 压缩包，解压后文件夹名为：houmo-tcim-runtime-xh2

2. 将 houmo-examples-xh2 文件夹挂载至后摩 docker 镜像并创建&启动容器，进入docker容器

3. 进入 houmo-examples-xh2 文件夹，执行: `source env.sh`，读取环境变量 `HOUMO_EXAMPLES_PATH`，通常为 houmo-examples-xh2 文件夹绝对路径

4. 安装 Ninjia

```bash
sudo apt update
sudo apt install ninja-build -y
# 检查是否成功安装 ninja
ninja --version
```

5. 在 `HOUMO_EXAMPLES_PATH` 文件夹下创建 `toolchains` 文件夹，下载官方NDK[https://developer.android.google.cn/ndk/downloads/index.html?hl=ro]，解压到创建的 toolchains 目录下（也可通过设置环境变量 `NDK_PATH` 指定 NDK 路径）。预期结果如下：

```bash
houmo-examples-xh2/toolchains/android-ndk-r28c$ ls
build         meta       ndk-gdb   ndk-stack  NOTICE            prebuilt         README.md     simpleperf         sources     wrap.sh
CHANGELOG.md  ndk-build  ndk-lldb  ndk-which  NOTICE.toolchain  python-packages  shader-tools  source.properties  toolchains
```

6. 配置 RUNTIME 环境变量 `TCIM_RUNTIME_PATH` 为 houmo-tcim-runtime-xh2 文件夹的绝对路径

注：本例在 android-ndk-r28c 版本上验证通过，用户的环境如果不一致请自行修改适配。

### 2.2 编译程序

设置完成后进入程序目录编译：

### 2.2.1 Linux 环境

```bash
cd houmo-examples-xh2/tools/tcim_perf/
./build.sh
```

编译生成tcim_perf可执行文件在 houmo-examples-xh2/tools/bin 目录下。

### 2.2.2 Windows 环境（仅支持Windows11）

Windows系统下使用env.bat设置好环境变量后，按照如下步骤可以编译tcim_perf.exe，编译好的可执行程序再houmo-examples-xh2/tools/bin目录下。
``` bat
cd houmo-examples-xh2/tools/tcim_perf/
build_win.bat
```

### 2.2.3 Android 环境

build_ndk.sh 脚本中预设了编译所需的 NDK 环境路径为 `${HOUMO_EXAMPLES_PATH}/toolchains/android-ndk-r28c`，如有变化，可使用环境变量 `NDK_PATH` 进行配置。

```bash
cd houmo-examples-xh2/tools/tcim_perf/
./build_ndk.sh
```

编译生成tcim_perf可执行文件在 houmo-examples-xh2/tools/android 目录下。

### 2.3 一键运行

通过命令行参数修改模型路径、执行次数、线程数和stream数等。

如果是在 android adb 环境执行，需要先将 tcim_perf 可执行文件和模型等拷贝到adb环境中，将runtime和hal库路径加入到环境变量 `LD_LIBRARY_PATH`。

进入tcim_perf可执行文件所在目录，然后执行：

```bash
./tcim_perf -m xxx.hmm
```

提供run脚本运行tcim_perf可执行文件，可将参数配置在run脚本中执行tcim_perf程序

run脚本支持的参数为：
- -m，--model [必选] hmm模型路径
- -i，--input [可选] 输入和输出golden文件夹，默认为空
- -o，--output [可选] 性能结果文件输出文件夹，默认为当前文件夹
- -w，--warm_up [可选] warm up次数，默认为1
- -b，--batch [可选] 模型batch数，仅用于计算正确的qps，默认为1
- -l，--loops [可选] 模型内循环数，仅用于内部测试，默认为1
- -t，--threads [可选] 推理线程数，默认为1
- -d，--devices [可选] 推理模型占用设备数，模型编译时决定，默认为1
- -s，--samples [可选] 测试样本数，默认为1
- -n，--name [可选] 模型名，用于寻找golden数据
- -e，--streams [可选] tcim stream数量，默认为4

将run.sh脚本拷贝到tcim_perf可执行文件的同级目录中，进入该目录，执行：

```bash
# linux环境
./run.sh
# android adb环境
/system/bin/sh run.sh
```

参考结果：

执行程序后，生成如下格式测试结果

```bash
[latency] Inference avg:    3.365 ms,       max:    6.331 ms,       min:    2.797 ms
[latency] Input     avg:    0.665 ms,       max:    1.095 ms,       min:    0.480 ms
[latency] Output    avg:    0.247 ms,       max:    0.984 ms,       min:    0.110 ms
[latency] End2End   avg:    4.278 ms,       max:    7.316 ms,       min:    3.469 ms
[Throughput] total:    110.483 ms, avg:    1.105 ms
[Throughput] qps:  905.117
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
