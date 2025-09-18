# Module Pool Example

## 目录

[TOC]

## 概述

本示例以yolov5s检测模型推理&resnet50分类模型推理为例展示多线程场景下模型池的基本使用方式。

本示例仅支持c++，仅支持后摩XH1平台。


## 快速上手

### 1. 获取芯片模型

通过get_model.py脚本下载预编译的模型，含yolov5s和resnet50两个模型。

```bash
python3 get_model.py
```

### 2. 模型推理

#### 2.1 编译并执行c++演示程序

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_module_pool
```

#### 2.2 参考结果

模型推理执行成功输出以下打印：

```bash
# Linux x86 环境
[2025-09-16 11:36:58.505] [tid 4045982] [info] ModulePool 0x55c499577650 stats, module_type_num:4, stream num:8, infer_task_num:0.
[2025-09-16 11:36:58.505] [tid 4045982] [info]   module name:./resnet50.hmm, loaded module num:4
[2025-09-16 11:36:58.505] [tid 4045982] [info]   module name:./yolov5s.hmm, loaded module num:4
[2025-09-16 11:36:58.505] [tid 4045982] [info]   module name:resnet50_b1_1core, loaded module num:4
[2025-09-16 11:36:58.505] [tid 4045982] [info]   module name:yolov5s_b1_1core, loaded module num:4
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:0, infer task num:1999
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:1, infer task num:2000
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:2, infer task num:2002
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:3, infer task num:1999
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:4, infer task num:2001
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:5, infer task num:2002
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:6, infer task num:2002
[2025-09-16 11:36:58.505] [tid 4045982] [info]   stream id:7, infer task num:1995
[2025-09-16 11:36:58.505] [tid 4045982] [info] ===> ModulePool Deinit start.
[2025-09-16 11:36:58.506] [tid 4046026] [info] <== ModulePool InferThread end, stream id:3
[2025-09-16 11:36:58.506] [tid 4046028] [info] <== ModulePool InferThread end, stream id:5
[2025-09-16 11:36:58.506] [tid 4046030] [info] <== ModulePool InferThread end, stream id:7
[2025-09-16 11:36:58.506] [tid 4046023] [info] <== ModulePool InferThread end, stream id:0
[2025-09-16 11:36:58.506] [tid 4046025] [info] <== ModulePool InferThread end, stream id:2
[2025-09-16 11:36:58.506] [tid 4046027] [info] <== ModulePool InferThread end, stream id:4
[2025-09-16 11:36:58.506] [tid 4046024] [info] <== ModulePool InferThread end, stream id:1
[2025-09-16 11:36:58.506] [tid 4046029] [info] <== ModulePool InferThread end, stream id:6
[2025-09-16 11:36:58.514] [tid 4045982] [warning] <=== module pool c++ example completed.
```

### 3. 一键执行

可以使用run脚本一键执行并查看结果，run脚本中不提供依赖环境安装。

linux下执行：

```bash
bash run.sh
```
