# TCIM Examples

## 目录

[TOC]

## 概述

TCIM Examples 展示TCIM API的使用方式，包括调用编译API编译模型，调用python和c++ runtime API进行模型推理。

每个示例以模型为单位，说明每个模型从量化后模型开始编译和推理的全过程。模型目录内的主要文件说明如下：

| 文件名         | 说明                                     |
| ------------- | --------------------------------------- |
| run.sh        | 一键执行脚本，完成模型编译、推理全过程      |
| build.py      | 模型编译示例代码                          |
| CMakeLists.txt | 构建c++示例的CMake项目的配置文件          |
| [模型名].py    | python runtime API推理示例代码           |
| [模型名].cc    | c++ runtime API推理示例代码              |

## 软件依赖

示例中使用了一些第三方库实现程序编译、图像和数据处理、结果显示等功能，需要安装第三方软件，请自行安装。
以下为公共依赖，每个示例可能有其他依赖，请参考各示例的README.md文件。

- Python（3.9以上版本，需要和houmo-tcim-runtime支持的版本一致）
  - windows下将python可执行程序目录设置为环境变量PYTHON_DIR
- CMake（建议3.16.3以上版本），主要用于c++示例编译
  - linux下可通过apt等包管理工具直接安装
  - windows下可下载安装包安装，将bin目录设置为环境变量CMAKE_DIR
- OpenCV库（4.x版本），主要用于c++示例图像读取和处理，结果渲染显示
  - linux下可通过apt等包管理工具直接安装
  - windows下可下载安装包解压，将安装目录设置为环境变量OPENCV_DIR（目录下存在OpenCVConfig.cmake）并将dll目录加入PATH

python依赖可通过requirements.txt安装：

```bash
pip install -r requirements.txt
```

此外，示例运行需要依赖houmo-tcim-runtime，参考后摩大道软件平台快速入门配置runtime环境。

## 示例列表

| 示例名                | 目录        | 说明                                         | 支持的芯片类型  |
| --------------------- | ---------- | -------------------------------------------- | -------------- |
| resnet50              | inferences | resnet50图像输入分类模型推理示例               | 2core/4core    |
| yolov5s               | inferences | yolov5s图像输入目标检测模型推理示例            | 2core/4core    |
| resnet50_multistreams | inferences | resnet50图像输入模型多stream推理示例           | 2core/4core   |
| resnet50_pipeline     | inferences | resnet50图像输入模型pipeline推理示例           | 2core/4core   |
| qwen2                 | inferences | qwen2 大语言模型推理示例                       | 4core         |
| qwen3                 | inferences | qwen3 大语言模型推理示例                       | 2core/4core   |
| video_detect          | scenes     | 视频解码+yolov5s检测+resnet50分类多阶段推理示例 | 2core/4core   |

## 快速上手

参考各示例的README.md文件。
