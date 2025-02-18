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

## 示例列表

| 示例名                | 说明                                     |
| --------------------- | ---------------------------------------- |
| resnet50              | resnet50图像输入模型推理示例               |
| resnet50_multistreams | resnet50图像输入模型多stream推理示例       |
| resnet50_pipeline     | resnet50图像输入模型pipeline推理示例       |
| qwen2                 | qwen2 混合量化模型推理示例                 |

## 快速上手

参考各示例的README.md文件。
