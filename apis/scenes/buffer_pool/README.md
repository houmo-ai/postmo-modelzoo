# Buffer Pool Example

## 目录

[TOC]

## 概述

该示例展示了一个简单的应用层的内存池的使用方法。支持`HOST`、`DRM`, `RESERVED`三种类型内存。

## 运行

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./example_buffer_pool
```



