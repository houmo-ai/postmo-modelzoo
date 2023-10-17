#!/usr/bin/env python3

from enum import Enum


class PaddingMode(Enum):
    LEFT_TOP = 0
    CENTER = 1


class PixelFormat(Enum):
    NONE = -1
    BGR = 0
    RGB = 1
    GRAY = 2
    YUV420P = 3
    YUV420SP_NV12 = 4


class DataLayout(Enum):
    NONE = -1
    NCHW = 0
    NHWC = 1


class DataType(Enum):
    UINT8 = 0
    INT8 = 1
    FLOAT16 = 2
    FLOAT32 = 3
    TVM_UINT8 = 4
    TVM_INT8 = 5
    TVM_FLOAT16 = 6
    TVM_FLOAT32 = 7

