# Yolov5s Example

当前支持Linux、Android、OpenHarmony (OHOS) 平台。

## 目录

[TOC]

## 概述

TCIM Examples 以 yolov5s 模型为例展示 TCIM API 推理的基本使用方式。

1. yolov5s.py 展示 python 下单线程推理过程，包括：
    - python 单线程推理，使用默认 stream（无需指定）
2. yolov5s.cc 展示 c++下单线程推理过程，包括：
    - c++下单线程推理，使用默认 stream（无需指定）

对于 yolov5s 模型中的后处理部分，本示例提供两种实现方法：

1. （默认）基于代码实现后处理逻辑。
2. 将 yolov5s 后处理部分独立为一个 onnx 模型，使用 onnxruntime 推理得到结果。

其中方法 2 在本示例中仅提供 Linux 系统运行方法，如需在 windows 系统中运行可自行安装配置 onnxruntime 环境，则可使用相同步骤运行方法 2 示例。

## 软件依赖

本示例依赖 opencv 库进行图像预处理和可视化，依赖 onnxruntime 库推理 yolov5s 后处理模型。

## 快速开始

1. 获取模型文件，执行：

```bash
# 后处理实现为方法1
python3 get_model.py
# 后处理实现为方法2 (仅支持Linux)，其中将安装执行C++示例所需的onnxruntime环境
python3 get_model.py --enable_ort
```

2. 执行 python 演示程序，执行命令：

```bash
# 后处理实现为方法1
python3 yolov5s.py
# 后处理实现为方法2 (仅支持Linux)
python3 yolov5s.py --enable_ort
```

3. 编译并执行 c++演示程序，执行命令：

```bash
# 后处理实现为方法1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make -j
make install
cd ..
./example_yolov5s
```

```bash
# 后处理实现为方法2 (仅支持Linux)
if [ -z "${HOUMO_EXAMPLES_PATH}" ]; then
    echo "HOUMO_EXAMPLES_PATH is not set."
    exit 1
fi
mkdir build
cd build
cmake -DENABLE_ORT=ON -DCMAKE_INSTALL_PREFIX=../
make -j
make install
cd ..
# 将onnxruntime的动态库路径加入LD_LIBRARY_PATH，执行C++用例时需要依赖
export LD_LIBRARY_PATH="${HOUMO_EXAMPLES_PATH}/3rdparty/onnxruntime/lib:${LD_LIBRARY_PATH}"
# 执行C++示例
./example_yolov5s --enable_ort
```

**OpenHarmony (OHOS) 平台交叉编译：**

```bash
export OHOS_SDK=/path/to/OpenHarmony/release/6.0-Release/linux/native
export HOUMO_TARGET=xh2
export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
./build_OHOS.sh
cd bin
./example_yolov5s
```

## 参考结果

### Python 示例结果

Python 演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
===> yolov5s python example start...
Found model file: ./yolov5s_xh2_b1_1roi_1core_O2_dynamic_v2.hmm
input image shape: (426, 640, 3)
Model input info:
  Input[images] shape = [1, 3, 1080, 1920], dtype = <class 'numpy.uint8'>, mem_size = 3110400, stride = [3110400, 1920, 1, 3110400, 1920, 2, 1, 2073600], format = YUV420SP
  Input[resizer_crop_images] shape = [1, 10], dtype = <class 'numpy.int32'>, mem_size = 40, stride = [40, 4], format = ND
pad input image to (1080, 1920, 3) height = 1080, width = 1920
input_data shape: (1, 3, 1080, 1920), dtype: uint8, dyn_info: [[  0   0 426 640 426 640 106   0 108   0]]
output[340] shape = [1, 3, 80, 80, 85], dtype = <class 'numpy.float32'>, format = ND.
output[378] shape = [1, 3, 40, 40, 85], dtype = <class 'numpy.float32'>, format = ND.
output[416] shape = [1, 3, 20, 20, 85], dtype = <class 'numpy.float32'>, format = ND.
box num = 21
x1:7, y1:167, x2:153, y2:261, conf:0.903196, cls:62
x1:293, y1:215, x2:353, y2:318, conf:0.852535, cls:56
x1:411, y1:154, x2:465, y2:299, conf:0.829967, cls:0
x1:557, y1:211, x2:640, y2:285, conf:0.764457, cls:62
x1:449, y1:169, x2:515, y2:289, conf:0.723631, cls:72
x1:167, y1:232, x2:185, y2:267, conf:0.654429, cls:75
x1:550, y1:297, x2:587, y2:400, conf:0.606098, cls:39
x1:480, y1:361, x2:638, y2:425, conf:0.600883, cls:60
x1:361, y1:220, x2:421, y2:314, conf:0.599389, cls:56
x1:448, y1:120, x2:462, y2:140, conf:0.595322, cls:74
x1:403, y1:221, x2:442, y2:307, conf:0.576745, cls:56
x1:550, y1:298, x2:587, y2:399, conf:0.535768, cls:75
x1:231, y1:172, x2:267, y2:212, conf:0.519052, cls:58
x1:332, y1:176, x2:369, y2:223, conf:0.458111, cls:58
x1:411, y1:157, x2:472, y2:294, conf:0.443604, cls:72
x1:358, y1:212, x2:374, y2:229, conf:0.401225, cls:75
x1:491, y1:173, x2:513, y2:282, conf:0.363853, cls:72
x1:241, y1:193, x2:254, y2:211, conf:0.341569, cls:75
x1:349, y1:211, x2:362, y2:229, conf:0.340402, cls:75
x1:355, y1:216, x2:421, y2:316, conf:0.326122, cls:60
x1:384, y1:171, x2:404, y2:212, conf:0.255194, cls:58
demo results saved to demo_results/python/000000000139.jpg
<=== yolov5s python example completed.
```

检测效果图片保存在 demo_results/python 目录下。

### C++示例结果

C++演示程序推理完成后会进行后处理，获得分类结果，执行成功输出以下打印：

```bash
======> yolov5s c++ example start...
Found .hmm file: ./yolov5s_xh2_b1_1roi_1core_O2_dynamic_v2.hmm
Load yolov5s model from file ./yolov5s_xh2_b1_1roi_1core_O2_dynamic_v2.hmm
Model ./yolov5s_xh2_b1_1roi_1core_O2_dynamic_v2.hmm loaded.
Count of Input: 2
Input[images] info: TensorInfo: shape: [1,3,1080,1920], stride: [3110400,1920,1,3110400,1920,2,1,2073600], dtype: UINT8, format: YUV420SP, size: 3110400, memsize: 3110400
Input[resizer_crop_images] info: TensorInfo: shape: [1,10], stride: [40,4], dtype: INT32, format: ND, size: 40, memsize: 40
input image shape: [426 x 640 x 3]
pad input image to [1080 x 1920 x 3], height=1080, width=1920
dyn_info: [0, 0, 426, 640, 426, 640, 106, 0, 108, 0]
Count of Output: 3
Output[340] info: TensorInfo: shape: [1,3,80,80,85], stride: [6528000,2176000,27200,340,4], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
Output[378] info: TensorInfo: shape: [1,3,40,40,85], stride: [1632000,544000,13600,340,4], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
Output[416] info: TensorInfo: shape: [1,3,20,20,85], stride: [408000,136000,6800,340,4], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000
detect num: 18.
box[6, 166, 153, 263], conf: 0.90319633, cls: 62.
box[292, 216, 352, 319], conf: 0.84127283, cls: 56.
box[411, 156, 464, 299], conf: 0.8292532, cls: 0.
box[448, 170, 513, 289], conf: 0.73832417, cls: 72.
box[556, 210, 639, 285], conf: 0.7210939, cls: 62.
box[360, 220, 420, 314], conf: 0.63769984, cls: 56.
box[550, 295, 587, 401], conf: 0.6210923, cls: 39.
box[480, 362, 637, 425], conf: 0.5836165, cls: 60.
box[448, 120, 461, 141], conf: 0.58207154, cls: 74.
box[403, 221, 442, 307], conf: 0.569859, cls: 56.
box[167, 233, 185, 267], conf: 0.54097867, cls: 75.
box[232, 173, 265, 212], conf: 0.5020509, cls: 58.
box[332, 176, 367, 223], conf: 0.4430598, cls: 58.
box[349, 211, 362, 230], conf: 0.39989126, cls: 75.
box[489, 173, 513, 282], conf: 0.37056684, cls: 72.
box[241, 193, 255, 212], conf: 0.31034493, cls: 75.
box[384, 170, 404, 213], conf: 0.29339647, cls: 58.
box[355, 213, 372, 229], conf: 0.263983, cls: 75.
demo results saved to demo_results/cpp/000000000139.jpg.
<=== yolov5s c++ example completed.
```

检测效果图片保存在 demo_results/cpp 目录下。

_注：由于 c++与 python 版本的前后处理不完全一致，算法结果可能有少量偏差。_

### 一键执行

可以使用 run 脚本一键执行查看结果

linux 下执行：

```bash
# 后处理实现为方法1
bash run.sh
# 后处理实现为方法2
bash run.sh enable_ort
```

windows 下执行：

```bash
bash run.bat
```

## 附录：Python Demo 流程说明

Python demo 的主流程位于 `yolov5s.py` 的 `__main__` 入口，整体可以分为模型加载、输入准备、动态 resize 信息构建、推理和后处理几个阶段。

1. 加载模型和样例图片

   demo 会在当前示例目录下自动查找 `.hmm` 文件并加载模型，然后读取样例图片 `../../data/000000000139.jpg`。同时保留一份原始 BGR 图片 `cv_image`，用于最后绘制检测框。

2. 解析模型输入信息

   模型通常包含两类输入：

   - `images`：图像输入，格式为 `YUV420SP`，示例中 shape 为 `[1, 3, 1080, 1920]`。
   - `resizer_crop_images`：动态 resize/crop/pad 参数输入，格式为 `int32`，shape 为 `[1, 10]`。

   demo 通过输入名中是否包含 `resizer_crop` 来区分这两类输入。非 `resizer_crop` 输入的 H/W 会作为上传给硬件 resizer 的最大图像画布尺寸，即 `max_img_height` 和 `max_img_width`。

3. 准备图像画布和有效 crop 区域

   Python demo 不再在 CPU 侧直接做 YOLOv5 的完整 letterbox 输入，而是把原始图像上传到较大的 YUV 画布，并通过 `dyn_info` 告诉模型内部的硬件 resizer 如何裁剪、缩放和 padding。

   如果原图尺寸小于模型图像输入画布：

   - 使用灰色值 `(114, 114, 114)` 在右侧和底部 padding 到 `max_img_height x max_img_width`。
   - `crop_height` 和 `crop_width` 保留原图的有效区域大小。
   - padding 出来的区域不作为有效图像参与 resize。

   如果原图尺寸超过模型图像输入画布：

   - 先把图像 resize 到 `max_img_width x max_img_height`。
   - `crop_height` 和 `crop_width` 使用整个上传画布大小。

4. 转换图像输入格式

   demo 会把 BGR 图像从 HWC 排布转成 CHW，再使用 `BGR2YUV(fmt="YUV420")` 转为运行时图像输入需要的 `YUV420` 数据，最终输入 tensor 类型为 `uint8`。

5. 构建 `dyn_info`

   `dyn_info` 是一个 `int32` 的 `[1, 10]` tensor，字段顺序如下：

   ```text
   [
     crop_offset_h,
     crop_offset_w,
     crop_h,
     crop_w,
     resize_h,
     resize_w,
     pad_top,
     pad_left,
     pad_bottom,
     pad_right
   ]
   ```

   在当前 YOLOv5 demo 中，crop 从上传画布左上角开始，所以：

   ```text
   crop_offset_h = 0
   crop_offset_w = 0
   crop_h = crop_height
   crop_w = crop_width
   ```

   由于硬件 resizer 处理的是 YUV420 数据，crop 宽高需要是偶数。demo 会将 crop 尺寸向下取偶数：

   ```python
   crop_height = crop_height - (crop_height % 2)
   crop_width = crop_width - (crop_width % 2)
   ```

   YOLOv5 使用等比例 letterbox 预处理，因此 resize 尺寸不是直接设成 `640 x 640`，而是先计算保持宽高比的缩放比例：

   ```python
   scale = min(target_height / crop_height, target_width / crop_width)
   ```

   然后根据该比例计算硬件 resizer 的输出尺寸，并向下对齐到偶数：

   ```python
   resizer_height = int(round(crop_height * scale)) & ~1
   resizer_width = int(round(crop_width * scale)) & ~1
   resizer_height = min(resizer_height, target_height)
   resizer_width = min(resizer_width, target_width)
   ```

   剩余空间作为 padding，尽量居中分配到上下左右：

   ```python
   pad_height = target_height - resizer_height
   pad_width = target_width - resizer_width
   pad_h_top = (pad_height // 2) & ~1
   pad_w_left = (pad_width // 2) & ~1
   pad_h_bottom = pad_height - pad_h_top
   pad_w_right = pad_width - pad_w_left
   ```

   最终构造：

   ```python
   dyn_info = np.array(
       [
           0,
           0,
           crop_height,
           crop_width,
           resizer_height,
           resizer_width,
           pad_h_top,
           pad_w_left,
           pad_h_bottom,
           pad_w_right,
       ],
       dtype=np.int32,
   )
   dyn_info = np.expand_dims(dyn_info, 0)
   ```

   以 README 中的样例图为例，原始有效图像大小为 `426 x 640`，目标输入为 `640 x 640`，因此等比例缩放后仍为 `426 x 640`，上下方向剩余 `214` 像素，padding 被分配为：

   ```text
   pad_top = 106
   pad_bottom = 108
   pad_left = 0
   pad_right = 0
   ```

   对应的 `dyn_info` 为：

   ```text
   [0, 0, 426, 640, 426, 640, 106, 0, 108, 0]
   ```

6. 设置模型输入并执行推理

   demo 遍历所有模型输入：

   - 如果输入名包含 `resizer_crop`，则设置 `dyn_info`。
   - 否则设置转换后的 YUV 图像输入。

   设置完成后调用 `module.run()` 和 `module.sync()` 执行推理。

7. 后处理和结果保存

   demo 读取三个 YOLO 输出特征图，经过 YOLO decode、NMS 和坐标缩放后得到检测框。检测结果会绘制到原始图片 `cv_image` 上，并保存到 `demo_results/python/000000000139.jpg`。
