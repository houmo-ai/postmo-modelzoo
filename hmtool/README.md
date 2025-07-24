
# Hmtool

## 安装

```bash
python setup.py install
```

## 使用

### 量化
```bash
hmexec quant -c config.yml -t xh1
```

### 编译
```bash
hmexec build -c config.yml -t xh1
```

### 比较
```bash
hmexec compare -c config.yml -t xh1 --data_path your_img_or_npz_path
```

### 性能
```bash
hmexec perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 8
```

### 模型演示

需要用户实现模型实现，可参考`modelzoo`中示例

```bash
hmexec demo -c config.yml -t xh1         # 芯片
hmexec demo -c config.yml -t xh1 --onnx  # onnx
```

### 模型评估

同**模型演示**需要用户实现模型实现，可参考`modelzoo`中示例

```bash
hmexec eval -c config.yml -t xh1         # 芯片
hmexec eval -c config.yml -t xh1 --onnx  # onnx
```

## 配置文件

```yaml
# 模型信息相关描述
model:
  # 模型名称，用来命名编译后模型
  name:
  # [必填] 输出目录
  save_dir:
  # [必填] 模型路径
  model_path:
  # 模型输入信息
  inputs:
    # 模型输入名称，注意需与ONNX一致
    input_name:
      # [必填] 模型输入形状
      shape: [1, 3, 640, 640]
      # [必填] 模型输入如果是:
      #            图像，支持像素格式 RGB/BGR/GRAY
      #          非图像，填null
      data_format: RGB
      # [可选] 预处理均值，输入为图像必选
      mean: [0.0, 0.0, 0.0]
      # [可选] 预处理方差，输入为图像必选
      std: [255.0, 255.0, 255.0]
      # [可选] resize类型，0-长宽分别resize，1-等比例resize，输入为图像必选
      #      注意：等比例缩放时会使能dynamic_resize，量化编译后模型会增加shape为[10,]的输入来传入图像处理参数
      resize_type: 1
      # [可选] padding类型：0-左上角(LEFT_TOP)，1-中心点(CENTER)，等比例resize必选
      padding_mode: 1
      # [可选] padding填充数值，等比例resize必选，对应通道数
      padding_values: [114, 114, 114]
      # [可选] xh1 resizer相关参数，仅在输入为图像的可用，默认禁用
      #      xh2 暂不支持
      resizer:
        # [可选] 将模型输入转为YUV输入，目前仅支持YUV400、YUV420SP、YUV422SP、YUV444SP，使能resizer必选
        toYUV_format: YUV420SP
        # [可选] 使能resizer必选，可利用芯片resizer做crop->reisze->padding
        #       该输入为送入resizer单元的输入，也是量化编译后模型的输入size，输入格式为toYUV_format所配格式
        #       注意：当实际应用场景模型输入数据(预处理前)可能是不同分辨率时需要设置不能超过的输入size，如果超过需要在外部自行缩放处理
        #            一般为码流的最大可能分辨率较好 
        max_input_size: [1080, 1920]  # HW
        # [可选] 输入为图像时可设置，表示使用静态resizer，默认不使用
        #     慎用，这个方式仅在下列情况下可用：
        #     1. 实际应用场景中输入的图像分辨率固定，
        #     2. resize_type为0
        #     3. max_input_size和原模型shape必须一致
        enable_static_resizer: false
        # [可选] 当输入为图像且输入size较小时需要设置为true，反之false，默认为false
        insert_pad_scatter: false
  # [可选] 模型python实现模块，eval和demo功能必须设置，且必须与yml配置文件同级目录
  #        用于发现模型实现并导入
  model_impl_module:
  model_impl_cls:

# 模型量化相关选项
quant:
  # [可选] 校准数据目录，可填null、缺省、留空，表示随机数据
  #  真实数据：
  #      单输入非图像、多输入两种情况下数据以npz格式存储，且数据均为预处理后的数据
  #  格式如下：{"input_name0": np.ndarray, "input_name1": np.ndarray, ...}
  # npz保存方式如下：
  #  import numpy as np
  #  a = np.random.rand(1, 3, 28, 28)  # 预处理后数据
  #  b = np.random.rand(1, 3, 64, 64)  # 预处理后数据
  #  c = {'a': a, 'b': b}
  #  np.savez_compressed('1.npz', **c)
  # [xh2] 暂不需要校准数据可设置为null
  calib_data: your_calib_data_dir
  # [必填] 校准数据数量
  calib_num: 50
  
  #####################
  # [可选] 混合量化配置，此处仅作为示例，请根据实际情况配置
  config:
    # [可选] 混合量化auto搜索的配置
    mix_search:
      # [可选] 是否开启激活值混合量化，默认false
      activation_mix: false
      # [可选] 混合量化的搜索方法，可选all、topk、smart、smart_v2，默认smart
      # all            表示全int16
      # smart/smart_v2 表示搜索更具敏感性分析结果进行分配
      # topk           表示直接将敏感度的较高的前(#top_k_ratio)设为int16类型
      #  仅对activation有效
      method: smart
      # [可选] 是否开启权重混合量化，默认false
      weight_mix: false
      # [可选] 混合量化过程中用来校准的样本数，默认1
      mix_calib_samples: 1
      # [可选] 混合量化过程提前停止的容忍次数，默认5
      patience: 5
      # [可选] weight混合量化的阈值, 越小越多 conv\linear的权重设为int16. 默认0.0004
      w_thresh_ratio: 0.0004
    advanced_cfg:
      # [可选] 是否使用GPTQ进行量化
      gptq: false 
    # [可选] 按节点量化，优先级最高
    node_wise_cfg:
      node_name0:
        weight_quantize:
          # [可选] 量化的数据类型，目前支持int8、int16，默认int8
          dtype: int8
          # [可选] 校准方法，目前支持minmax、kl、percent-0.99、mse、ema
          #     其中percent-0.99，还可以为percent-0.999，percent-0.99999等
        output_quantize:
          dtype: int8
          calib_method: percentile9999
    # [可选] 按算子量化，优先级次于按节点
    op_wise_cfg:
      Conv:
        weight_quantize:
          dtype: int8
        output_quantize:
          dtype: int8
          calib_method: percentile9999
      Gemm:
        weight_quantize:
          dtype: int8
        output_quantize:
          dtype: int8
          calib_method: minmax
      Mul:
        output_quantize:
          dtype: int8
          calib_method: minmax
    # [可选] 按全局量化
    global_wise_cfg:
      weight_quantize:
        dtype: int8
      output_quantize:
        dtype: int8
        calib_method: minmax

# 模型编译相关选项
build:
  # [可选] 表示编译后模型使用几个IPU核进行推理，支持[1, 2, 4]，默认1
  ncore: 1
  # [可选] 编译优化等级，目前支持[0, 1, 2]，默认2
  opt_level: 2
  # [可选] 模型编译时可指定batch，在原模型输入batch的基础上乘以编译batch，默认为1
  # 比如原模型输入batch为4，编译batch为2，则最终编译后模型输入batch为2*4=8
  # 注意，慎用，模型修改batch维度容易失败
  batch: 1
  # [可选] 模型编译时可指定roi_num，表示输入图片上可以crop多ROI，默认为1
  #  当前编译后模型输入batch为1时，可设置roi_num>1，比如8、16、32
  #  注意仅对dynmiac_reiszer有效
  roi_num: 1

# 模型演示相关选项
demo:
  # [必选] 图片或者npz数据目录
  data_dir: imagenet
  # [可选] 模型演示数量，默认0，表示全部
  num: 0

# 模型评估相关选项
eval:
  # [必选] 数据集目录
  data_dir: imagenet
  # [可选] 选择数据集数量，默认0，表示全部
  num: 0
  # [必选] 数据集python实现模块，且必须与yml配置文件同级目录
  #        用于发现数据集实现并导入，用于提供图像或npz数据，以及gt信息
  dataset_module: 
  dataset_cls:
```