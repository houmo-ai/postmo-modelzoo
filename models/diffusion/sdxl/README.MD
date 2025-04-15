# sdxl

本示例展示如何把sdxl模型编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 sdxl 实现来源于[https://modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0](https://modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0)。

## 2 快速开始

目前sdxl模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

如果仅编译芯片模型，可选择下载量化模型，下载后的模型存放在`output/$HOUMO_TARGET/hmquant`目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type quant
```

也可以直接下载以上所有模型（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py
```

### 2.2 量化

该模型目前不支持用户自行量化，如有需要，请联系我们。

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在`output/$HOUMO_TARGET`目录：
sd_unet.hmm

### 2.3 演示

使用python API进行演示。目前demo中仅unet阶段使用后摩设备进行推理，text_encoder和vae推荐使用cpu进行推理。
因此演示过程需要下载原始模型，可通过脚本下载：

```bash
python3 get_model.py --type raw
```

1. 执行脚本，推理完成将输出性能结果。

可配置的演示参数：
- model_path：指定推理hmm模型路径
- sdxl_ckpt：指定SDXL模型路径
- lora_weights：LoRA（Low-Rank Adaptation，低秩自适应）模型的权重
- test_num：生成图片个数，默认为3
- nstep：unet推理次数，默认为10。推理次数越多图片生成效果越好
- eta：随机噪声参数，取值为[0,1]，值越大生成图像多样性越高
- prompt：正向提示词，如果未配置则使用预设提示词
- neg_prompt：反向提示词，如果未设置则不使用

```bash
python3 demo.py
```

**注：demo运行过程中如果报找不到文件错误，如`OSError: Error no file named model.safetensors found in directory stable-diffusion-xl-base-1.0/text_encoder.`，可能是原始模型没有下载完全，请使用get_model.py脚本重新下载或者尝试手动下载缺少的文件。**

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

输出图像保存到demo_results目录下。

```bash
input_num: 3
input[input] shape = [1, 4, 128, 128], dtype = <class 'numpy.int16'>, format = ND
input[input.9] shape = [1, 1280], dtype = <class 'numpy.int8'>, format = ND
input[encoder_hidden_states] shape = [1, 77, 2048], dtype = <class 'numpy.int16'>, format = ND
output_num: 1
output[17103] shape = [1, 4, 128, 128], dtype = <class 'numpy.int16'>, format = ND
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:21<00:00,  2.17s/it]
demo results saved in demo_results/0.jpg, cost 39743.521 ms.
unet infer 10 steps cost 21679.461 ms, average 2167.946 ms.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:21<00:00,  2.17s/it]
demo results saved in demo_results/1.jpg, cost 39724.767 ms.
unet infer 10 steps cost 21692.498 ms, average 2169.250 ms.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:21<00:00,  2.17s/it]
demo results saved in demo_results/2.jpg, cost 39659.114 ms.
unet infer 10 steps cost 21685.145 ms, average 2168.514 ms.
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
