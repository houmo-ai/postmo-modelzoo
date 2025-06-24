# sd3

本示例展示如何把sd3模型，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的 sd3 实现来源于[https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)。

## 2 快速开始

目前sd3模型仅提供python脚本demo方式。

### 2.1 获取模型和数据

可以直接下载预编译好的所有模型（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.2 量化

该模型目前不支持用户自行量化，如有需要，请联系我们。

### 2.3 编译

该模型目前不支持用户自行编译，如有需要，请联系我们。

编译好的模型存放在`output/$HOUMO_TARGET`目录：
clip.hmm
clip_l.hmm
mmdit.hmm
t5.hmm
vae.hmm

### 2.4 演示

使用python API进行演示。目前demo中支持全阶段使用后摩设备进行推理，text_encoder3（t5）模型推荐使用cpu进行推理。
因此演示过程需要下载原始模型，可通过脚本下载：

```bash
python3 get_model.py --type raw
```

1. 执行脚本，推理完成将输出性能结果。

可配置的演示参数：
- model_path：指定推理hmm模型路径
- sd3_ckpt：指定SD3模型路径
- test_num：生成图片个数，默认为1
- nstep：mmdit推理次数，默认为10。推理次数越多图片生成效果越好
- prompt：正向提示词，如果未配置则使用预设提示词
- neg_prompt：反向提示词，如果未设置则不使用

```bash
python3 demo.py
```

**注：demo运行过程中如果报找不到文件错误，如`OSError: Error no file named model.safetensors found in directory stable-diffusion-3-medium-diffusers/text_encoder.`，可能是原始模型没有下载完全，请使用get_model.py脚本重新下载或者尝试手动下载缺少的文件。**

### 2.4 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

输出图像保存到demo_results目录下。

```bash
module /houmo/llm/models/SD3/clip.hmm loaded success. cost 121.605 ms.
input_num: 1
input[input_ids.hmcc.format] shape = [1, 77], dtype = <class 'numpy.int32'>, format = ND
output_num: 2
output[text_embeds] shape = [1, 1, 768], dtype = <class 'numpy.float16'>, format = ND
output[hidden_states] shape = [1, 77, 768], dtype = <class 'numpy.float16'>, format = ND
module /houmo/llm/models/SD3/clip_l.hmm loaded success. cost 735.056 ms.
input_num: 1
input[input_ids.hmcc.format] shape = [1, 77], dtype = <class 'numpy.int32'>, format = ND
output_num: 2
output[text_embeds] shape = [1, 1, 1280], dtype = <class 'numpy.float16'>, format = ND
output[hidden_states] shape = [1, 77, 1280], dtype = <class 'numpy.float16'>, format = ND
module /houmo/llm/models/SD3/t5.hmm loaded success. cost 3510.761 ms.
input_num: 1
input[input_ids.hmcc.format] shape = [1, 256], dtype = <class 'numpy.int32'>, format = ND
output_num: 1
output[text_embeds] shape = [1, 256, 4096], dtype = <class 'numpy.float16'>, format = ND
module /houmo/llm/models/SD3/mmdit.hmm loaded success. cost 2283.831 ms.
input_num: 4
input[hidden_states.hmcc.format] shape = [2, 16, 64, 64], dtype = <class 'numpy.float16'>, format = ND
input[encoder_hidden_states.hmcc.format] shape = [2, 333, 4096], dtype = <class 'numpy.float16'>, format = ND
input[pooled_projections.hmcc.format] shape = [2, 2048], dtype = <class 'numpy.float16'>, format = ND
input[timestep.hmcc.format] shape = [2], dtype = <class 'numpy.float16'>, format = ND
output_num: 1
output[output] shape = [2, 16, 64, 64], dtype = <class 'numpy.float16'>, format = ND
module /houmo/llm/models/SD3/vae.hmm loaded success. cost 280.296 ms.
input_num: 1
input[input.hmcc.format] shape = [1, 16, 64, 64], dtype = <class 'numpy.float16'>, format = ND
output_num: 1
output[output] shape = [1, 3, 512, 512], dtype = <class 'numpy.float16'>, format = ND
A cute young girl in a pink dress, smiling in a garden. Photo - realistic, 8k.
['Blurry, low - quality, distorted, ugly colors, bad composition, cartoonish.']
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:10<00:00,  1.08s/it]
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:346 - demo results saved in demo_results/0.jpg, cost 13141.620 ms.
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:347 - clip b infer cost 31.365 ms.
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:348 - clip l infer cost 118.261 ms.
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:349 - t5 infer cost 895.414 ms.
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:350 - dit infer 10 steps cost 10778.469 ms, average 1077.847 ms.
2025-06-24 14:26:45.205 | SUCCESS  | __main__:<module>:351 - vae infer cost 1273.996 ms.
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
