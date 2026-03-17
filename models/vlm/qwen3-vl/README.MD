# Qwen3-VL(通义千问3-VL)

本示例展示如何把Qwen3VL模型量化和编译，部署到后摩鸿途芯片的设备上。

## 预训练模型

本项目使用从 ModelScope 下载的预训练模型：

- 模型名称: Qwen3-VL-8B-Instruct
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
- 许可: Apache License 2.0 （https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md）

预训练模型在运行时下载，工程发布**不包含**该模型。
同时，原有的预处理代码 image_processing_qwen2_vl.py 和 processing_qwen3_vl.py 已进行专门的改编和修改，并包含在项目提交文件中。特此声明。

## 数据集使用声明

本项目使用了由LAION提供的"220k-GPT4Vision-captions-from-LIVIS"数据集。该数据集包含22万条通过GPT-4 Vision生成的图像描述文本，源自LIVIS数据集。
- 数据集获取方式：

    数据集可通过Hugging Face平台公开获取：

    https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS

- 许可信息： 本数据集采用Apache License 2.0开源协议授权。根据该许可条款，使用者可以自由地：

    使用、复制和分发数据集

    修改和创建衍生作品

    将数据集用于商业用途

    同时需遵守协议中规定的署名要求和责任限制条款。
- 合规性说明：

  本项目的使用方式完全符合Apache License 2.0的规定要求，特此声明。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)。

## 2 快速开始

目前Qwen3VL模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```

### 2.2 量化

#### 2.2.1 自定义数据集量化

下载自定义数据集到当前目录下
```bash
python3 get_model.py --type raw
mkdir -p data/calib_data/LMUData
ln -s ../../../data/datasets/LMUData/* data/calib_data/LMUData
```

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python ptq.py --data_files ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 64 --calib_dataset vllm_custom_data
```

#### 2.2.1 默认公开数据集数据集量化

量化下载的原始模型。在当前目录下，执行脚本：

```bash
# 量化依赖transformers==4.57.3
pip3 install transformers==4.57.3
python ptq.py
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录，包括3个文件：
qwen3-vl_prefill.hmm
qwen3-vl_decoder.hmm
qwen3-vl_visual.hmm

### 2.4 演示

Qwen3-vl模型使用python API进行演示

1. 将编译好的模型放在当前目录，同时将对应量化模型的`quant_embedding.pt`文件放在量化模型目录下

2. 执行脚本，推理完成将输出性能结果。

注意：由于ubuntu18.04下qwen_vl_utils无法正常安装，导致无法正常运行。

```bash
# Qwen3-VL-4B 推荐惩罚系数是4
# Qwen3-VL-8B 推荐惩罚系数是1.5
python3 demo.py --repetition_penalty 1.5
```

### 2.5 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
question:
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
请描述图片内容。
response:
这是一张充满温馨与宁静氛围的海滩日落照片。

画面中，一位年轻女性和一只金毛犬坐在沙滩上。女性侧身对着镜头，面带微笑，目光温柔地注视着身旁的狗狗。她穿着一件蓝白格子的衬衫，双腿交叉，姿态放松。狗狗则安静地坐在她旁边，戴着一个蓝色的胸背带，正抬起前爪，似乎在与主人互动或准备握手。

背景是广阔的海洋，海浪正轻轻拍打着沙滩，远处的海平线与天空在柔和的夕阳下融为一体，呈现出温暖的橙黄色调。阳光从画面右侧斜射过来，为整个场景镀上了一层金色的光晕，营造出温暖、治愈的氛围。

整个画面构图和谐，人物与宠物之间的互动充满爱意，展现了人与宠物在自然美景中共享宁静时光的美好瞬间。
```

### 3.2 性能结果
#### M50 2core
```bash
Total Images: 1, Total Input: 211 tokens, Output 186 tokens, Vision Cost 157.396 ms, Prefill Cost 228.211 ms, Decode Cost 13778.114 ms
Vision Cost 157.396 ms/image
Prefill Speed: 924.58 tokens/s; Decode Speed: 13.43 tokens/s
TTFT (Time to First Token): 464.260 ms
TPOT (Time Per Output Token): 74.476 ms/token
E2E Latency (End-to-End Latency): 15.824 seconds
E2E TPS (End-to-End Tokens Per Second): 11.75 tokens/s
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
