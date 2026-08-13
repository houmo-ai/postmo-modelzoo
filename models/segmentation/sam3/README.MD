# SAM3

本示例展示如何将 SAM3（Segment Anything Model 3）模型量化、编译并部署到后摩芯片设备上，通过文本提示或框提示生成图像分割结果。

[TOC]

## 1 模型说明

本例使用的模型实现和预训练权重来源于 Meta 的 SAM3 项目：

- 模型名称：SAM3
- 默认模型规格：`sam3-0.8b`
- ModelScope：[`facebook/sam3`](https://www.modelscope.cn/models/facebook/sam3)
- 官方代码仓库：[`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)
- 模型许可：Meta SAM License，详见随模型下载的 `sam3/LICENSE` 以及上游仓库许可说明

SAM3 模型和上游代码在运行时获取，工程发布默认**不包含**预训练权重。SAM3 的模型许可独立于本仓库示例代码的 Apache License 2.0；使用、下载或再分发模型及其衍生物前，请阅读并遵守上游 Meta SAM License 的全部条款。本仓库不对第三方模型的许可适用性作额外授权或保证。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yml` 作为默认配置来源，默认配置如下：

- `model_name=sam3`
- `model_size=0.8b`
- `quant_type=w8a8`
- `batch=1`
- `max_size_w=1008`
- `max_size_h=1008`
- `num_boxes=16`：单次推理支持输入的提示框最大个数，不是模型可检测的最大目标个数；如需支持更多提示框，可在 `config.yml` 中增大该参数，并重新导出、量化和编译模型
- `ncore=2`
- `ndevice=1`

### 模型架构

SAM3 支持基于开放词汇文本提示以及视觉提示进行目标检测和实例分割。本示例使用 HMM 后端完成图像输入、提示处理和分割结果输出，默认文本提示为 `shoe`。

### 资源要求

**后摩设备运行要求：**

- 目标平台：xh2。
- 设备型号、运行时显存和内存要求：暂无数据，待后续实测补充。

**量化硬件要求：**

- 需要 GPU 和 CUDA 12.8 兼容环境。
- `ptq.py` 会优先选择空闲显存不少于 20 GiB 的 GPU；不满足时可回退到 CPU 导出，但完整量化流程仍建议使用 GPU。
- 量化主机的实际显存、内存和耗时：暂无数据，待后续实测补充。

**编译时 host 内存要求：**

- 编译要求 x86_64 Linux host。
- 内存要求：暂无数据，待后续实测补充。

## 2 快速开始

SAM3 使用 Python 脚本完成模型获取、量化、编译和演示。量化与编译建议在 Linux 环境以及项目提供的工具链 Docker 中执行。

### 2.1 环境准备

在当前目录创建 Python 虚拟环境并安装依赖：

```bash
cd models/segmentation/sam3
python3 -m venv sam3_venv
source sam3_venv/bin/activate
pip install -r requirements.txt
```

HMM/HMONNX 推理不依赖额外的 SAM3 源码仓库。示例通过 Transformers 从本地模型目录 `sam3/` 加载标准 CLIP tokenizer 资产，并使用本仓库独立实现的图像前处理、提示框组装和结果后处理。可通过 `demo.py --model_dir`指定其他包含完整 tokenizer 文件的本地模型目录。

前后处理接口依据本地 SAM3 processor 配置及导出模型的输入输出约定实现，没有复制上游 `tokenizer_ve.py` 或 `visualization_utils.py` 源码。模型配置、tokenizer 资产和权重仍属于运行时获取的 SAM3 模型资产，适用上游 Meta SAM License；Transformers 作为用户安装的运行时依赖，适用其原始许可证。

### 2.2 获取模型和数据

获取 ModelScope 原始模型：

```bash
cd models/segmentation/sam3
python3 get_model.py --type raw
```

原始模型默认下载到当前目录下的 `sam3/`。如果模型已存在，可使用 `test.sh --skip_download` 跳过下载步骤。

获取预编译 HMM 模型：

```bash
cd models/segmentation/sam3
python3 get_model.py --type hmm
```

预编译模型默认下载到 `output/xh2/`。示例默认使用的 HMM 文件为：

- `output/xh2/sam3_0.8b_1008x1008.hmm`

演示默认使用的测试图片为：

- `${HOUMO_EXAMPLES_PATH}/data/pic/sam3_test_image.jpg`

如果未设置 `HOUMO_EXAMPLES_PATH`，脚本会使用仓库目录推导该图片路径；也可以通过 `demo.py --image` 显式指定输入图片。

### 2.3 量化

量化需要 GPU 和 CUDA 12.8。请在宿主机启动工具链 Docker：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 Docker 后执行：

```bash
source env.sh
cd models/segmentation/sam3
python3 ptq.py
```

量化输出文件为：

- `work_dirs/sam3_0.8b_1008x1008_sim.onnx`
- `output/xh2/hmquant/hmquant_sam3_0.8b_1008x1008_w8a8.onnx`
- `output/xh2/hmquant/hmquant_sam3_0.8b_1008x1008_w8a8_external_data`

本流程默认不使用独立校准数据集。

### 2.4 编译

在 x86_64 Linux host 上，将量化后的 HMONNX 模型编译为 HMM：

```bash
cd models/segmentation/sam3
python3 build.py
```

默认输入文件为：

- `output/xh2/hmquant/hmquant_sam3_0.8b_1008x1008_w8a8.onnx`

默认输出文件为：

- `output/xh2/sam3_0.8b_1008x1008.hmm`

### 2.5 演示

确保本地 `sam3/` 模型目录、Python 依赖、输入图片和 HMM 文件均已准备完成，然后执行：

```bash
cd models/segmentation/sam3
python3 demo.py
```

默认参数如下：

- HMM 模型：`output/xh2/sam3_0.8b_1008x1008.hmm`
- 输入图片：`${HOUMO_EXAMPLES_PATH}/data/pic/sam3_test_image.jpg`
- 文本提示：`shoe`
- 输出图片：`demo_hmm_result.png`
- 性能测试：默认开启，`warmup=1`、`repeat=1`

使用自定义图片和文本提示：

```bash
cd models/segmentation/sam3
python3 demo.py \
  --image /path/to/image.jpg \
  --prompt "person" \
  --output demo_person_result.png
```

使用 `--mode 1` 保存文本提示、单框提示和多框提示的全部五类结果：

```bash
cd models/segmentation/sam3
python3 demo.py --mode 1 --output_dir demo_results
```

也可以显式关闭性能测试：

```bash
cd models/segmentation/sam3
python3 demo.py --perf false
```

## 3 一键评估

当前 `test.sh` 支持下载、量化、编译和演示步骤。默认执行演示步骤；如需执行完整流程，请显式指定 `all`：

```bash
cd models/segmentation/sam3
bash test.sh
```

常用命令如下：

```bash
# 仅运行量化：会先下载原始模型
bash test.sh -s quant

# 仅运行编译：要求量化输出已存在
bash test.sh -s build

# 仅运行演示：会先下载预编译 HMM 模型
bash test.sh -s demo

# 完整流程：下载原始模型、量化、编译和演示
bash test.sh -s all

# 使用已有本地模型，跳过下载
bash test.sh -s demo --skip_download

# 指定模型名称和规格
bash test.sh -s demo -name sam3 -size 0.8b
```

`test.sh` 当前实际使用的参数为：

```bash
-s, --step              Step to run. Choices: demo, build, quant, all.
-name, --model_name     Model name. Default: sam3.
-size, --model_size     Model size. Default: 0.8b.
--skip_download         Skip model download steps.
-h, --help              Show this help message.
```

## 4 参考结果

### 4.1 演示结果

默认演示会生成：

- `demo_hmm_result.png`

使用 `--mode 1 --output_dir demo_results` 时，会生成文本提示、单框提示和多框提示对应的五个结果文件。当前仓库尚未提供经过统一环境验证的 SAM3 参考图片结果，参考结果：暂无结果，待后续实测补充。

### 4.2 性能结果

========== SAM3 Performance ==========
warmup=1, repeat=1
set_input : 9.755 ms
infer     : 2603.973 ms
get_output: 21.672 ms
total     : 2635.400 ms
throughput: 0.379 images/s
======================================


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护，后摩算法团队不对其合法性、准确性、完整性、有效性、安全性做任何形式的保证。您因使用或依赖以下链接中的软件、数据或模型而导致的任何直接、间接、偶然、特殊、后续、惩罚性损害（包括但不限于数据丢失、利润损失、业务中断等）不承担任何责任。

- SAM3 官方代码仓库：<https://github.com/facebookresearch/sam3>
- SAM3 ModelScope 模型：<https://www.modelscope.cn/models/facebook/sam3>
- SAM3 模型许可：随模型下载的 `sam3/LICENSE`，以及上游仓库中的许可说明

您应当自行判断是否使用上述第三方提供的软件、数据或模型，并自行承担相关风险。您还应在使用、下载或再分发 SAM3 模型及其衍生物前，阅读并遵守 Meta SAM License 的全部条款。如果您不同意上述声明或相关第三方许可，请不要使用相关软件、数据或模型。
