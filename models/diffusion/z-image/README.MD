# Z-Image-Turbo

本示例展示如何把 Z-Image-Turbo 文生图模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Z-Image-Turbo
- 来源: [https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
- 许可: Apache License 2.0 ([https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0))

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=z-image-turbo`
- `model_size=6b`
- `context_length=4k`
- `prefill_length=256`
- `ncore=2`
- `ndevice=1`
- `quant_type=w8a8h1_sefp`

当前示例包含三个子模型：

- `dit`: Z-Image-Turbo transformer / diffusion backbone
- `vae`: VAE decoder
- `text_encoder`: 文本编码器

### 资源要求

**后摩设备运行要求：**

- `z-image-turbo-6b`: 单卡24GB可运行

**量化硬件要求：**

- `z-image-turbo-6b`: 需要 GPU，显存最小 28GB

**编译时 host 内存要求：**

- `z-image-turbo-6b`: 需要空闲 host 内存最小 60GB

## 2 快速开始

目前 Z-Image-Turbo 模型仅提供 python 脚本编译方式。芯片模型只能在 linux 环境下量化和编译。

### 2.1 环境准备

设置目标平台：

```bash
export HOUMO_TARGET=xh2
```

安装示例运行所需的 python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

如果运行演示时环境缺少图像相关依赖，可安装 `python3-tk`：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

如果需要自己量化，可选择下载原始模型：

```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型和演示依赖，下载后的文件存放在当前目录或 `output/${HOUMO_TARGET}` 相关目录中：

```bash
python3 get_model.py --type hmm
```

`get_model.py` 支持从 `config.yaml` 读取默认模型配置，也可显式指定：

```bash
python3 get_model.py --type hmm --model_name z-image-turbo --model_size 6b
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA12.8 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu2404-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/diffusion/z-image` 目录，执行 `ptq.py` 脚本：

```bash
source env.sh
cd models/diffusion/z-image
python3 ptq.py
```

`ptq.py` 默认量化 `dit`、`vae` 和 `text_encoder` 三个子模型。量化完成的模型和演示依赖存放在 `output/${HOUMO_TARGET}/hmquant` 目录：

```bash
output/${HOUMO_TARGET}/hmquant
|-- dit
|   |-- hmquant_z-image-turbo_6b_w8a8h1_sefp_256_2048_dit.onnx
|   `-- zimage_dit-XH2a-w8a8h1_sefp_external_data
|-- text_encoder
|   |-- hmquant_z-image-turbo_6b_w8a8h1_sefp_256_2048_text_encoder.onnx
|   `-- zimage_llm-XH2a-2k-w8a8h1_sefp_prefill_external_data
|-- vae
|   `-- hmquant_z-image-turbo_6b_w8a8h1_sefp_256_2048_vae.onnx
|-- hf_config
|   |-- scheduler
|   |   `-- scheduler_config.json
|   |-- tokenizer
|   |   |-- added_tokens.json
|   |   |-- chat_template.jinja
|   |   |-- merges.txt
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer_config.json
|   |   `-- vocab.json
|   |-- t_embedder.pt
|   `-- transformer_config.json
`-- quant_embedding.pt
```

如只需要导出演示依赖，可执行：

```bash
python3 ptq.py --export_demo_deps only
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。默认从 `output/${HOUMO_TARGET}/hmquant` 读取量化结果，并输出到 `output/${HOUMO_TARGET}`：

```bash
python3 build.py
```

默认输出名称分别为：

- `z-image-turbo-6b_dit.hmm`
- `z-image-turbo-6b_vae.hmm`
- `z-image-turbo-6b_encoder.hmm`

默认编译命令使用 `config.yaml` 中的 `model_name`、`model_size`、`ncore`、`ndevice`、`context_length` 和 `prefill_length`。也可以显式指定模型规格：

```bash
python3 build.py --model_name z-image-turbo --model_size 6b
```

### 2.5 演示

Z-Image-Turbo 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 读取模型名称和规格，并默认读取：

  - `output/${HOUMO_TARGET}/z-image-turbo-6b_encoder.hmm`
  - `output/${HOUMO_TARGET}/z-image-turbo-6b_dit.hmm`
  - `output/${HOUMO_TARGET}/z-image-turbo-6b_vae.hmm`
  - `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
  - `output/${HOUMO_TARGET}/hmquant/hf_config/tokenizer`
  - `output/${HOUMO_TARGET}/hmquant/hf_config/scheduler/scheduler_config.json`
  - `output/${HOUMO_TARGET}/hmquant/hf_config/t_embedder.pt`
  - `output/${HOUMO_TARGET}/hmquant/hf_config/transformer_config.json`

2. `demo.py` 会从 VAE HMM 输入 shape 推导生成图像的 `height` 和 `width`，并从 DIT HMM 输入 shape 推导图像 token 长度和文本 token 长度，不需要用户手动传入分辨率。

3. 执行脚本，推理完成默认输出图像到 `output_turbo_xh2a.png`：

```bash
python3 demo.py
```

可显式指定单个 prompt：

```bash
python3 demo.py --prompt "一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒"
```

也可以指定多个 prompt 或每个 prompt 生成多张图：

```bash
python3 demo.py --prompt "一朵花" "一只猫" --num_images_per_prompt 2
```

当生成多张图片时，输出文件会追加序号，例如 `output_turbo_xh2a_0.png`、`output_turbo_xh2a_1.png`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。默认执行 demo 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：执行下载步骤 + 量化 + 编译 + 演示
bash test.sh -s all
```

`test.sh` 支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: z-image-turbo.
  -size, --model_size     Model size. Default: 6b.
  --context_length        Model context length. Used by build.py.
  --quant_type            Quantization type, e.g. w8a8h1_sefp.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`build.py`、`demo.py`，并在下载芯片模型时把 `quant_type` 透传给 `get_model.py`。默认值分别是 `z-image-turbo` 和 `6b`。

示例：

```bash
# 仅运行 quant，会先执行下载步骤再量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（执行下载步骤、量化、编译、演示）
bash test.sh -s all

# 显式指定模型名和模型规格（当前默认就是 z-image-turbo 6b）
bash test.sh -s demo -name z-image-turbo -size 6b

# 组合执行多个步骤
bash test.sh -s quant,build

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

## 4 参考结果

以下为单 prompt 演示输出参考：

```bash
python3 demo.py --prompt "一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒"
```

### 4.1 演示结果

![Z-Image-Turbo demo result](ref_result_image.png)

生图过程中的参考日志如下：

```bash
Model: z-image-turbo-6b
Loading demo dependencies from output/xh2/hmquant/hf_config...
Inference steps: 9
Guidance scale: 0.0
Number of images per prompt: 1
Tokenizer directory: output/xh2/hmquant/hf_config/tokenizer
Scheduler config path: output/xh2/hmquant/hf_config/scheduler/scheduler_config.json
T embedder path: output/xh2/hmquant/hf_config/t_embedder.pt
Transformer config path: output/xh2/hmquant/hf_config/transformer_config.json
Loading encoder runtime from ./output/xh2/z-image-turbo-6b_encoder.hmm...
Loading dit runtime from ./output/xh2/z-image-turbo-6b_dit.hmm...
Loading vae runtime from ./output/xh2/z-image-turbo-6b_vae.hmm...
Inferred image size from VAE input shape [1, 16, 128, 128]: 1024x1024
Inferred DIT input lengths from HMM: image_token_len=4096, text_token_len=256
Demo initialized with model z-image-turbo-6b.
Generating image with prompt: 一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒
100%|███████████████████████████████| 9/9 [00:32<00:00,  3.63s/it]
Image 0 generated with shape: (1024, 1024), saving image to output_turbo_xh2a.png...
```

### 4.2 性能结果

`demo.py` 会在图片保存完成后输出性能统计。性能统计使用一张 `Performance summary` 总表展示初始化和生成阶段的层级耗时。表格字段含义如下：

- `Stage`: 当前统计项。缩进表示层级关系，例如 `load_runtime_models` 下包含各 HMM runtime 加载耗时，`denoise_total` 下包含 DIT 和 scheduler 耗时。
- `Count`: 当前统计项执行的次数。
- `Time(s)`: 当前统计项总耗时，单位为秒。
- `Avg(s)`: 当前统计项平均耗时（总耗时/次数），单位为秒。
- `Percent`: 当前统计项相对其上层总耗时的占比。顶层项为 100%。
- `Notes`: 当前统计项说明。

总表中的初始化阶段指标：

- `init_total`: demo 初始化总耗时。
- `load_demo_dependencies`: 加载 tokenizer、scheduler config、transformer config、t_embedder 等演示依赖的耗时。
- `load_runtime_models`: 加载全部 HMM runtime 的总耗时。
- `load_encoder_runtime`: 加载 text encoder HMM runtime 的耗时，占 `load_runtime_models` 的比例。
- `load_dit_runtime`: 加载 DiT HMM runtime 的耗时，占 `load_runtime_models` 的比例。
- `load_vae_runtime`: 加载 VAE HMM runtime 的耗时，占 `load_runtime_models` 的比例。

总表中的生成阶段指标：

- `generation_end_to_end`: `generate_image()` 的端到端耗时，不包含图片保存耗时。
- `text_encode`: prompt 编码耗时。
- `prepare_latents`: 初始化 latent 的耗时。
- `denoise_total`: 全部 denoising steps 的总耗时。
- `dit_total`: DiT 推理总耗时，占 `denoise_total` 的比例。
  - `dit_preprocess`: DiT 预处理耗时，包括 t_embedder、patchify 和 masks 操作。
  - `dit_runtime`: DiT HMM runtime 推理耗时。
  - `dit_postprocess`: DiT 后处理耗时，包括 crop 和 unpatchify 操作。
- `scheduler_total`: scheduler step 总耗时，占 `denoise_total` 的比例。
- `vae_decode`: VAE 解码耗时。
- `postprocess`: 图像后处理耗时。

表格后的补充指标：

- `Generation config`: 当前生成配置，包括分辨率、prompt 数量、输出图片数量和推理步数。
- `Generation average`: 平均耗时统计。
  - `denoise_per_step`: 平均每个 denoising step 耗时。
  - `dit_per_step`: 平均每个 step 中 DiT 推理耗时。
- `Generation throughput`: 吞吐统计。
  - `latency_per_image`: 平均每张图片端到端生成耗时。
  - `throughput`: 图片生成吞吐，单位为 images/s。

参考性能日志如下：

```bash
================================================================================
Performance summary:
+--------------------------+-------+---------+---------+---------+----------------------------+
| Stage                    | Count | Time(s) |  Avg(s) | Percent | Notes                      |
+--------------------------+-------+---------+---------+---------+----------------------------+
| init_total               |     1 | 108.282 | 108.282 |  100.0% | model setup end-to-end     |
|   load_demo_dependencies |     1 |   0.405 |   0.405 |    0.4% | tokenizer/scheduler/config |
|   load_runtime_models    |     1 |  98.441 |  98.441 |   90.9% | all HMM runtimes           |
|     load_encoder_runtime |     1 |  32.968 |  32.968 |   33.5% | share of runtime loading   |
|     load_dit_runtime     |     1 |  60.269 |  60.269 |   61.2% | share of runtime loading   |
|     load_vae_runtime     |     1 |   4.627 |   4.627 |    4.7% | share of runtime loading   |
| generation_end_to_end    |     1 |  35.174 |  35.174 |  100.0% | generate_image total       |
|   text_encode            |     1 |   1.112 |   1.112 |    3.2% | prompt encoder             |
|   prepare_latents        |     1 |   0.002 |   0.002 |    0.0% | latent init                |
|   denoise_total          |     9 |  32.658 |   3.629 |   92.8% | all denoising steps        |
|     dit_total            |     9 |  32.649 |   3.628 |  100.0% | share of denoising         |
|       dit_preprocess     |     9 |   0.018 |   0.002 |    0.1% | t_embedder/patchify/masks  |
|       dit_runtime        |     9 |  32.625 |   3.625 |   99.9% | HMM runtime                |
|       dit_postprocess    |     9 |   0.004 |   0.000 |    0.0% | crop/unpatchify            |
|     scheduler_total      |     9 |   0.002 |   0.000 |    0.0% | share of denoising         |
|   vae_decode             |     1 |   1.370 |   1.370 |    3.9% | latent to image tensor     |
|   postprocess            |     1 |   0.032 |   0.032 |    0.1% | image processor            |
+--------------------------+-------+---------+---------+---------+----------------------------+
Generation config: resolution=1024x1024, prompts=1, images=1, steps=9
Generation average: denoise_per_step=3.629 s, dit_per_step=3.628 s
Generation throughput: latency_per_image=35.174 s, throughput=0.028 images/s
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## 附录：`demo.py` 推理流程速览

`demo.py` 的核心类是 `HmZImage`。它把 Z-Image-Turbo 文生图推理拆成三个 HMM 子模型和少量 host 侧处理逻辑：

- `text_encoder`: 将 prompt 编码成文本 hidden states。
- `dit`: 在扩散去噪循环中预测 latent 空间的噪声残差。
- `vae`: 将最终 latent 解码成图像。

整体流程如下：

```text
[初始化]
参数/config.yaml
  -> 加载 tokenizer、scheduler、transformer config、t_embedder
  -> 加载 text_encoder / DiT / VAE 三个 HMM
  -> 从 HMM shape 推导分辨率和 token 长度

[文本编码]
Prompt
  -> host: chat template + tokenizer
  -> host: token embedding lookup
  -> text_encoder HMM
  -> host: 根据 attention_mask 截掉 padding hidden states
  -> prompt hidden states

[扩散去噪]
随机 latent 噪声 + prompt hidden states
  -> host: scheduler 生成 timesteps
  -> 循环 num_inference_steps 次:
       host: timestep embedding
       host: latent patchify + position ids + mask/padding
       -> DiT HMM 预测 noise residual
       -> host: crop + unpatchify，恢复 latent 布局
       -> host: scheduler.step 更新 latent
  -> 最终 latent

[图像解码]
最终 latent
  -> host: latent scale 还原
  -> VAE HMM 解码
  -> host: VaeImageProcessor 后处理
  -> 保存图片
```

上图中的 `host` 表示在 Host 侧 CPU 中完成的轻量处理；`HMM` 表示运行在后摩设备上的已编译子模型。

### 1. 初始化阶段

运行 `python3 demo.py` 后，脚本先解析命令行参数和 `config.yaml`。如果用户没有显式传入模型路径，脚本默认按以下规则查找编译后的 HMM 文件：

```text
output/${HOUMO_TARGET}/${model_name}-${model_size}_encoder.hmm
output/${HOUMO_TARGET}/${model_name}-${model_size}_dit.hmm
output/${HOUMO_TARGET}/${model_name}-${model_size}_vae.hmm
```

随后 `HmZImage` 会加载两类内容：

- host 侧依赖：`tokenizer`、`scheduler_config.json`、`transformer_config.json`、`t_embedder.pt`。
- 设备侧模型：`text_encoder`、`dit`、`vae` 三个 HMM runtime。

其中 host 侧依赖来自 `output/${HOUMO_TARGET}/hmquant/hf_config` 和 `quant_embedding.pt`。这些文件不是完整模型主体，但决定了 prompt 如何 tokenize、scheduler 如何生成时间步、latent 和 image shape 如何换算，以及 DiT 的 timestep embedding 如何生成。

### 2. 自动推导分辨率和序列长度

`demo.py` 不要求用户手动传入 `height`、`width` 或 token 长度，而是直接从编译后的 HMM 输入 shape 推导：

- 从 `vae` 输入 shape 读取 latent 的 `height` 和 `width`，再乘以 `vae_scale_factor` 得到最终图像分辨率。
- 从 `dit` 的 `latent_mask` 输入 shape 读取图像 token 长度。
- 从 `dit` 的 `cap_feats` 输入 shape 读取文本 token 长度。

这样做可以保证 demo 的预处理逻辑和实际编译出来的 HMM 模型保持一致。如果 HMM 是固定 `1024x1024` 分辨率编译的，demo 就会按该分辨率准备 latent 和 token，不需要用户额外指定。

### 3. Prompt 编码阶段

用户输入的 `--prompt` 会先经过 tokenizer 的 chat template 处理，再 tokenize 成固定长度 token ids。之后流程如下：

```text
prompt 文本
  -> apply_chat_template
  -> tokenizer
  -> token embedding lookup
  -> text_encoder HMM
  -> prompt hidden states
```

这里有两个关键点：

- `quant_embedding.pt` 在 host 侧用 PyTorch 加载，先把 token ids 转成 embeddings。
- 编译后的 `text_encoder` HMM 接收的是 embeddings 和序列长度信息，而不是原始 token ids。

如果 prompt 超过单次 prefill 长度，`demo.py` 会把 token 分块送入 text encoder，并在同一个 prompt 内复用 HMM KV cache，最终得到该 prompt 的 hidden states。`attention_mask` 会用于截掉 tokenizer padding 产生的无效 hidden states。

### 4. Latent 初始化

文本编码完成后，脚本会根据推导出的图像分辨率和 VAE scale factor 创建初始 latent 噪声：

```text
batch_size = prompt 数量 * num_images_per_prompt
latents = torch.randn(batch_size, latent_channels, latent_height, latent_width)
```

如果用户设置 `--num_images_per_prompt` 大于 1，demo 会扩展 latent batch，并复制对应的 prompt hidden states，让每张图都有一份文本条件。

### 5. Scheduler 时间步

Z-Image-Turbo 使用 `FlowMatchEulerDiscreteScheduler`。去噪前，demo 会根据 latent token 数量调用 `calculate_shift`，得到与分辨率相关的 `mu`，再调用 `retrieve_timesteps` 生成实际推理时间步。

这一阶段的作用是确定 DiT 要执行多少个 denoising step，以及每一步对应的 timestep。默认 `--num_inference_steps` 为 9。

### 6. DiT 去噪循环

每个 denoising step 中，demo 会执行以下逻辑：

```text
当前 latents
  -> 转成 DiT dtype
  -> 增加 frame 维度
  -> patchify 成 image tokens
  -> 结合 prompt hidden states 和 mask
  -> 计算 timestep AdaLN embedding
  -> 调用 DiT HMM 预测 noise residual
  -> unpatchify 回 latent 布局
  -> scheduler.step 更新 latents
```

其中 `ZImageTransformerHelper` 负责 DiT HMM 前后的 host 侧形状处理：

- `patchify_and_embed`: 将 latent 切成 patch token，并为 image token 和 text token 准备 position ids、padding mask。
- `_pad_with_ids`: 将序列 padding 到编译时要求的静态长度。
- `unpatchify`: 将 DiT 输出的 patch token 还原成 latent 的 `C/F/H/W` 布局。
- `t_embedder`: 根据当前 timestep 生成 DiT 需要的 AdaLN 条件输入。

DiT HMM 的输出会被转换成 `noise_pred`，再交给 scheduler 更新 latent。这个过程重复 `num_inference_steps` 次。

### 7. VAE 解码和图像保存

去噪完成后，demo 会先把 latent 从模型内部缩放还原到 VAE 输入分布：

```python
latents = (latents.to(torch.float16) / 0.3611) + 0.1159
```

然后逐张调用 `vae` HMM 解码。当前 demo 逐张 latent 解码，是因为编译后的 VAE HMM batch shape 固定。VAE 输出再经过 `VaeImageProcessor.postprocess` 转成 PIL 图像，最后保存到 `--output` 指定路径。

当生成多张图时，输出文件名会自动追加序号，例如：

```text
output_turbo_xh2a_0.png
output_turbo_xh2a_1.png
```