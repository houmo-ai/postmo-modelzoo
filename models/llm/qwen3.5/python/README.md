# Qwen3.5 / Qwen3.6 Python Engine 示例

本目录提供三个基于 `utils/python/houmo_engine` 的 Python 推理示例，分别用于标准生成、Qwen3.6 MTP 推测生成以及文本/视觉前缀缓存。

[TOC]

## 1 示例概览

| 示例 | 用途 | 输入 | 适用模型 | 主要特点 |
| --- | --- | --- | --- | --- |
| `demo.py` | 标准生成 | 文本、图片 | Qwen3.5、Qwen3.6 | 支持单次推理、交互模式和多轮历史 |
| `demo_mtp.py` | MTP 推测生成 | 文本 | Qwen3.6 | 使用 draft/verify 流程加速生成，并输出 MTP 统计信息 |
| `demo_prefix_caching.py` | 文本/视觉前缀缓存 | 文本，或图片和多个问题 | Qwen3.5、Qwen3.6 | 文本和图片请求均可复用公共 prompt 前缀；相同图片还可复用视觉特征 |

三个示例均通过源码方式使用 Houmo Python Engine，不需要安装额外的 Python package。脚本会把仓库中的 `utils/python` 加入当前进程的 `sys.path`。

当前示例仅支持 `batch=1`，并仅适用于 xh2。

## 2 运行准备

### 2.1 环境和依赖

先按照上级目录的 [README.MD](../README.MD) 完成依赖安装、模型下载、量化和编译，并确保已加载 Dadao 环境：

```bash
source /path/to/imodelzoo/env.sh
cd models/llm/qwen3.5/python
```

建议设置仓库根目录：

```bash
export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
```

`HOUMO_TARGET` 未设置时默认为 `xh2`。

### 2.2 默认配置和模型路径

三个脚本均从上级目录的 `config.yaml` 读取模型配置。当前默认模型为：

- `model_name=qwen3.6`
- `model_size=35b-a3b`
- `ndevice=1`
- `batch=1`

标准生成和前缀缓存示例默认读取：

- `../output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `../output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`
- `../output/${HOUMO_TARGET}/${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`
- `../output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

如果带尺寸的 visual 模型不存在但 `visual.hmm` 存在，脚本会使用后者。

MTP 示例额外读取：

- `../output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill_mtp.hmm`
- `../output/${HOUMO_TARGET}/${model_name}-${model_size}_decode_mtp.hmm`

当 `ndevice>1` 时，文本模型路径会自动从 `.hmm` 切换为 `.hmms`。标准示例和前缀缓存示例的 visual 模型仍使用 `.hmm`。

可通过以下命令查看每个脚本的完整参数：

```bash
python3 demo.py --help
python3 demo_mtp.py --help
python3 demo_prefix_caching.py --help
```

## 3 标准生成示例

`demo.py` 使用公共 `Qwen35Engine`，支持文本和图片生成。默认输入示例图片，并执行一次视觉问答。

### 3.1 基本运行

```bash
python3 demo.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b
```

指定问题和图片：

```bash
python3 demo.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --question "请描述图片中的人物和动物" \
  --image_path "${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg"
```

通过 `--image_path` 可以传入一张或多张图片：

```bash
python3 demo.py \
  --image_path image_1.jpg image_2.jpg \
  --question "比较这两张图片"
```

### 3.2 交互和历史

启用交互模式：

```bash
python3 demo.py --it true
```

启用交互模式并在后续轮次保留历史：

```bash
python3 demo.py --it true --history true
```

交互模式下，每轮可重新输入问题和图片路径。输入 `stop`、`exit` 或 `quit` 结束运行。

### 3.3 常用参数

- `--question`：用户问题，默认值为 `描述这些图片`。
- `--system_prompt`：自定义 system prompt。
- `--image_path`：一张或多张图片路径。
- `--max-new-tokens`：最大生成 token 数。
- `--it`：启用交互模式。
- `--history`：交互模式下保留后续轮次的会话历史。
- `--perf`：是否输出性能报告，默认开启。
- `--temperature`、`--topk`、`--topp`：确定性 sampling 的 logits 处理参数，最终仍使用 argmax 选取 token。
- `--presence-penalty`、`--repetition-penalty`：生成 token 的惩罚参数。

## 4 MTP 推测生成示例

`demo_mtp.py` 使用 `Qwen36MtpEngine`，通过 MTP draft/verify 流程进行文本生成。该示例只接受 `model_name=qwen3.6`，不处理图片输入。

运行前需要准备标准 prefill/decode 模型和 MTP prefill/decode 模型。

### 4.1 基本运行

```bash
python3 demo_mtp.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b
```

指定问题和最大生成长度：

```bash
python3 demo_mtp.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --question "请介绍一下存算一体技术的优势" \
  --max-new-tokens 512
```

### 4.2 图文件对应关系

| 参数 | 默认模型文件 | 作用 |
| --- | --- | --- |
| `--prefill_path` | `${model_name}-${model_size}_prefill.hmm` | 主模型 prefill |
| `--prefill_mtp_path` | `${model_name}-${model_size}_prefill_mtp.hmm` | MTP prefill |
| `--decode_mtp_path` | `${model_name}-${model_size}_decode_mtp.hmm` | draft token 生成 |
| `--decode_verify_path` | `${model_name}-${model_size}_decode.hmm` | 主模型 verify |

### 4.3 输出指标

除常规 TTFT、E2E、prefill 和 decode 性能外，MTP 示例还会统计：

- speculative rounds
- draft tokens
- accepted draft tokens
- acceptance rate
- average accepted tokens per round
- drafts per round

可使用 `--debug true` 输出额外的运行诊断信息。

## 5 文本/视觉前缀缓存示例

`demo_prefix_caching.py` 支持纯文本和图片输入，适用于“连续提交多个具有公共开头的问题”场景。前缀缓存扩展只定义在该 Demo 内，不修改公共 `houmo_engine` 的已有 Engine、Process 或 Module。

纯文本和图片请求均支持 prompt prefix cache。图片请求在图片路径完全一致时还会复用 vision embedding。

### 5.1 缓存行为

每次请求仍会清理普通生成 session，但 Demo 会在请求之间保存用于前缀复用的独立状态：

- 比较上一轮和当前轮的 token IDs，计算最长公共前缀。
- 在每个 prefill chunk 结束后快照 `conv_cache` 和 `recurrent_state`。
- 恢复不超过公共前缀长度的最近快照。
- 从恢复位置 replay 尚未快照的公共前缀 token。
- 只对分叉后的 suffix 执行剩余 prefill。
- 图片路径完全一致时复用 vision embedding。
- 图片路径变化时不复用上一轮的前缀快照和 vision embedding。

全注意力 KV cache 保留在原有位置缓冲区中，replay 和 suffix prefill 会覆盖发生变化的后缀位置。

### 5.2 基本运行

不传 `--question` 时，脚本会对默认图片连续执行三个具有公共前缀的问题，以展示缓存复用：

```bash
python3 demo_prefix_caching.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b
```

不输入图片时，可将 `--image_path` 显式设置为 `None` 或 `null`。脚本会连续执行三个默认纯文本问题，并复用它们的公共 prompt 前缀：

```bash
python3 demo_prefix_caching.py \
  --model_name qwen3.5 \
  --model_size 2b \
  --image_path None
```

默认纯文本问题包含一段明显超过默认 `prefill_length=256` 的公共任务说明，只在末尾的具体问题处发生分叉；同时要求模型将答案控制在六个汉字以内。使用 Qwen3.5-2B tokenizer 实测时，三个默认问题约为 634、635、634 tokens，前两个问题的最长公共前缀约为 621 tokens，可恢复到 512-token 快照并 replay 剩余公共前缀。不同 tokenizer 版本的实际 token 数可能略有差异。

也可以配合重复的 `--question` 自定义纯文本问题。为了产生可恢复快照，自定义问题的公共 token 前缀需要至少覆盖一个完整的 prefill chunk；只有几个相同开头词的短问题通常只能得到少量 `matched_tokens`，无法得到非零的 `restored_tokens`。

以下命令直接复用脚本内置的长公共前缀，再在末尾追加不同问题：

```bash
PREFIX="$(python3 -c 'from demo_prefix_caching import TEXT_QUESTION_PREFIX; print(TEXT_QUESTION_PREFIX)')"

python3 demo_prefix_caching.py \
  --image_path None \
  --question "${PREFIX}它最主要的优势是什么？" \
  --question "${PREFIX}它最典型的应用是什么？" \
  --question "${PREFIX}它最主要的挑战是什么？"
```

自定义多个问题时，重复使用 `--question`：

```bash
python3 demo_prefix_caching.py \
  --image_path "${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg" \
  --question "请仔细观察这张图片，并描述主要主体。" \
  --question "请仔细观察这张图片，并描述拍摄环境。" \
  --question "请仔细观察这张图片，并说明人物和动物的互动。"
```

问题的公共 token 前缀越长，可复用的 prefill 范围通常越大。快照只在 prefill chunk 结束后保存，因此可直接恢复的位置取决于 chunk 边界；从最近快照到最长公共前缀之间的 token 仍需 replay。

### 5.3 开关和性能

前缀缓存默认开启。可关闭缓存，用相同问题序列对比完整 vision 和 prefill 流程：

```bash
python3 demo_prefix_caching.py --prefix-cache false
```

启用 `--perf true` 时，`PerfTracker` 会记录以下 prefix 原始指标：

- `matched_tokens`：两轮 prompt 的最长公共 token 前缀。
- `restored_tokens`：直接从快照恢复的 token 数。
- `replay_tokens`：从恢复位置到最长公共前缀之间需要重新执行的 token 数。

当前默认终端性能格式不会单独展开上述三个原始指标，但会显示实际发生过的 prefix timing scope：

- `llm.prefix.restore`：设备状态恢复耗时。
- `llm.prefix.replay`：公共前缀 replay 耗时。
- `llm.prefix.suffix_prefill`：差异后缀 prefill 耗时。
- `llm.prefix.snapshot`：设备状态快照耗时。

某个 scope 在当前请求中未执行时，不会出现在该轮性能报告中。例如首轮没有可恢复快照，因此不会出现 `llm.prefix.restore` 和 `llm.prefix.replay`。

### 5.4 使用限制

- 支持纯文本和图片输入；两种请求均支持 prefix cache 状态恢复，vision embedding 复用仅对图片请求生效。
- 缓存只在同一个 Python 进程和同一个模型实例内有效。
- 连续纯文本请求使用空图片路径元组，可以相互复用文本前缀状态。
- 对图片请求，图片路径元组必须完全一致才能复用图片特征和前缀状态。
- 当前实现按图片路径判断图片是否相同，不检查文件内容是否在请求间发生变化。
- `--question` 的顺序会影响相邻两轮之间可匹配的公共前缀。
- 该示例用于验证单实例顺序请求，不提供跨进程、持久化或并发请求缓存。

## 6 公共参数

三个示例均支持以下模型选择和路径覆盖参数：

- `--config`：配置文件路径，默认使用上级目录的 `config.yaml`。
- `--model_name`、`--model_size`：选择模型配置。
- `--tokenizer_dir`：覆盖 tokenizer 和 processor 目录。
- `--embedding_path`：覆盖 embedding 权重路径。
- `--ndevice`：设备数量。
- `--batch`：推理 batch，当前仅支持 `1`。
- `--perf`：开启或关闭性能报告。

模型图路径可通过各 Demo 对应的 `--prefill_path`、`--decode_path`、`--vision_path` 或 MTP 图参数覆盖。例如：

```bash
python3 demo.py \
  --prefill_path /path/to/prefill.hmm \
  --decode_path /path/to/decode.hmm \
  --vision_path /path/to/visual.hmm \
  --embedding_path /path/to/quant_embedding.pt \
  --tokenizer_dir /path/to/tokenizer
```

## 7 选择建议

- 普通文本或视觉问答使用 `demo.py`。
- 需要交互输入或保留多轮会话历史时使用 `demo.py`。
- 已准备 Qwen3.6 MTP 模型并希望验证推测生成时使用 `demo_mtp.py`。
- 连续提交多个具有长公共前缀的文本问题，或针对同一图片回答多个相似问题，并希望验证 prompt prefix cache 时使用 `demo_prefix_caching.py`；图片请求还可验证 vision embedding 复用。
