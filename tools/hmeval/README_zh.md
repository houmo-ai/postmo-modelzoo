# hmeval（中文说明）

`hmeval` 是一个内部使用的命令行评测工具，用于基于 EvalScope 评测自定义大模型。

核心流程：
- 加载**自定义 Python 模型脚本**（`--model`）
- 指定模型权重目录（`--model-dir`）
- 在一个或多个数据集上执行评测（`--dataset`）

---

## 1）安装

在当前目录（`imodelzoo/hmeval`）执行：

```bash
# 仅安装 CLI（轻量）
pip install -e .

# 安装评测依赖（建议真实评测时使用）
pip install -e .[eval]
```

说明：
- `requirements.txt` 通过 `eval` 可选依赖安装。
- 这样可以避免基础安装被重依赖阻塞。

### 示例模型依赖策略（重要）

- `hmeval` 工具本体依赖保持最小化。
- 各示例模型依赖**不做全局统一**，因为版本可能冲突（尤其是 `transformers`）。
- 示例脚本中**不要**在运行时自动安装/升级依赖。
- 若版本不匹配，应直接报错并提示切换到兼容环境，而不是在当前环境里动态改包版本。

---

## 2）命令行用法

```bash
hmeval --model <model_script.py|python.module> \
       --model-dir <model_artifact_dir> \
       --dataset <dataset1> [dataset2 ...] \
       [--limit N] \
       [--output ./outputs] \
       [--model-args KEY=VALUE]...
```

### 参数说明

- `--model`（必填）
  - 自定义模型脚本路径（如 `examples/qwen3/hmm_xh2_qwen3.py`）或模块名。
- `--model-dir`（必填）
  - 模型权重/产物目录。
  - 会写入 `TaskConfig.model`，并同时透传到 `TaskConfig.model_args["model_dir"]`。
- `--dataset`（必填，支持多个）
  - 一个或多个数据集，例如：`--dataset mmlu gsm8k`。
- `--limit`（可选）
  - 最多评测样本数。`0` 表示全量。
- `--output`（可选）
  - 输出目录（默认 `./outputs`）。
- `--model-args`（可选，可重复）
  - 透传给自定义模型构造函数的扩展参数。
  - 例如：`--model-args tokenizer_dir=/path/to/tokenizer --model-args temperature=0.7`

### `--model-args` 自动类型解析

`hmeval` 会自动解析标量：
- `true/false` → `bool`
- `none/null` → `None`
- 整数/浮点数 → 数值类型
- 其他 → `str`

---

## 3）自定义模型接入（详细）

这是最关键部分。

### 3.1 必须满足的条件

你的模型脚本必须：

1. 定义全局常量 `API_NAME`
2. 使用 `@register_model_api(name=API_NAME)` 注册
3. 模型类继承 `ModelAPI`
4. 实现 `generate()` 并返回 `ModelOutput`

### 3.2 推荐模板

```python
from typing import List, Dict, Any, Optional
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api

API_NAME = "my_custom_model"


@register_model_api(name=API_NAME)
class MyCustomModel(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)

        # hmeval 会自动传入
        self.model_dir = model_args.get("model_dir")
        if not self.model_dir:
            raise ValueError("`model_dir` is required")

        # 来自 --model-args
        self.tokenizer_dir = model_args.get("tokenizer_dir")

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 在这里实现推理逻辑
        text = "hello"
        return ModelOutput.from_content(model="my_custom_model", content=text)
```

### 3.3 `hmeval` 如何加载你的脚本

当 `--model` 是 `.py` 文件时，`hmeval` 会：
- 动态导入该脚本
- 将脚本目录加入 `sys.path`
- 将脚本目录添加到 `PYTHONPATH`

因此脚本同目录下的依赖模块（如 `from xxx_impl import ...`）可直接导入。

### 3.4 常见错误

- 脚本里没有 `API_NAME`
- 没有使用 `@register_model_api(name=API_NAME)`
- 没有实现 `generate()`
- `model_dir` / `tokenizer_dir` 路径不存在
- `generate()` 没有返回 `ModelOutput`

---

## 4）示例命令

### 下载模型（hmm）

评测前先下载 hmm 模型文件：

```bash
cd examples/qwen3
python3 get_model.py --download-dir ./models
```

qwen3-vl 同理：

```bash
cd examples/qwen3-vl
python3 get_model.py --download-dir ./models
```

### Qwen3 文本模型

```bash
hmeval \
  --model examples/qwen3/hm_xh2_qwen3.py \
  --model-dir examples/qwen3/models/hmm_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v1.1.0/ \
  --dataset gsm8k \
  --limit 2 \
  --model-args tokenizer_dir=examples/qwen3/models/tokenizers
```

### Qwen3-VL 多模态模型

```bash
hmeval \
  --model examples/qwen3-vl/hm_xh2_qwen3_vl.py \
  --model-dir examples/qwen3-vl/models/hmm_xh2_qwen3-vl_4b_256_32k_b1_1chip_2cores_v1.1.0/ \
  --dataset mm_bench \
  --limit 2 \
  --model-args tokenizer_dir=examples/qwen3-vl/models/tokenizers
```

### 多数据集

```bash
hmeval \
  --model examples/qwen3/hm_xh2_qwen3.py \
  --model-dir examples/qwen3/models/hmm_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v1.1.0/ \
  --dataset mmlu gsm8k ceval \
  --model-args tokenizer_dir=examples/qwen3/models/tokenizers
```

---

## 5）输出与报告

评测结果和报告输出到 `--output`（默认 `./outputs`）。

如果报告中模型名显示异常，请检查：
- 自定义模型中 `ModelOutput.from_content(model=...)` 是否返回稳定名称
- `model_dir` 是否已规范化（避免末尾 `/` 导致 basename 异常）
- `TaskConfig.model` 是否设置为期望展示名称
