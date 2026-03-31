# hmeval

`hmeval` is an internal CLI tool for evaluating custom large models with EvalScope.

It focuses on one workflow:
- Load a **custom Python model script** (`--model`)
- Read model artifacts from `--model-dir`
- Run one or more datasets via EvalScope

---

## 1) Installation

From this directory (`imodelzoo/hmeval`):

```bash
# Base CLI installation (lightweight)
pip install -e .

# With evaluation dependencies (recommended for real runs)
pip install -e .[eval]
```

> `requirements.txt` is installed through the optional extra `eval`.

### Dependency policy for example models

- `hmeval` CLI keeps its own dependencies minimal.
- Example model dependencies are **not** unified globally because versions may conflict (especially `transformers`).
- Example scripts should **not** auto-install/upgrade packages at runtime.
- If an example requires a specific version, it should fail fast with a clear error and you should switch to a compatible environment.

---

## 2) CLI Usage

```bash
hmeval --model <model_script.py|python.module> \
			 --model-dir <model_artifact_dir> \
			 --dataset <dataset1> [dataset2 ...] \
			 [--limit N] \
			 [--output ./outputs] \
			 [--model-args KEY=VALUE]...
```

### Parameters

- `--model` (required)
	- Python script path (e.g. `examples/qwen3/hm_xh2_qwen3.py`) or module name.
- `--model-dir` (required)
	- Model artifact directory, passed into `TaskConfig.model` and `TaskConfig.model_args["model_dir"]`.
- `--dataset` (required, supports multiple)
	- One or more datasets, e.g. `--dataset mmlu gsm8k`.
- `--limit` (optional)
	- Max samples to evaluate. `0` means full dataset.
- `--output` (optional)
	- Output directory (default: `./outputs`).
- `--model-args` (optional, repeatable)
	- Extra key-value pairs forwarded to your model constructor.
	- Example: `--model-args tokenizer_dir=/path/to/tokenizer --model-args temperature=0.7`

### Type parsing for `--model-args`

`hmeval` parses scalar values automatically:
- `true/false` -> bool
- `none/null` -> None
- integers/floats -> numeric types
- otherwise -> string

---

## 3) Custom Model Guide (Detailed)

This is the key part of the integration.

### 3.1 Minimal requirements

Your custom script must:

1. Define a global `API_NAME`
2. Register a class with `@register_model_api(name=API_NAME)`
3. Inherit from `ModelAPI`
4. Implement `generate()` and return `ModelOutput`

### 3.2 Recommended file skeleton

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

				# model_dir is passed by hmeval automatically
				self.model_dir = model_args.get("model_dir")
				if not self.model_dir:
						raise ValueError("`model_dir` is required")

				# Optional args from --model-args
				self.tokenizer_dir = model_args.get("tokenizer_dir")

		def generate(
				self,
				input: List[ChatMessage],
				tools: List[ToolInfo],
				tool_choice: ToolChoice,
				config: GenerateConfig,
		) -> ModelOutput:
				# Your inference logic
				text = "hello"
				return ModelOutput.from_content(model="my_custom_model", content=text)
```

### 3.3 How `hmeval` loads your script

When `--model` is a `.py` file:
- `hmeval` imports it dynamically
- adds the script directory to `sys.path`
- prepends the script directory to `PYTHONPATH`

So local imports in the same folder (e.g. `from xxx_impl import ...`) work by default.

### 3.4 Common pitfalls

- Missing `API_NAME` in model script
- Not using `@register_model_api(name=API_NAME)`
- Missing `generate()` implementation
- `model_dir`/tokenizer paths not existing
- Returning non-`ModelOutput` value in `generate()`

---

## 4) Examples

### Download model (hmm)

Before running evaluation, download the hmm model files:

```bash
cd examples/qwen3
python get_model.py --download-dir ./models
```

For qwen3-vl:

```bash
cd examples/qwen3-vl
python get_model.py --download-dir ./models
```

### Qwen3 text example

```bash
hmeval \
	--model examples/qwen3/hm_xh2_qwen3.py \
	--model-dir examples/qwen3/models/hmm_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v1.1.0/ \
	--dataset gsm8k \
	--limit 2 \
	--model-args tokenizer_dir=examples/qwen3/models/tokenizers
```

### Qwen3-VL example

```bash
hmeval \
	--model examples/qwen3-vl/hm_xh2_qwen3_vl.py \
	--model-dir examples/qwen3-vl/models/hmm_xh2_qwen3-vl_4b_256_32k_b1_1chip_2cores_v1.1.0/ \
	--dataset mm_bench \
	--limit 2 \
	--model-args tokenizer_dir=examples/qwen3-vl/models/tokenizers
```

### Multiple datasets

```bash
hmeval \
	--model examples/qwen3/hm_xh2_qwen3.py \
	--model-dir examples/qwen3/models/hmm_xh2_qwen3_8b_256_8k_b1_1chip_2cores_v1.0.0/ \
	--dataset mmlu gsm8k ceval \
	--model-args tokenizer_dir=examples/qwen3/models/tokenizers
```

---

## 5) Output

Evaluation artifacts and reports are written under `--output` (default `./outputs`).

If report model naming is abnormal, ensure your custom model returns a stable model identifier in `ModelOutput.from_content(model=...)` and that `model_dir` is normalized.


