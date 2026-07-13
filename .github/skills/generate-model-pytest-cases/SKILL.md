---
name: generate-model-pytest-cases
description: 为 imodelzoo 中指定模型生成 `tests/models_tests` 风格的 pytest 测例，包括模型配置 JSON、test.sh 参数组与 Python demo 开关、pytest 入口、marker 注册、`imodelExampleConfig.yaml` 聚合配置。适用于新增模型测例、补齐 get_model/quant/compile/demo/perf/eval/compare 流程。
---

# 为指定模型生成 pytest 测例

## 存放规则

1. `SKILL.md` 必须放在 `.github` 目录下，推荐路径为 `.github/skills/<skill-name>/SKILL.md`。
2. 不要把 `SKILL.md` 放在仓库根目录。
3. 如果历史上已经在仓库根目录创建过 `SKILL.md`，迁移到 `.github/skills/` 后应删除旧文件，避免出现多个版本。
4. 与技能相关的模板、参考材料和辅助脚本，也优先放在 `.github/skills/<skill-name>/` 子目录中统一管理。

## 目标

为指定模型接入 `tests/models_tests` 统一测试框架，并保证：

1. 模型有独立配置文件 `model_cfg_<model>.json`。
2. 对应 flow 在 `test_*.py` 中有 pytest 入口。
3. 模型名已注册到 `model_names.txt`。
4. `config/imodelExampleConfig.yaml` 中使用 `models_tests` 风格，而不是直接跑 `test.sh`。
5. 不要修改这些文件:
    - `tests/conftest.py`
    - `tests/models_tests/conftest.py`
    - `tests/models_tests/test_models_utils.py`
    - `tests/models_tests/update_test_py.py`
    - `tests/tests_utils/tests_common_utils.py`
    - `tests/tests_utils/tests_pyvenv_utils.py`

## 先读这些文件

1. `tests/models_tests/README.md`
2. `tests/models_tests/test_models_utils.py`
3. `tests/models_tests/update_test_py.py`
4. `config/imodelExampleConfig.yaml`
5. 目标模型目录下的 `README.md`、`get_model.py`、`ptq.py`、`build.py`、`demo.py`、`perf.py`、`test.sh`

## 必须修改的文件

按需修改以下文件：

1. `tests/models_tests/model_configs/model_cfg_<model>.json`
2. `tests/models_tests/model_names.txt`
3. `tests/models_tests/test_get_models.py`
4. `tests/models_tests/test_quant_models.py` （仅当支持 quant）
5. `tests/models_tests/test_compile_models.py`
6. `tests/models_tests/test_demo_models.py`
7. `tests/models_tests/test_perf_models.py`（仅当支持 perf）
8. `tests/models_tests/test_eval_models.py`（仅当支持 eval）
9. `tests/models_tests/test_compare_models.py`（仅当支持 compare）
10. `config/imodelExampleConfig.yaml`

## 执行顺序（强制）

1. 先阅读目标模型目录，确认实际支持的 flow 和脚本参数。
2. 再创建/修改 `model_cfg_<model>.json`。
3. 把模型 marker 写入 `model_names.txt`。
4. 在对应 `test_*.py` 中补 pytest 函数。
5. 若 `imodelExampleConfig.yaml` 里已有该模型但走的是脚本直跑，改成 `models_tests` 风格。
6. 最后跑 `pytest --collect-only` 做最小验证。

## 第一步：识别模型能力

需要从模型目录判断以下信息：

### 1. 基础信息

- 模型目录：如 `models/tts/cosyvoice3`
- 模型类型：`cv` 或 `llm`
- 支持 backend：如 `xh2`
- 支持平台：通常从 README 和脚本中的 `HOUMO_TARGET` / 平台断言判断
- 设备依赖：如 `ndevice=1`、`dev_mem=12g`

### 2. 支持 flow

结合脚本是否存在来判断：

- `get_model.py` -> `get_model`
- `ptq.py` -> `quant`
- `build.py` -> `compile`
- `demo.py` -> `demo`
- `test.sh` -> 在 ASIC 环境的 `demo` flow 中优先执行；根据脚本接口生成 `test_sh_params`
- `demo_multibatch.py` -> 在 `support_flow` 中额外加入 `demo_multibatch`，并补 `demo_multibatch_params`；它作为 `demo` flow 的附加执行步骤，不新增独立 `test_*.py`
- `demo.py` 中带 perf 输出，且测试框架可解析 perf 输出日志 -> `perf`
- `hmatc` compare/eval 配置 -> `compare` / `eval`

### 3. 参数名

重点核对脚本参数是否和框架默认假设一致：

- `ptq.py` 常见参数：`--out-dir` 或 `--output_dir`
- `ptq.py` 模型路径参数：`--model` 或 `--model_dir`
- `build.py` 常见参数：`--model_dir`、`--output_dir`、`--context_length`
- `demo.py` 常见参数：hmm 路径、embedding 路径、tokenizer 路径
- 若模型 demo 入口不是 `demo.py`，优先在 `demo_params` / `demo_multibatch_params` 中增加 `script` 字段指定脚本名，而不是为单个模型在共享逻辑里写特判

优先规则：

- `model_cfg_<model>.json` 中的参数名，应优先与模型脚本的真实对外接口保持一致。
- 如果 `ptq.py` 对外接口是 `--output_dir`，则 `quant_params` 中就写 `output_dir`；不要为了迁就历史习惯，强行把模型脚本改成 `--out-dir`。
- 只有在现有统一框架无法通过配置表达、且最小兼容修改可以明确降低维护成本时，才考虑改模型脚本或框架。

## 第二步：编写 `model_cfg_<model>.json`

### 1. 命名规则

文件命名必须为：

```text
tests/models_tests/model_configs/model_cfg_<模型名>.json
```

例如：

```text
tests/models_tests/model_configs/model_cfg_cosyvoice3.json
```

### 2. 最小字段集合

```json
{
    "obsolete": false,
    "model_dir": "models/tts/cosyvoice3",
    "model_type": "llm",
    "dependencies": {
        "ndevice": [1],
        "dev_mem": ["12g"]
    },
    "support_platform": ["x86_64"],
    "support_backend": ["xh2"],
    "support_core_num": {
        "xh2": [2]
    },
    "support_flow": {
        "xh2": ["get_model", "compile", "demo"]
    },
    "support_hmatc": null,
    "get_model_params": {},
    "compile_params": {},
    "demo_params": {}
}
```

### 3. context_length 规则（非常重要）

如果模型的量化/编译依赖固定上下文长度，配置文件中必须显式写出来，并反映到缓存目录名中。

例如：

- `ptq.py` 默认 `context_length=2048`
- 则目录命名不应写成 `hmquant_xh2`
- 应显式写成 `hmquant_xh2_2k`

推荐映射：

- `2048` -> `2k`
- `8192` -> `8k`
- `16384` -> `16k`
- `32768` -> `32k`

补充规则：

- `get_model_params` 中 `hmm` 下载项的 `context_length`，必须与 `get_model.py` 的默认值或该测试项实际传入值一致。
- `get_model_params.extract_dir` 的目录后缀，也必须与该 `context_length` 一致，例如 `8k` 应写成 `cached_models/hmm_xh2_8k`。
- `compile_params.context_length` 应优先与 `get_model_params` 中对应 `hmm` 产物的上下文长度保持一致。
- `compile_params.model_dir` 指向的量化输入目录，可以与 `compile_params.output_dir` 的上下文后缀不同；也就是说，允许“输入 `hmquant_xh2_2k`，输出 `hmm_xh2_8k`”。
- 一旦 `compile_params.output_dir` 的上下文后缀发生变化，`demo_params`、`perf` 读取路径中引用的 hmm/embedding 路径也要同步更新，避免仍指向旧目录。

以 `cosyvoice3` 为例：

```json
"get_model_params": {
    "xh2": {
        "type": ["raw", "raw", "hmm"],
        "download_dir": ["cached_models", "cached_models", "cached_models"],
        "extract_dir": [null, null, "cached_models/hmm_xh2_2k"],
        "source_type": [null, "modelscope", null],
        "context_length": [null, null, "2k"]
    }
},
"quant_params": {
    "xh2": {
        "output_dir": ["cached_results/hmquant_xh2_2k"],
        "context_length": ["2048"]
    }
},
"compile_params": {
    "xh2": {
        "model_dir": ["cached_results/hmquant_xh2_2k"],
        "context_length": ["2048"],
        "output_dir": ["cached_results/hmm_xh2_2k"]
    }
}
```

如果是类似 `glm-ocr` 这种“下载的预编译 hmm 默认是 `8k`，但量化输入目录仍是 `2k`”的场景，可以写成：

```json
"get_model_params": {
    "xh2": {
        "type": ["raw", "hmm"],
        "download_dir": ["cached_models", "cached_models"],
        "extract_dir": [null, "cached_models/hmm_xh2_8k"],
        "context_length": [null, "8k"]
    }
},
"quant_params": {
    "xh2": {
        "output_dir": ["cached_results/hmquant_xh2_2k"],
        "max_sequence_length": ["2048"]
    }
},
"compile_params": {
    "xh2": {
        "model_dir": ["cached_results/hmquant_xh2_2k"],
        "context_length": ["8192"],
        "output_dir": ["cached_results/hmm_xh2_8k"]
    }
},
"demo_params": {
    "xh2": {
        "embedding_path": ["cached_results/hmm_xh2_8k/hmquant/quant_embedding.pt"],
        "prefill_path": ["cached_results/hmm_xh2_8k/<model>_prefill.hmm"],
        "decode_path": ["cached_results/hmm_xh2_8k/<model>_decode.hmm"]
    }
}
```

### 4. LLM 模型的经验规则

- `model_type` 通常写 `llm`。
- `quant` / `compile` 很可能依赖 GPU，框架会在 `execute_quant_flow()` / `execute_compile_flow()` 中按 `llm` 分支处理。
- `compare` / `eval` 不要盲目加，先确认模型确实有这两类脚本或 `hmatc` 配置。
- 如果模型目录下存在 `demo_multibatch.py`，默认应在配置里加入 `demo_multibatch` 与 `demo_multibatch_params`。若 multibatch 依赖单独的 batch 编译产物，应在 `compile_params` 中额外增加一组 `batch` / `output_dir`，并让 `demo_multibatch_params` 指向对应目录。

### 5. perf 配置规则

如果支持 `perf`：

- `support_flow` 要包含 `perf`
- 必须加 `perf_metrics`
- 如果 perf 走 `demo.py` 输出性能日志，则写：

```json
"perf_params": "demo"
```

如果 demo 走的不是默认脚本名，可在 `demo_params` 中补充：

```json
"demo_params": {
    "xh2": {
        "script": ["demo_asr.py"]
    }
}
```

`perf_params: "demo"` 时，perf 默认复用 `demo_params` 中的 `script`。`demo_multibatch.py` 也支持通过 `demo_multibatch_params` 中的 `script` 覆盖脚本名。

### 6. `test.sh` 与 Python demo 配置规则

如果模型目录存在 `test.sh`，检查其参数接口和 README 中的典型调用方式，并按需生成 `test_sh_params`。该字段只在 ASIC 环境且模型临时工作目录存在 `test.sh` 时生效。

使用参数数组表示多次执行，每个内层数组对应一次 `test.sh`：

```json
"test_sh_params": [
    ["--model_size", "0.8b"],
    ["--model_size", "27b", "--model_name", "qwen3.6"]
]
```

以上配置依次执行：

```bash
bash test.sh --model_size 0.8b
bash test.sh --model_size 27b --model_name qwen3.6
```

需要同时覆盖默认参数和其他参数时，把空数组作为一组测试参数，并按期望执行顺序放置：

```json
"test_sh_params": [
    [],
    ["--model_size", "0.8b"],
    ["--model_size", "27b", "--model_name", "qwen3.6"]
]
```

以上配置依次执行：

```bash
bash test.sh
bash test.sh --model_size 0.8b
bash test.sh --model_size 27b --model_name qwen3.6
```

需要按 backend 区分时，在 backend 下分别配置参数数组：

```json
"test_sh_params": {
    "xh1": [
        [],
        ["--model_size", "7b"]
    ],
    "xh2": [
        [],
        ["--model_size", "7b"],
        ["--model_size", "14b"]
    ]
}
```

也可使用与其他参数矩阵一致的按参数列格式，但该格式不适合表达“默认参数组 + 其他参数组”；有默认参数组时必须优先使用参数数组：

```json
"test_sh_params": {
    "xh2": {
        "model_size": ["7b", "14b"],
        "use_cache": [true, false]
    }
}
```

未配置 `test_sh_params` 或配置为空数组时，执行一次默认命令 `bash test.sh`。所有参数组都会执行，任意一组失败都会使 demo 测例失败。

使用 `enable_demo_test` 控制 `test.sh` 后是否继续执行 Python demo 测试。该字段缺省为 `true`：

```json
"enable_demo_test": true
```

如果只验证默认参数及其他参数的 `test.sh`，不执行 `demo.py`、`demo_multibatch.py`、模型准备和 Python 虚拟环境安装，配置：

```json
"test_sh_params": [
    [],
    ["--model_size", "7b"]
],
"enable_demo_test": false
```

即使设置 `enable_demo_test: false`，也要在 `support_flow.<backend>` 中保留 `demo`，并生成 `test_demo_models.py` 的 pytest 入口；`test.sh` 由 `execute_demo_flow()` 内部执行，不要在 `imodelExampleConfig.yaml` 中改回直接运行脚本。

## 第三步：注册 marker

把模型 marker 追加到：

```text
tests/models_tests/model_names.txt
```

规则：

- `-` -> `_`
- `.` -> `dot`

例如：

- `qwen2.5` -> `qwen2dot5`
- `deepseek-r1-qwen3-8b` -> `deepseek_r1_qwen3_8b`
- `cosyvoice3` -> `cosyvoice3`

## 第四步：补 pytest 入口

### 1. `get_model`

在 `tests/models_tests/test_get_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_tts_cosyvoice3_get_model")
def test_tts_cosyvoice3_get_model(setup_logging) -> None:
    """test_tts_cosyvoice3_get_model"""
    model_name = "cosyvoice3"
    _get_model_func(model_name, setup_logging)
```

### 2. `quant`

在 `tests/models_tests/test_quant_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_tts_cosyvoice3_quant",
    depends_on=["test_get_models.py::test_tts_cosyvoice3_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_tts_cosyvoice3_quant(setup_logging) -> None:
    """test_tts_cosyvoice3_quant"""
    model_name = "cosyvoice3"
    _quant_func(model_name, setup_logging)
```

### 3. `compile`

在 `tests/models_tests/test_compile_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_tts_cosyvoice3_compile",
    depends_on=["test_quant_models.py::test_tts_cosyvoice3_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_tts_cosyvoice3_compile(setup_logging) -> None:
    """test_tts_cosyvoice3_compile"""
    model_name = "cosyvoice3"
    _compile_func(model_name, setup_logging)
```

### 4. `demo`

在 `tests/models_tests/test_demo_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_tts_cosyvoice3_demo(setup_logging) -> None:
    """test_tts_cosyvoice3_demo"""
    model_name = "cosyvoice3"
    _demo_func(model_name, setup_logging)
```

### 5. `perf`

如果支持 `perf`，在 `tests/models_tests/test_perf_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_tts_cosyvoice3_perf(setup_logging) -> None:
    """test_tts_cosyvoice3_perf"""
    model_name = "cosyvoice3"
    _perf_func(model_name, setup_logging)
```

### 6. `eval`

如果支持 `eval`，在 `tests/models_tests/test_eval_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.eval
def test_tts_cosyvoice3_eval(setup_logging) -> None:
    """test_tts_cosyvoice3_eval"""
    model_name = "cosyvoice3"
    _eval_func(model_name, setup_logging)
```

### 7. `compare`

如果支持 `compare`，在 `tests/models_tests/test_compare_models.py` 中增加目标模型测试用例，参考如下：

```python
@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.compare
def test_tts_cosyvoice3_compare(setup_logging) -> None:
    """test_tts_cosyvoice3_compare"""
    model_name = "cosyvoice3"
    _compare_func(model_name, setup_logging)
```

## 第五步：修改 `imodelExampleConfig.yaml`

如果该模型在 `config/imodelExampleConfig.yaml` 中原来直接执行 `test.sh`，应改为与现有 `models_tests` 一致的风格。
如果该模型不在 `config/imodelExampleConfig.yaml` 中，应为模型新增测试配置，保持与现有 `models_tests` 一致的风格。

### 不推荐

```yaml
tts_cosyvoice3:
    example_case:
        - tts_cosyvoice3_test:
              script: ../models/tts/cosyvoice3/test.sh
```

### 推荐

```yaml
tts_cosyvoice3:
    include:
        - models/tts/cosyvoice3
    exclude:
        - models/tts/cosyvoice3/README.MD
    example_case:
        - tts_cosyvoice3_test:
              test_type: models_tests
              script: cosyvoice3
              args: all
    test:
        - tts_cosyvoice3_test
```

## 最低交付物

新增一个模型 pytest 测例时，最低要交付：

1. `model_cfg_<model>.json`
2. `model_names.txt` 中的 marker
3. 至少 `get_model` / `compile` / `demo` 的 pytest 入口
4. 如果模型支持，则补 `quant` / `perf` / `eval` / `compare`
5. `config/imodelExampleConfig.yaml` 中对应 `models_tests` 风格配置
6. 一次 `pytest --collect-only` 验证结果

## 推荐验证命令

### 1. 收集指定模型用例

```bash
pytest tests/models_tests/test_get_models.py \
  tests/models_tests/test_quant_models.py \
  tests/models_tests/test_compile_models.py \
  tests/models_tests/test_demo_models.py \
  -k <model_name> --collect-only -q
```

例如：

```bash
pytest tests/models_tests/test_get_models.py \
  tests/models_tests/test_quant_models.py \
  tests/models_tests/test_compile_models.py \
  tests/models_tests/test_demo_models.py \
  -k cosyvoice3 --collect-only -q
```

## 生成新模型测例时的检查清单

- [ ] 已阅读目标模型目录中的脚本和 README
- [ ] 已确定模型类型是 `cv` 还是 `llm`
- [ ] 已确认实际支持哪些 flow
- [ ] `quant_params` / `compile_params` 的参数名与脚本真实接口一致（如 `output_dir` vs `out-dir`）
- [ ] `context_length` 已显式写入配置
- [ ] `get_model_params` 的 `hmm` 默认 `context_length` 与 `get_model.py` 默认值一致
- [ ] 缓存目录名已反映 `2k` / `8k` / `16k` 等信息
- [ ] `compile_params.context_length` 与目标 hmm 上下文一致
- [ ] `demo_params` / `perf` 读取路径已跟随 `compile_params.output_dir` 同步
- [ ] 如模型存在 `test.sh`，已核对是否需要生成多组 `test_sh_params`
- [ ] 如需覆盖默认命令和带参数命令，`test_sh_params` 已使用 `[]` 表示默认参数组
- [ ] `test_sh_params` 未使用对象数组或字符串参数组
- [ ] 已根据是否继续运行 Python demo 正确设置 `enable_demo_test`；未配置时确认默认开启符合预期
- [ ] 即使 `enable_demo_test` 为 `false`，`support_flow` 和 pytest 入口仍保留 `demo`
- [ ] `model_names.txt` 已追加 marker
- [ ] 对应 `test_*.py` 已补 pytest 入口
- [ ] `imodelExampleConfig.yaml` 已改为 `models_tests` 风格
- [ ] 如有必要，`test_models_utils.py` 已做最小兼容改动
- [ ] `pytest --collect-only` 已通过

## 不要这样做

1. 不要只写 `model_cfg_<model>.json`，却不补 `test_*.py`。
2. 不要忘记更新 `model_names.txt`。
3. 不要在 `imodelExampleConfig.yaml` 里继续用 `test.sh` 替代 `models_tests`，除非该模型明确不接入统一框架。
4. 不要把 `hmquant_xh2`、`hmm_xh2` 这种不含上下文长度的信息用于有固定 context 的 LLM 模型。
5. 不要因为历史经验里常见 `--out-dir`，就把实际只支持 `--output_dir` 的模型脚本强行改接口。
6. 不要只改 `compile_params.output_dir` 的上下文后缀，却忘了同步 `demo_params` / `perf` 对应路径。
7. 不要修改 `apis/common/` 或 `tools/common/` 下的 vendored 内容。

## `cosyvoice3` 经验总结

本次接入 `cosyvoice3` 时，实际落地经验如下：

1. 该模型应走 `models_tests` 统一框架，而不是在 `imodelExampleConfig.yaml` 中直接跑 `test.sh`。
2. `ptq.py` 默认 `context_length=2048`，因此量化/编译/下载的目录需要统一命名为 `*_2k`。
3. `ptq.py` 使用的是 `model_dir` / `output_dir` 参数名，不是典型的 `model` / `out-dir`，框架侧需要兼容。
4. `demo.py` 的 perf 日志字段与通用 LLM 略有差异，必要时需要补 perf 解析兼容。
5. 对 TTS/多组件模型，`demo_params` 往往比普通 LLM 更长，必须逐个核对 hmm 和 embedding 路径。

## `glm-ocr` 经验补充

本次接入 `glm-ocr` 时，额外确认了以下规则：

1. `get_model.py` 的 `--context_length` 默认值是 `8k`，因此 `get_model_params` 中的 `hmm` 下载项必须写成 `8k`，并对应 `cached_models/hmm_xh2_8k`。
2. `ptq.py` 的真实对外接口是 `--output_dir`，因此 `quant_params` 应写 `output_dir`，不应为了适配历史习惯强制改脚本为 `--out-dir`。
3. `compile_params.context_length` 最好与 `get_model_params` 中目标 hmm 的上下文保持一致；对 `glm-ocr`，应使用 `8192`。
4. `compile_params.model_dir` 可以继续指向 `hmquant_xh2_2k`，而 `compile_params.output_dir` 使用 `hmm_xh2_8k`，这是允许的。
5. 当 `compile_params.output_dir` 改为 `hmm_xh2_8k` 后，`demo_params` 中的 `embedding_path`、`vit_path`、`prefill_path`、`decode_path` 也必须同步改到 `hmm_xh2_8k`。
