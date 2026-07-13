# `models_tests` 测试说明

`tests/models_tests` 是 `imodelzoo` 中面向模型用例的统一回归测试入口，基于 `pytest` + JSON 配置驱动，覆盖模型资源获取、量化、编译、推理、结果对比、精度评测和性能评测。


## 1. 目录与文件职责

| 路径 | 说明 |
| --- | --- |
| `model_configs/` | 每个模型的测试配置，决定支持哪些 flow、跑哪些参数、需要哪些资源 |
| `model_names.txt` | 所有模型 marker 名单；`conftest.py` 会读取它并注册成 pytest marker |
| `test_get_models.py` | 模型获取测试入口 |
| `test_quant_models.py` | 量化测试入口 |
| `test_compile_models.py` | 编译测试入口 |
| `test_demo_models.py` | 推理demo测试入口 |
| `test_compare_models.py` | 结果对比测试入口 |
| `test_eval_models.py` | 精度评测测试入口 |
| `test_perf_models.py` | 性能评测测试入口 |
| `test_models_utils.py` | 所有 flow 的核心执行逻辑、资源准备、结果校验、缓存恢复 |
| `update_test_py.py` | 根据 `model_configs` 自动为主测试文件追加新模型用例 |
| `conftest.py` | 注册 flow marker / model marker，并固定测试文件执行顺序 |

## 2. 测试框架是怎么工作的

### 2.1 配置驱动

每个模型对应一个配置文件：

- 命名规则：`model_cfg_<模型名>.json`
- 路径：`tests/models_tests/model_configs/`
- 代码入口：`test_models_utils.py::_load_model_cfg()`

测试执行时，公共逻辑会先读取配置，再判断：

- 模型是否废弃：`obsolete`
- 当前 backend 是否支持：`support_backend`
- 当前平台是否支持：`support_platform`
- 当前 flow 是否支持：`support_flow`
- 当前设备资源是否满足：`dependencies` + pytest marker

### 2.2 固定执行顺序

`tests/models_tests/conftest.py` 会按下列顺序重排测试文件：

1. `test_get_models.py`
2. `test_quant_models.py`
3. `test_compile_models.py`
4. `test_demo_models.py`
5. `test_compare_models.py`
6. `test_eval_models.py`
7. `test_perf_models.py`

同时，很多测试函数还通过 `@pytest.mark.dependency(...)` 显式依赖前置 flow，例如：

- `quant` 依赖 `get_model`
- `compile` 依赖 `quant` 或 `get_model`

因此推荐直接跑目录或按 marker 过滤，而不要手工改变 flow 顺序。

### 2.3 临时工作目录与缓存

每条用例都会先复制模型目录到临时目录，再在临时目录中执行测试，结束后删除临时目录。目录名格式如下：

```text
<model_dir>_<flow>_<timestamp>
```

这样可以避免污染原始模型目录。

此外，框架还使用两类缓存：

- 原始模型缓存：`IMODELZOO_MODELS_PATH` 指向的目录
- 测试结果缓存：`tests/model_results_{HOUMO_TARGET}/<ndevice_x>_<dev_mem_xxg>/...`

缓存目录会结合文件锁 `ModelResourceLock` 进行保护，避免并发测试互相覆盖。

## 3. 当前实现的功能全景

### 3.1 七个主测试 flow

| Flow | 作用 | 常见执行路径 | 结果判定 |
| --- | --- | --- | --- |
| `get_model` | 测试模型资源下载/准备 | `get_model.py` | 命令正常结束 |
| `quant` | 测试量化 | `hmatc quant` 或 `ptq.py` | 命令正常结束，输出不含失败标记 |
| `compile` | 测试编译 | `hmatc build` 或 `build.py` | 编译成功，且 golden / cosine 等结果通过阈值校验 |
| `demo` | 测试推理 | `hmatc demo` 或 `demo.py` | 推理正常结束 |
| `compare` | 测试结果一致性 | `hmatc compare` | 解析输出表格并校验阈值 |
| `eval` | 测试精度 | `hmatc eval` | 解析评测结果并按阈值比较 |
| `perf` | 测试性能回归 | `hmatc perf` 或 `demo.py` / `perf.py` | 解析性能结果并与 benchmark 比较 |

### 3.2 支持`demo_multibatch`

支持 llm 中的 multibatch demo：

- `execute_demo_flow()` 在跑完 `demo` 后，会继续尝试执行 `demo_multibatch.py`
- 若模型目录存在 `demo_multibatch.py`，模型配置应在 `support_flow` 中包含 `demo_multibatch`，并提供 `demo_multibatch_params`
- 前提是模型配置中声明了：
  - `support_flow` 包含 `demo_multibatch`
  - 且存在 `demo_multibatch_params`
- 当前它没有单独的 `test_demo_multibatch_models.py`
- `update_test_py.py` 也会显式跳过为 `demo_multibatch` 生成独立 pytest 用例

也就是说：`demo_multibatch` 是 `demo` flow 的附加执行步骤，而不是单独的第八类 pytest flow。

### 3.3 支持非 `demo.py` 脚本名

部分模型的推理入口并不叫 `demo.py`，例如 `demo_asr.py`、`demo_forcealigner.py`。这类模型可在配置中声明：

- `demo_params.<backend>.script`：覆盖 `demo` flow 的脚本名
- `demo_multibatch_params.<backend>.script`：覆盖 `demo_multibatch` 的脚本名

示例：

```json
"demo_params": {
  "xh2": {
    "script": ["demo_asr.py"]
  }
}
```

当 `perf_params: "demo"` 时，perf 默认复用 `demo_params.<backend>.script`；未配置脚本名时，框架仍保持现有默认行为：`demo` 对应 `demo.py`，`demo_multibatch` 对应 `demo_multibatch.py`。

### 3.3 CV 与 LLM 的差异

当前代码对 CV 和 LLM 走的是不同分支：

| 维度 | CV | LLM |
| --- | --- | --- |
| 量化准备 | `_prepare_quantized_cv_model()` | `_prepare_quantized_llm_model()` |
| 编译准备 | `_prepare_compiled_cv_model()` | `_prepare_compiled_llm_model()` |
| 常见量化方式 | `hmatc quant` 或模型内量化脚本 | 主要走 `ptq.py` |
| 常见编译方式 | `hmatc build` 或 `build.py` | 主要走 `build.py` |
| compare / eval | 主要是 CV 模型支持 | 当前 LLM 基本不支持 |
| GPU 依赖 | 视模型而定 | 量化/编译常要求 GPU |

补充说明：

- 当前代码中，LLM 测试通常不走 `hmatc` compare / eval。
- 对 LLM，`quant` / `compile` 在不少场景下要求 `x86_64 + GPU`，且 release 模式下会被跳过。

### 3.4 `hmatc` 路径与 Python 脚本路径

框架会根据模型配置自动选择执行路径：

- 如果配置中存在 `hmquant_params` / `hmbuild_params` / `hmdemo_params` / `hmcompare_params` / `hmeval_params` / `hmperf_params`，优先走 `hmatc`
- 否则退化到模型目录内的 Python 脚本，如：
  - `get_model.py`
  - `ptq.py`
  - `build.py`
  - `demo.py`
  - `perf.py`

Python 脚本型 flow 在需要时会尝试安装/使用独立虚拟环境。

## 4. 主要功能点说明

### 4.1 设备资源依赖 marker：`dependencies`

`model_configs/*.json` 现在普遍带有 `dependencies` 字段，`update_test_py.py` 会把它转成 pytest marker：

- `dependencies.ndevice: [1]` -> `@pytest.mark.ndevice_1`
- `dependencies.dev_mem: ["24g"]` -> `@pytest.mark.dev_mem_24g`

测试执行时，`test_models_utils.py::check_device_markers()` 会读取这些 marker，并把它们拼到结果缓存路径中：

```text
<ndevice_x>_<dev_mem_xxg>
```

这意味着当前框架已经不只是“模型功能测试”，还显式携带了“资源规格约束”。

### 4.2 自动注册模型 marker

`tests/models_tests/conftest.py` 会读取 `model_names.txt`，把每个模型名注册成 pytest marker。这样可以直接使用：

```bash
pytest -m "qwen3"
pytest -m "deepseek_r1_qwen3_8b"
```

注意：marker 名不是原始模型名，而是脚本转换后的名字：

- `-` 会转成 `_`
- `.` 会转成 `dot`

例如：

- `qwen2.5` -> `qwen2dot5`
- `deepseek-r1-qwen3-8b` -> `deepseek_r1_qwen3_8b`

### 4.3 分阶段测试

框架支持把“量化/编译”和“推理/评测”拆到不同机器执行，这是当前实现里非常重要的一点。

核心环境变量：

- `SKIP_INFER`: 只要设置为 `ON` 或 `OFF` 中任一值，就会启用分阶段模式
- `HDPL_PLATFORM=ISIM`: 当前进程视为“无推理阶段”，即 `SEPARATE_NO_INFER`
- `HDPL_PLATFORM=ASIC`: 当前进程视为“推理阶段”，即 `SEPARATE_INFER`

代码行为如下：

- `SEPARATE_NO_INFER` 阶段：
  - 执行量化/编译准备
  - 把结果保存到 `model_results_{HOUMO_TARGET}`
  - 跳过真正的推理/compare/eval/perf 执行
- `SEPARATE_INFER` 阶段：
  - 从缓存目录恢复模型结果
  - 在推理侧执行 `demo` / `compare` / `eval` / `perf`

### 4.4 release 模式

`USE_RELEASED_MODELS=ON` 时，代码会启用 release 模式。当前可见行为包括：

- `get_model` 对 LLM 会跳过 `raw` 下载分支
- `quant` / `compile` 的部分 LLM 测试会直接跳过
- `demo` 在 ASIC 环境下，如果模型目录存在 `test.sh`，会优先执行 `test.sh`

因此 release 模式更适合“验证已交付模型是否可运行”，而不是完整开发流程回归。

### 4.5 `test.sh` 优先执行

在 `demo` flow 中，如果满足以下条件：

- `HDPL_PLATFORM == "ASIC"`
- 当前模型目录存在 `test.sh`

框架会先执行 `test.sh`，然后根据 `enable_demo_test` 决定是否继续执行框架标准的 Python demo 测试。

#### 4.5.1 为 `test.sh` 配置多组参数

模型配置可通过 `test_sh_params` 指定多组参数。每组参数会独立执行一次 `test.sh`；所有执行结果会聚合，任意一次执行失败都会使当前 demo 测试失败。

推荐直接使用参数数组：

```json
"test_sh_params": [
  ["--model_size", "7b"],
  ["--model_size", "14b", "--ndevice", "1"]
]
```

以上配置会依次执行：

```bash
bash test.sh --model_size 7b
bash test.sh --model_size 14b --ndevice 1
```

也可以按 backend 分别配置：

```json
"test_sh_params": {
  "xh1": [
    ["--model_size", "7b"]
  ],
  "xh2": [
    ["--model_size", "7b"],
    ["--model_size", "14b"]
  ]
}
```

此外，还支持与其他模型参数一致的按参数列配置方式：

```json
"test_sh_params": {
  "xh2": {
    "model_size": ["7b", "14b"],
    "use_cache": [true, false],
    "ndevice": ["0", "1"]
  }
}
```

该配置同样按数组索引组合参数，对应执行：

```bash
bash test.sh --model_size 7b --ndevice 0
bash test.sh --model_size 14b --ndevice 1
```

按参数列配置时：

- 普通值生成 `--参数名 参数值`
- `true` 生成不带值的布尔开关 `--参数名`
- `false` 或 `null` 跳过该参数
- 已经以 `-` 开头的参数名会直接使用，不再自动添加 `--`

未配置 `test_sh_params`，或者将其配置为空数组时，保持原有行为，只执行一次默认命令：

```bash
bash test.sh
```

#### 4.5.2 控制是否执行 Python demo 测试

`enable_demo_test` 用于控制 `test.sh` 完成后是否继续执行 Python demo 测试，默认值为 `true`：

```json
"enable_demo_test": true
```

如只需执行一组或多组 `test.sh`，可配置：

```json
"test_sh_params": [
  ["--model_size", "7b"],
  ["--model_size", "14b"]
],
"enable_demo_test": false
```

设置为 `false` 后，框架执行完所有 `test.sh` 并检查结果，随后清理临时目录并结束当前 flow，不再准备编译模型、安装 Python 虚拟环境或执行 `demo.py` / `demo_multibatch.py` 等后续 Python demo 测试。

## 5. 环境与前置要求

### 5.1 基本要求

- 当前仅支持 `Linux`
- 测试框架依赖 `pytest`
- 部分用例使用 `@pytest.mark.dependency(...)`，因此运行环境应具备对应 pytest 插件能力
- 如果模型 flow 依赖 `hmatc` / `hm_smi` / 芯片 runtime，需要先完成整套后摩环境初始化

建议先在仓库根目录完成环境初始化，例如：

```bash
cd imodelzoo
source env.sh
```

### 5.2 测试侧会自动设置的环境变量

`tests/conftest.py` 会自动设置：

- `HOUMO_MODELZOO_URL=http://artifactory.houmo.ai/artifactory/Dadao`
- `HOUMO_DATASETS_PATH=<repo>/data/datasets/`
- `HOUMO_VERSION=2.4.2`（若外部未设置）
- `LD_LIBRARY_PATH` 会附加 runtime / torch / onnxruntime 相关路径

### 5.3 关键运行环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `HOUMO_TARGET` | `xh2` | 当前 backend，影响配置选择、缓存目录和命令参数 |
| `HDPL_PLATFORM` | 空 | 区分 `ASIC` / `ISIM`，影响是否跑真实推理 |
| `SKIP_INFER` | 空 | 设置为 `ON` 或 `OFF` 会启用分阶段模式 |
| `IMODELZOO_MODELS_PATH` | `tests/models_<HOUMO_TARGET>/` | 原始模型缓存根目录 |
| `USE_RELEASED_MODELS` | `ON` | 是否走 release 模式 |

## 6. 日志、输出与结果判定

### 6.1 日志目录

每条测试会生成独立日志，路径格式如下：

```text
tests/test_logs/YYYYMMDD/<module>_<test_name>_<timestamp>.log
```

### 6.2 结果检查逻辑

当前框架并不是只看返回码，还会做额外结果解析：

- `compile`: 解析表格中的 cosine / golden 结果
- `compare`: 解析 `onnx vs hmquant` 表格，并按阈值校验
- `eval`: 解析评测输出字段，再与 `eval_threshold` 比较
- `perf`: 读取性能结果，与 `perf_metrics` 基线比较

如果命令失败，框架还会尝试执行芯片 reset。

## 7. 如何执行测试

### 7.1 跑全部模型测试

```bash
cd imodelzoo/tests
pytest -s -v models_tests/
```

### 7.2 按模型筛选

```bash
cd imodelzoo/tests
pytest -s -v models_tests/ -m "resnet50"
pytest -s -v models_tests/ -m "qwen2dot5"
pytest -s -v models_tests/ -m "deepseek_r1_qwen3_8b"
```

### 7.3 按 flow 筛选

```bash
cd imodelzoo/tests
pytest -s -v models_tests/ -m "get_model"
pytest -s -v models_tests/ -m "compile"
pytest -s -v models_tests/ -m "perf"
pytest -s -v models_tests/ -m "eval"
```

### 7.4 组合筛选

```bash
cd imodelzoo/tests
pytest -s -v models_tests/ -m "resnet50 and perf"
pytest -s -v models_tests/ -m "ndevice_2 and dev_mem_24g"
pytest -s -v models_tests/ -k "_llm_"
```

### 7.5 指定 backend

```bash
cd imodelzoo/tests
HOUMO_TARGET=xh2 pytest -s -v models_tests/ -m "qwen3"
```

### 7.6 分阶段执行示例

第一阶段：在编译/量化侧执行（例如 `ISIM` 环境）

```bash
cd imodelzoo/tests
HOUMO_TARGET=xh2 HDPL_PLATFORM=ISIM SKIP_INFER=ON pytest -s -v models_tests/
```

第二阶段：在真实推理侧执行（例如 `ASIC` 环境）

```bash
cd imodelzoo/tests
HOUMO_TARGET=xh2 HDPL_PLATFORM=ASIC SKIP_INFER=ON pytest -s -v models_tests/
```

说明：分阶段模式下，实际执行/跳过哪些步骤由 `test_models_utils.py` 内部逻辑决定，不同模型可能因配置不同而自动 `skip`。

## 8. 模型配置文件字段说明

### 8.1 通用字段

| 字段 | 是否常用 | 说明 |
| --- | --- | --- |
| `obsolete` | 必需 | 是否废弃；为 `true` 时整模型跳过 |
| `model_type` | 常用 | 当前代码主要区分 `cv` / `llm` |
| `model_dir` | 必需 | 模型目录，相对仓库根目录 |
| `dependencies` | 必需且重要 | 资源依赖，当前主要用到 `ndevice`、`dev_mem` |
| `support_platform` | 必需 | 支持的平台，如 `x86_64`、`aarch64` |
| `support_backend` | 必需 | 支持的 backend，如 `xh1`、`xh2` |
| `support_core_num` | 常用 | demo / hmm 相关的核心数限制 |
| `support_flow` | 必需 | 每个 backend 支持哪些测试 flow |
| `support_hmatc` | 可选 | 支持哪些 `hmatc` 子命令 |
| `perf_metrics` | `perf` 必需 | 性能基线 |
| `eval_threshold` | `eval` 必需 | 精度阈值 |

### 8.2 获取与准备类字段

| 字段 | 说明 |
| --- | --- |
| `get_model_params` | `get_model.py` 的测试参数矩阵 |
| `quant_params` | Python 量化脚本 `ptq.py` 的参数矩阵 |
| `compile_params` | Python 编译脚本 `build.py` 的参数矩阵 |
| `demo_params` | `demo.py` 的参数矩阵 |
| `demo_multibatch_params` | `demo_multibatch.py` 的参数矩阵；在 `demo` flow 内部顺带执行 |
| `test_sh_params` | ASIC 环境下 `test.sh` 的参数组；支持参数数组、按 backend 配置或按参数列配置；缺省时执行一次无额外参数的 `test.sh` |
| `enable_demo_test` | CI环境中，`test.sh` 后是否继续执行 Python demo 测试，默认为 `true`；设为 `false` 时仅检查 `test.sh` 并结束当前 flow |
| `perf_params` | Python 侧性能测试参数；部分 LLM 会复用 `demo` |

### 8.3 `hmatc` 类字段

| 字段 | 对应命令 |
| --- | --- |
| `hmquant_params` | `hmatc quant` |
| `hmbuild_params` | `hmatc build` |
| `hmdemo_params` | `hmatc demo` |
| `hmcompare_params` | `hmatc compare` |
| `hmeval_params` | `hmatc eval` |
| `hmperf_params` | `hmatc perf` |

### 8.4 参数矩阵规则

无论是 `hmatc` 还是 Python 脚本，当前实现都遵循“按索引并排组合参数”的规则：

- 第 `0` 组参数组成第 `0` 条命令
- 第 `1` 组参数组成第 `1` 条命令
- 值为 `null` 或 `default` 的参数会被跳过，表示使用脚本默认值
- 路径里写成 `cached_models` / `cached_results` 时，运行时会替换成真实缓存目录

## 9. 新增模型如何接入测试

### 9.1 新增配置文件

在 `tests/models_tests/model_configs/` 下新增：

```text
model_cfg_<模型名>.json
```

推荐从模板复制并修改：

- `model_cfg_template_cv.json`
- `model_cfg_template_llm.json`

### 9.2 生成 pytest 入口

```bash
cd imodelzoo/tests/models_tests
python3 update_test_py.py
```

脚本会：

- 扫描新增的配置文件
- 自动把模型名写入 `model_names.txt`
- 按支持的 flow 追加到对应 `test_*_models.py`
- 自动追加 `ndevice_*` / `dev_mem_*` marker

注意：

- `demo_multibatch` 不会生成独立 pytest 用例
- 模型名会被转换为 marker 名，规则见上文

### 9.3 运行新增模型测试

```bash
cd imodelzoo/tests
pytest -s -v models_tests/ -m "<转换后的模型名>"
```

## 10. 排查建议

如果用例和预期不一致，优先检查以下几项：

1. `HOUMO_TARGET` 是否与模型配置匹配
2. `HDPL_PLATFORM` 是否正确，是否意外进入分阶段模式
3. `dependencies` 对应的 `ndevice_*` / `dev_mem_*` marker 是否正确
4. `model_results_{HOUMO_TARGET}` 中是否已有旧缓存结果
5. 当前模型是走 `hmatc` 还是走 Python 脚本路径
6. 对 LLM，当前机器是否具备 GPU，且是否处于 release 模式
