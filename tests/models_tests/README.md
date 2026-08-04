# `models_tests` 模型回归测试说明

`tests/models_tests` 是由模型 JSON 驱动的回归测试框架。pytest 测试函数负责选择模型和 flow，具体的模型下载、量化、编译、artifact 复用、demo、精度和性能校验由 flow handler 完成。

按使用目标阅读即可：

- 运行现有模型：阅读“快速开始”和“运行环境”；
- 新增或修改模型：阅读“核心 flow”和“JSON 配置”；
- 使用 no-infer/infer：阅读“运行模式、缓存和 artifact”；
- 测试失败：阅读“日志和排错”；
- 维护框架内部代码：阅读 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 快速开始

以下命令从仓库根目录 `/data02/services/imodelzoo` 执行，并提前按项目要求加载运行环境。

```bash
# 运行全部模型功能测试
pytest -s -v tests/models_tests/

# 按模型或 flow marker 筛选
pytest -s -v tests/models_tests/ -m "sam2"
pytest -s -v tests/models_tests/ -m "demo"
pytest -s -v tests/models_tests/ -m "sam2 and demo"

# 按常用设备资源 marker 筛选
pytest --log-cli-level=INFO -s \
  -m "imodelzoo and ndevice_1 and dev_mem_12g"

# 生成测试入口，或只检查 JSON 与生成文件是否一致
python -m tests.models_tests.update_test_py
python -m tests.models_tests.update_test_py --check

# 只检查收集结果
pytest tests/models_tests --collect-only -q
```

测试 flow 的默认收集顺序是：

```text
get_model, quant, compile, demo, compare, eval, perf
```

该顺序主要用于稳定 pytest 展示和 producer flow 的执行顺序，不表示所有 flow 组成一条强制依赖链。`demo`、`compare`、`eval` 和 `perf` 会由各自 handler 准备或复用所需 artifact。

当前不支持 `pytest -n 2` 等多个 xdist worker 并发执行，因为不同 flow 可能共享模型 cache 和 artifact。不要使用 xdist，或仅使用一个 worker。

## 2. 核心 flow

CV 注册七类 flow；LLM 当前注册 `get_model`、`quant`、`compile`、`demo` 和 `perf`。`demo_multibatch` 是 `demo` 的附加 case，不生成独立 pytest 文件。

| flow | family | Python/模型脚本配置 | HMATC 配置 | pytest 静态依赖 | 主要职责 |
| --- | --- | --- | --- | --- | --- |
| `get_model` | CV/LLM | `get_model_params` → `get_model.py` | 无 | 无 | 下载 raw、quant、HMM 或配套资源。 |
| `quant` | CV/LLM | `quant_params` → `ptq.py` | `hmquant_params` | 支持 get_model 时依赖 get_model | 生成量化模型。 |
| `compile` | CV/LLM | `compile_params` → `build.py` | `hmbuild_params` | 优先依赖 quant，否则依赖 get_model | 生成 compiled HMM。 |
| `demo` | CV/LLM | `demo_params` → `demo.py` 或自定义脚本 | `hmdemo_params` | 无 | 可选执行 `test.sh`，再执行标准 demo。 |
| `compare` | CV | 无通用 Python runner | `hmcompare_params` | 无 | 校验 HMATC Cosine Distance。 |
| `eval` | CV | 无通用 Python runner | `hmeval_params` | 无 | 比较 ONNX 与 HM 精度指标。 |
| `perf` | CV/LLM | `perf_params: "demo"` 或代码侧 custom runner | `hmperf_params` | 无 | 解析并校验性能指标。 |

一个模型的不同 flow 可以自由混合 Python 脚本和 HMATC。例如 SAM2 可以配置为：

```text
get_model -> get_model.py
quant     -> HMATC
compile   -> HMATC
demo      -> demo.py
```

不需要增加“混合模式”开关。`support_flow` 决定是否生成和运行对应 pytest flow；实际 runner 由相应的 `hm*params` 或普通 `*_params` section 决定。

`support_hmatc` 仅保留为兼容性能力描述，不是 runner 开关。新增配置时不要依赖它决定执行方式。

## 3. 运行环境

### 3.1 环境变量

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `HOUMO_TARGET` | `xh2` | 当前 backend。默认值会同步到父进程环境，模型子进程也能读取到。 |
| `USE_RELEASED_MODELS` | `ON` | `ON/on` 为 release 模式，其他值为 development 模式。 |
| `SKIP_INFER` | 未设置 | `ON` 或 `OFF` 都表示启用 separate；当前机器是否为 ASIC 决定 no-infer 或 infer 阶段。 |
| `IMODELZOO_MODELS_PATH` | `tests/models_<backend>` | `cached_models` 的物理根目录。 |
| `HOUMO_EXAMPLES_PATH` | 仓库根目录 | 查找 `hm_gptq` requirements 的基准路径。 |
| `HOUMO_DATASETS_PATH` | `<repo>/data/datasets` | 数据集路径，也用于解析 quant 的 `prerequisites.py_reqs`。 |
| `HOUMO_FULL_DATASET` | 未设置 | 设置后 eval 使用完整阈值；未设置时阈值乘 `0.5`。 |
| `IMODELZOO_ALLOW_XH2_NCORE4` | `OFF` | `ON` 时允许默认被过滤的 xh2 `ncore=4` HMATC build case。 |
| `IMODELZOO_MIRROR_COMMAND_OUTPUT` | `ON` | 是否把模型命令输出在写日志的同时镜像到 pytest 终端。 |

`cached_results` 当前没有独立环境变量，物理根目录固定为：

```text
tests/model_results_<backend>
```

### 3.2 release 模式

当前默认是 release 模式，主要差异如下：

- LLM `get_model` 过滤 `type=raw`，所有 family 都过滤 `source_type=modelscope` 的 release case；
- LLM Python quant 仅在 development + GPU 执行；
- LLM compile 在 release 下 skip，CV compile 不因此 skip；
- inference 可将匹配的 `get_model type=hmm` 下载映射到最终 `cached_results` case；目标目录有效时不会重复下载；
- `enable_demo_test=false` 只在 release 下阻止 `test.sh` 之后的标准 demo，development 仍执行标准 demo；
- perf 最低通过比例：development 为基线的 `95%`，release 为 `10%`。

例如在命令前设置 `USE_RELEASED_MODELS=OFF` 可切换到 development 模式。

## 4. 运行模式、缓存和 artifact

### 4.1 default、no-infer 和 infer

| 条件 | 模式 | 行为 |
| --- | --- | --- |
| `SKIP_INFER` 未设置或不是 `ON/OFF` | `DEFAULT` | 当前机器准备 artifact 并执行完整 flow。 |
| `SKIP_INFER=ON/OFF`，当前不是 ASIC | `SEPARATE_NO_INFER` | 执行 producer flow；inference flow 只准备和持久化 artifact，不执行推理，并以 prepared/skip 结束。 |
| `SKIP_INFER=ON/OFF`，当前是 ASIC | `SEPARATE_INFER` | get_model/quant/compile skip；恢复 no-infer artifact 后执行 demo、perf 等推理 flow。 |

`ON` 和 `OFF` 都只是兼容旧接口的“启用 separate”值，不分别代表两个阶段。

### 4.2 两个逻辑缓存

JSON 命令参数只写逻辑路径，框架在执行前替换为当前模型的物理目录：

```text
cached_models/<case-or-resource>
cached_results/<case>
```

| 逻辑路径 | 典型内容 | producer | consumer |
| --- | --- | --- | --- |
| `cached_models` | raw ONNX、tokenizer、Hugging Face 资源、get_model 对 workspace 的副作用文件 | get_model | quant、HMATC、demo |
| `cached_results` | 量化结果、compiled HMM、HMATC inference bundle | quant、compile、inference preparation | demo、compare、eval、perf |

例如配置：

```json
{
  "model_dir": "models/llm/example",
  "dependencies": {
    "ndevice": [1],
    "dev_mem": ["12g"]
  }
}
```

默认对应：

```text
cached_models
  -> tests/models_<backend>/models/llm/example

cached_results
  -> tests/model_results_<backend>/ndevice_1_dev_mem_12g/models/llm/example
```

路径替换按完整路径段执行，不会误替换 `old_cached_models` 这类只包含相同字符串的名称。

如果 Python `compile_params.output_dir` 位于 `cached_models/<case>`，框架会把有效 HMM 镜像到 `cached_results/<case>`。infer 阶段最终校验和消费 `cached_results`。

### 4.3 separate 阶段的数据流

no-infer 保存 raw get_model 产生或修改的 workspace 文件，并持久化 quant/compiled artifact；infer 在新 workspace 中按原相对目录恢复。HMATC compiled bundle 即使可复用，框架仍会确保 raw artifact 可用；是否真正执行 `get_model.py` 取决于 raw cache 状态。

infer 执行 demo/perf 前还会检查参数中引用的 `cached_models/<folder>`：

1. 目录存在并包含非空 artifact 文件时直接复用；
2. 目录缺失或为空时，根据同一参数 case 的模型规格匹配 `get_model_params.type=hmm`；
3. 只执行匹配的 HMM get_model case；
4. 下载后再次校验引用目录，仍缺失时报告具体路径。

`cached_models` 引用通常是 tokenizer 或 Hugging Face 资源，与 `cached_results` 中的推理 HMM 是两种需求。

### 4.4 cache 有效性

artifact 目录可以带有类型和 case 对应的 manifest。框架会校验 artifact identity、配置 fingerprint 和 required files，再决定复用、兼容采用或重新生成。

Python quant/build、HMM 镜像和部分下载发布使用 staging + backup 原子发布；失败不会直接覆盖旧的有效目录。manifest 字段、fingerprint 和恢复机制见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 5. JSON 配置

配置文件位于：

```text
tests/models_tests/model_configs/model_cfg_<model_name>.json
```

文件名中的 `<model_name>` 决定传给测试框架的模型名和模型 marker。新增模型应从 CV 或 LLM 模板复制，不要从零编写：

```text
model_cfg_template_cv.json
model_cfg_template_llm.json
```

### 5.1 顶层公共字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `obsolete` | boolean | `true` 时保留生成的测试函数和 marker，但运行时所有声明 flow 都 skip。 |
| `model_dir` | string | 相对仓库根目录的模型源码路径；不能是绝对路径或包含 `..`。 |
| `model_type` | `cv` / `llm` | 模型 family。缺省按 `cv` 解析，新配置应显式填写。 |
| `dependencies` | object | 资源 marker，目前使用 `ndevice`、`dev_mem` 数组；生成器读取每个数组的第一个值。 |
| `support_platform` | string[] | 支持的平台，例如 `x86_64`、`aarch64`。 |
| `support_backend` | string[] | 支持的 backend，当前为 `xh1`、`xh2`。 |
| `support_core_num` | object | backend 到 core 数量列表或 null；用于 aarch64 demo 设备检查。 |
| `support_flow` | object | backend 到 flow 列表；合法值为七类 flow 和附加的 `demo_multibatch`。 |
| `support_hmatc` | object/null | 兼容性能力描述，可列 `hmquant/hmbuild/hmdemo/hmcompare/hmeval/hmperf`；不决定 runner。 |
| `validation` | object | 可选 backend 阈值覆盖；支持 `compile_cosine_threshold` 和 `compare_cosine_threshold`，值在 `[0,1]`。 |
| `perf_metrics` | object | backend/platform 对应的性能基线；声明 perf 时必须能解析出非空数字指标。 |
| `eval_threshold` | object/null | eval 指标名到 HM/ONNX 最低比例；声明 eval 时必须为对象。 |
| `enable_demo_test` | boolean | release 模式下，执行 `test.sh` 后是否继续标准 demo，默认 `true`。 |
| `test_sh_params` | array/object | `test.sh` 参数，格式见“test.sh”小节。 |

### 5.2 flow 必需配置

active 配置加载时会检查：

| `support_flow` | 至少需要 |
| --- | --- |
| `get_model` | 当前 backend 的 `get_model_params` |
| `quant` | `hmquant_params` 或当前 backend 的 `quant_params` |
| `compile` | 当前 backend 的 `hmbuild_params` 或 `compile_params` |
| `demo` | `hmdemo_params` 或当前 backend 的 `demo_params` |
| `demo_multibatch` | 当前 backend 的 `demo_multibatch_params` |
| `compare` | `hmcompare_params`，且 family 必须为 CV |
| `eval` | `hmeval_params` + `eval_threshold`，且 family 必须为 CV |
| `perf` | `hmperf_params`、`perf_params: "demo"` 或代码侧 custom runner，同时需要 `perf_metrics` |

### 5.3 Python 和 get_model 参数

这些 section 通常使用：

```text
section -> backend -> 列式参数对象
```

| section | 含义 |
| --- | --- |
| `get_model_params` | 传给 `get_model.py`；常用 `type`、`model_dir`、`download_dir`、`extract_dir`、`quant_model_dir`、`build_model_dir`、`source_type`。 |
| `quant_params` | 传给 `ptq.py`；特殊键 `prerequisites` 不渲染为命令参数。 |
| `compile_params` | 传给 `build.py`；常用 `model_dir`、`output_dir`、`stage`、`batch`、`ncore`、`ndevice`、`context_length`、`j`、`model_size`。 |
| `demo_params` | 传给 demo；`script` 可覆盖默认 `demo.py`，且不会渲染为 `--script`。 |
| `demo_multibatch_params` | 传给 multibatch demo；默认脚本为 `demo_multibatch.py`。 |
| `perf_params` | 通用值为字符串 `"demo"`，表示复用当前 backend 的第一个 `demo_params` case。 |

quant 可额外配置依赖：

```json
"prerequisites": {
  "hm_gptq": true,
  "py_reqs": ["path/to/extra-requirements.txt"]
}
```

requirements 搜索和安装顺序见 [ARCHITECTURE.md](ARCHITECTURE.md)。

### 5.4 HMATC 参数

quant/demo/compare/eval/perf 使用共享结构：

```json
"hmquant_params": {
  "params": {
    "required": {},
    "optional": {}
  }
}
```

对应 section 为：

```text
hmquant_params
hmdemo_params
hmcompare_params
hmeval_params
hmperf_params
```

HMATC build 按 backend 配置，少一层 `params`：

```json
"hmbuild_params": {
  "xh2": {
    "required": {},
    "optional": {}
  }
}
```

`required` 和 `optional` 会合并为一个参数矩阵，所有列表列长度必须一致。框架会补充 `hmatc <subcommand> --target <backend>`，并跳过 `target`、`onnx` 等由 handler 内部处理的字段。

### 5.5 参数矩阵规则

例如：

```json
{
  "model_dir": ["cached_results/q1", "cached_results/q2"],
  "ncore": [1, 2],
  "fast": [false, true],
  "stage": ["build", "default"]
}
```

生成：

```text
case 0: --model_dir <q1真实路径> --ncore 1 --stage build
case 1: --model_dir <q2真实路径> --ncore 2 --fast
```

规则如下：

- 同一列索引组成同一个 case；
- 所有列表列必须等长且不能为空；
- `null` 和字符串 `"default"` 省略该选项；
- boolean `true` 生成无值 flag，`false` 省略；
- 其他值生成 `--<JSON key> <value>`，key 不自动转换下划线或短横线；
- `cached_models`、`cached_results` 在渲染前替换为真实路径；
- `prerequisites`、`script` 和部分 HMATC 内部字段由对应 handler 跳过。

## 6. 高级流程

### 6.1 HMATC inference bundle 和多 YAML

存在 `hmbuild_params` 的 demo/perf/eval/compare 会准备 HMATC inference bundle：

1. 从 `hmquant_params` 和当前 backend 的 `hmbuild_params` 收集所有 `config`；
2. 去重后对每个 YAML 执行 inference 所需的 HMATC quant/build；
3. YAML 必须位于 workspace 内；
4. 每个 YAML 必须声明非空且相同的 `model.save_dir`；当前框架只接受 workspace 内相对路径；
5. 最终 bundle 位于 `<model.save_dir>/<backend>`，要求存在非空 `.hmm` 或 `.hmms`，以及 `hmquant/*with_act.onnx`；
6. 有效 bundle 可以通过 cache 和 manifest 复用。

当前框架仍采用 workspace-relative 的 `model_path/save_dir` 布局，并通过 cache 复制完成 separate 阶段恢复。这是当前框架实现约束，不代表 HMATC 工具本身不支持自定义输入和输出路径。

### 6.2 `test.sh`

`test.sh` 是 demo flow 中可选的端到端功能检查：

- 仅在 ASIC 且模型目录存在 `test.sh` 时执行；
- SEPARATE_NO_INFER 不执行，DEFAULT ASIC 和 SEPARATE_INFER 执行；
- 使用独立 workspace，不替代拆分的 get_model/quant/compile；
- `test.sh` 后标准 demo 会独立准备或恢复 artifact；
- 两个阶段的失败最终合并到同一个 demo flow；
- `enable_demo_test=false` 仅在 release 下阻止后续标准 demo。

`test_sh_params` 支持完整 argv 列表，每个内层数组是一条命令：

```json
"test_sh_params": [
  [], ["--model_size", "1.7b"], ["--demo_mode", "streaming"]
]
```

也可以按 backend 包装或使用列式对象：

```json
"test_sh_params": {"xh2": [[]]}

"test_sh_params": {"model_size": ["0.6b", "1.7b"],
                   "streaming": [false, true]}
```

空值或未配置时等价于执行一次无额外参数的 `bash test.sh`。

### 6.3 compare、eval 和 perf 校验

compare 从输出中解析 `Cosine Distance` 表格。第一列必须为 `name`，且至少包含两个 `X vs Y` 指标列；框架不匹配固定的 `onnx/hmquant/hmonnx/hmm` 列名，而是校验所有 output 行和实际指标。缺表、列数不足、非数字或任意值低于阈值都会失败。默认阈值 xh1=`1.0`、xh2=`0.90`，可由 `validation` 覆盖。

eval 对每个 case 分别执行 ONNX 和 HM，并要求 `hm_metric >= onnx_metric * eval_threshold`；未设置 `HOUMO_FULL_DATASET` 时配置阈值再乘 `0.5`。

perf 默认指标为：

| JSON 指标 | 默认日志 key | 方向 | 聚合 |
| --- | --- | --- | --- |
| `qps` | `[Throughput] qps` | 越高越好 | max |
| `prefill` | `Prefill Speed` | 越高越好 | max |
| `decode` | `Decode Speed` | 越高越好 | max |
| `end2end` | `E2E TPS` | 越高越好 | max |

特殊日志、lower-is-better 或 custom runner 属于代码 policy，不在 JSON 中扩展为执行 DSL。

## 7. 新增或修改用例

### 7.1 新增模型

1. 复制 `model_cfg_template_cv.json` 或 `model_cfg_template_llm.json`；
2. 保存为 `model_cfg_<model_name>.json`；
3. 设置 `model_dir`、`model_type`、资源依赖、platform、backend 和 `support_flow`；
4. 为每个 flow 添加第 5.2 节要求的参数 section；
5. 对齐所有参数列表的 case 数量；
6. cache 参数使用 `cached_models/...` 或 `cached_results/...`，不要写机器绝对路径；
7. 重新生成并检查：

   ```bash
   python -m tests.models_tests.update_test_py
   python -m tests.models_tests.update_test_py --check
   pytest tests/models_tests --collect-only -q
   ```

8. 在目标 backend/platform 上执行定向 smoke。

### 7.2 修改现有用例

- 只修改命令参数但不改变 `support_flow` 时，pytest 函数通常不变，仍建议执行 `--check`；
- 增删 `support_flow` 后必须重新运行生成器；
- `obsolete=true` 不会删除测试函数和 marker，运行时只会 skip；
- JSON 与 `test_<flow>_models.py` 不一致时，普通执行 `update_test_py.py` 会按全部配置全量更新；
- `--check` 只报告漂移，不修改文件；
- 不要手工修改带有 `Generated by update_test_py.py. Do not edit manually.` 的测试文件；
- 不要把 flow dependency、perf regex、指标方向、聚合方式或通用失败词例外写进 JSON，这些属于代码 policy。

## 8. 目录和生成文件

主目录只需要区分以下几类：

| 路径 | 功能 |
| --- | --- |
| `model_configs/` | 模型 JSON 和 CV/LLM 模板。 |
| `test_*_models.py` | 自动生成的 pytest 入口，不手工修改。 |
| `update_test_py.py` | 全量生成入口文件和 `model_names.txt`；`--check` 检查漂移。 |
| `test_models_utils.py` | pytest 与 flow framework 的调用边界。 |
| `test_flows/` | get_model、quant、compile、demo、compare、eval、perf 的业务编排。 |
| `model_workflow/` | 配置、contract、参数、cache、artifact、环境和指标等基础机制。 |
| `model_names.txt` | 自动生成的模型 marker 列表，包含 active 和 obsolete 模型。 |

详细文件职责、调用链和模块边界见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 9. 日志和排错

模型命令的 stdout/stderr 会实时读取、写入当前 pytest case 日志，并默认镜像到 pytest 进程。终端要实时显示镜像内容，还需要使用 `-s` 或 `--capture=tee-sys`。

推荐命令：

```bash
pytest --log-cli-level=INFO -s \
  -m "imodelzoo and ndevice_1 and dev_mem_12g"
```

完整日志位于 `tests/test_logs/<date>/`。

如果只希望写日志、不镜像到终端，设置 `IMODELZOO_MIRROR_COMMAND_OUTPUT=OFF`。

常见问题：

| 现象 | 优先检查 |
| --- | --- |
| 模型 import 时断言 `HOUMO_TARGET` | 当前 backend 和子进程环境；未设置时应自动同步为 `xh2`。 |
| `compiled artifact case ... is missing` | `compile_params.output_dir`、最终 `cached_results/<case>`、manifest 和非空 `.hmm/.hmms`。 |
| `cached_models` 资源目录缺失 | demo 参数引用、匹配的 `get_model type=hmm` case，以及 get_model 是否实际生成目标目录。 |
| HMATC 找不到 ONNX/YAML | raw artifact 是否缓存和恢复、infer 是否使用相同 cache、YAML 是否位于 workspace 内。 |
| HMATC bundle 不复用 | YAML 内容/路径、backend、`model.save_dir`、fingerprint 和 required files 是否变化。 |
| pytest 只显示 venv 日志 | 使用 `-s` 或 `--capture=tee-sys`，确认 `IMODELZOO_MIRROR_COMMAND_OUTPUT` 未关闭，再检查完整 case log。 |
| 所有 xh2 compile case 被跳过 | 是否只有 `ncore=4`；需要时设置 `IMODELZOO_ALLOW_XH2_NCORE4=ON`。 |
| eval 阈值比预期宽松 | 未设置 `HOUMO_FULL_DATASET` 时阈值会乘 `0.5`。 |
| release 下 LLM quant/compile skip | 当前 policy 如此；将 `USE_RELEASED_MODELS` 设置为非 `ON/on` 进入 development。 |

提交前至少执行：

```bash
python -m tests.models_tests.update_test_py --check
pytest tests/models_tests --collect-only -q
pytest -q tests/unit_tests
git diff --check
```

排错时依次确认 runtime 环境、日志中的 `model/backend/flow/case_id/phase`、逻辑 cache 路径和 manifest 状态。不要为了排错直接删除整个缓存根目录。
