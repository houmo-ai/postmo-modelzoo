# `unit_tests` 测试框架单元测试说明

本目录测试 `tests/` 下 API、HMATC 和模型测试框架自身的逻辑。这里的用例应保持轻量，不依赖加速卡、已下载模型、外部数据集或预先生成的模型产物。

## 1. 收集和执行方式

`unit_tests` 默认不参与 `pytest tests/` 的递归收集，避免与设备功能测试混合执行。可以显式指定目录：

```bash
pytest -q tests/unit_tests
```

也可以从完整测试树中通过 `unit` marker 选择：

```bash
pytest -q tests -m unit
pytest -q tests -m "unit and not slow"
pytest -q tests -m "unit or compare"
```

只要 marker 表达式中引用了独立的 `unit` marker，根 `conftest.py` 就允许收集本目录；最终的组合表达式筛选仍由 pytest 完成。

还可以只执行某个域或文件：

```bash
pytest -q tests/unit_tests/apis
pytest -q tests/unit_tests/hmatc
pytest -q tests/unit_tests/models
pytest -q tests/unit_tests/models/test_compare_flow.py
```

本目录的 [conftest.py](conftest.py) 覆盖功能测试的 autouse 日志 fixture，避免轻量单元测试为每个 case 创建 `test_logs` 文件。`tests/models_tests/history_codes/` 则始终由根 `tests/conftest.py` 排除，不参与任何 pytest 递归收集。

## 2. 根目录文件

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 将 `unit_tests` 声明为 Python package。 |
| `conftest.py` | 覆盖功能测试的 `setup_logging` fixture，使单元测试不创建逐用例日志文件。 |
| `test_collection_policy.py` | 测试根 pytest 收集策略对显式 unit 路径、单 marker 和组合 marker 表达式的识别。 |
| `README.md` | 说明单元测试目录、文件职责、收集规则和执行方式。 |

## 3. `apis/`：API 测试框架单元测试

| 文件 | 功能 |
| --- | --- |
| `apis/__init__.py` | 将 API 单元测试目录声明为 Python package。 |
| `apis/test_api_runner.py` | 测试 API runner 的命令成功判定、stdout/stderr legacy 失败标记处理，以及 C++ API demo 的 CMake 配置、构建和安装命令。 |

## 4. `hmatc/`：HMATC 测试框架单元测试

| 文件 | 功能 |
| --- | --- |
| `hmatc/__init__.py` | 将 HMATC 单元测试目录声明为 Python package。 |
| `hmatc/test_hmatc_runner.py` | 测试独立 HMATC runner 的 `CommandSpec` 属性、return code/legacy stdout 失败判定、quant/build/demo/compare/eval/perf 参数生成、xh1/xh2 perf `ncore`、失败短路、配置排序、功能矩阵聚合和 xh1-only perf 流程。 |

## 5. `models/`：模型测试框架单元测试

模型测试按生产代码组件拆分。测试脚本应直接导入对应的生产模块；共享 support 只保存跨文件复用的 builder、测试产物生成函数和源码目录常量，不作为生产模块的转发 facade。

| 文件 | 功能 |
| --- | --- |
| `models/__init__.py` | 将模型单元测试目录声明为 Python package。 |
| `models/_flow_contract_support.py` | 提供 `TESTS_DIR`、`MODELS_TESTS_DIR`、`CONFIG_DIR`，以及 demo/HMATC request builder、HMATC 测试配置和产物生成函数、Python import 扫描 helper；文件名不以 `test_` 开头，不会被 pytest 当作测试模块。 |
| `models/test_artifact_cache_store.py` | 测试 artifact manifest、fingerprint、required files、文件角色扫描、原子 staging/commit/backup/recovery，以及 compiler 中间产物清理。 |
| `models/test_artifact_preparation.py` | 测试 separate-infer 阶段已有 HMATC raw ONNX 的恢复、缺失 raw 模型时执行 get_model，以及下载后仍缺失时的错误诊断。 |
| `models/test_artifact_preparer.py` | 测试 inference compile 失败后的 release HMM 回退、raw workspace side effects 恢复，以及已有 quant input 的复用。 |
| `models/test_compare_flow.py` | 测试 `Cosine Distance` 表格结构解析、任意两个 `X vs Y` 列、列重排、多 output、多种 name 格式、状态列、科学计数法、ANSI/时间前缀和阈值失败。 |
| `models/test_compile_flow.py` | 测试 Python compile HMM 发布、model-cache 输出所有权、demo mirror、输入缺失错误和 xh2 `ncore=4` case 过滤。 |
| `models/test_demo_flow.py` | 测试 demo artifact 引用分析、release HMM 映射/下载/mirror、`test.sh` 参数和失败词兼容、virtualenv 传递及 demo workspace 生命周期。 |
| `models/test_flow_architecture.py` | 测试 flow registry/policy seam、模块依赖方向、共享参数 renderer、quant/compile 拆分、pytest 入口签名、显式导出以及禁止恢复 legacy 执行模式。 |
| `models/test_get_model_flow.py` | 测试 LLM release get_model case 过滤、HMM manifest 发布、CV workspace side-effect 生命周期和 LLM side-effect 恢复。 |
| `models/test_hmatc_flow_support.py` | 测试 separate workspace 持久化/恢复、默认与多组件 HMATC config、inference bundle manifest 复用、legacy adoption 和配置变化重建。 |
| `models/test_hmatc_v2_flow.py` | 测试显式 HMATC v2 schema、YAML 深度合并和只读源配置、递归 cache 路径、quant/build artifact 发布与复用、按 config 选择 raw get_model、build 环境变量和 `.pt`/`hf_config` sidecar 复制。 |
| `models/test_metric_validation.py` | 测试默认/custom perf 指标提取、lower-is-better 延迟、HMATC QPS 聚合、eval 部分数据集系数，以及 compile/compare 阈值。 |
| `models/test_model_config_repository.py` | 测试全部模型 JSON 加载、active flow handler 注册、固定 backend policy、模型阈值覆盖、必需参数 section 和禁止 JSON 控制代码侧 perf 规则。 |
| `models/test_parameter_matrix.py` | 测试参数矩阵列长度、命令行参数渲染、`cached_models`/`cached_results` 路径替换和 cache case reference 解析。 |
| `models/test_python_environment.py` | 测试 virtualenv 环境继承、requirements 安装顺序、pip timeout/retry、activated 环境命令和 `py_reqs` workspace/dataset 查找规则。 |
| `models/test_quant_flow.py` | 测试 quant 原子 staging、CV quant 结构化命令与 artifact 发布，以及 HMATC quant 对 legacy 失败关键词的兼容。 |
| `models/test_runtime_and_command.py` | 测试 `CommandResult`、diagnostic context、模型命令默认超时、跨 suite 命令超时隔离、runtime context、设备 marker、实时输出和启动失败日志。 |
| `models/test_update_test_py.py` | 测试生成文件中的跨 flow dependency 名称、obsolete marker/case 保留，以及 disabled backend 的 flow section 忽略规则。 |
| `models/test_workspace.py` | 测试 workspace 创建位置、相对目录布局、复制失败清理和 ownership marker 清理保护。 |

## 6. 新增单元测试的约定

- 新测试文件必须使用包含 copyright、`File`、`Description`、Apache-2.0 license 和 SPDX 的完整 HOUMO AI 文件头。
- 测试模块应声明 `pytestmark = pytest.mark.unit`。
- 测试应直接导入被测生产模块，不要通过公共 support 转发生产 API。
- 只有至少两个测试文件真正复用的 builder、fake 或路径常量才放入 support 文件。
- 不依赖设备和网络；需要命令执行时使用 recording/fake runner。
- 不读取或修改真实 `cached_models`、`cached_results` 和模型源码目录，使用 `tmp_path` 构造隔离数据。
- 新增或迁移测试后至少运行：

```bash
pytest -q tests/unit_tests
```
