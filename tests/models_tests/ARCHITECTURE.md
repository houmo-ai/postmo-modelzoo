# `models_tests` 当前架构说明

本文档面向维护 `tests/models_tests` 框架代码的开发者。模型配置、运行命令和常见排错请先阅读 [README.md](README.md)。历史重构方案只记录设计过程，本文件描述当前代码职责和调用关系。

## 1. 调用链

生成的测试函数最终执行：

```text
test_<flow>_models.py
  -> test_models_utils.execute_<flow>_flow()
  -> execute_model_flow()
  -> ModelConfigRepository.load()
  -> TestRuntimeContext.from_environment()
  -> create_flow_context()
  -> FLOW_REGISTRY.resolve(family, backend, flow)
  -> handler.run(FlowRequest, FlowServices)
  -> FlowResult
  -> pytest pass / skip / fail
```

每个命令携带 `run_id/model/family/backend/flow/case_id/phase` 诊断字段。配置、artifact 和命令基础设施异常在 pytest 边界格式化；业务校验通过 `FlowResult` 转换为 pass、skip 或 failure。

## 2. pytest 入口和生成器

| 文件 | 职责 |
| --- | --- |
| `conftest.py` | 注册 marker、按 flow 排序、禁止多个 xdist worker 竞争共享 artifact。 |
| `test_models_utils.py` | 加载配置和 runtime、创建 `FlowContext`、解析 handler、转换 `FlowResult`。 |
| `test_get_models.py` | 自动生成的 get_model 测试入口。 |
| `test_quant_models.py` | 自动生成的 quant 测试入口。 |
| `test_compile_models.py` | 自动生成的 compile 测试入口。 |
| `test_demo_models.py` | 自动生成的 demo 测试入口，handler 内可附加 multibatch。 |
| `test_compare_models.py` | 自动生成的 CV compare 测试入口。 |
| `test_eval_models.py` | 自动生成的 CV eval 测试入口。 |
| `test_perf_models.py` | 自动生成的 perf 测试入口。 |
| `update_test_py.py` | 从全部非模板 JSON 全量生成七类入口和 `model_names.txt`。 |

生成器规则：

- `model_cfg_<name>.json` 决定模型名；
- marker 将 `-` 转为 `_`、`.` 转为 `dot`；
- 函数类别取 `model_dir` 第二段；
- 每个模型、每个支持 flow 只生成一个 pytest 函数，参数 case 在 handler 内执行；
- obsolete 模型仍生成函数和 marker，运行时 skip；
- `--check` 不写文件，只输出漂移摘要；
- 普通执行全量重写生成文件。

静态 pytest dependency 只有：

```text
quant   -> get_model（模型支持 get_model 时）
compile -> quant，或在无 quant 时依赖 get_model
```

inference flow 不创建静态 dependency，由 handler 声明和准备 artifact need。

## 3. `model_workflow/`

| 文件 | 职责 |
| --- | --- |
| `flow_contracts.py` | family、flow、disposition、请求、上下文、结果、命令、校验、诊断和结构化异常。 |
| `backend_flow_policies.py` | flow 顺序/依赖、family/backend policy、阈值、release 下载规则、ncore 过滤和输出例外。 |
| `model_config_repository.py` | 发现、读取和校验 JSON；根据 `hmatc_flow_version` 明确区分 HMATC v1/v2 schema。 |
| `parameter_matrix.py` | 将列式配置转换为 `ParameterCase` 并渲染命令参数。 |
| `hmatc_v2_config.py` | 解析 v2 行式 case，物化 override YAML，处理 nested cache 路径、case id 和 fingerprint。 |
| `cache_path_resolver.py` | 替换 `cached_models/cached_results` 逻辑路径并提取 case 引用。 |
| `artifact_cache_store.py` | artifact 类型、manifest、fingerprint、状态检查和原子 writer。 |
| `artifact_file_scanner.py` | 扫描 ONNX、HMM/HMMS、普通文件和 required file role。 |
| `artifact_publication.py` | 发布 quant/compiled artifact，构造 manifest，兼容采用旧目录。 |
| `artifact_workspace.py` | 比较 workspace 前后快照，持久化和恢复 get_model 副作用。 |
| `python_environment.py` | 模型侧 requirements 策略，复用共享 virtualenv 实现。 |
| `perf_metric_validation.py` | runner 选择、指标提取、聚合和 development/release 基线校验。 |

`models_tests` 的通用命令、workspace、runtime、平台、marker 和 virtualenv 能力来自 `tests/tests_utils/`，不要在 `model_workflow` 内重新实现第二套基础设施。

## 4. `test_flows/`

| 文件 | 职责 |
| --- | --- |
| `flow_registry.py` | 注册 family + backend + flow handler，并组装 flow services。 |
| `artifact_preparation.py` | 准备 raw、quant、compiled need；调度上游 flow、release HMM、separate 恢复和 v2 nested raw 引用。 |
| `get_model_flow.py` | 在隔离 workspace 执行 get_model；支持按 artifact id 或 config 路径筛选 case。 |
| `quant_flow.py` | 按显式版本选择 HMATC v2 quant；否则执行原有 HMATC v1 或 Python ptq。 |
| `compile_flow.py` | 按显式版本选择 HMATC v2 build；否则执行原有 v1 pipeline 或 Python build。 |
| `hmatc_flow_support.py` | HMATC v1 quant/demo/compare/eval/perf、config 解析和 inference bundle。 |
| `hmatc_v2_flow_support.py` | HMATC v2 quant/build、artifact 复用、环境变量、sidecar 和原子发布。 |
| `inference_flow_support.py` | inference 共用 skip、结果封装、release HMM 匹配、镜像和 compiled 校验。 |
| `demo_flow.py` | `test.sh` 和标准 demo 编排、multibatch、release policy 和失败诊断。 |
| `compare_flow.py` | 解析 Cosine Distance 表格并按 backend 阈值校验。 |
| `eval_flow.py` | 执行 ONNX/HM eval，提取指标并比较阈值。 |
| `perf_flow.py` | 选择 HMATC/demo/custom runner 并调用性能指标校验。 |

## 5. HMATC v1/v2 边界

`support_hmatc` 只描述能力，quant/build 协议由 JSON 顶层字段显式选择：

```text
hmatc_flow_version 缺失或为 1
  -> HMATC v1
  -> required/optional 参数矩阵
  -> 同一 workspace quant + build
  -> v1 inference bundle

hmatc_flow_version == 2
  -> HMATC v2
  -> config + override + ENV_* 行式 case
  -> 独立 quant/build cached_results artifact
  -> Python demo.py
```

框架不根据 list/object、模型名或 family 推断版本。v1/v2 只共享命令执行、cache 路径、锁、manifest 和原子 writer 等基础设施，不共享 case parser、workspace 和产物校验语义。

v2 build 通过 `override.save_dir` 关联 quant artifact，校验 config 一致；有效时复用，缺失时只补跑匹配的 quant case。JSON 中 `ENV_*` 是环境字段标记，注入子进程前移除前缀，例如：

```text
ENV_HMATC_BUILD_OUTPUT_DIR -> HMATC_BUILD_OUTPUT_DIR
```

有效 YAML 直接生成到 artifact staging 的 `.imodelzoo_configs/`，原始 YAML 保持只读。build 后只从 quant `<backend>/hmquant` 复制 `.pt` 和 `hf_config`。

## 6. artifact 和 manifest

artifact 目录可包含类型和 case 对应的 manifest：

```text
artifact_manifest.<artifact_type>.<case_id>.json
```

旧 `artifact_manifest.json` 继续兼容。核心字段包括：

- `schema_version`、`fingerprint_version`；
- `artifact_type`、`model_name`、`model_family`、`backend`、`case_id`；
- `producer_flow`、`source_type`；
- `config_fingerprint`；
- `required_files`。

cache 检查区分 valid、legacy、missing 和 invalid。复用要求 identity、fingerprint 和 required files 满足当前 need。

`AtomicArtifactWriter` 使用 staging 目录生成新 artifact，校验成功后再替换正式目录。替换期间保留受框架标记的 backup；异常不会直接破坏旧的有效 artifact，后续运行可以恢复中断的替换。

HMATC v2 使用 `local_hmatc_v2_quant`/`local_hmatc_v2_build` 标识来源，manifest 只要求框架生成的有效 YAML，不枚举模型特定 HMM、ONNX 或组件文件。build fingerprint 包含上游 quant fingerprint。

## 7. workspace 和 separate 恢复

flow 在模型源码目录旁创建带 `.imodelzoo-workspace` sentinel 的临时 workspace。清理前同时验证 sentinel 和允许根目录，避免删除非框架目录。

get_model 作为上游调用时，`artifact_workspace.py` 比较执行前后快照，将新增或修改文件按相对路径保存到 `cached_models`。SEPARATE_INFER 将目录树恢复到新 workspace，并排除 lock 和 manifest 协调文件。

HMATC v1 inference bundle 从 quant/build 参数引用的多个 YAML 推导。当前要求：

- YAML 位于 workspace；
- `model.save_dir` 非空、相同且为 workspace-relative；
- `<save_dir>/<backend>` 同时包含 HMM/HMMS 和 `hmquant/*with_act.onnx`。

YAML 内容、相对路径、backend 和命令参与 fingerprint。当前的 copy/restore 方式是框架实现约束，不代表 HMATC 工具不支持绝对 `model_path/save_dir`。

HMATC v2 不使用该 bundle。no-infer 直接发布 quant/build `cached_results`；infer 优先复用同步后的非空 compiled artifact，目录缺失或为空时才回退到匹配的 `get_model type=hmm`。v2 raw 准备递归扫描 override 中的 `cached_models`，并按 config 路径选择 raw get_model case。

## 8. Python 环境策略

模型侧环境适配层最终复用 `tests/tests_utils/python_environment.py`。quant requirements 顺序为：

1. workspace 中的 `requirements_ptq.txt`；
2. workspace 中的 `requirements.txt`；
3. `hm_gptq=true` 时追加 `$HOUMO_EXAMPLES_PATH/hmodel/gptqmodel/requirements.txt`；
4. `py_reqs` 相对路径先查 workspace，再查 `HOUMO_DATASETS_PATH`。

缺失的 requirements 文件会被忽略，不会传给模型脚本。模型测试侧保留的是依赖选择策略，共享层负责 virtualenv 创建、复用、激活和安装命令。

## 9. 维护约束

- 不手工修改自动生成的 `test_*_models.py` 和 `model_names.txt`；
- `support_hmatc` 不选择 runner；HMATC quant/build 只按 `hmatc_flow_version` 分派；
- 不根据 JSON 容器类型、模型名或 family 自动推断 v1/v2；
- v2 YAML、artifact 和 sidecar 逻辑不得进入 v1 `hmatc_flow_support.py`；
- v1 inference bundle 不处理 v2 case；v2 首版不接管 hmdemo/compare/eval/perf；
- 通用 policy、失败词例外和指标解析不要下沉成任意 JSON DSL；
- 新增通用能力前先检查 `tests/tests_utils` 是否已有实现；
- flow handler 返回结构化结果，不在内部直接调用 pytest pass/fail；
- artifact producer 必须先在 staging 中生成和校验，再发布到正式 cache；
- 相关逻辑应在 `tests/unit_tests/models` 中增加不依赖设备和网络的单元测试。

提交前执行：

```bash
python -m tests.models_tests.update_test_py --check
pytest tests/models_tests --collect-only -q
pytest -q tests/unit_tests
git diff --check
```
