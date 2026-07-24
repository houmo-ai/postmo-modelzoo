# iModelzoo API 示例评审细则

本文件补充 `imodelzoo-api-review/SKILL.md`。评审 `apis/converts/**`、
`apis/inferences/**`、`apis/scenes/**` 或 `tests/apis_tests/**` 时，按变更涉及的章节执行；
不要把所有示例强制套入同一种目录或测试结构。

## 1. API 示例类型与评审单元

### 1.1 Conversion 示例

典型评审单元包括：

- `apis/converts/<example>/**`；
- 目录内直接相关的 `get_model.py`、`ptq.py`、`build.py`、`config.yml`、`test.sh` 和 README；
- golden input/output、量化配置和编译配置；
- 直接消费该转换产物的 inference 或 scene 示例。

只有仓库中确实存在对应的 `tests/apis_tests` 配置时，才将该 JSON 和 pytest 入口加入评审单元；
不要假设每个 conversion 示例都必须有 `apis_cfg_<example>.json`。

### 1.2 Inference 示例

典型评审单元包括：

- `apis/inferences/<example>/**`；
- 对应的 `tests/apis_tests/apis_configs/*.json`；
- `tests/apis_tests/test_inferences_apis.py`；
- 相关 pytest marker、runner、CMake、平台脚本和 README。

### 1.3 Scene 示例

典型评审单元包括：

- `apis/scenes/<example>/**`；
- 对应的 `tests/apis_tests/apis_configs/*.json`；
- `tests/apis_tests/test_scenes_apis.py`；
- 组合场景依赖的 conversion/inference 示例、marker、runner 和 README。

除单模型规则外，还要检查跨模型或跨阶段的 tensor 名、顺序、shape、dtype、layout，
VPU 等硬件依赖，以及中间 buffer 和 runtime resource 的生命周期。

测试配置中的 `example_dir` 必须指向真实目录。若目录已移除，确认配置是否应标记为
`obsolete`、删除或恢复实现；不得让测试因为路径不存在而长期处于不可解释的跳过状态。

## 2. Conversion 产物契约

按生产者到消费者追踪完整链路：

```text
get_model.py
    -> ptq.py / HMATC quant
    -> build.py / HMATC compile
    -> HMONNX / HMM / golden artifacts
    -> inference or scene example
```

检查：

- 原始模型、量化模型、HMONNX、HMM 的文件名和目录在生产者、消费者、脚本、配置与 README 中一致；
- 量化输出确实被编译步骤使用，推理示例读取的是目标编译产物；
- CLI 默认值、显式参数和 `test.sh` 调用表达同一工作流；
- 单独执行某个 stage 时，缺失前置产物会明确失败；
- `--model_path`、`--output_dir`、设备数等参数真正影响执行，而不是仅完成解析；
- xh2 和 HMATC 两套入口若声称等价，其输入、输出和精度口径一致；
- conversion 依赖本次修改的 HMATC 公共行为时，叠加 `imodelzoo-hmatc-review`。

### 2.1 Golden 与结果比较

- golden input/output 的 shape、dtype、layout 和 tensor 顺序与 runtime 一致；
- dynamic shape 在生成 golden 与运行时采用相同取值；
- 多输出模型覆盖所有必须比较的输出；
- cosine similarity 或误差阈值真正决定成功或失败；
- 比较失败不能只打印日志后返回成功。

### 2.2 Pipeline

- block 或 layer 数能够按 `ndevice`、stage 或 partition 规则合法划分；
- stage 编号连续，设备映射与 inference 侧一致；
- 相邻 stage 的 tensor 名、顺序、shape 和 dtype 一致；
- README 中的设备数、分割关系与实际构建参数一致。

### 2.3 Speculative decoding

- target 与 draft 的模型路径、配置和产物没有互换；
- tokenizer、vocab、embedding 和必要模型规格兼容；
- prefill、decode、verify 产物命名清楚且由正确角色消费；
- inference 示例和 README 能明确区分 target/draft 的准备、编译和运行步骤。

## 3. API 测试配置协议

结合以下文件核对真实 runner 行为，不要只根据 JSON 字段名推测：

- `tests/apis_tests/apis_configs/apis_cfg_template.json`；
- `tests/apis_tests/test_apis_utils.py`；
- `tests/apis_tests/test_inferences_apis.py`；
- `tests/apis_tests/test_scenes_apis.py`；
- `tests/apis_tests/conftest.py`。

重点字段包括：

```text
obsolete
example_dir
support_platform
support_backend
support_core_num
dependency
get_model_params
py_example_params
cpp_example_params
```

### 3.1 参数按列组合

参数数组按相同 index 形成一条测试命令，不是笛卡尔积。review 时必须检查：

- 同一组参数数组的列数和 index 语义一致；
- 新增或删除参数没有造成已有测试列错位；
- Python、C++ 和 CMake `defines` 的对应列描述同一场景；
- `get_model_params` 的起始 index 与 runner 的特殊处理一致，首列没有被误用。

### 3.2 参数编码

按 runner 的实现确认以下编码：

- `"default"` 和 `null` 表示不生成该参数；
- boolean `true` 生成 flag，`false` 不生成；
- `name` 表示脚本名或 executable 名；
- key 以 `#` 开头时，value 作为位置参数加入命令；
- key 以 `-` 开头时原样作为 option；
- 其他 key 转换为 `--<key>`；
- `defines` 和 `envs` 不作为普通运行参数加入；
- `defines` 按对应的 C++ command index 传给 CMake。

若 runner 实现发生变化，以修改后的实现为准，并同步更新模板、已有配置和本细则。

### 3.3 支持矩阵与 marker

- `support_backend` 正确决定 Python/C++ demo 是否执行；
- `support_platform` 与 CMake、平台脚本及 README 的支持声明一致；
- `support_core_num` 与模型产物和运行时限制一致；
- platform、device、backend marker 完整注册且不会静默跳过有效用例；
- `dependency: ["vpu"]` 等依赖能触发对应的环境检查；
- 新增 example 时同步 pytest 入口和 `conftest.py` marker；
- `obsolete` 的含义明确，且没有继续被有效聚合配置或脚本依赖。

## 4. `run.sh`、平台脚本与 pytest runner

ASIC 流程可能先执行示例的 `run.sh`，随后清理并重新准备测试目录，再分别执行模型准备、
Python demo、C++ build 和 C++ demo。评审时检查：

- `run.sh` 非交互、可重复执行，任一步失败返回非零；
- 脚本不依赖上次运行遗留的模型、build 目录或临时文件；
- 脚本不会删除示例工作区之外的用户文件；
- 获取模型失败后不会继续使用旧产物并报告成功；
- `run.sh` 与 pytest 后续阶段使用相同的默认模型、输入、backend 和产物；
- Python 与 C++ 示例若声称等价，使用相同预处理、输入和结果口径；
- `run.bat` 正确传播失败状态并正确处理 Windows 路径；
- Android/NDK 脚本的 ABI、API level、环境变量和 install 输出与 CMake、README 一致。

## 5. CMake、C++ 与资源生命周期

- CMake target、install prefix、README 中的 executable 路径和测试配置的 `name` 一致；
- include/library 搜索路径不依赖作者机器上的绝对路径；
- backend、core 数、dynamic shape 和自定义算子选项在编译与运行时一致；
- runtime、stream、model、device memory 和 host memory 在所有退出路径上释放；
- 异步执行在读取输出或释放资源前完成必要同步；
- 错误码被检查并传播，失败路径不会继续使用未初始化的 handle 或 buffer；
- 输入输出大小按实际 dtype、shape 和 batch 计算，没有错误的固定字节数。

## 6. API README

API README 不强制套用模型 README 的章节模板，但必须与示例形成可复制的闭环。

共同检查：

- 命令的工作目录、参数、环境变量和产物路径与脚本一致；
- 支持的 Linux、Windows、Android、backend 和 core 数不超过已有实现与验证范围；
- Python/C++ 若声称等价，其模型产物、输入、预处理和结果口径一致；
- CMake install 位置与 README 的 executable 路径一致；
- 示例输出、性能或精度结果属于当前模型，未残留其他示例的复制内容。

Conversion README 还要说明原始模型、量化/编译命令、输入格式、输出 HMM、golden/compare
方法，以及下游 inference 消费路径。Pipeline/speculative README 还要说明模型角色、stage
关系、硬件限制、设备数和各类产物的用途。

## 7. 公共 helper 与生成物

- 先应用全局 exclusions；`apis/common/**` 中被排除的 vendored 内容不做内容级评审；
- 只有任务明确包含且确认属于 first-party 的公共 helper 才纳入评审；
- 公共 runtime/build/config 契约变化时，搜索所有 API 调用方；
- 不把单个模型专有逻辑下沉到公共 helper；
- 对本次新增或修改的模型、二进制、`build/`、生成代码和运行产物，只判断是否误提交，
  不对排除内容逐行评审；
- 历史已存在且本次未触碰的生成物不作为本次 finding；finding 应定位到引入产物的变更
  或仓库范围问题，而不是二进制内容内部。
