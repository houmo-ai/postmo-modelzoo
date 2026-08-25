# iModelzoo Code Review Routing

本文件是 iModelzoo code review 的路由入口。它先确定无需评审的路径，再为其余变更选择适用的 review skill。严重程度、通用检查项、finding 写法和输出模板统一由 `imodelzoo-code-review` 定义。

## AI Reviewer 能力边界

iModelzoo code review 由纯 AI 大模型 reviewer 基于评审系统提供的 changed paths、diff 和仓库上下文完成。Reviewer 不执行命令、测试、编译、模型流程或设备验证，也不访问 Git 工作区、Python/C++ 环境、依赖、SDK、模型、数据或硬件。

这些限制是 review 的固定边界，不是每次变更的 validation gap。不要在输出中枚举缺少的工具、依赖、模型或设备，不要逐项列出未运行的检查，也不要声称已经执行任何验证。Review 只根据静态语义、控制流、数据流、配置和跨文件 contract 报告具有明确证据的 finding。

## Review Exclusions

在选择 review skill 前，先按以下分组过滤 changed paths。排除规则仅控制 code review 范围，不代表这些文件可以被任意修改；实现任务仍须遵守用户指令、所有权和仓库安全规则。

### 1. 数据、模型和静态资产

默认不评审以下目录中的内容：

- `data/**`
- `apis/data/**`
- `apis/models/**`

这些文件可以作为评审其他源码时的只读上下文，但不要对其内容或格式提出 finding。

### 2. 直接排除的目录

默认不评审：

- `hmodel/**`

### 3. 顶层批量运行入口

默认不评审：

- `run_all.py`

### 4. 生成物、构建输出和缓存

不逐行评审以下非源码内容：

- `build/**`、`builds/**`、`dist/**`
- `**/build/**`、`**/builds/**`、`**/dist/**`
- `**/__pycache__/**`、`**/.pytest_cache/**`
- `*.pyc`、`*.whl`、`*.inc`、`*.gen`

若这些文件不应出现在变更中，可以针对“误提交生成物/构建产物”报告一个仓库范围 finding，但不要评审其生成内容本身。

### 5. Third-party 和 vendored code

默认不评审以下目录的实现内容：

- `3rdparty/**`
- `apis/3rdparty/**`
- `hmatc/3rdparty/**`
- `apis/common/eigen3/**`
- `apis/common/yaml-cpp/**`
- `apis/common/nlohmann/**`
- `apis/common/hpp/spdlog/**`
- `tools/common/spdlog/**`

`apis/common/**` 和 `tools/common/**` 同时包含 first-party 与 vendored code；未列入上方清单的路径应先确认所有权，再决定是否评审。

### 排除规则的执行方式

- 如果所有 changed paths 都属于排除范围，停止专项 review，并输出 `No review required: all changed files match review exclusions.`。
- 如果 diff 同时包含排除路径和可评审路径，只过滤排除部分，继续评审其余文件。
- 在 Review Basis 或 Summary 中列出被排除的路径组，不要把排除项写成 finding。
- 排除文件可以用于理解可评审代码，但 finding 不应定位到排除文件的行。
- 文件从排除路径移动到可评审路径时，按目标路径评审；从可评审路径移动到排除路径时，检查非排除侧产生的直接影响。
- 用户明确要求评审某个排除路径时，以用户要求为准；不要将这一例外扩展到其他排除路径。

## 通用规则

任何存在可评审路径的 iModelzoo code review 都必须先加载：

- `imodelzoo-code-review`

再根据变更所属子系统加载一个或多个专项 review skill。测试、配置和文档默认跟随其实现所属的子系统，不要仅按文件扩展名或 `tests/` 路径孤立评审。

## 1. Model Example Review

以下变更使用 `imodelzoo-model-review`：

- `models/**`
- `tests/models_tests/**`
- `config/imodelExampleConfig.yaml`
- 与具体模型接入直接相关的 `imodelzoo.yaml`、`imodelzoo_xh2.yaml`
- 与模型工作流直接相关的顶层模型清单或文档

将模型目录、模型测试 JSON、pytest 入口、聚合配置和 README 作为同一个评审单元。沿模型实际支持的 get/convert/quant/build/demo/compare/eval/perf 流程检查，不要求每个模型实现全部阶段。

## 2. HMATC Review

以下变更使用 `imodelzoo-hmatc-review`：

- `hmatc/hmatc/**`
- `tests/hmatc_tests/**`
- HMATC 的 `setup.py`、entry point、配置、README 和直接耦合的 package 文件
- `models/**` 中依赖被修改 HMATC 行为的调用方

排除列表中的 `hmatc/3rdparty/**`、`hmatc/build/**`、`hmatc/dist/**` 和其他生成产物不进入 HMATC review。

HMATC 是公共工具。评审其 CLI、配置或执行行为变化时，必须搜索并检查下游模型示例、Shell 命令和 README。

## 辅助 skill 路由

专项 review skill 可以与以下实现/规范 skill 叠加：

- 新增或重构 `tests/models_tests` 接入：`generate-model-pytest-cases`
- 大模型 README：`large-model-readme-generation`
- 单文件 Python Demo 迁移到 Demo/Engine/Process/Module：`houmo-python-engine-support`

辅助 skill 不替代 `imodelzoo-code-review` 和所属子系统的专项 review skill。

## 未命中专项路由的变更

以下未被排除的变更默认只使用 `imodelzoo-code-review`，并读取最近的组件文档：

- `utils/**`
- `tools/**`
- `tests/tools_tests/**`
- 顶层环境、CI、许可证和仓库维护文件
- 尚未建立专项 review skill 的其他子系统

## Mixed Review

当过滤排除路径后的 diff 跨越多个子系统时：

1. 按所有权拆分为 model、HMATC 或通用子评审。
2. 分别应用对应专项 skill，同时共享 `imodelzoo-code-review` 的 severity 和输出规则。
3. 检查跨子系统的 CLI、配置 schema、模型产物和公共 API contract。
4. 合并 findings，并按 severity、文件和行号排序。
5. 在 Review Basis 或 Summary 中记录被排除的路径组。
6. 不要用一个笼统结论掩盖尚未完成静态语义审查的可评审子系统。

典型组合：

- `models/**` + `tests/models_tests/**`：只作为一个 model review unit，不算 mixed review。
- `hmatc/**` + `tests/hmatc_tests/**`：过滤 HMATC 排除路径后，作为一个 HMATC review unit。
- `hmatc/**` + `models/**`：同时使用 `imodelzoo-hmatc-review` 和 `imodelzoo-model-review`，重点检查公共行为变化及调用方迁移。

## Review 与实施边界

普通 code review 只报告 findings、Review Basis 和必要假设，不修改代码、不发布评论。只有用户明确要求修复或发布时，才进入对应实施或 Gerrit 发布流程。
