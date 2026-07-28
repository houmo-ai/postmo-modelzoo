---
name: imodelzoo-code-review
description: "Perform static semantic and cross-file feature-design review of non-excluded iModelzoo application-layer changes, including model demos, model acquisition/conversion, quantization, compilation, inference, evaluation, performance, tests, configurations, shell workflows, state/flag design, and README examples. Use for AI code review or review summaries based on the supplied diff and repository context after applying the exclusions declared in .github/guidance/review-guidelines.md."
---

# iModelzoo Code Review

## 定位与边界

iModelzoo 位于编译器、量化工具和运行时之上，主要提供可执行的模型示例与端到端工作流。评审时关注模型应用是否正确、可用、可复现、易维护，以及仓库内各阶段是否一致。

重点评审：

- 模型获取、转换、量化、编译、推理、评测和性能测试流程。
- Python/C++ Demo、Shell 入口、配置文件、测试和 README。
- 对 Houmo 量化、编译、运行时及公共工具 API 的调用方式。
- 模型路径、参数、tensor、产物、命令和指标在各阶段之间的一致性。

开始评审时说明：`我正在使用 imodelzoo-code-review skill 来评审代码。`

## Review exclusions

先应用 `.github/guidance/review-guidelines.md` 中的 `Review Exclusions`：

- 所有 changed paths 都被排除时，停止专项 review 并输出 `No review required: all changed files match review exclusions.`。
- mixed diff 只评审未排除路径，并在 Review Basis 或 Summary 中记录被排除的路径组。
- 排除文件可作为理解其他源码的只读上下文，但不要对其内容产生 finding。
- 用户明确要求评审某个排除路径时，以用户要求为准，但不将例外扩展到其他排除路径。

## 必读上下文

评审前：

1. 读取 `.github/guidance/repo-layout.md` 和 `.github/guidance/coding-style.md`。
2. 读取 `.github/guidance/review-guidelines.md`，先应用 `Review Exclusions`。
3. 如果所有 changed paths 都被排除，按全排除模板结束；否则读取 [`references/review-detail-rules.md`](references/review-detail-rules.md)。
4. 变更包含 Python、C/C++、Bash、CMake、Windows/MSVC、Android/NDK 或新增 first-party 源文件时，读取 [`references/static-source-review-rules.md`](references/static-source-review-rules.md)。
5. 按未排除的变更路径应用 review 路由：命中时加载对应专项 skill，未命中时仅使用本 skill；detail rules 在两种情况下都适用。
6. 检查未排除的完整 diff，并打开与变更直接耦合的配置、脚本、测试和 README。
7. 当代码使用环境变量时，读取仓库根 `env.sh`、适用的平台初始化脚本和顶层 README 环境准备说明，先还原标准执行入口已经建立的环境契约，再判断变量是否缺失。
8. 优先参考同模型族或同类示例中已工作的相邻实现和仓库现有约定。
9. 若存在适用于该模型或任务的辅助 skill，同时加载并遵循它。

## 评审方法

沿用户实际执行路径检查改动，不要孤立评审单个文件：

```text
配置 / CLI
    -> 模型获取 / 转换
    -> 量化
    -> 编译
    -> 部署 / 推理
    -> 评测 / 性能
    -> tests / README
```

除阶段链路外，还要为本次新增或修改的每个功能建立跨文件“功能切片”：从 changed identifier、CLI option、配置 key、环境变量或字段出发，联合检查其定义与默认值、所有赋值/覆盖、跨脚本或配置传递、条件分支、最终消费者、测试和文档。不能只确认每个文件局部语法正确。

对每个候选问题确认：

- 问题由本次变更引入，或本次变更扩大了其影响。
- 存在具体、可触发的失败条件，而非纯推测。
- 会造成错误结果、流程失败、兼容性破坏、误导性指标、不可复现行为或明显维护风险。
- 新增的状态、flag、配置或中间变量具有独立且可说明的语义；若其值完全可由另一个值推导，确认没有引入可达的矛盾状态、静默忽略或多处同步责任。
- 能定位到最窄的相关代码行，并给出聚焦的修复方向。

不要提出主观风格偏好、与本次变更无关的旧问题、没有失败路径的猜测或大范围顺手重构。

## 严重程度

按用户影响、触发概率和是否存在合理绕过方式定级，不按改动行数定级。

### P0（必须修复）

会导致数据破坏、敏感信息泄露、广泛安全/稳定性问题，或使仓库提供的标准用户入口确定性不可用。

以下入口视为标准用户入口：

- 模型目录的 `test.sh`，包括其声明支持的 `demo`、`quant`、`build`、`all` 等阶段。
- API 示例的主 `run.sh`、`run.bat` 或等价一键入口。
- README Quick Start 或默认端到端流程给出的主命令。
- 被上述脚本和文档直接调用的公共 CLI、Python/C++ 入口。

只要本次变更造成这些入口在满足仓库已声明的统一环境准备、依赖安装和必要前置步骤后，仍会在其声明支持的参数或默认配置下必然因参数不兼容、入口缺失、产物契约断裂或直接调用错误而失败，就定为 P0。用户可以手工修改脚本、改写参数或绕开仓库入口不算合理绕过方式；但执行仓库统一要求的 `source env.sh` 等初始化步骤属于正常使用前提，不是绕过方式。

示例：

- `ptq.py` 删除或改名某个 option，但 `test.sh` 仍传入旧 option，导致 `bash test.sh -s quant` 在参数解析阶段失败。
- `test.sh`、主 `run.sh` 或 README Quick Start 调用不存在的脚本、subcommand、option 或产物。
- 默认 Demo 因参数或必需产物名不一致而无法启动。
- 默认命令覆盖非目标目录中的用户数据；提交或打印凭据；公共入口在正常使用中破坏共享模型产物。

### P1（必须修复）

会使受支持的主要流程失败、产生根本错误的模型结果或指标，且影响不满足上述标准入口 P0 升级条件。

示例：

- 量化或编译实际处理了错误模型、错误配置或错误产物。
- tensor、KV cache、shape、dtype、mask、position 或输出解析错误，导致推理结果无效。
- 评测逻辑产生明显错误的指标但仍报告成功。
- 非主入口的受支持工作流稳定失败，或文档中的专项命令错误但不阻断 Quick Start/标准一键入口。

### P2（应该修复）

会影响一部分受支持场景、造成可复现性或兼容性问题，或使重要回归缺少合理保护，但影响范围有限或存在绕过方式。

示例：

- 非默认但已支持或已文档化的参数被忽略或错误透传，但不会导致标准用户入口失败。
- 特定 batch、sequence length、动态 shape、精度或多卡配置失败。
- `config.yaml`、`test.sh`、Python/C++ Demo、测试和 README 存在不阻断流程的默认值漂移。
- 子进程失败被吞掉，流程继续使用旧产物。
- 行为改动缺少可在现有测试框架中实现的聚焦回归用例。

不要把纯格式、个人偏好、可选清理或没有实际影响的建议列为 finding。

## 检查清单

只应用与变更相关的条目。

### 端到端流程与产物

- 确认每个阶段消费的是上一个阶段实际生成的产物。
- 对齐模型名、tokenizer/processor、模型变体、精度、batch、sequence length、device 数及产物命名。
- 检查输出目录、覆盖策略和旧产物复用，避免不同配置的产物被静默混用。
- 检查相对路径是否以文档规定的执行目录为基准，避免依赖个人工作目录。
- 检查新增阶段是否同步到直接相关的 `test.sh`、测试配置、聚合配置和 README。

### 配置、CLI 与 Shell

- 对齐 `config.yaml`、命令行默认值、Shell 参数、Python/C++ 参数和 README 示例。
- 确认 CLI override 以正确的名称、类型和值传递到最终调用点。
- 检查 boolean、list、空值、数字和路径经过 Shell 转发后语义不变。
- 保留既有公开 CLI、默认值、输出格式和产物名，除非变更明确要求破坏兼容性。
- 检查 Shell 引号、退出码和错误传播，避免失败后继续执行。
- 标记个人绝对路径、硬编码设备、私有凭据或未说明的内网环境假设。
- 判定 Shell/CMake/README 中的环境变量未定义前，必须按 `仓库根 env.sh / 平台初始化脚本 -> 父级启动脚本或容器入口 -> 当前脚本` 的实际执行链追踪定义。仓库文档要求用户预先执行的初始化脚本所 `export` 的变量，视为标准流程已定义，不能仅因当前子脚本没有再次赋值而形成 finding。
- 对 `HOUMO_EXAMPLES_PATH`、`HOUMO_PATH`、`HOUMO_SDK_PATH`、`TCIM_RUNTIME_PATH`、`HOUMO_DATASETS_PATH`、`HOUMO_MODEL_PATH` 等仓库级变量，检查其在根 `env.sh` 或对应平台环境准备中的最终展开值以及下游路径是否真实一致；不要假设它们为空。只有标准初始化后仍未定义、被中途 unset/覆盖成无效值、用于不受该初始化约束的独立入口，或文档把命令声明为可直接运行却遗漏必要环境准备时，才报告问题。
- 例如，`run.sh` 使用 `$HOUMO_EXAMPLES_PATH/3rdparty/...`，而根 `env.sh` 已将 `HOUMO_EXAMPLES_PATH` 导出为仓库根目录时，不能以“run.sh 本地未定义变量”为由报告路径展开为空；应继续检查初始化后的目标目录、库名和平台分支是否正确。

### 跨文件功能设计与状态最小化

- 对每个新增或修改的功能概念，联合查看所有直接定义方、赋值方、透传层、判断分支和最终消费者；同时检查相关测试配置、Shell/CLI、README。不要把一个提交拆成互不关联的逐文件检查。
- 为新增状态列出实际可达组合，并确认每个组合都有明确语义。重点检查 `enable_x + x_value`、`x_set + x`、`has_x + x`、`count + collection`、缓存值与源值、多个 boolean 模拟一个 mode 等重复表达同一事实的设计。
- 如果一个 flag 能完全从 payload 是否存在、集合是否为空、对象是否为 `None`、枚举值或其他权威状态推导，且没有文档、调用方或测试证明它需要表达独立状态，优先使用原始值作为单一真值来源。报告重复状态时必须指出当前可达的矛盾组合及后果，例如“值已提供但 flag 为 false，导致值被静默忽略”或“flag 为 true 但值为空，导致传入无效参数”。
- 不要仅因两个变量相关就认定冗余。若“未设置”“显式空值”“使用默认值”“禁用功能”确实是不同状态，或存在兼容性、延迟加载、安全控制等独立 contract，应保留独立状态；确认该区别在命名、文档、校验和测试中被明确表达。
- 检查同一信息是否在环境变量、Shell 局部变量、CLI flag、配置 key、Python/C++ 字段中重复保存并需要人工同步。若只需要传递，避免重新编码成第二套 boolean 或默认值；若必须转换，检查转换只有一个权威位置。
- 检查新增抽象是否真正减少分支和重复，还是只增加 wrapper、透传字段、镜像配置或永远一起变化的参数。只有存在具体错误路径、可达非法状态、调用方遗漏或明确的持续同步负担时形成 finding；纯粹“可以少写几行”的偏好不报告。
- 跨文件设计问题通常定为 P2；若矛盾状态会稳定产生错误模型结果或阻断受支持流程，按实际影响升为 P1；若使标准用户入口在正常前置条件下确定性失败，按 P0 规则处理。

### 转换、量化与编译

- 验证模型输入、校准数据和预处理与目标模型匹配。
- 检查量化精度、排除项、校准样本、shape 和目标设备配置是否传递正确。
- 检查编译接口、输入 shape/dtype、动态/静态维度、产物路径和目标配置是否符合公开 API。
- 确认下层工具或子进程失败能被显式报告，并在需要时返回非零退出码。
- 只判断 iModelzoo 是否正确调用公开接口；不要推测编译器内部优化或 Kernel 实现问题。

### Demo 与推理

- 检查输入/输出 tensor 的名称、顺序、shape、dtype、layout 和语义。
- 检查预处理、后处理、padding、batching 和结果解析是否与参考实现一致。
- 对自回归模型检查 prefill/decode、KV cache、mask、position、stop token、最大长度和 token 计数。
- 检查 greedy/sampling 分支、随机种子、sampling 参数、流式输出和文本解码。
- 检查多模态、ASR、TTS、Embedding、Reranker、OCR 等模型特有的数据处理与输出语义。
- 检查设备初始化、资源生命周期、同步、warm-up 和清理是否正确。

### 评测与性能

- 检查数据集、split、prompt/template、预处理、prediction/reference 配对和 metric 实现。
- 避免将失败或跳过的样本静默计为成功。
- 确认精度比较对象语义一致，threshold/tolerance 合理。
- 区分 warm-up 与正式测量，并在计时边界执行必要的设备同步。
- 对齐 latency、throughput、TTFT、TPOT、token 数、batch 和显存等指标定义。
- 不要根据未经验证的数据宣称精度或性能收益。

### 测试与文档

- 行为变化、bugfix 和易回归逻辑应添加或更新最小相关测试；不要一律将“缺测试”定为 P0。
- 优先复用 `tests/models_tests`、`tests/apis_tests`、`tests/hmatc_tests` 或相邻目录的既有模式。
- 确认测试真正执行变更路径，而非仅检查 import 或进程退出成功。
- 检查 pytest marker、模型 JSON、`test.sh` 参数组、Demo 开关和 `config/imodelExampleConfig.yaml` 是否需要同步。
- README 命令必须与当前 CLI、默认值、路径、产物名和阶段顺序一致。
- 检查复制的文档是否残留其他模型的名称、shape、路径、命令或指标。

### 仓库边界与维护性

- 保持 diff 小且聚焦，避免无关重构和格式化。
- 不修改 vendored dependency、生成文件或 build output。
- 不无故新增依赖、全局开关、重复配置或公共 API/ABI 变化。
- 检查 import/include、错误处理、资源释放和直接耦合调用方是否同步。
- 对变更的 Python、C/C++、Bash 和 CMake 应用静态语法检查；对声明支持 Windows/MSVC 或 Android/NDK 的 C/C++ 组件应用平台可移植性检查。
- 检查新增 first-party Python/C/C++ 源文件是否具有符合仓库规范的版权、文件名、Description、Apache-2.0 和 SPDX 文件头。
- 若改动影响其他 skill 的路径、命令、source anchor 或权威来源，指出并同步检查 skill 是否失效。

## AI reviewer 能力边界

本 skill 面向仅由 AI 大模型承担的 reviewer。Reviewer 只基于评审系统提供的 changed paths、diff、仓库源码上下文、配置、测试定义和文档进行静态语义分析，不执行仓库命令，也不假设能够访问终端、Git 工作区、Python/C++ 环境、依赖、SDK、模型、数据或设备。

以下限制是 reviewer 的固定能力边界，不是某次变更特有的 validation gap：

- 不能运行 Shell、Python、C++、Git、formatter、linter、pytest、构建或编译命令。
- 不能安装或导入 FunASR、TCIM、Houmo SDK 或其他依赖。
- 不能下载或加载模型、数据集、ONNX/HMONNX/HMM 等产物。
- 不能执行量化、编译、推理、评测、性能测试或访问 GPU/XH2 设备。

不要在评审结果中枚举上述缺失条件，不要逐项列出未运行的检查，也不要输出类似内容：

- `当前环境缺少 python3、git。`
- `缺少 FunASR、TCIM、模型资源和 XH2 设备。`
- `未运行语法检查、pytest、量化、编译或推理。`
- `由于环境限制，无法验证本次修改。`

不要声称已经执行、通过或失败了任何命令、测试、编译、模型流程或设备验证。评审结论只表示：在提供的 diff 和仓库上下文中，是否发现了具有明确静态证据的可操作问题。

## 静态语义评审策略

1. 检查完整的未排除 diff，并打开与变更直接耦合的源码、配置、测试定义和 README 上下文。
2. 沿用户实际执行路径追踪参数、配置、控制流、数据流和产物契约。
3. 以 changed identifier、option、配置 key、环境变量或字段为锚点建立跨文件功能切片，追踪定义、赋值、覆盖、透传、判断和最终消费。
4. 比较生产者与消费者之间的名称、类型、shape、dtype、layout、路径、默认值和错误语义。
5. 对新增状态建立可达状态组合，检查是否有可由权威值推导的重复 flag、互相矛盾的镜像状态或需要多处同步的默认值。
6. 检查 CLI、Shell、配置、Python/C++ 入口、测试配置和 README 是否一致。
7. 对环境变量先应用仓库根初始化脚本、平台环境配置、父级入口和 README 声明的标准前置条件，再检查展开后的路径和值；不要把子脚本未重复定义仓库级变量当成缺陷。
8. 根据源码检查边界条件、异常处理、失败传播、资源生命周期、并发状态和兼容性。
9. 检查测试设计是否覆盖变更分支并断言真实结果；不要声称测试已经运行。
10. 只根据评审上下文中的具体代码证据形成 finding，不用运行时猜测填补证据空白。

可以直接形成 finding 的静态证据包括：

- parser、调用方、配置、测试或 README 使用不兼容的 option、默认值或类型。
- 上游产物名、路径或格式与下游消费者不一致。
- tensor、shape、dtype、layout、buffer size 或处理顺序在相邻阶段矛盾。
- 合法的 `False`、`0`、空字符串、空列表或 `None` 被错误覆盖。
- 两个或多个状态重复表达同一事实且能够独立变化，产生“payload 已提供但 enable/set flag 为 false”“flag 为 true 但 payload 无效”等可达矛盾状态。
- 明确的不可达分支、空值解包、越界、资源泄漏、错误码吞掉或失败后复用旧产物。
- 测试参数、marker 或 skip 条件无法触发其声称覆盖的代码路径。

依赖未提供外部 contract 的问题，例如 HMM 真实 tensor metadata、某个 TCIM 版本的未展示 API 行为或设备对特定 dtype 的支持，不能靠猜测形成 finding。只有该 contract 会实质影响候选 finding 是否成立时，才在 Questions / Assumptions 中写出最具体的条件；不要添加“假设依赖已安装”“假设存在设备”等通用运行前提。

## Finding 写法

每条 finding 必须包含：

- `[P0]`、`[P1]` 或 `[P2]`。
- 简洁、直接描述缺陷的标题。
- 最窄的变更文件和行号。
- 可触发的条件或受支持输入。
- 对用户或模型工作流的具体影响。
- 必要时给出最小修复方向。

对于冗余状态或抽象设计 finding，还必须说明哪个值是合理的单一真值来源、额外状态是否具有独立语义、至少一个可达矛盾组合及其当前行为后果。不要只写“代码可以简化”或“变量多余”。

一条 finding 只描述一个问题。先按严重程度、再按文件/行号排序。不要把 actionable finding 藏在摘要中。

## 输出模板

```markdown
## Findings

- [P1] 标题 — `path/to/file.py:123`
  说明触发条件、用户影响和最小修复方向。

## Review Basis

- 说明已审阅的未排除 diff 和直接耦合的源码、配置、测试定义与文档上下文。
- 被排除的路径组（例如 `Excluded from review: data/**, hmodel/**.`）；没有则写 `None`。
- 不列出 reviewer 不具备的工具、环境、依赖、模型或设备，不声称执行了任何命令、测试或运行时验证。

## Questions / Assumptions

- 仅列出会影响结论的问题或假设；没有则写 `None`。

## Summary

- 简述变更范围与总体风险。
```

没有 actionable finding 时明确写：`No actionable findings.`，不要为了填充模板而制造问题。

如果所有 changed paths 都被排除，使用以下简化输出：

```markdown
## Findings

No review required: all changed files match review exclusions.

## Review Basis

- Scope classification only; code content was not reviewed.
- Excluded paths: `...`.

## Questions / Assumptions

- None.

## Summary

- All changed files are outside the default iModelzoo review scope.
```

仅在用户要求评审 commit 时检查 commit message。仅在用户要求发布 review 时调用相应发布流程；普通 code review 不应修改代码或发布评论。
