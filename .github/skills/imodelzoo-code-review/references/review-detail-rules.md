# iModelzoo Review Detail Rules

本文件补充 `imodelzoo-code-review` 的跨子系统细则。只在应用 `Review Exclusions` 后仍存在可评审路径时读取；它不替代 model、API、HMATC 专项 skill。

## 使用方式

- 只应用与本次变更和实际执行链相关的规则，不为覆盖清单而制造 finding。
- 优先遵循用户要求、适用的 `.github/guidance/review-guidelines.md` 和仓库编码规范。
- 专项 skill 负责模型、API 或 HMATC 的领域规则；本文件负责跨目录、跨语言和跨阶段的一致性。
- Python、C/C++、Bash、CMake、MSVC、Android NDK 和 first-party 文件头的详细检查使用 [`static-source-review-rules.md`](static-source-review-rules.md)。

## 变更范围与仓库卫生

1. 区分语义改动与纯格式噪声。只有空行、对齐或无语义格式变化的文件通常不构成 finding；如果噪声掩盖真实改动、误改生成物或显著增加维护风险，再报告具体影响。

2. 不要为局部需求引入新的全局开关、全局状态或重复配置。优先复用现有入口，并明确配置的单一真值来源。

3. 不要顺手重构与本次目标无关的代码。只有当现有结构会直接导致本次行为错误、调用方遗漏或无法可靠验证时，才将结构问题写成 finding。

4. 对 vendored code、生成物、构建输出和其他排除路径，遵循全局 exclusions；可以把它们作为只读上下文，但不要逐行评审其实现。

## 配置、参数与产物链路

1. 对新增或修改的参数，沿 `CLI / Shell -> config -> Python/C++ 调用 -> 量化/编译/推理入口` 追踪到最终消费点，不能只确认入口层已声明。

2. 检查显式 CLI、配置文件和代码默认值的优先级。特别确认 `False`、`0`、空字符串、空列表和 `None` 不会因 truthy/falsy 判断被错误覆盖。

3. 同一概念在不同阶段应使用一致的名称、类型和语义，包括模型名、precision、batch、shape、sequence length、device/core 数、tokenizer/processor 和 backend。

4. 每个阶段必须消费上游实际生成的产物。检查 ONNX/HMONNX/HMM、缓存目录和输出文件名，避免静默复用旧产物或混用不同配置的结果。

5. 路径应以文档声明的执行目录或显式配置为基准。不要依赖个人工作目录、个人绝对路径或未说明的环境变量。

## 实现与错误处理

1. 注释、命名和日志必须准确表达当前行为，不保留从其他模型或示例复制来的错误名称、shape、命令和指标。

2. 如果逻辑依赖输入范围、文件存在、shape/dtype、设备数或运行模式等前提，应通过校验、条件或清晰错误显式表达，不能只依赖隐含假设。

3. 对等价 backend、模型变体或 Python/C++ 分支执行相同语义时，检查它们没有默默采用不同的默认值、预处理、产物或失败策略。

4. 完整处理可选值、空值、异常、子进程返回码和部分产物。失败后不得继续消费旧文件并报告成功。

5. 简单、直接且可验证的实现优先。不要仅为消除少量重复而引入更难理解的抽象；也不要复制会持续漂移的关键配置或执行逻辑。

## 测试、文档与结论可信度

1. 行为变化和 bugfix 应有最小相关回归保护。测试必须执行发生变化的分支，并断言结果、产物或错误，而不只是 import 成功或进程退出。

2. 参数、默认值、阶段和产物发生变化时，检查直接相关的 `test.sh`、pytest 配置、聚合配置和 README 是否同步。

3. 测试输入、模型变体、backend、marker、skip 条件和设备要求必须能触发声称覆盖的路径，避免测试永久跳过或只覆盖旧路径。

4. Reviewer 不执行测试、模型流程或设备验证，也不把缺少模型、数据、SDK、命令或设备列为 validation gap。只根据测试定义和静态代码证据判断覆盖是否充分，并在存在可触发影响时形成 finding。

5. 性能和精度结论必须在变更和直接上下文中对应可复现的命令、输入和统计口径。不要声称 reviewer 运行或验证了这些结果；只检查声明与代码、配置和已有证据是否一致。

## Finding 门槛与 mixed diff

1. 每条 finding 都要说明本次变更引入的问题、可触发条件、具体用户影响和最小修复方向。纯风格偏好、没有失败路径的猜测和无关旧问题不进入 findings。

2. 如果静态证据表明仓库自带的 `test.sh`、主 `run.sh`、README Quick Start 或其直接调用的公共入口会在声明支持的参数下确定性失败，按通用规则定为 P0；不能因为修复只需改一个 option、用户可以手工改命令或故障只发生在某个 `test.sh` 阶段而降级。

3. 优先把 finding 定位到导致问题的最窄变更行。排除路径只作为上下文，不作为 finding 定位位置。

4. mixed diff 先过滤 exclusions，再按 model、API、HMATC 和通用变更拆分评审；最终统一按 severity、文件和行号合并结果。

5. 如果问题来自跨阶段不一致，finding 应落在引入错误契约或错误默认值的变更处，并说明受影响的下游阶段。

## 不写入本通用细则的内容

- 编译器内部 IR、Pass、lowering、Kernel 或设备原语规则；iModelzoo review 只检查公开接口的调用是否正确。
- 单个模型独有的 tensor、prompt、sampling、音频或图像处理约束；这些内容属于 `imodelzoo-model-review` 或其 model convention reference。
- API 示例或 HMATC CLI 的专有契约；分别由 `imodelzoo-api-review` 和 `imodelzoo-hmatc-review` 定义。
- 一次性的 CI 时间权衡、临时环境限制或没有形成仓库约定的个人偏好。
