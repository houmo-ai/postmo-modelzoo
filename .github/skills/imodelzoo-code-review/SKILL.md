---
name: imodelzoo-code-review
description: "Review non-excluded iModelzoo application-layer changes, including model demos, model acquisition/conversion, quantization, compilation, inference, evaluation, performance, tests, configurations, shell workflows, and README examples. Use for code review or review summaries after applying the exclusions declared in .github/guidance/review-guidelines.md."
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
- mixed diff 只评审未排除路径，并在 Validation 或 Summary 中记录被排除的路径组。
- 排除文件可作为理解其他源码的只读上下文，但不要对其内容产生 finding。
- 用户明确要求评审某个排除路径时，以用户要求为准，但不将例外扩展到其他排除路径。

## 必读上下文

评审前：

1. 读取 `.github/guidance/repo-layout.md` 和 `.github/guidance/coding-style.md`。
2. 读取 `.github/guidance/review-guidelines.md`，先应用 `Review Exclusions`。
3. 如果所有 changed paths 都被排除，按全排除模板结束；否则读取 [`references/review-detail-rules.md`](references/review-detail-rules.md)。
4. 按未排除的变更路径应用 review 路由：命中时加载对应专项 skill，未命中时仅使用本 skill；detail rules 在两种情况下都适用。
5. 检查未排除的完整 diff，并打开与变更直接耦合的配置、脚本、测试和 README。
6. 优先参考同模型族或同类示例中已工作的相邻实现和仓库现有约定。
7. 若存在适用于该模型或任务的辅助 skill，同时加载并遵循它。

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

对每个候选问题确认：

- 问题由本次变更引入，或本次变更扩大了其影响。
- 存在具体、可触发的失败条件，而非纯推测。
- 会造成错误结果、流程失败、兼容性破坏、误导性指标、不可复现行为或明显维护风险。
- 能定位到最窄的相关代码行，并给出聚焦的修复方向。

不要提出主观风格偏好、与本次变更无关的旧问题、没有失败路径的猜测或大范围顺手重构。

## 严重程度

按用户影响、触发概率和是否存在合理绕过方式定级，不按改动行数定级。

### P0（必须修复）

会导致数据破坏、敏感信息泄露、广泛安全/稳定性问题，或使核心默认流程普遍不可用。P0 在 iModelzoo 中应很少出现。

示例：默认命令覆盖非目标目录中的用户数据；提交或打印凭据；公共入口在正常使用中破坏共享模型产物。

### P1（必须修复）

会使受支持的主要流程失败、产生根本错误的模型结果或指标，且没有合理绕过方式。

示例：

- 默认 Demo 因参数、依赖或产物名不一致而无法启动。
- 量化或编译实际处理了错误模型、错误配置或错误产物。
- tensor、KV cache、shape、dtype、mask、position 或输出解析错误，导致推理结果无效。
- 评测逻辑产生明显错误的指标但仍报告成功。
- 文档中的默认端到端流程稳定失败。

### P2（应该修复）

会影响一部分受支持场景、造成可复现性或兼容性问题，或使重要回归缺少合理保护，但影响范围有限或存在绕过方式。

示例：

- 非默认但已支持或已文档化的参数被忽略或错误透传。
- 特定 batch、sequence length、动态 shape、精度或多卡配置失败。
- `config.yaml`、`test.sh`、Python/C++ Demo、测试和 README 默认值不一致。
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
- 若改动影响其他 skill 的路径、命令、source anchor 或权威来源，指出并同步检查 skill 是否失效。

## 测试与验证策略

按风险选择最小有效验证集：

1. 检查 diff 及直接耦合文件。
2. 对变更文件运行语法、静态检查或格式检查。
3. 运行最相关的 unit/pytest 用例。
4. 在不需要模型和硬件时检查 `--help`、参数解析和配置合并。
5. 仅在环境、模型、数据和设备可用且成本合理时运行量化、编译、推理、评测或性能流程。

不要声称未执行的硬件相关行为已验证。缺少模型、数据集、授权资产、设备或内部服务时，将其记录为 validation gap；只有存在具体代码证据时才将其写成 finding。

## Finding 写法

每条 finding 必须包含：

- `[P0]`、`[P1]` 或 `[P2]`。
- 简洁、直接描述缺陷的标题。
- 最窄的变更文件和行号。
- 可触发的条件或受支持输入。
- 对用户或模型工作流的具体影响。
- 必要时给出最小修复方向。

一条 finding 只描述一个问题。先按严重程度、再按文件/行号排序。不要把 actionable finding 藏在摘要中。

## 输出模板

```markdown
## Findings

- [P1] 标题 — `path/to/file.py:123`
  说明触发条件、用户影响和最小修复方向。

## Validation

- 已执行的检查及结果。
- 未执行的检查及具体环境/资产限制。
- 被排除的路径组（例如 `Excluded from review: data/**, hmodel/**.`）。

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

## Validation

- Scope classification only; code content was not reviewed.
- Excluded paths: `...`.

## Questions / Assumptions

- None.

## Summary

- All changed files are outside the default iModelzoo review scope.
```

仅在用户要求评审 commit 时检查 commit message。仅在用户要求发布 review 时调用相应发布流程；普通 code review 不应修改代码或发布评论。
