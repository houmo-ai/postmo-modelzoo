---
name: imodelzoo-hmatc-review
description: "Review HMATC public CLI and mode resolution, configuration generation and loading, first-party ONNX optimization, quantization, compilation, inference, comparison, evaluation, performance, benchmark, golden/check workflows, packaging, native extensions, and tests in iModelzoo. Use for changes under hmatc/hmatc/**, hmatc/setup.py and editable package entrypoints, tests/hmatc_tests/**, HMATC configuration files, or model/API changes that depend on modified HMATC behavior."
---

# iModelzoo HMATC Review

## 基础规则与影响范围

先加载 `imodelzoo-code-review`，使用其中的 severity、finding 格式、验证策略和仓库边界。本 skill 只补充 HMATC 专项规则。

评审 HMATC CLI、配置、optimizer、benchmark、packaging 或 `tests/hmatc_tests/**` 时，必须读取 `references/hmatc-cli-config-conventions.md`，按变更涉及的章节检查命令模式、路径解析、结果状态和测试覆盖。

先应用 `.github/guidance/review-guidelines.md` 的全局 `Review Exclusions`。`hmatc/3rdparty/**`、`hmatc/build/**`、`hmatc/dist/**`、wheel、native binary、缓存和其他生成产物不做内容级评审；只检查本次新增或修改的生成物是否误提交。

HMATC 是多个模型示例共享的公共工具。评审 `hmatc/hmatc/**` 和 `tests/hmatc_tests/**` 时，搜索调用它的 `models/**`、`apis/**`、Shell 命令和 README，只把真正依赖被修改公共契约的调用方加入 review unit。本次同时修改模型或 API 文件时，分别叠加 `imodelzoo-model-review` 或 `imodelzoo-api-review`。

排除路径可作为确认下游兼容性的只读上下文，但不评审其实现内容，也不将 finding 定位到其中。

## 沿 CLI 调用链评审

对 CLI 变化追踪完整调用链：

```text
parser
    -> request/mode resolution
    -> config loading and CLI override
    -> command dispatch
    -> BaseModel/BaseExec/BaseInfer or task implementation
    -> gen/quant/build/golden/check/demo/compare/eval/benchmark/perf result
    -> exit status, logs and output artifacts
```

检查：

- subcommand、option、alias、required/default/choices 和 help text 是否一致。
- parser 接受的参数是否被 resolver 和 dispatcher 完整处理。
- mutually exclusive mode 是否真正互斥，partial argument 是否产生清晰错误。
- `--config`、直接模型/HMONNX/HMM 模式和隐藏兼容参数是否进入正确分支。
- CLI override 是否覆盖正确字段，且不会因 falsy value、类型转换或命名差异而丢失。
- 错误是否返回非零状态，避免打印错误后继续执行或报告成功。

## 公共兼容性

- 保留已有 subcommand、option、默认值、输出目录、产物名和日志/结果格式，除非任务明确要求破坏兼容性。
- 检查 Python import surface、entry point、公开 class/function 和配置 schema 的调用方。
- 检查新增 mode 是否与旧 config-driven mode、直接 artifact mode 和 ONNX/chip backend 清晰区分。
- 检查 deprecated/hidden option 是否仍满足现有模型脚本，避免“CLI 可解析但行为已失效”。
- 检查一处公共默认值变化对多个模型类别、backend、device/core 数和 CI 命令的影响。
- 不无故在 HMATC 添加只服务单个模型的硬编码；模型特有逻辑应留在模型目录或通过既有扩展点接入。

## 配置和执行对象

- 检查 config schema、`gen` 默认生成、加载、校验、路径解析和 CLI override 的单一真值关系。
- 检查 model path、input/output、preprocess/postprocess、dataset、precision、shape 和 target 字段。
- 检查 factory/registry 和动态 plugin 路径是否为每种 task 选择正确的 model、dataset、dataloader、exec 和 infer 实现。
- 检查 BaseModel/BaseExec/BaseInfer 契约是否保持一致，子类是否实现必需行为。
- 检查输出目录、临时文件、缓存和结果文件不会跨任务或并发运行相互污染。
- 检查异常清理、资源释放和部分产物处理，避免失败后复用无效结果。

## 各命令专项检查

### Quant / Build / Golden / Check / Gen

- 检查量化输入、校准数据、precision、排除项、CUDA/target 选择和输出 HMONNX。
- 检查 build 的 config/HMONNX 两种模式、batch/core/opt/LLM override 和 HMM 命名。
- 检查 golden/check 的输入模式、必需参数、per-layer 选项、比较对象和阈值。
- 检查 `gen` 从 ONNX 生成的 config 能被当前 loader 和 validator 直接接受，并与 README schema 同步。
- `hmatc/hmatc/optimizer/**` 是 first-party 应用层 ONNX 图优化，评审其图重写正确性和数值语义；只排除下层编译器 IR、Pass、lowering 和 Kernel 实现。

### Demo / Compare / Infer

- 检查 ONNX/HMONNX/chip backend 是否使用语义等价的输入和预后处理。
- 检查 tensor name/order/shape/dtype/layout、device id、同步和结果解析。
- 检查 compare 指标、threshold、失败判定和结果汇总，避免部分失败仍返回成功。
- 检查 `--onnx`、隐藏 HMONNX 模式和 config 模式不会选择错误的执行对象。

### Eval / Benchmark / Perf

- 检查 small-model config eval 与 large-model EvalScope mode 的参数互斥和必需字段。
- 检查 model implementation、model-dir、dataset、limit、model-args 和 output 的解析与透传。
- 检查 dataset split、prediction/reference、metric 和失败样本计数。
- 检查 warm-up、sample、loop、thread、stream、infer-only 和设备同步语义。
- 对 benchmark 检查 child process/queue/timeout、cwd 恢复、产物复用、失败汇总和 Excel report。
- 对齐 latency、throughput、TTFT、TPOT、token 数和分布式指标，避免更改统计口径却不更新调用方。

## Tests 与下游验证

- 检查 `tests/hmatc_tests/**` 是否覆盖发生变化的 subcommand、mode resolution、config override 和错误路径；现有设备流程主要覆盖 quant/build/demo/compare/eval/perf，不要假设 gen/golden/check/benchmark 和 direct-artifact mode 已有同等保护。
- 优先添加不依赖设备的 parser/resolver/config/unit 测试，再补必要的设备集成测试。
- 确认测试断言真实结果、产物或错误，而非只断言命令成功退出。
- 检查 model-specific HMATC 配置、backend、platform 和 skip 条件是否仍有效。
- 搜索受影响的 `hmatc ...` 命令、Python import 和配置字段，检查模型 Demo、API 示例和 README。
- 公共行为变化至少选择一个代表性小模型进行端到端验证；若设备或模型不可用，明确记录未验证范围。

## Packaging 与依赖

- 检查 `setup.py`、entry point、package data、版本和 native extension 的直接耦合变化。
- 对本次新增或修改的 `build/`、`dist/`、wheel、`.so`、`.pyd`、DLL 和 cache，只判断是否误提交，不逐行评审生成内容；历史未触碰的产物不作为本次 finding。
- 不新增依赖或改变安装方式，除非用户明确批准；检查 import 是否会让未使用的子命令也强制依赖可选组件。
- 检查 Linux/Windows、Python 版本和目标平台条件分支，避免在 import 阶段破坏不相关功能。

## 验证与报告

优先运行 parser/resolver/config/gen 和 `tests/hmatc_tests` 的聚焦测试，再根据风险运行代表性 quant/build/golden/check/demo/eval/benchmark/perf 流程。缺少 SDK、量化环境、编译环境、数据集或设备时，将其列为 validation gap。

按 `imodelzoo-code-review` 输出 findings。公共行为问题要说明受影响的 subcommand、调用方和模型范围；不要只描述 HMATC 内部实现细节。
