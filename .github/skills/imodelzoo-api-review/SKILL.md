---
name: imodelzoo-api-review
description: "Perform static semantic review of iModelzoo model conversion, compilation, deployment, inference, and scene API examples. Use for changes under apis/converts/**, apis/inferences/**, apis/scenes/**, editable first-party API helpers explicitly in scope, tests/apis_tests/**, API example configurations, CMake/build/run scripts, Python or C++ demos, and their README files."
---

# iModelzoo API Review

## 基础规则与评审单元

先加载 `imodelzoo-code-review`，使用其中的 severity、finding 格式、静态语义评审策略和仓库边界。本 skill 只补充 API 示例专项规则。

评审 `apis/converts/**`、`apis/inferences/**`、`apis/scenes/**` 或 `tests/apis_tests/**` 时，必须读取 `references/api-example-conventions.md`，按变更涉及的章节检查 conversion 产物契约、测试配置协议、平台脚本和 API README。

先应用 `.github/guidance/review-guidelines.md` 的全局 `Review Exclusions`。`apis/data/**`、`apis/models/**`、`apis/3rdparty/**` 和已列明的 vendored `apis/common/**` 不进入 API review unit。

按示例类型建立评审单元，不要假设所有 API 示例都具有相同文件：

- Conversion unit：`apis/converts/<example>/**` 内直接相关的 `get_model.py`、`ptq.py`、`build.py`、`config.yml`、`test.sh`、README、golden/compare 配置，以及直接消费其产物的 inference 或 scene 示例。只有实际存在对应 API pytest 配置时，才纳入该 JSON 和 pytest 入口。
- Inference unit：`apis/inferences/<example>/**`、对应 `tests/apis_tests/apis_configs/*.json`、pytest 入口与 marker、Python/C++ source、CMake、平台脚本和 README。
- Scene unit：`apis/scenes/<example>/**`、对应测试配置和 pytest 入口，以及直接组合的 conversion/inference 示例。额外检查跨模型 tensor 契约、执行顺序、硬件依赖和资源生命周期。

`apis/common/**` 混合公共代码与 vendored dependency；对未列入全局 exclusions 的 first-party helper，只有任务明确包含该路径且已确认所有权时才纳入评审。

公共 first-party helper 修改 runtime、build 或配置契约时，搜索所有 API 调用方，并把受影响的 conversion、inference 和 scene 作为 mixed API review unit。不要因为 API 示例调用 HMATC 就默认评审 HMATC 内部实现；只有本次修改了 HMATC 公共行为，或 API 变更依赖被修改的 HMATC 契约时，才叠加 `imodelzoo-hmatc-review`。

## 区分转换和推理契约

转换/编译示例重点检查：

- 输入模型、转换参数、input/output tensor、shape、dtype、dynamic dimension 和目标平台。
- 转换或编译输出的文件名、目录和格式是否与下游推理示例一致。
- Python API 是否按公开调用顺序初始化、配置、执行并处理错误。
- 示例是否展示了当前支持的接口，而非内部、废弃或偶然可用的实现细节。
- 生成产物是否会安全覆盖，以及失败后是否遗留可被误用的半成品。
- raw model -> quantized/HMONNX -> HMM -> inference 的生产者/消费者是否一致。
- golden input/output 的 shape、dtype、layout、tensor 顺序和比较阈值是否与 runtime 一致，比较失败是否真正返回失败。
- pipeline 的 stage/partition/设备映射，以及 speculative target/draft、prefill/decode/verify 产物是否由正确角色消费。

推理示例重点检查：

- runtime/HAL 初始化、device/core/stream 选择和 context 生命周期。
- model load、input buffer、output buffer、enqueue/run、同步和资源释放顺序。
- tensor name/order/shape/dtype/layout、host/device 内存和字节大小计算。
- batch、multibatch、multistream、pipeline、prefix cache 或 speculative 等示例的并发与状态隔离。
- 预处理、后处理、sampling、结果解码和输出保存是否符合目标模型语义。
- 失败状态、异常、返回码和日志是否向用户明确暴露。

不要评审编译器或 runtime 内部实现；只判断示例是否遵守其公开 API contract。

## Python/C++ 与平台行为

- 检查 Python 和 C++ 示例声称等价时是否使用相同模型、输入、预处理和输出语义。
- 检查对象所有权、RAII/cleanup、指针/size 计算和错误码，避免泄漏、越界或 use-after-free。
- 检查 include、library、CMake target、install path 和 `tcim_runtime.cmake` 的使用是否与仓库约定一致。
- 检查 Linux `run.sh`、Windows `run.bat`、Android/NDK 脚本的参数和产物名是否同步。
- 对本次变更涉及的 Python/C/C++、Bash 和 CMake 文件，静态检查语言语法与条件分支配对；新增 first-party 源文件同时检查 HOUMO AI Apache-2.0 文件头、`File:`、`Description:` 和 SPDX 字段。Reviewer 不执行语法检查或编译。
- 对声明支持 Windows 的示例应用 [`static-source-review-rules.md`](../imodelzoo-code-review/references/static-source-review-rules.md) 中的 MSVC 清单；对声明支持 Android 的示例应用 Android NDK 清单，静态确认源码、CMake、目标库、ABI/API level 和脚本没有确定性编译冲突。
- 如果主 `run.sh`、`run.bat`、README Quick Start 或其直接调用的 CLI 因 option、subcommand、入口或产物契约不一致而确定性失败，按 `imodelzoo-code-review` 定为 P0。
- 检查环境变量、动态库路径和工作目录是否在 README 中说明，避免依赖个人机器状态。
- 保留公开 CLI、CMake option、输出格式和示例目录结构，除非任务明确要求兼容性变化。

## 测试配置与支持矩阵

- 检查 `apis_cfg_*.json` 中的 `support_backend`、`support_platform`、`support_core_num`、dependency 和 obsolete 状态。
- 按 `test_apis_utils.py` 的真实逻辑还原命令；参数数组按相同 index 形成测试列，不是笛卡尔积，检查列数、index 语义和 CMake `defines` 是否对齐。
- 检查 boolean、defines、envs、name 和以 `#`/`-` 开头的特殊参数是否被 runner 正确处理。
- 检查测试配置的 `example_dir` 指向真实目录；目录已移除时，确认配置应标记 obsolete、删除还是恢复实现。
- 确认测试不会把不支持的平台错误标记为通过，也不会因 skip 条件错误而永远不执行。
- 检查 C++ build 测试是否使用正确的 source/build/install 目录并验证实际可执行文件。
- 对 multibatch、multistream、pipeline 和 scene 示例检查测试是否覆盖其核心并发或组合行为。
- 检查新增示例是否同步 pytest 参数化入口、marker 和必要的聚合文档。
- 新增或修改 `apis/converts/<example>/`、`apis/inferences/<example>/` 或 `apis/scenes/<example>/` 时，必须读取仓库根 `README.md` 的 `## API 示例` 表，确认对应示例已列出；未列出按 `imodelzoo-code-review` 定为 P0。保持现有排除路径：不要因 `apis/data/**`、`apis/models/**` 等排除路径单独触发本项。
- 删除上述 API 示例目录时，必须检查根 `README.md` 的 `## API 示例` 是否同步删除对应行或描述；残留登记定为 P0。仅改测试配置、聚合文档或排除路径且没有 API 示例目录变更时，不要凭空报告本项。

## 示例可复制性

API 示例是用户可能直接复制的参考实现，因此额外检查：

- README 命令能从声明的目录执行，且模型和数据准备步骤完整。
- 示例没有硬编码个人路径、内部凭据、固定设备或未声明的私有服务。
- 错误处理不会教用户忽略失败、泄漏资源或继续消费无效输出。
- 性能示例区分 warm-up 与测量，并在异步设备执行后正确同步。
- 示例数据、图片、音频、视频和模型文件与代码默认值匹配。
- `run.sh` 非交互、可重复执行并传播失败；其默认值和产物与 pytest 后续的模型准备、Python demo、C++ build/demo 一致。
- 对本次新增或修改的二进制、模型、build 目录或生成产物，只判断是否误提交，不逐行评审排除内容；历史已存在且本次未触碰的产物不作为本次 finding。

## 静态评审与报告

Reviewer 不执行参数解析命令、CMake、Python syntax、pytest、C++ build 或推理，也不访问 SDK、runtime、模型和设备。不要枚举缺失工具或环境，不要将未执行项写成 validation gap。

根据 diff 和直接上下文静态检查 parser、CMake、平台脚本、Python/C++ API 调用、测试配置和 README 的契约是否一致。检查测试设计是否能够覆盖目标 backend、platform、core、并发或 scene 组合，但不要声称这些测试已经运行。

外部 SDK/runtime contract 只有在评审上下文中有明确依据时才能支撑 finding；没有依据时不要推测其内部行为。只有某个缺失 contract 会影响候选 finding 是否成立时，才在 Questions / Assumptions 中精确说明。

按 `imodelzoo-code-review` 输出 findings。每条 finding 指明违反的 API contract、触发平台/参数，以及用户复制该示例后会观察到的具体后果。
