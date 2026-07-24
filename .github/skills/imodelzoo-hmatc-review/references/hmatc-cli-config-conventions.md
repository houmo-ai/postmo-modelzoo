# HMATC CLI、配置与执行契约评审细则

本文件补充 `imodelzoo-hmatc-review/SKILL.md`。评审 HMATC 公共 CLI、配置、ONNX 优化、
benchmark、packaging 或测试时，按变更涉及的章节执行，不要求每次加载无关细节。

## 目录

1. [命令与模式矩阵](#1-命令与模式矩阵)
2. [调用链、结果与退出状态](#2-调用链结果与退出状态)
3. [配置、路径与动态扩展](#3-配置路径与动态扩展)
4. [Gen 与配置单一真值](#4-gen-与配置单一真值)
5. [HMATC ONNX Optimizer](#5-hmatc-onnx-optimizer)
6. [各执行流程专项规则](#6-各执行流程专项规则)
7. [测试覆盖与清理安全](#7-测试覆盖与清理安全)
8. [Packaging 与 Native Extension](#8-packaging-与-native-extension)
9. [README 与下游兼容性](#9-readme-与下游兼容性)

## 1. 命令与模式矩阵

以 `hmatc/hmatc/cli/parser.py`、`hmatc/hmatc/cli/resolve.py` 和
`hmatc/hmatc/main.py` 为当前权威实现，检查 parser、resolver、dispatcher、README 和测试
是否表达同一组合法模式。

| 命令 | 合法模式 | 关键参数与约束 |
|---|---|---|
| `quant` | config | 需要 `--config`；检查 `--cuda`、target 和隐藏兼容参数 |
| `build` | config 或 direct HMONNX | `--config` / `--hmonnx` 二选一；direct 模式包含 HMM 名称、输出目录和 LLM build options |
| `compare` | config | 需要 `--config` 和 `--data_path` |
| `perf` | config 或 direct model | `--config` / `--model` 二选一；检查 warmup/sample/loop/thread/stream/infer-only |
| `demo` | config + backend | config 驱动；chip、`--onnx`、隐藏 `--hmonnx` 的选择必须明确 |
| `eval` | small-model config 或 large-model EvalScope | `--config`，或者同时提供 `--model`、`--model-dir`、`--dataset` |
| `benchmark` | config | 需要 `--config`；批量模型和报告逻辑由 benchmark 配置驱动 |
| `check` | config 或 direct HMM | `--config`，或者 `--hmm` + `--golden` |
| `gen` | direct ONNX | 需要 `--onnx`；`--output` 默认生成 config 文件 |
| `golden` | direct HMONNX | 需要 `--hmonnx` 和 `--output`；可选 `--data_path`、`--layers` |

检查：

- parser 接受的组合能被 resolver 解析为唯一 `CommandRequest.kind`；
- 每个 request kind 都有 dispatcher 分支，且不会继续访问另一模式才存在的参数；
- mutually exclusive 和 required 约束在 parser/resolver 中真实生效；
- partial argument、冲突 backend flag 和无效 mode 给出清晰错误并返回非零状态；
- 公共 parent option 只出现在语义有效的命令中，或明确说明无影响；
- `--target` 与 `HOUMO_TARGET` 的 required/default/choices 和 README 一致；
- deprecated/hidden option 只用于兼容已有调用方，不应成为新 README 的推荐入口；
- CLI option、alias、默认值或 mode 变化时，搜索 `models/**`、`apis/**`、tests 和 README 调用方。

## 2. 调用链、结果与退出状态

沿完整链路验证失败能到达调用者：

```text
parser
    -> resolver
    -> config / direct-artifact preparation
    -> dispatcher
    -> Exec / Infer / task implementation
    -> dict / bool / exception / artifact
    -> main()
    -> sys.exit()
    -> subprocess return code
    -> pytest / Shell / CI
```

重点检查：

- `logger.error()`、`logger.fatal()` 是否仅记录日志；若不会终止，必须有后续异常或失败返回；
- 返回 `success: false`、空结果、`False` 或 `None` 时，CLI 不会仍以 0 退出；
- compare/check 多输出或多样本部分失败时，整体失败状态不会被成功项覆盖；
- quant/build/demo/eval/perf 的结果类型变化不会破坏 dispatcher 或测试判断；
- early return 分支与 config-driven 主流程使用相同的成功/失败约定；
- benchmark child process、queue 结果和父进程退出状态一致；
- Python import API 直接调用 `main()` 时，不会依赖只有 console entry point 才成立的副作用；
- 日志、结果文件和退出状态不互相矛盾，不能“日志失败但命令成功”。

## 3. 配置、路径与动态扩展

### 3.1 配置单一真值

检查 `gen_default_config.py`、`check.py`、README、模型 `config.yml` 和执行对象对 schema、
默认值及约束是否一致。重点字段包括：

- `model.name`、`save_dir`、`model_path`、`inputs`；
- input name、shape、dtype/data format、mean/std、resize/padding；
- resizer mode、input size、crop、YUV format；
- `model_impl_module/cls`、`dataloader_module/cls`；
- `quant.calib_data`、`quant_type`、advanced config、`mix_search`；
- `build.batch/ncore/opt_level/roi_num/parallel_jobs/cpp_backend`；
- `demo`、`eval`、dataset module/class 和样本数量。

高风险约束必须由校验和执行共同保证，例如：

- model input shape 与 ONNX input 一致；
- image/non-image、多输入和 dynamic shape 的处理不混用；
- resizer mode、硬件对齐、ROI 和 batch/core 约束一致；
- `mix_search` 字段类型、范围、候选 bit 和互斥条件正确；
- CLI override 覆盖正确字段，并保留 `False`、`0`、空列表和 `None` 的预期语义；
- 未知字段、缺失必填字段和旧 schema 的兼容策略清晰。

### 3.2 路径解析

检查以下来源的优先级和工作目录语义：

```text
CLI 显式路径
config 文件所在目录
当前工作目录
HOUMO_MODEL_PATH
HOUMO_DATASETS_PATH
HOUMO_EXAMPLES_PATH
```

- 相对路径必须有唯一、文档化的基准；
- config 加载后写入的 `_config_dir` 应被所有相关组件一致使用；
- fallback 只在原路径不存在时发生，并记录最终 resolved path；
- 空环境变量不能意外把路径解析到当前目录或根目录；
- 输出目录、日志、缓存和中间产物不能依赖作者机器的当前工作目录；
- config 从模型目录之外执行时，model、dataset 和 plugin 仍按声明规则解析。

### 3.3 动态 Python 扩展

对 model、dataloader、dataset 和 EvalScope plugin 检查：

- `.py` 路径、无后缀模块名和 importable module 的支持范围明确；
- module/class 必须成对配置，缺失时清晰失败；
- 构造函数参数与 BaseModel/BaseDataset/DataLoader 契约兼容；
- 不因相同文件名污染 `sys.modules` 或错误复用旧 module；
- import 异常、class 不存在和路径错误会传播为命令失败；
- plugin 不应要求修改 HMATC 公共代码来接入单个模型特例。

## 4. Gen 与配置单一真值

`hmatc gen` 生成的配置必须能被当前 HMATC 直接加载和校验。检查：

- `--onnx`、`--output` 的 required/default/help 和 README 一致；
- input name、shape、dtype 等信息来自目标 ONNX，而不是固定模板残留；
- dynamic dimension 生成可理解、可编辑的配置，不会错误固化；
- 生成文件包含当前最小必需 schema，并使用与 `check_cfg()` 相同的默认值；
- 输出已存在时的覆盖行为明确，失败时不留下看似有效的半文件；
- schema 新增、删除或重命名后同步生成器、校验、README、模型配置和测试；
- 对生成文件至少执行一次 load + validation 的无设备测试。

## 5. HMATC ONNX Optimizer

`hmatc/hmatc/optimizer/**` 是 HMATC first-party 应用层代码，属于本 skill 的内容级评审范围。
不要把它与编译器内部 IR、Pass、lowering 或 Kernel 混为一谈；后者不在 iModelzoo review 范围。

检查 HMATC ONNX 图重写：

- pattern 只匹配目标子图，不因同名节点或可选 input 误匹配；
- fusion/replace/delete 后输入输出名称、顺序、shape、dtype 和 graph output 保持正确；
- initializer、Constant、attribute、opset 和 domain 没有丢失或错误复用；
- 删除节点后引用完整，图保持拓扑有序并通过 ONNX checker；
- dynamic shape、多输出、共享 initializer 和 external data model 得到正确处理；
- shape inference 或 ONNX Runtime validation 的失败真正阻止输出被当作成功；
- 必要时使用代表性输入比较优化前后的数值结果和容差；
- 优化重复执行保持幂等，失败时不会覆盖原始模型或遗留半成品；
- `model.app_onnx_opt` 配置、manager/registry 和实际优化实现保持一致；
- 新增模型专有优化优先通过既有扩展点接入，不把单模型假设扩散到通用 optimizer。

## 6. 各执行流程专项规则

### 6.1 Quant / Build / Golden / Check

- quant 的原始模型、校准数据、预处理、quant type、mix search、CUDA 和 target 一致；
- 输出 HMONNX/HMQuant 文件名与 build、compare、README 和下游模型配置一致；
- build config/direct-HMONNX 两种模式的默认值和输出目录清晰隔离；
- batch、model batch、ROI、core、opt level、parallel jobs 和 LLM build options 真正透传；
- HMM 命名包含的 batch/core/resizer/ROI 等信息与真实编译参数一致；
- build 后自动 check 和 `--skip_check` 行为清晰，check 失败能使整体失败；
- golden/check 的随机或显式输入、per-layer 模式、tensor 对象和 threshold 一致；
- 量化或编译底层 API 失败后不继续复用旧 HMONNX/HMM。

### 6.2 Demo / Compare / Infer

- ONNX、HMONNX 和 chip backend 使用语义等价的输入、预处理和后处理；
- backend flag 冲突不会静默选择某一个分支；
- tensor name/order/shape/dtype/layout、device id、同步和输出解析一致；
- model implementation 和 dataloader 返回契约适用于目标 backend；
- compare 指标、threshold、样本汇总和失败判定真实反映精度；
- 失败样本、缺失 output 或不匹配 tensor 不会被静默跳过。

### 6.3 Eval

- small-model config eval 与 large-model EvalScope mode 真正互斥；
- large-model 模式完整要求 `model`、`model-dir` 和 `dataset`；
- `model-args` 的重复 KEY=VALUE 解析、类型和覆盖顺序清晰；
- `model-dir` 不与 small-model config 的 `model.model_path` 混用；
- dataset、split、limit、prediction/reference 和 metric 对齐；
- import/register EvalScope model 的副作用和错误只影响 large-model 模式；
- 失败或跳过样本不会被计入成功指标。

### 6.4 Perf

- config/direct-model 两种模式的 batch、device 和模型路径一致；
- warmup、sample、loop、thread、stream 和 infer-only 的边界值被校验；
- 计时边界包含正确的设备同步，不把 IO 混入 infer-only；
- latency、throughput、TTFT、TPOT、token、H2D/D2H 等单位和统计口径稳定；
- 多线程/多流的状态、buffer 和结果相互隔离；
- native perf extension 的接口变化同步 Python 调用方和 packaging。

### 6.5 Benchmark

- 模型列表、location、config 和平台解析正确；
- 下载、quant、build、perf 和可选 eval 的子步骤返回码被检查；
- child process 异常、超时、退出码和 queue 缺失结果能被父进程识别；
- `os.chdir()` 和其他进程全局状态在所有异常路径恢复；
- 已有 HMONNX/HMM 只在配置匹配时复用，不因文件存在就静默跳过；
- x86 与非 x86 的阶段差异、core/batch/thread/resizer 组合符合支持范围；
- success summary 不包含失败或不完整结果，full report 与 summary 的列名、单位和模型标识一致；
- Excel/report 写入失败会使命令失败，不能仅打印警告后返回成功；
- 并发模型的输出、日志和临时目录互不污染。

## 7. 测试覆盖与清理安全

当前 `tests/hmatc_tests` 的主要设备集成流程固定覆盖 `quant`、`build`、`demo`、`compare`、
`eval` 和 `perf`。不要据此假设其他命令已经受到同等回归保护。

### 7.1 无设备 unit test

优先为以下变化增加聚焦测试：

- parser option、alias、required、choices 和 help；
- `resolve_command_request()` 的全部合法与非法 mode；
- CLI override、config load/check 和 default generation；
- 路径解析与动态 plugin 加载；
- result/exception 到非零退出状态；
- build direct HMONNX、perf direct model、check direct HMM；
- gen、golden、benchmark 和 large-model eval 参数解析。

### 7.2 Mock/fake execution test

- 断言 dispatcher 调用正确的 Exec/task 方法及参数；
- 覆盖成功、失败、空结果和部分失败；
- 不用真实量化、编译或设备即可验证公共契约；
- 测试结果、产物或错误语义，不只断言 import 或进程退出。

### 7.3 设备集成 test

- 选择最小代表模型验证受影响的 quant/build/demo/compare/check/eval/perf 链路；
- platform、backend、device/core marker 和 skip 条件能触发声明路径；
- 设备、模型、数据或 SDK 不可用时记录 validation gap，不伪造通过结论。

### 7.4 清理安全

- 递归删除前确认目标是测试框架创建的明确工作目录；
- 不对仓库根、模型源目录、用户工作目录、空路径或环境变量展开结果执行删除；
- 测试失败、异常和中断路径仍能恢复 cwd 并安全清理；
- 清理不能删除共享模型缓存或其他并行测试的输出；
- 复用缓存时使用锁或隔离目录，避免并发污染。

## 8. Packaging 与 Native Extension

检查 `hmatc/setup.py`、requirements、entry point 和 native sources 的直接契约：

- `hmatc = hmatc:main` 指向可导入且返回状态一致的入口；
- setup/build/wheel/help 命令不会被自定义参数解析破坏；
- 版本、commit、build time 生成文件在源码安装和 wheel 中可用且不会误改源码；
- `tcim_lite`、`HOUMO_TARGET`、`HOUMO_VERSION`、`HOUMO_SDK_PATH` 的要求清晰；
- `hmatc.python.perf` 和可选 `hmatc.python.smi` 的 module name、source、library、ABI 和 package data 一致；
- `--enable_smi_support` 仅影响 SMI extension，不成为默认安装强依赖；
- Linux/Windows compile/link 参数和 Windows DLL copy 与 runtime 布局一致；
- optional dependency 不会在 import 阶段破坏未使用的 subcommand；
- requirements 的空行、注释和 platform filtering 不会形成无效依赖；
- 本次新增或修改的 build/dist、wheel、`.so`、`.pyd`、DLL 和 cache 只检查是否误提交，
  不逐行评审生成内容；本次未触碰的历史产物不作为 finding。

## 9. README 与下游兼容性

### 9.1 HMATC README

- 所有 public subcommand、mode、alias、required/default/choices 与 parser 一致；
- gen 输出 schema、config 示例和 `check_cfg()` 一致；
- config/direct-artifact、small/large eval 和 ONNX/HMONNX/chip backend 的边界清晰；
- resizer、ROI、batch/core、mix search、plugin 和路径规则与实现一致；
- 命令从声明的目录可执行，不依赖个人绝对路径或未说明环境变量；
- 性能、精度和平台支持只陈述已验证范围。

### 9.2 下游调用方

公共 CLI、schema、默认值、产物名、Python import 或 plugin 契约变化时，搜索：

```text
models/**
apis/**
tests/**
Shell scripts
README files
```

只把真正依赖被修改契约的调用方加入 HMATC review unit。若本次同时修改模型或 API 文件，
分别叠加 `imodelzoo-model-review` 或 `imodelzoo-api-review`；只读确认兼容性时不必无边界加载
全部下游专项规则。finding 应定位到引入错误契约的可修改源文件，排除路径只作为上下文。
