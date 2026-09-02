---
name: imodelzoo-model-review
description: "Perform static semantic review of end-to-end model example changes in iModelzoo. Use for changes under models/**, related tests/models_tests/** configurations and pytest entries, config/imodelExampleConfig.yaml, model aggregation manifests, large-model config.yaml/get_model.py/ptq.py/build.py/demo.py or python/demo.py workflows, CV config.yml/model_impl.py/dataset.py HMATC workflows, test.sh, or model README files."
---

# iModelzoo Model Review

## 基础规则与评审单元

先加载 `imodelzoo-code-review`，使用其中的 severity、finding 格式、静态语义评审策略和仓库边界。本 skill 只补充模型示例专项规则。

先应用 `.github/guidance/review-guidelines.md` 的全局 `Review Exclusions`。被排除的路径不进入 model review unit，也不因与模型工作流相关而重新纳入。

评审文件命名、参数解析、产物名、`test.sh` 或测试 JSON 时，必须读取 [`references/model-example-conventions.md`](references/model-example-conventions.md)。先根据目录结构和实际执行入口区分“大模型脚本体系”与“CV/HMATC 体系”，再应用对应规则。

评审模型目录主 README 时，必须读取 [`references/model-readme-conventions.md`](references/model-readme-conventions.md)，区分“大模型与生成式模型 README”和“CV/HMATC 模型 README”。涉及 LLM、TTS、VLM、Diffusion 等大模型体系 README 时，同时使用 `large-model-readme-generation`；不要把该结构机械套到 Detection、Backbone 等 CV 示例。

不要把一套体系的要求机械套到另一套体系：大模型通常使用 `config.yaml` 和独立阶段脚本；CV 通常使用 `config.yml`、`model_impl.py`、`dataset.py` 和 HMATC 公共命令。混合或迁移中的示例按实际调用链检查。

不要孤立评审 `models/<category>/<model>/` 中的单个文件。将以下直接耦合内容视为同一个评审单元：

- `models/<category>/<model>/**`
- `tests/models_tests/model_configs/model_cfg_<model>.json`
- `tests/models_tests/test_*.py`、marker 和共享测试入口
- `config/imodelExampleConfig.yaml`
- 直接相关的 `imodelzoo.yaml`、`imodelzoo_xh2.yaml` 或顶层模型清单
- 模型 README、示例命令和测试参数

只检查本次变更影响的文件，不要求每个模型具备所有阶段。

## 沿模型工作流评审

按模型实际支持的阶段追踪数据、配置和产物：

```text
config / CLI
    -> get_model / convert
    -> ptq / quant
    -> build / compile
    -> demo / inference
    -> compare / eval / perf
    -> tests / README / aggregate config
```

确认每个阶段消费上游实际生成的模型、tokenizer/processor、校准数据和中间产物。检查失败后是否可能继续使用旧产物并产生假成功。

## 配置和命名一致性

- 先按 reference 识别 `config.yaml` 大模型体系或 `config.yml` CV/HMATC 体系，确认对应配置源承担单一真值职责。
- 对齐模型目录名、模型 ID、revision、tokenizer/processor 路径和本地缓存目录。
- 对齐 FP、W8A8、W4A16 等精度名称以及 ONNX/HMONNX/HMM 等产物名。
- 对齐 batch、sequence/context/prefill length、dynamic shape、device/core 数和目标平台。
- 检查 `test.sh`、Python CLI、C++ CLI、测试 JSON 和 README 的默认值及 flag 拼写。
- 修改或新增 Python Demo 时，先按测试 runner 的解析规则确定有效入口：相对脚本名优先使用模型目录下的 `python/<script>`，不存在时才回退根目录 `<script>`；不要仅因根目录缺少 `demo.py` 报告 finding。
- 如果本次修改 Python 脚本或 C++ 程序的 CLI 入参名称、alias、`dest`、位置参数、required/default/类型或 boolean 语义，必须沿测试 runner 的命令生成逻辑检查对应 `tests/models_tests/model_configs/model_cfg_<model>.json`。确认受影响的 `get_model_params`、`quant_params`、`compile_params`、`demo_params`、`demo_multibatch_params`、`perf_params`、`test_sh_params` 或 HMATC 参数段是否仍生成目标 parser 接受的参数；有影响时要求在同一变更中同步更新 JSON。
- 如果 `test.sh` 在任一声明支持的阶段调用了 parser 不再接受的 option、错误入口或不存在的产物，使该阶段确定性失败，按 `imodelzoo-code-review` 定为 P0；不要因为默认 `STEP` 是其他阶段、修复只需统一 flag，或用户可以手工改命令而降级。
- 检查大模型的显式 CLI > `config.yaml` > 安全默认值优先级，或 CV 的显式 HMATC CLI override > `config.yml` 优先级没有被反转。
- 确认 CLI override 以正确类型和值传递到量化、编译或推理调用点。
- 检查不同 backend 或模型变体不会覆盖、误读或混用同一输出目录中的产物。

## 获取、量化和编译

- 检查下载来源、文件选择、缓存复用和离线/本地路径行为是否与文档一致。
- 检查导出或转换时的 input name、output name、opset、shape、dtype 和动态维度。
- 检查校准数据、预处理、样本数量和量化排除项是否适合目标模型。
- 检查量化配置是否真正传入 Houmo Quantization Tool，而非只停留在 CLI 或 YAML 层。
- 检查 build 阶段消费正确的 HMONNX/ONNX，且输出 HMM 名称与 Demo、测试和 README 一致。
- 只评审 iModelzoo 对量化/编译公开 API 的调用；不要推测下层 Pass、lowering 或 Kernel 内部实现。

## Demo 与模型语义

所有模型均检查：

- 输入/输出 tensor 的名称、顺序、shape、dtype、layout 和语义。
- 预处理、padding/batching、后处理和输出解码是否与参考模型一致。
- 设备初始化、资源生命周期、同步、warm-up、异常退出和输出落盘。
- ONNX/HMONNX/HMM 等运行模式是否使用等价输入，并按合理 tolerance 比较。

按模型类型补充检查：

- LLM/VLM/Omni：prompt template、prefill/decode、KV cache、mask、position、stop token、sampling、最大长度和多设备切分。
- VLM/OCR：图像 resize/normalize、视觉 placeholder、视觉 token、坐标和文本输出。
- ASR/音频：采样率、channel、feature extractor、chunk/streaming、时间戳和文本归一化。
- TTS：文本前处理、speaker/reference audio、codec/vocoder、采样率和音频写出。
- Embedding/Reranker：pooling、normalization、batching、sequence 截断和 score 语义。
- CV/Diffusion：图像 shape/layout、颜色空间、后处理、随机种子和输出尺寸。

## Compare、Eval 与 Perf

- 确认 compare 比较语义等价的输出，并使用适合 dtype 和任务的指标/阈值。
- 检查 eval 的 dataset、split、prompt/template、prediction/reference 配对和 metric 聚合。
- 避免将失败、空输出或跳过样本静默计入成功结果。
- 区分 warm-up 与正式测量，并在设备计时边界正确同步。
- 对齐 latency、throughput、TTFT、TPOT、token 数、batch 和显存等指标定义。
- 不接受没有可复现命令、环境和输入说明的精度或性能结论。

## 测试和文档耦合

- 按 reference 检查大模型 `test_common.sh` 阶段参数协议，或 CV 完整 HMATC workflow 的固定执行顺序。
- 按 README reference 检查两套章节结构、命令、配置、产物、结果指标和免责声明，不要跨体系复制模板。
- 检查模型配置 JSON 是否覆盖变更涉及的 get/quant/compile/demo/compare/eval/perf 阶段。
- 对所有 Python/C++ CLI 入参变更，检查模型 JSON 中直接生成 CLI option 的 key、`test_sh_params` 中的原始参数以及 runner 特殊消费字段是否需要同步；不能只检查 `test.sh` 和 README。
- 检查 `test_sh_params`、Python Demo 参数组、backend 分支、prerequisite 和 skip 条件是否与实现一致。
- 检查测试实际选中的 Python Demo，而不是只检查 JSON 字面路径。`script` 未配置时默认名为 `demo.py`；配置 `demo_asr.py` 等相对自定义入口时同样优先解析 `python/demo_asr.py`。只有入口 basename 或 CLI contract 改变时才要求同步 JSON，不要为了选择 `python/` 版本机械增加 `script` 列。
- 检查 pytest marker、model name、device 数和显存要求是否正确注册。
- 检查 `config/imodelExampleConfig.yaml` 和聚合模型清单是否需要同步。
- 确认测试执行了变更路径并验证产物或结果，而不只是进程成功退出。
- 检查 README 命令、执行目录、环境变量、模型路径、产物名和阶段顺序。
- 检查复制内容是否残留其他模型的名称、shape、命令、指标或免责声明。
- 新增或修改 `models/<category>/<example>/` 时，必须读取仓库根 `README.md` 的 `## 模型示例` 表，按仓库实际模型标识（交叉核对目录名、模型 README 名称和聚合清单，不要只比较任意内部脚本名）确认该模型已列出。先检查该行的 `support` 列：`✅️`（或等价持续支持标记）表示当前支持；版本号或明确的停止维护文本表示已停止维护。已停止维护的模型不要求当前仍存在于 `models/` 下；其他模型未列出按 `imodelzoo-code-review` 定为 P0。
- 如果本次变更新增根 `README.md` 的模型示例登记行，必须确认该行按实际目录路径 `models/<category>/<example>/` 和模型名称顺序插入；同一表内已有顺序与实际路径不一致的相邻/相关行也应同步调整，使 README 登记顺序与模型目录路径一致。新增行排序错误通常按 P2 报告；若导致登记缺失、重复或路径/模型标识错误，则按 P0 登记问题报告。
- 删除模型示例目录时，必须检查根 `README.md` 的 `## 模型示例` 是否同步删除对应行或描述；仍在支持中的模型残留登记定为 P0。若该行 `support` 已标明停止维护，允许 README 保留历史登记，不因目录缺失或登记残留报告 P0。仅改测试 JSON、聚合配置或排除路径且没有模型示例目录变更时，不要凭空报告本项。

涉及新增/重构模型 pytest 接入时，同时使用 `generate-model-pytest-cases`。

## 静态评审与报告

Reviewer 不执行模型获取、Python 导入、pytest、量化、编译、推理、评测或性能测试。不要把 Python、FunASR、TCIM、Houmo SDK、模型、数据集、GPU 或 XH2 设备不可用写成 validation gap；这些属于 reviewer 的固定能力边界。

沿模型工作流静态追踪配置、参数、模型资源和产物，检查每个阶段是否消费上游代码实际定义的输出，以及测试定义和 README 是否与实现一致。检查测试是否按设计覆盖变更路径，但不要声称测试已经运行。

对模型目录中本次变更涉及的 Python、C/C++、Bash 或 CMake 文件，必须执行（仅基于 diff 和上下文的）静态语法检查：检查括号/字符串/注释、缩进与 block、条件编译、宏续行、Shell 方言与 heredoc、CMake command/block 配对，以及声明/定义的直接结构一致性。不要把缺少运行时依赖或未执行工具写成语法验证缺口。

如果模型或其构建脚本、README、测试配置声明支持 Windows/MSVC 或 Android/NDK，必须静态检查对应平台可移植性：POSIX/GCC-only API 和编译/链接选项是否受 guard 限制，MSVC 的 CMake/DLL 产物是否一致，Android toolchain、ABI/API level、架构 intrinsic、目标库及 install/push 路径是否指向目标产物。只有从代码和仓库契约可确定的必然失败才报告，不要求 reviewer 实际交叉编译。

对新增或复制到 first-party 路径的 Python、C/C++ 源文件和头文件，检查仓库规定的 HOUMO AI Apache-2.0 文件头、`File:` basename、准确非空 `Description:` 和 SPDX 标识；既有历史缺失文件头不因本次普通修改单独报错，third-party/vendored 文件保留其原始许可。

只有存在具体静态代码证据时才报告 finding。对于评审上下文未提供的模型 metadata、外部 API 行为或设备能力，不要猜测；只有它会影响候选 finding 是否成立时，才在 Questions / Assumptions 中说明具体 contract。

按 `imodelzoo-code-review` 输出 findings。每条 finding 明确指出失败发生在哪个阶段，以及它如何影响下游 Demo、测试、评测或用户命令。
