# Model README Review Conventions

本 reference 定义 iModelzoo 模型主 README 的两套 review 规范。评审 README 时先识别示例体系，再检查文档与配置、脚本、测试和真实产物是否一致。

## 目录

- [适用范围与评审原则](#适用范围与评审原则)
- [大模型与生成式模型 README](#大模型与生成式模型-readme)
- [CV 与 HMATC 模型 README](#cv-与-hmatc-模型-readme)
- [共同的 finding 门槛](#共同的-finding-门槛)

## 适用范围与评审原则

- 大模型与生成式模型体系：重点覆盖 `models/llm/**`、`models/tts/**`、`models/vlm/**`、`models/diffusion/**` 的模型目录主 README；其他采用 `config.yaml` 和独立阶段脚本的模型按同一体系判断。
- CV 与 HMATC 体系：重点覆盖 `models/detection/**`、`models/backbone/**` 的模型目录主 README；其他采用 `config.yml`、`model_impl.py`、`dataset.py` 和 HMATC 公共命令的模型按同一体系判断。
- `cpp/README*`、下载模型目录中的上游 README 或子组件说明按其实际用途评审，不机械套用模型目录主 README 的章节编号。
- 新增或大幅重写 README 时应用完整结构；只修改一个命令或段落时，重点检查本次改动和直接耦合内容，不把未触及的历史格式差异扩展成 finding。
- 优先参考同体系、同任务、脚本结构最接近的当前示例，不要因模型名称相似而跨体系复制模板。

## 大模型与生成式模型 README

评审该体系的 README 时，同时使用 `large-model-readme-generation`。

### 文件名和章节结构

- 模型目录主文档使用 `README.MD`。不要据此重命名嵌套的上游模型卡或平台专用说明。
- 标题使用实际模型或模型族名称；简介说明模型获取、量化、编译和部署范围，不宣称脚本未支持的阶段。
- `[TOC]` 位于标题和简介之后。新建或大幅重写文档采用以下主结构：

```text
1 模型说明
2 快速开始
3 一键评估
4 参考结果
5 免责声明
```

- 第 1 章列出模型名称、可信来源和许可，说明预训练模型不随工程发布及适用平台；以 `config.yaml` 为默认值单一真值，并按实际情况列出资源要求和校准数据集许可。
- 第 2 章按模型实际能力包含环境准备、模型获取、量化、编译和演示。模型不支持某阶段时允许省略并连续编号，不要补无效占位命令。
- 第 3 章描述 `test.sh` 的真实默认行为、阶段依赖和模型专有参数。
- 第 4 章使用真实演示输出、图片或音频结果及性能数据；无可核对结果时明确写“暂无结果”，不要拼接或推测数据。
- 第 5 章使用仓库统一免责声明。附录和 C++/算法流程说明可以放在免责声明之后，但不能破坏前五章主流程。

### 配置、命令和产物一致性

- README 中的默认 model name/size、batch、context/prefill length、device 数、quant type 和路径必须与 `config.yaml` 及脚本解析结果一致。
- `get_model.py`、`ptq.py`、`build.py`、`demo.py` 的 option 拼写、默认值和执行目录必须可直接执行；不要假定连字符和下划线参数等价。
- 需要 GPU 的默认量化流程必须说明 CUDA 前提、带 `--gpus` 的 docker 命令，以及进入容器后执行 `source env.sh`、切换到当前模型目录并运行 `ptq.py` 的完整步骤。
- 编译章节列出 build 实际生成的全部关键 HMM/HMMS 文件；Demo 章节列出或解释其实际消费的模型、tokenizer/processor、embedding、scheduler 和其他依赖。
- `test.sh` 参数表只列当前模型脚本真正消费并透传的参数；默认 `demo`、`all`、`--skip_download` 和专有模式的说明必须与脚本一致。
- 文档中的环境变量、venv、Python/C++ Demo、输出目录和清理行为必须与 `test.sh` 一致，不能遗漏用户执行默认流程必需的步骤。

### 模型类型专项内容

- LLM：对齐 prefill/decode、tokenizer、context/prefill length、sampling、MTP/多卡模式和 TTFT/TPOT 等指标。
- VLM：说明 visual HMM、图片/视频输入、分辨率或帧参数，并保证示例命令与 Demo parser 一致。
- TTS：说明多子模型、speaker/reference audio、采样率、流式/非流式模式、音频输出位置和 TTS 性能指标。
- Diffusion：说明 text encoder/DiT/VAE 等子模型、prompt、分辨率、采样步数、随机种子或 scheduler、输出图片命名；参考图片必须存在且链接可解析，性能结果使用 image latency/throughput 等真实口径。

## CV 与 HMATC 模型 README

### 文件名和章节结构

- 仓库中 `README.MD` 与 `README.md` 均存在。保留被评审目录的既有大小写；新增模型主 README 优先跟随同类别多数示例使用 `README.MD`，不要仅因扩展名大小写产生 finding。
- 标题和简介准确描述模型与任务，避免分类、检测、特征提取等任务名称从其他 README 复制后未修改。
- 允许使用 `[TOC]` 或显式目录列表。新增或大幅重写文档采用以下结构，并保持编号唯一、连续，禁止重复的 `2.1` 等标题：

```text
1. 模型说明
2. 快速开始
3. 参考结果
4. 免责声明
```

- 第 1 章说明模型来源、权重或导出方式、必要许可和精度特殊处理；导出命令生成的 ONNX 文件必须等于 `config.yml` 的 `model.model_path`。
- 第 2 章使用 HMATC 公共命令描述实际支持的 quant、build、eval、perf、demo，按需补充 compare；不要套用大模型的 `config.yaml`、独立 `ptq.py/build.py/demo.py` 或“一键评估”章节。
- 第 3 章给出任务相关的精度和性能结果；第 4 章使用统一免责声明，并列出第三方模型和数据集来源。

### HMATC 命令和 config.yml 一致性

- 所有命令使用当前目录的同一 `config.yml`。`-t/--target`、`--onnx`、warmup/sample/thread 等 option 必须符合当前 HMATC CLI 和 `test.sh`，不要复制已经废弃或无效的参数。
- Eval 和 Demo 若同时声明 chip 与 ONNX 路径，两者必须使用等价输入和预后处理；输出目录如 `vis_xh2/`、`vis_onnx/` 应与实现一致。
- README 中的数据集目录、样本数、输入 shape/layout、颜色空间、resize/padding、模型名、ncore 和 opt level 应能在 `config.yml`、`model_impl.py` 或 `dataset.py` 中核对。
- README 不一定需要逐行复述 `test.sh`，但声称的完整流程、依赖 fallback 和命令参数不能与 `test.sh` 相冲突。
- 量化不可用时若 `test.sh` 会下载预编译 HMM，文档不得暗示默认流程一定执行本地 quant/build。

### 任务结果和指标

- Classification/Backbone：精度结果使用实际 top1/top5 或特征任务指标，数据集、样本数和输入尺寸与 Eval 配置一致。
- Detection/Face：精度结果使用实际 mAP、mAP50 等检测指标，数据集类别和可视化结果说明与 detector 后处理一致。
- ONNX 与 XH2 结果必须明确标注平台，不能交换数值或使用不同数据集却宣称可比。
- 性能结果说明测量配置，包括必要的 warmup、sample、thread/core 信息；README 命令与结果采用相同口径。
- 不接受从相邻模型复制但未核对的模型名、输入尺寸、数据集、结果字段、图片目录或免责声明链接。

## 共同的 finding 门槛

- README 命令、参数、路径或产物名会使用户执行失败，或文档化的默认端到端流程不可用时，按通用 severity 报告 finding。
- 错误的模型/数据许可、虚构结果、错误指标或将不等价结果宣称为可比，属于可操作的文档正确性问题。
- 纯排版、空格、标题风格和既有扩展名大小写差异通常不构成 finding；只有影响链接、自动化、命令解释或明显误导用户时才报告。
- finding 应定位到本次变更的 README 行，并说明与哪个脚本、配置、测试或真实产物不一致。
