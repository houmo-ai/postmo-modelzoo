# Houmo Python Engine 开发规范

本目录实现 imodelzoo 内共享的 Houmo NPU Python 推理引擎。新增或修改模型时，必须遵循 `Demo -> Engine -> Process / Module` 调用链，并保持 Engine、Process、Module 三层职责隔离。

当前实现是本规范的唯一基准：

- 基础接口：`core/`
- Qwen3.5 Engine：`engine/qwen3_5.py`
- Qwen3.5 Process：`process/qwen3_5/process.py`
- Qwen3.5 Module：`module/qwen3_5.py`
- Qwen3.6 MTP Engine：`engine/qwen3_6_mtp.py`
- Qwen3-ASR Engine：`engine/qwen3_asr.py`
- 层间类型：`core/types.py`
- Qwen3.5 Demo：`../../../models/llm/qwen3.5/python/demo.py`
- Qwen3.6 MTP Demo：`../../../models/llm/qwen3.5/python/demo_mtp.py`
- Qwen3-ASR Demo：`../../../models/asr/qwen3-asr/python/demo.py`

## 一、首版集成约束

首个版本直接使用 imodelzoo 源码：

- 不安装 `houmo-python-engine`。
- 不提供或维护 `pyproject.toml`。
- 不构建 wheel 或 source distribution。
- 不使用 editable install。
- 不修改全局 `PYTHONPATH`。
- Demo 只向当前 Python 进程的 `sys.path` 添加 `utils/python`。
- 原始模型 Demo 与新框架 Demo 并行保留，不得无依据覆盖或删除原实现。
- 模型文件、tokenizer、音频、图片、虚拟环境、设备日志和其他本地资产不得提交到公共引擎目录。

源码导入标准形式：

```python
import sys
from pathlib import Path

imodelzoo_root = Path(__file__).resolve().parents[4]
engine_src = (
    imodelzoo_root
    / "utils"
    / "python"
)
sys.path.insert(0, str(engine_src))
```

## 二、继承关系和公共接口

当前只保留具有实际公共契约的三个基础类，不建立空的 LLM、ASR、TTS 中间层：

```text
HoumoEngine
├── Qwen35Engine
├── Qwen36MtpEngine
└── Qwen3AsrEngine

ModelProcess
├── Qwen35Process
├── Qwen36MtpProcess
└── Qwen3AsrProcess

HoumoModule
├── Qwen35Module
├── Qwen36MtpModule
└── Qwen3AsrModule
```

禁止增加仅用于分类、没有公共实现或额外契约的空中间层。只有在至少两个模型已经出现稳定且可复用的真实实现时，才能从重复代码中提取新的基类。

### 2.1 `HoumoEngine`

定义位置：`core/houmo_engine.py`。

```python
class HoumoEngine(ABC):
    def __init__(self, batch: int = 1):
        ...

    @abstractmethod
    def generate(self, request, **kwargs):
        ...
```

规则：

- 所有具体 Engine 必须直接继承 `HoumoEngine`。
- 所有任务统一公开 `generate()`。
- 不得为 Engine 增加 `chat()`、`transcribe()`、`synthesize()` 等并行公开入口。
- 具体 Engine 必须调用 `super().__init__(batch=batch)`。
- 当前模型只支持 `batch=1`，具体 Engine 必须显式拒绝其他值。
- 不得重新引入 `Runner` 类型或兼容别名。

### 2.2 `ModelProcess`

定义位置：`core/model_process.py`。

```python
class ModelProcess(ABC):
    @abstractmethod
    def preprocess(self, *args, **kwargs):
        ...

    @abstractmethod
    def postprocess(self, state, *, final: bool = False):
        ...
```

规则：

- 所有具体 Process 必须直接继承 `ModelProcess`。
- `preprocess()` 将用户输入转换为模型专用请求对象。
- `postprocess(final=False)` 返回尚未发出的稳定增量。
- `postprocess(final=True)` 返回最终 remainder，并且不得重复或丢失内容。
- 阶段输入构造、融合等模型专属方法继续保留在具体 Process 中。

### 2.3 `HoumoModule`

定义位置：`core/houmo_module.py`。

```python
class HoumoModule(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        ...

    @abstractmethod
    def run(self, stage: Stage) -> None:
        ...

    @abstractmethod
    def get_output(self, stage: Stage) -> StageOutputs:
        ...
```

规则：

- 所有具体 Module 必须直接继承 `HoumoModule`。
- `__init__()` 保存共享依赖后调用 `load()`。
- `load()` 加载图、读取 shape、绑定 cache、初始化固定输入。
- `set_input()` 只绑定当前阶段输入。
- `run()` 只执行并同步当前阶段图。
- `get_output()` 只读取和适配当前阶段输出，返回 `StageOutputs`。
- 不得恢复把输入绑定、执行和输出读取合并在一个公开方法中的旧调用方式。
- 特殊模型操作可以保留具名扩展，例如 `run_vision()`、`prepare_verify_from_prefill()`、`commit_verify_cache()`。

## 三、标准调用链

普通阶段必须由 Engine 显式执行 Module 四步接口：

```text
Demo main()
  -> HmXxx.generate()
     -> XxxEngine.generate()
        -> XxxProcess.preprocess()
        -> [optional] XxxEngine._vision() / _encode()
        -> XxxEngine._prefill()
           -> XxxProcess.prepare_*()
           -> XxxModule.set_input(stage, inputs)
           -> XxxModule.run(stage)
           -> XxxModule.get_output(stage)
           -> Engine sampling
        -> Engine decode/speculative loop
           -> XxxProcess.prepare_*()
           -> XxxModule.set_input(stage, inputs)
           -> XxxModule.run(stage)
           -> XxxModule.get_output(stage)
           -> Engine sampling and stop decision
           -> XxxProcess.postprocess(state)
        -> XxxProcess.postprocess(state, final=True)
```

依赖方向只能是：

```text
Demo -> Engine -> Process
               -> Module -> tcim_lite / HMM
```

禁止反向依赖：

- Process 不得导入 Engine 或 Module。
- Module 不得导入 Engine 或 Process。
- Engine 不得被 Process 或 Module 回调来决定下一阶段。
- Demo 不得绕过 Engine 调用 Process、Module、HMM 或 Runtime。
- Process 和 Module 之间不得直接互相调用。

## 四、Demo 规范

新框架 Demo 放在对应模型目录的 `python/` 子目录，不放在公共引擎目录：

```text
models/<task>/<model>/python/<demo>.py
```

Demo 必须包含：

1. 常量和必要 CLI helper。
2. 面向用户的 `HmXxx` 类。
3. `get_args()`。
4. 动态默认参数解析 helper（如需要）。
5. `main()`。

### 4.1 `HmXxx`

`HmXxx` 只组合一个具体 Engine，并且只公开 `generate()`：

```python
class HmXxx:
    def __init__(self, ...):
        from houmo_engine import XxxEngine

        self.engine = XxxEngine(...)

    def generate(self, request, ...):
        yield from self.engine.generate(request, ...)

    def print_perf(self) -> None:
        self.engine.perf.print_summary()
```

`HmXxx` 必须负责：

- 将用户参数完整转发给 Engine。
- 转发 `batch`、`ndevice`、sampling 和模型路径。
- 对支持历史的模型，使用 `keep_history=True` 表示。
- 直接转发 Engine 的 generator。
- 暴露必要的性能打印方法。

`HmXxx` 不得负责：

- 创建 Process 或 Module。
- 调用 `Stage` 或 Engine 内部阶段。
- 实现 prefill、decode、draft、verify 循环。
- Sampling、EOS 或 context 判断。
- Tokenization、图片 resize、音频重采样。
- 调用 `tcim_lite`。

### 4.2 延迟导入

具体 Engine 必须在 `HmXxx.__init__()` 内延迟导入：

```python
from houmo_engine import XxxEngine
```

执行 Demo `--help` 时不得加载：

- `torch`。
- `tcim_lite`。
- `hmatc` Runtime。
- Processor、Tokenizer 或 HMM。
- NPU 设备。

允许在模块顶层导入轻量的 sampling 参数类型。

### 4.3 CLI 参数

`get_args()` 必须返回 `ArgumentParser`，不得调用 `parse_args()`：

```python
def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ...
    return parser
```

每个 `add_argument()` 必须显式包含：

1. 参数名。
2. `dest`。
3. `type`。
4. `default`。
5. `help`。

布尔参数使用具名解析函数，并支持 `--perf` 和 `--perf false`：

```python
parser.add_argument(
    "--perf",
    dest="perf",
    type=_parse_bool,
    default=False,
    help="enable performance reporting",
    nargs="?",
    const=True,
)
```

动态默认值必须在 `parse_args()` 后由单独 helper 解析，`--help` 不能读取模型或访问设备。

所有当前 Demo 必须公开：

```text
--batch
--ndevice
```

当前模型 `--batch` 默认值必须为 1。

### 4.4 输出格式

终端输出统一为：

```python
print(f"\033[1;95m\nQ: {user_input}\nA: ", end="", flush=True)
for chunk in model.generate(...):
    print(f"\033[1;95m{chunk}", end="", flush=True)
print()
```

ASR、TTS 等非问答任务也使用 `Q:`/`A:` 标签，`Q:` 后显示音频路径、文本或其他用户输入。

## 五、Engine 规范

具体 Engine 位于：

```text
engine/<model_name>.py
```

类名必须明确表达模型：

```text
Qwen35Engine
Qwen36MtpEngine
Qwen3AsrEngine
```

### 5.1 初始化

标准初始化顺序：

```python
class XxxEngine(HoumoEngine):
    def __init__(self, ..., batch: int = 1, perf: bool = False):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("XxxEngine only supports batch=1")

        self.perf = PerfTracker.create(perf)
        self.sampler = Sampler(...)
        self.module = XxxModule(..., perf=self.perf)
        self.process = XxxProcess(
            ...,
            embedding_size=self.module.embedding_size,
            perf=self.perf,
        )
        self.state = GenerationState()
```

如果 Process 依赖 HMM shape，必须先创建 Module，再将必要元数据显式传给 Process。Process 不得读取 HMM。

### 5.2 `generate()`

`generate()` 是唯一公开推理入口，负责：

- 校验用户请求和 `max_new_tokens`。
- 设置 system prompt 等请求默认值。
- 根据会话语义决定是否清理状态和设备 cache。
- 重置请求级 CPU 状态。
- 调用 `Process.preprocess()`。
- 编排所有模型阶段。
- Sampling。
- EOS、长度和 context 停止判断。
- 调用 `Process.postprocess()` 并 `yield`。
- 记录请求级性能指标。

普通生成顺序：

```text
generate()
  1. validate request
  2. apply defaults
  3. reset request perf
  4. clear/reset session when required
  5. reset request-only state
  6. request = process.preprocess(...)
  7. optional vision/encode
  8. prefill and first-token sampling
  9. emit stable delta
 10. decode/draft/verify loop
 11. emit final remainder
 12. record metrics
```

### 5.3 阶段方法

独立图阶段必须拆成具名 Engine 方法：

```text
_vision()
_encode()
_prefill()
_decode()
_draft()
_verify()
```

每个普通阶段必须显式调用：

```python
self.module.set_input(stage, inputs)
self.module.run(stage)
outputs = self.module.get_output(stage)
```

Decode 或 speculative 的完整循环属于 Engine，不属于 Module。

### 5.4 状态

Engine 拥有 CPU 请求和会话状态：

- `context_length`。
- `generated_ids`。
- `emitted_text`。
- `rope_deltas`。
- finish reason。
- MTP pending token、anchor hidden 等模型专属 CPU 状态。

Module 拥有所有设备状态和 Runtime handle。

`clear_session()` 通常同时重置：

```python
self.state = GenerationState()
self.module.clear_session()
```

模型需要专用状态 dataclass 时，应继承 `GenerationState`，例如 Qwen3.6 MTP。

### 5.5 Engine 禁止事项

Engine 不得：

- 导入或调用 `tcim_lite`。
- 直接调用 HMM `set_input()`、`run()`、`sync()`、`get_output()`。
- 根据 HMM tensor 名称或下标绑定输入。
- 绑定、复制或清理具体设备 cache。
- 加载 tokenizer、embedding 或 Processor。
- 实现图片、音频业务前处理。

## 六、Process 规范

具体 Process 位于：

```text
process/<model_name>/process.py
```

### 6.1 `preprocess()`

`preprocess()` 负责：

- 构造 messages。
- 应用 chat template。
- Tokenization。
- 读取和处理图片、视频或音频。
- CPU embedding lookup。
- 构造并返回模型专用 PreparedRequest dataclass。

不得返回含义不明的多层 tuple。

### 6.2 阶段输入

Process 为具体阶段构造 `StageInputs`：

```python
def prepare_prefill_chunk(...) -> StageInputs:
    ...

def prepare_decode(...) -> StageInputs:
    ...
```

包括：

- Embedding padding。
- Position IDs。
- Attention mask。
- Valid/current length。
- 模型要求的 dtype 和 CPU tensor 结构。

`StageInputs.tensors` 可以保持模型语义顺序；Module 负责将其映射到实际 HMM tensor 名称。

### 6.3 融合和后处理

Vision/encode 输出的业务融合属于 Process，例如：

```text
merge_vision()
merge_encode()
```

`postprocess()` 必须：

- 将 token IDs 转换为用户输出。
- `final=False` 时只返回稳定且尚未发出的后缀。
- `final=True` 时返回剩余内容。
- 更新 `state.emitted_text`，避免重复输出。
- 不直接 `print()`。

### 6.4 Process 禁止事项

Process 不得：

- 导入 `tcim_lite`。
- 创建设备、Runtime option 或加载 HMM/HMMS。
- 调用 Module 或 HMM 的执行方法。
- 操作设备 tensor 或 cache。
- 控制完整阶段循环。
- Sampling 或决定停止条件。

## 七、Module 规范

具体 Module 位于：

```text
module/<model_name>.py
```

### 7.1 `load()`

Module 的 `__init__()` 应保存共享对象并调用 `load()`：

```python
def __init__(self, ..., perf: PerfTracker):
    self.perf = perf
    self._stage_metadata = {}
    self.load(...)
```

`load()` 负责：

- 创建 `DevManager`、`WeightManager` 和 Runtime option。
- 加载 HMM/HMMS 图。
- 配置 dummy tensors。
- 读取固定 shape 和容量。
- 绑定 KV、conv、recurrent、draft/verify cache。
- 初始化固定输入和初始设备状态。

Module 向 Engine 暴露必要的只读元数据，例如：

```text
embedding_size
prefill_length
context_max_length
encode_feature_length
verify_length
draft_block_size
```

### 7.2 四步执行接口

职责必须严格分开：

```text
set_input(stage, inputs)
  -> 选择 stage 对应图
  -> shape/dtype 适配
  -> 绑定 Runtime 输入
  -> 保存 StageInputs.metadata

run(stage)
  -> model.run()
  -> model.sync()

get_output(stage)
  -> 读取 Runtime 输出
  -> 必要的 shape 截断或 copy
  -> 恢复 metadata
  -> 返回 StageOutputs
```

`get_output()` 不得返回 Runtime 原始对象给 Engine 或 Demo。

### 7.3 特殊阶段

特殊阶段可以使用具名扩展，但必须保持层级边界：

- `run_vision()` 可以在 Module 内处理多张图片的 vision 图循环并返回 `StageOutputs`。
- MTP cache 准备和 commit 属于 Module。
- 是否调用这些扩展、调用顺序和循环次数仍由 Engine 决定。

### 7.4 Session 和 cache

设备状态属于 Module：

- KV cache。
- Conv cache。
- Recurrent state。
- Draft/verify cache。
- Runtime model handle。
- Device buffer。

具体 Module 可提供 `clear_session()`，但不得自行决定请求级清理时机；是否清理由 Engine 决定。

### 7.5 Module 禁止事项

Module 不得：

- 构造 messages 或 chat template。
- Tokenization 或 detokenization。
- 图像、音频业务前处理。
- CPU embedding lookup。
- Sampling。
- EOS、长度和 context 停止判断。
- 决定 prefill chunk 数、decode 次数或 speculative 轮数。
- `yield` 或打印用户结果。
- 解析 CLI 参数。

## 八、层间数据规范

公共定义位于 `core/types.py`。

### 8.1 `Stage`

当前阶段：

```text
VISION
ENCODE
PREFILL
DECODE
MTP_PREFILL
DRAFT
VERIFY
```

新增可执行图阶段时必须扩展 `Stage`，不得使用无语义字符串散落在 Engine 中。

### 8.2 `StageInputs` 和 `StageOutputs`

```python
@dataclass
class StageInputs:
    tensors: tuple[Any, ...]
    metadata: dict[str, Any]

@dataclass
class StageOutputs:
    tensors: tuple[Any, ...]
    metadata: dict[str, Any]
```

规则：

- `tensors` 表示当前阶段的语义 tensor 顺序。
- 非显然信息放入 `metadata`。
- Module 必须把输入 metadata 传递到输出。
- 跨层不得传递含义不明的裸 tuple。
- 复杂请求和状态使用模型专用 dataclass。

### 8.3 `GenerationState`

只保存 Engine 管理的 CPU 状态，不保存设备 cache、Runtime model 或 device tensor。

## 九、性能统计规范

每个 Engine 创建一个 `PerfTracker`，并将同一实例注入 Process 和 Module。

| 指标 | 归属 |
| --- | --- |
| init 总阶段、E2E、TTFT | Engine |
| vision、encode、prefill、decode、draft、verify 总阶段 | Engine |
| set_input、infer、get_output | Module |
| tokenize、embedding、图片/音频 preprocess、文本 postprocess | Process |

禁止多个层记录同名总阶段。

### 9.1 计时边界

修改性能代码时必须明确以下操作是否计入 TTFT/E2E：

- Session/cache 清理。
- Tokenization 和多模态前处理。
- 首 token sampling 和文本后处理。
- Generator `yield` 期间的消费者等待。

不得仅比较指标名称而忽略计时边界。新旧 Demo 对比时，应增加独立 scope 定位未统计耗时。

当前 Qwen3.5 和 Qwen3.6 MTP Engine 在请求 TTFT/E2E 计时开始前执行 `clear_session()`；调整该顺序会改变指标口径，必须同步说明和验证。

## 十、源码和文件规则

### 10.1 Python header

本目录所有 `.py` 文件必须使用 2026 Apache-2.0 header：

```python
# Copyright (c) 2026 HOUMO AI
#
# File: <file_name>.py
# Description:
#   <concise description>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
```

不得重复添加 header，不得保留错误年份。

### 10.2 命名

同一模型三层类型必须对应：

```text
Qwen35Engine / Qwen35Process / Qwen35Module
Qwen36MtpEngine / Qwen36MtpProcess / Qwen36MtpModule
Qwen3AsrEngine / Qwen3AsrProcess / Qwen3AsrModule
```

禁止含义不明确的公开名称，例如 `Runner`、`VLMEngine`、`ASREngine`。

### 10.3 缓存和生成物

不得提交：

```text
__pycache__/
*.pyc
.pytest_cache/
build/
dist/
*.egg-info/
```

普通 Python 导入会自动生成 `__pycache__`。需要避免生成字节码时使用：

```bash
PYTHONDONTWRITEBYTECODE=1 python <command>
```

语法检查优先使用不写入字节码的 `compile()`：

```bash
python -c 'from pathlib import Path; [compile(p.read_text(), str(p), "exec") for p in Path("src").rglob("*.py")]'
```

如执行会生成缓存的工具，完成后必须删除本任务产生的 `__pycache__`。

## 十一、测试和验证

当前公共引擎目录没有独立硬件测试套件。改动至少完成以下无硬件验证。

### 11.1 基础接口

- `HoumoEngine` 只抽象 `generate()`。
- `ModelProcess` 只抽象 `preprocess()` 和 `postprocess()`。
- `HoumoModule` 抽象 `load()`、`set_input()`、`run()`、`get_output()`。
- 所有具体类不应残留未实现抽象方法。
- 具体类必须直接继承对应基础类。

### 11.2 Engine 调用顺序

使用 fake Process 和 fake Module 验证：

```text
preprocess
  -> prepare stage inputs
  -> set_input
  -> run
  -> get_output
  -> sampling
  -> postprocess
```

还应验证：

- Process 生成的 `StageInputs` 原样交给 Module。
- Module logits 只由 Engine sampling。
- EOS 不进入不必要的下一阶段。
- `max_new_tokens` 和 context 边界。
- Session 清理语义。
- 增量输出和最终 remainder 不重复、不丢失。
- MTP draft/verify 接受逻辑和 cache commit。

### 11.3 Demo 验证

至少执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python models/llm/qwen3.5/python/demo.py --help
PYTHONDONTWRITEBYTECODE=1 python models/llm/qwen3.5/python/demo_mtp.py --help
PYTHONDONTWRITEBYTECODE=1 python models/asr/qwen3-asr/python/demo.py --help
```

并确认 `--help` 后：

- `torch` 未加载。
- `tcim_lite` 未加载。
- 未访问 NPU。
- 未创建 `__pycache__`。

### 11.4 真实设备回归

有模型资产和 NPU 时，根据改动范围回归：

- Qwen3.5 文本输入。
- Qwen3.5 图片输入。
- Qwen3.5 多轮会话和 `keep_history=True/False`。
- Qwen3.6 MTP 输出文本、token 数、draft 数、接受率和 cache 行为。
- Qwen3-ASR 转写文本、chunk 数、token 数和 RTF。
- 默认参数和显式 CLI 参数。
- `batch=1`。
- 性能报告及新旧 Demo 统计边界。

## 十二、合入检查清单

- [ ] Demo 只组合一个具体 Engine，并统一调用 `generate()`。
- [ ] Demo `--help` 不加载模型、Runtime 或 NPU。
- [ ] CLI 参数包含 `dest/type/default/help`。
- [ ] `batch` 已从 Demo 转发到 Engine，当前模型只接受 1。
- [ ] 具体 Engine 直接继承 `HoumoEngine`。
- [ ] 具体 Process 直接继承 `ModelProcess`。
- [ ] 具体 Module 直接继承 `HoumoModule`。
- [ ] 未增加空任务中间层或 Runner。
- [ ] Engine 显式调用 Module 四步接口。
- [ ] Engine 不直接调用 HMM/Runtime。
- [ ] Process 不导入 `tcim_lite`。
- [ ] Module 不处理 tokenizer、sampling 和生成循环。
- [ ] 请求 CPU 状态和设备状态已分离。
- [ ] 跨层使用 `StageInputs`、`StageOutputs` 和明确 dataclass。
- [ ] PerfTracker 在三层共享，指标边界没有重复。
- [ ] 所有 Python 文件包含 2026 Apache-2.0 header。
- [ ] 没有 `__pycache__`、模型资产或构建产物进入变更。
- [ ] 语法检查、Demo `--help` 和必要的真实设备回归通过。

## 十三、职责判断

职责不明确时按以下问题判断：

- “用户如何调用？”属于 Demo / `HmXxx`。
- “下一步执行哪个阶段？”属于 Engine。
- “是否继续生成、是否停止、如何 Sampling？”属于 Engine。
- “用户数据如何转换为请求或阶段输入？”属于 Process。
- “模型 token 如何转换为稳定增量或最终输出？”属于 Process。
- “HMM 如何加载、绑定输入、执行和读取输出？”属于 Module。
- “设备 cache 如何绑定、复制和清理？”属于 Module，由 Engine 决定调用时机。
