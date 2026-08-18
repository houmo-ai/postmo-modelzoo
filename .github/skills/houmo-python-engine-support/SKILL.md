---
name: houmo-python-engine-support
description: 将 imodelzoo 中已有的单文件 Python 推理 Demo 等价迁移到 `utils/python/houmo_engine` 的 Demo/Engine/Process/Module 新框架。Use when 用户要求迁移或重构现有 `demo.py`、`demo_mtp.py`、`demo_asr.py`，并需要保留原始 CLI、图执行顺序、Sampling、cache、流式输出和性能口径。
---

# 现有单文件 Demo 迁移到 Houmo Python Engine

## 目标

使用本 Skill 将 imodelzoo 中一个已经可运行的单文件 Python Demo 等价拆分为：

```text
模型目录/python/demo.py
  -> XxxEngine
     -> XxxProcess
     -> XxxModule
```

迁移请求通常只会提供以下任意一项：

- 目标模型目录。
- 原始 Demo 路径。
- “把这个 Demo 迁移到 houmo-python-engine”。

收到请求后，主动阅读源码、分析行为、完成改造和验证。不要要求用户预先整理完整设计。只有原始代码和仓库信息无法确定关键语义时，才提出一个简短、具体的问题。

## 适用范围

典型触发场景：

- “把 `models/.../demo.py` 迁移到新框架”
- “参考 Qwen3.5，把现有单文件 Demo 拆成 Engine/Process/Module”
- “迁移 `demo_mtp.py`，保留 MTP cache 和性能行为”
- “迁移 ASR Demo，统一对外使用 `generate()`”
- “旧 Demo 保留，新建 `python/demo.py`”
- “对齐新旧 Demo 的输出、token 数、TTFT 或 E2E”

不适用于：

- 没有任何现有 Python Demo 的全新模型设计。
- C++ Engine 迁移。
- 只修改量化、编译或模型下载流程。
- 只修复原始 Demo 的局部 bug，且不接入公共框架。

## 规范真值源

开始工作前必须阅读：

1. `utils/python/houmo_engine/CLAUDE.md`
2. `utils/python/houmo_engine/README.md`
3. 原始单文件 Demo
4. 目标模型目录的 `config.yaml`、`test.sh`、`README.MD`
5. 与目标任务最接近的标准实现

标准参考：

```text
Qwen3.5 文本/图片
├── models/llm/qwen3.5/python/demo.py
├── utils/python/houmo_engine/engine/qwen3_5.py
├── utils/python/houmo_engine/process/qwen3_5/process.py
└── utils/python/houmo_engine/module/qwen3_5.py

Qwen3.6 MTP
├── models/llm/qwen3.5/python/demo_mtp.py
├── utils/python/houmo_engine/engine/qwen3_6_mtp.py
├── utils/python/houmo_engine/process/qwen3_6_mtp/process.py
└── utils/python/houmo_engine/module/qwen3_6_mtp.py

Qwen3-ASR
├── models/asr/qwen3-asr/python/demo.py
├── utils/python/houmo_engine/engine/qwen3_asr.py
├── utils/python/houmo_engine/process/qwen3_asr/process.py
└── utils/python/houmo_engine/module/qwen3_asr.py
```

若本 Skill 与 `CLAUDE.md` 或当前源码不一致，以 `CLAUDE.md` 和当前源码为准。

## 迁移硬规则

1. 原始单文件 Demo 是行为真值源。
2. 第一阶段只做等价拆分，不同时做算法优化。
3. 原始 Demo 默认只读，不删除、不覆盖、不重命名。
4. 新 Demo 放在模型目录的 `python/` 子目录，与原 Demo 并行存在。
5. 先对齐输出和状态，再对齐性能，最后才允许优化。
6. 所有模型统一公开 `generate()`，不增加 `chat()`、`transcribe()`、`synthesize()`。
7. 具体 Engine、Process、Module 直接继承基础类，不增加空任务中间层或 Runner。
8. 无法从原 Demo 证明的行为不得自行补充。
9. 缺少模型资产或 NPU 时，只能声明完成静态迁移和无硬件验证，不能声明真实行为或性能已对齐。
10. 不修改与迁移无关的量化、编译、下载或测试框架代码。当前模型测试会对相对 Demo basename 自动优先选择 `python/` 下同名脚本，不要为单个模型重复实现路径特判。

默认不得修改：

```text
原始 demo.py / demo_mtp.py / demo_asr.py
build.py
ptq.py
get_model.py
test.sh
```

只有用户明确要求、新 Demo 接入测试流程属于任务范围，或原文件存在明确阻塞问题时才能修改，并在最终报告中单独说明。

## 执行方式

收到迁移请求后，直接执行以下流程：

1. 定位并阅读原 Demo 和相关配置。
2. 提取迁移基线和职责映射。
3. 创建或修改新 Demo、Engine、Process、Module。
4. 更新必要导出和公共类型。
5. 执行无硬件验证。
6. 有设备和资产时运行新旧 Demo 对比。
7. 输出统一迁移报告。

不要停留在方案描述；除非用户只要求计划，否则应完成实际代码修改和验证。

## Gate 1：建立原始 Demo 基线

### 1.1 识别入口和用户行为

必须确认：

- 原 Demo 路径和运行命令。
- 用户输入类型：文本、图片、音频、视频或组合。
- 输出类型和流式方式。
- 是否支持交互模式。
- 是否支持多轮历史。
- CLI 参数、默认值和动态配置来源。
- `test.sh` 是否调用该 Demo，以及使用哪些参数。

### 1.2 识别模型图和阶段

列出：

- 所有 HMM/HMMS 路径。
- 图的输入、输出和固定 shape。
- 图的执行顺序。
- 每个阶段的循环条件和次数。
- 首 token 来源。
- Sampling、EOS、最大长度和 context 判断的位置。

典型阶段包括：

```text
VISION
ENCODE
PREFILL
DECODE
MTP_PREFILL
DRAFT
VERIFY
VOCODE
```

### 1.3 识别 cache

必须记录：

- Cache 类型：KV、conv、recurrent、draft、verify 等。
- Cache tensor 名称、shape 和 dtype。
- Prefill、Decode、Draft、Verify 之间的绑定关系。
- 使用 `set_input`、`set_dev_input` 还是设备输出直连。
- 初始化时是否清零。
- 每请求、每 chunk、每轮会话是否清零。
- 多轮会话是否保留。
- Valid length 如何决定 cache 的有效区域。
- MTP accepted step 如何选择和提交 cache。

硬规则：

```text
valid_length=0 不等于必须物理清零整个 cache。
```

必须复刻原 Demo 的实际语义，不得因为 cache 中存在旧数据就擅自增加清零。

### 1.4 识别 Sampling 和停止条件

记录：

- Logits shape 和选取位置。
- 是否 reshape 或 squeeze。
- Temperature。
- Top-k、top-p。
- Repetition penalty、presence penalty。
- 是否执行 softmax。
- 最终使用 argmax 还是随机采样。
- Previous tokens 的范围。
- EOS/stop token 集合。
- 首 token 是否计入输出长度。

迁移阶段必须保持原算法和默认参数，不要先做“等价优化”。

### 1.5 识别流式输出

记录原 Demo 是否：

- 每步 decode 全部历史 token。
- 使用滑动窗口。
- 保留末尾 N 个不稳定 token。
- 按字符范围判断稳定输出。
- 过滤特殊 token 或标签。
- 在结束时补发 remainder。

### 1.6 识别性能边界

必须定位代码中的开始和结束语句，而不是只看日志名称：

- TTFT 起点和终点。
- E2E 起点和终点。
- Cache/session 清理是否计入。
- Tokenization、embedding、首 token 后处理是否计入。
- 终端打印是否计入。

### 1.7 形成职责映射表

在开始编码前，至少在工作记录中形成：

| 原 Demo 代码 | 原行为 | 新层归属 |
| --- | --- | --- |
| CLI/config | 参数和动态默认值 | Demo |
| Tokenizer/Processor | 用户数据预处理 | Process |
| Embedding lookup | CPU embedding | Process |
| HMM load | Runtime 图加载 | Module |
| set_input/run/get_output | Runtime 执行 | Module |
| Cache bind/reset/commit | 设备状态 | Module，Engine 决定时机 |
| Prefill/decode loop | 阶段顺序 | Engine |
| Sampling/EOS | 生成控制 | Engine |
| Streaming decode | 用户输出 | Process |
| TTFT/E2E | 请求级统计 | Engine |

一个旧函数可能同时包含三层职责，不能按函数整体复制，必须按代码块拆分。

Gate 1 完成标准：原始 Demo 的用户接口、图、cache、Sampling、输出和计时边界都能够说明。

## Gate 2：创建并行新 Demo

新 Demo 默认放在：

```text
models/<task>/<model>/python/demo.py
```

独立模式使用明确文件名，例如：

```text
python/demo_mtp.py
```

### 2.1 源码导入

```python
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_SRC = (
    IMODELZOO_ROOT
    / "utils"
    / "python"
)
sys.path.insert(0, str(ENGINE_SRC))
```

首版不安装 package，不修改全局 `PYTHONPATH`。

### 2.2 用户封装

```python
class HmXxx:
    def __init__(self, ..., batch: int = 1):
        from houmo_engine import XxxEngine

        self.engine = XxxEngine(..., batch=batch)

    def generate(self, request, ...):
        yield from self.engine.generate(request, ...)

    def print_perf(self) -> None:
        self.engine.perf.print_summary()
```

规则：

- 只组合一个 Engine。
- 具体 Engine 在 `HmXxx.__init__()` 中延迟导入。
- 对外只提供 `generate()`。
- 支持历史时设置 `keep_history=True`。
- 不创建 Process 或 Module。
- 不实现内部阶段、Sampling 或停止判断。

### 2.3 CLI 对齐

尽量保持原 Demo 参数名和默认行为，避免迁移后用户命令失效。

`get_args()` 必须只返回 Parser：

```python
def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ...
    return parser
```

每个参数显式设置：

```text
dest
type
default
help
```

必须包含：

```text
--batch
--ndevice
```

当前模型 `batch` 默认且仅支持 1。

动态默认值在 `parse_args()` 后解析，`--help` 不得加载模型、Processor、Runtime 或 NPU。

### 2.4 输出格式

```python
print(f"\033[1;95m\nQ: {user_input}\nA: ", end="", flush=True)
for chunk in model.generate(...):
    print(f"\033[1;95m{chunk}", end="", flush=True)
print()
```

Gate 2 完成标准：新 Demo 用户接口清晰，参数与原 Demo 对齐，`--help` 可在无设备环境运行。

## Gate 3：提取 Process

路径：

```text
utils/python/houmo_engine//process/<model_name>/
├── __init__.py
└── process.py
```

```python
class XxxProcess(ModelProcess):
    def preprocess(self, ...) -> PreparedRequest:
        ...

    def postprocess(self, state, *, final: bool = False) -> str:
        ...
```

从原 Demo 搬入：

- Tokenizer、Processor、chat template。
- 图片、视频、音频前处理。
- CPU embedding 权重和 lookup。
- PreparedRequest dataclass。
- Position IDs、attention mask、padding、valid/current length。
- Vision/encode 业务融合。
- Token decode、稳定增量和最终 remainder。

按需增加：

```text
prepare_encode()
merge_encode()
prepare_prefill_chunk()
prepare_decode()
prepare_mtp_prefill_chunk()
prepare_draft()
prepare_verify()
merge_vision()
```

Process 禁止：

- 导入 `tcim_lite`。
- 加载 HMM/HMMS。
- 调用 Runtime 或 Module。
- 操作设备 cache。
- Sampling、EOS 和停止判断。
- 控制完整阶段循环。
- 直接打印。

### Process 输出对齐

使用固定 token IDs 验证：

```python
state.generated_ids = [...]
first = process.postprocess(state)
final = process.postprocess(state, final=True)

assert first + final == expected_text
assert process.postprocess(state, final=True) == ""
```

必须满足：

- 不重复。
- 不丢失。
- 最终调用幂等。
- 特殊 token 和标签过滤与原 Demo 一致。

Gate 3 完成标准：Process 纯 CPU、无 Runtime 依赖，并且给定相同输入或 token IDs 时与原 Demo 数据处理一致。

## Gate 4：提取 Module

路径：

```text
utils/python/houmo_engine/module/<model_name>.py
```

```python
class XxxModule(HoumoModule):
    def load(self, ...) -> None:
        ...

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        ...

    def run(self, stage: Stage) -> None:
        ...

    def get_output(self, stage: Stage) -> StageOutputs:
        ...
```

从原 Demo 搬入：

- 设备和 manager 创建。
- Runtime option。
- HMM/HMMS 加载。
- Dummy tensor。
- 输入输出名称和 shape。
- `set_input`、`run/sync`、`get_output`。
- KV、conv、recurrent、draft/verify cache。
- Cache reset、bind、copy 和 commit。

职责分开：

```text
load
  -> 加载图、读取 metadata、绑定和初始化 cache

set_input
  -> stage dispatch、shape/dtype 适配、绑定输入

run
  -> model.run、model.sync

get_output
  -> 读取输出、必要的 copy/截断、返回 StageOutputs
```

不得重新聚合成：

```python
module.run(stage, inputs) -> StageOutputs
```

特殊模型行为使用具名方法，例如：

- `run_vision()`。
- `prepare_verify_from_prefill()`。
- `commit_verify_cache()`。

Module 禁止：

- Tokenization、detokenization。
- 图片或音频业务处理。
- CPU embedding lookup。
- Sampling 和停止判断。
- 决定循环次数。
- `yield` 或打印。

Gate 4 完成标准：图签名、输入顺序、dtype、输出 shape、cache 绑定和清理行为与原 Demo 一致。

## Gate 5：实现 Engine 编排

路径：

```text
utils/python/houmo_engine/engine/<model_name>.py
```

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

Engine 负责从原 Demo 搬入：

- 参数校验和默认值。
- 是否清理 session 的决策。
- 请求状态重置。
- 图阶段顺序。
- Prefill chunk、decode、draft/verify 循环。
- Sampling。
- EOS、最大长度和 context 判断。
- 流式 `yield`。
- TTFT、E2E 和阶段总时间。

每个普通阶段显式执行：

```python
inputs = self.process.prepare_*(...)
self.module.set_input(stage, inputs)
self.module.run(stage)
outputs = self.module.get_output(stage)
```

独立阶段拆为：

```text
_vision()
_encode()
_prefill()
_decode()
_draft()
_verify()
_vocode()
```

Engine 禁止：

- 导入 `tcim_lite`。
- 直接调用 HMM API。
- 按 HMM 名称或下标绑定输入。
- 实现 tokenizer、图片或音频前处理。
- 操作具体设备 cache。

Gate 5 完成标准：阶段顺序、循环、Sampling、停止条件和 session 决策与原 Demo 一致。

## Gate 6：导出和并行接入

按需更新：

```text
src/houmo_engine/engine/__init__.py
src/houmo_engine/process/__init__.py
src/houmo_engine/module/__init__.py
src/houmo_engine/__init__.py
```

根 package 对具体 Engine 使用延迟导入，保证：

```python
import houmo_engine
```

不会加载模型或访问 NPU。

命名必须对应：

```text
XxxEngine / XxxProcess / XxxModule
```

文件名使用模型规范名称，例如：

```text
qwen3_5.py
qwen3_6_mtp.py
qwen3_asr.py
```

原始 Demo 继续保留；除非用户明确要求，本 Gate 不修改 `test.sh` 的默认入口。

检查 `tests/models_tests/model_configs/model_cfg_<model>.json` 的 Demo 配置：默认 `demo.py` 和 `script` 指定的相对 basename 都会优先解析到模型目录的 `python/` 子目录，不存在才回退根目录。新 Demo 保持原 basename 时通常不需要修改 JSON；只有 basename 或 CLI contract 改变时才同步 `script` 和参数列。README 若介绍新入口，应使用实际命令，例如 `python3 python/demo.py`。

Gate 6 完成标准：新 Demo 可独立执行，旧 Demo 仍存在，导入路径和公开类型正确。

## Cache 对齐检查表

- [ ] 所有 cache tensor 已列出。
- [ ] Shape 和 dtype 已记录。
- [ ] Prefill/Decode/Draft/Verify 绑定关系一致。
- [ ] 初始化清理行为一致。
- [ ] 每请求清理行为一致。
- [ ] 每 chunk 清理行为一致。
- [ ] 多轮保留行为一致。
- [ ] Valid length 语义一致。
- [ ] Host copy 与 device tensor 直连行为一致。
- [ ] MTP accepted step 和 commit 输出一致。

## Sampling 对齐检查表

- [ ] Logits 选取位置一致。
- [ ] Temperature 一致。
- [ ] Top-k、top-p 一致。
- [ ] Repetition/presence penalty 一致。
- [ ] Previous tokens 范围一致。
- [ ] Softmax/argmax 或随机策略一致。
- [ ] Stop token 集合一致。
- [ ] 首 token 和 EOS 的计数口径一致。

对齐前不得把 `softmax + argmax` 改成快速 argmax，即使数学上看似等价；优化必须在行为一致后单独进行并验证。

## Streaming 对齐检查表

- [ ] 滑动窗口大小一致。
- [ ] 稳定字符判断一致。
- [ ] 特殊 token 和标签过滤一致。
- [ ] 每次增量不重复。
- [ ] 最终 remainder 不丢失。
- [ ] `final=True` 重复调用返回空字符串。

## 性能对齐流程

按以下顺序分析，不要先看 TTFT/E2E 就判断 NPU 变慢。

### 1. 比较底层 infer

```text
Vision/Encode infer
Prefill infer
Decode infer
Draft infer
Verify infer
```

如果底层 infer 基本一致，优先排查 Python、Runtime 管理和统计边界。

### 2. 比较非 infer 阶段

```text
cache reset
cache propagation
tokenize
embedding
set_input
get_output
sampling
postprocess
```

### 3. 对齐 TTFT/E2E 起止点

必须明确：

```text
原 Demo TTFT 起点：
新 Engine TTFT 起点：
原 Demo TTFT 终点：
新 Engine TTFT 终点：

原 Demo E2E 起点：
新 Engine E2E 起点：
原 Demo E2E 终点：
新 Engine E2E 终点：
```

如果 TTFT 差异超过约 10%，但 infer 基本一致，优先检查 cache reset、tokenization、get_output 和计时边界。

Generator 在 `yield` 时会暂停；如果 E2E 排除消费者等待，必须在报告中明确说明。

## 无硬件验证

即使没有模型和 NPU，也必须完成：

### 语法

```bash
python -c 'from pathlib import Path; [compile(p.read_text(), str(p), "exec") for p in Path("utils/python/houmo_engine").rglob("*.py")]'
```

### Demo help

```bash
PYTHONDONTWRITEBYTECODE=1 python <new_demo.py> --help
```

确认：

- 成功退出。
- 未加载 `torch`。
- 未加载 `tcim_lite`。
- 未访问 NPU。
- 未生成 `__pycache__`。

### 继承和接口

- `XxxEngine.__bases__ == (HoumoEngine,)`。
- `XxxProcess.__bases__ == (ModelProcess,)`。
- `XxxModule.__bases__ == (HoumoModule,)`。
- 具体类没有残留 abstractmethod。
- 只有统一 `generate()` 任务入口。

### 静态边界

- 没有 Runner。
- 没有旧的 `prepare_request()`、`stream_delta()`、`final_delta()`。
- 没有 `module.run(stage, inputs)`。
- Process 不导入 `tcim_lite`。
- Engine 不调用 HMM API。
- 所有公共引擎 Python 文件包含 2026 Apache-2.0 header。
- `git diff --check` 通过。

### Fake 调用顺序

至少验证：

```text
preprocess
-> prepare stage inputs
-> set_input
-> run
-> get_output
-> sampling
-> postprocess
```

Fake Module 最小形式：

```python
class FakeModule:
    def __init__(self):
        self.calls = []

    def set_input(self, stage, inputs):
        self.calls.append(("set_input", stage, inputs))

    def run(self, stage):
        self.calls.append(("run", stage))

    def get_output(self, stage):
        self.calls.append(("get_output", stage))
        return StageOutputs(tensors=(fake_logits,))
```

## 真实设备对照

有模型资产和 NPU 时，先运行原 Demo，再运行新 Demo。为减少冷启动偏差，必要时交换运行顺序或重复多轮并报告中位数。

通用对比：

- 完整输出。
- Input Tokens。
- Output Tokens。
- 图执行次数。
- EOS 和最大长度行为。
- Cache 清理和复用。
- TTFT、E2E。
- 各图 infer 时间。

MTP 额外对比：

- Speculative Rounds。
- Draft Tokens。
- Accepted Draft Tokens。
- Acceptance Rate。
- Drafts Per Round。

ASR 额外对比：

- Audio Duration。
- Chunk Count。
- Encode/Prefill/Decode 次数。
- Overall RTF。
- Inference RTF。

只有以下项目都能够解释时，才能声明迁移完成：

```text
输出一致
token/阶段计数一致或差异有明确口径说明
cache 行为一致
性能差异已定位
```

## 缺少设备或资产时

如果缺少 HMM/HMMS、tokenizer、Processor、embedding、输入样本、Runtime 或 NPU，最终必须明确区分：

```text
已完成静态迁移
已完成无硬件验证
未完成真实设备回归
```

不得声称：

- 输出已经一致。
- Cache 已经对齐。
- 性能已经对齐。

## 文件和生成物

公共引擎所有 `.py` 文件使用 2026 Apache-2.0 header。具体格式以 `utils/python/houmo_engine/CLAUDE.md` 为准。

不得提交：

```text
__pycache__/
*.pyc
.pytest_cache/
build/
dist/
*.egg-info/
模型文件
tokenizer 资产
虚拟环境
设备日志
```

优先使用：

```bash
PYTHONDONTWRITEBYTECODE=1 python <command>
```

## 最终迁移报告模板

完成后在最终响应中按以下结构汇报：

```markdown
## 迁移结果

### 原始入口
- 原 Demo：
- 原运行命令：
- 原 Demo 是否修改：否/是（原因）

### 新框架入口
- 新 Demo：
- Engine：
- Process：
- Module：

### 阶段和状态
- 阶段顺序：
- 循环方式：
- Sampling：
- Stop tokens：
- Session 语义：

### Cache
- Cache 类型：
- 绑定关系：
- 清理时机：
- 多轮/chunk 行为：

### 行为对齐
- 输出：
- Input Tokens：
- Output Tokens：
- 阶段次数：
- 特殊指标：

### 性能对齐
- TTFT：
- E2E：
- 各阶段 infer：
- 已知统计边界差异：

### 验证
- 无硬件验证：
- 真实设备回归：
- 未完成项：
```

## 完成交付清单

- [ ] 原始 Demo 已审计且保留。
- [ ] 已记录图、cache、Sampling、streaming 和性能边界。
- [ ] 新 Demo 位于模型目录 `python/` 下。
- [ ] 已确认 models_tests 按 `python/` 优先规则选中新 Demo；仅在 basename 或 CLI 改变时更新 JSON。
- [ ] Demo 只组合一个 Engine。
- [ ] Demo 和 Engine 统一使用 `generate()`。
- [ ] CLI 与原 Demo 兼容或差异已说明。
- [ ] `--help` 不加载 Runtime/NPU。
- [ ] 已实现 `XxxEngine/XxxProcess/XxxModule`。
- [ ] 三个具体类直接继承基础类。
- [ ] Engine 显式执行 Module 四步接口。
- [ ] Process、Module、Engine 职责未越界。
- [ ] 请求 CPU 状态与设备 cache 分离。
- [ ] Cache 绑定和清理时机与原 Demo 一致。
- [ ] Sampling 和 stop 行为一致。
- [ ] Streaming 不重复、不丢失。
- [ ] 层间使用 `StageInputs/StageOutputs` 和明确 dataclass。
- [ ] PerfTracker 在三层共享且统计边界明确。
- [ ] 公共引擎 Python 文件包含 2026 header。
- [ ] 无 `__pycache__` 或资产进入提交。
- [ ] 无硬件验证通过。
- [ ] 有设备时完成新旧 Demo 对照。
- [ ] 最终报告明确已完成和未完成项。

## 职责快速判断

- 用户调用、CLI、终端打印：Demo。
- 阶段顺序、循环、Sampling、停止条件：Engine。
- Tokenizer、前处理、阶段输入、输出文本：Process。
- HMM load/set_input/run/get_output、设备 cache：Module。
