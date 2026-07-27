# Houmo Python Engine

`houmo-python-engine` 是 imodelzoo 仓库内共享的 Houmo NPU Python 推理引擎。
它将用户 Demo、推理流程编排、CPU 数据处理和 TCIM Runtime 图执行分离，供不同模型复用统一的分层结构和公开接口。

首个版本直接使用仓库源码，不安装 Python package，不构建 wheel，也不修改全局 `PYTHONPATH`。

## 目录结构

```text
houmo_engine/
├── core/       # 基础接口和跨层数据类型
├── engine/     # 请求校验、阶段编排、Sampling 和生成状态
├── process/    # 前处理、阶段输入构造和用户输出后处理
├── module/     # Runtime 图加载、执行和设备 cache 管理
├── perf/       # 统一性能统计和报告
└── sampling/   # 确定性 Sampling 和 logits 处理
```

## 当前支持模型

- `Qwen35Engine`：Qwen3.5 文本和图片生成。
- `Qwen36MtpEngine`：Qwen3.6 MTP 推测生成。
- `Qwen3AsrEngine`：Qwen3-ASR 音频输入生成。

所有模型统一使用一个公开推理方法：

```python
generate(...)
```

不再为不同任务分别提供 `chat()`、`transcribe()` 或 `synthesize()` 等 Engine 接口。

## 继承关系

当前只保留具有明确公共契约的三类基础接口，不增加没有公共实现的 LLM、ASR、TTS 空中间层。

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

如果后续多个模型出现稳定且可复用的任务级公共实现，再从真实重复代码中提取对应中间层。

## Engine

`HoumoEngine` 提供公共 `batch` 字段，并要求具体 Engine 实现 `generate()`：

```python
class HoumoEngine(ABC):
    def __init__(self, batch: int = 1):
        ...

    @abstractmethod
    def generate(self, request, **kwargs):
        ...
```

具体 Engine 是完整推理流程的唯一编排者，负责：

- 请求参数校验。
- 创建和管理请求状态。
- 决定阶段执行顺序。
- Sampling。
- EOS、最大生成长度和 context 停止判断。
- 流式输出。
- TTFT、E2E 和阶段总耗时统计。

当前接入的三个模型仅支持 `batch=1`，传入其他值时会直接报错。

## Process

`ModelProcess` 定义两个公共接口：

```python
preprocess(...)
postprocess(state, final=False)
```

具体 Process 负责：

- Tokenization 和 chat template。
- 图片、音频等用户输入的 CPU 前处理。
- CPU embedding lookup。
- 构造各模型阶段的 `StageInputs`。
- 多模态 embedding 融合。
- 将生成 token 转换为增量输出或最终输出。

模型专属方法继续保留在具体 Process 中，例如：

```text
prepare_prefill_chunk()
prepare_decode()
prepare_encode()
merge_encode()
prepare_draft()
prepare_verify()
merge_vision()
```

## Module

`HoumoModule` 定义四个相互独立的 Runtime 操作：

```python
load(...)
set_input(stage, inputs)
run(stage)
get_output(stage)
```

Engine 执行一个普通模型阶段时使用以下顺序：

```text
Process 构造 StageInputs
  -> Module.set_input()
  -> Module.run()
  -> Module.get_output()
  -> Engine Sampling 并决定下一阶段
```

具体 Module 独占以下职责：

- 创建设备和 Runtime manager。
- 加载 HMM/HMMS 图。
- Runtime 输入绑定。
- 图执行和同步。
- Runtime 输出读取。
- KV、conv、recurrent、draft/verify 等设备 cache 管理。

模型专属 Runtime 操作仍作为具体 Module 的扩展方法保留，例如：

- Qwen3.5 的 `run_vision()`。
- Qwen3.6 MTP 的 `prepare_verify_from_prefill()`。
- Qwen3.6 MTP 的 `commit_verify_cache()`。

## 层间数据

公共层间类型定义在：

```text
houmo_engine/core/types.py
```

主要类型包括：

```python
Stage
StageInputs
StageOutputs
GenerationState
```

跨层阶段输入和输出必须使用 `StageInputs`、`StageOutputs`，不得传递含义不明的裸 tuple。

## 源码使用方式

模型 Demo 在导入 `houmo_engine` 前，将公共源码目录加入当前进程的 `sys.path`：

```python
import sys
from pathlib import Path

imodelzoo_root = Path(__file__).resolve().parents[4]
engine_src = (
    imodelzoo_root
    / "common"
    / "python"
)
sys.path.insert(0, str(engine_src))
```

具体 Engine 从 package 根目录延迟导入：

```python
from houmo_engine import Qwen35Engine
from houmo_engine import Qwen36MtpEngine
from houmo_engine import Qwen3AsrEngine
```

基础接口和层间类型可以按需导入：

```python
from houmo_engine import HoumoEngine, HoumoModule, ModelProcess
from houmo_engine.core import GenerationState, Stage, StageInputs, StageOutputs
```

仅执行 `import houmo_engine` 不会加载模型图或访问 NPU。

## Demo

当前基于新框架的 Demo 位于：

```text
models/llm/qwen3.5/python/demo.py
models/llm/qwen3.5/python/demo_mtp.py
models/asr/qwen3-asr/python/demo.py
```

查看参数：

```bash
python models/llm/qwen3.5/python/demo.py --help
python models/llm/qwen3.5/python/demo_mtp.py --help
python models/asr/qwen3-asr/python/demo.py --help
```

Demo 将具体 Engine 导入延迟到模型构造阶段，因此执行 `--help` 时不会加载 HMM、初始化 Runtime 或访问 NPU。

## 依赖边界

公共引擎源码使用以下共享 Python 依赖：

- `loguru`
- `numpy`
- `torch`

以下平台 Runtime 依赖由 Dadao 环境提供，不通过本目录安装：

- `tcim_lite`
- `hmatc`

模型专属依赖由对应模型目录的 `requirements.txt` 管理。例如 Qwen3.5 和 Qwen3-ASR 使用不同版本的 `transformers`，公共引擎不统一声明或覆盖该依赖。

## 性能统计

每个具体 Engine 创建一个 `PerfTracker`，并将同一实例注入对应 Process 和 Module。

- Engine 记录请求级 E2E、TTFT 和阶段总耗时。
- Process 记录前处理、embedding 和输出后处理。
- Module 记录 `set_input`、图推理和 `get_output`。

对比新旧 Demo 的性能数据时，必须先确认统计边界一致，尤其需要确认以下操作是否计入 TTFT 或 E2E：

- Session/cache 清理。
- Tokenization 和多模态前处理。
- 首 token 后处理。
- Generator `yield` 期间的消费者等待时间。

## 首版约束

- 不安装 `houmo-python-engine`。
- 不构建 wheel 或 source distribution。
- 不使用 editable install。
- 不修改全局 `PYTHONPATH`。
- 不替换或删除模型目录中的原始 Demo。
- 新 Demo 与原始 Demo 并行存在，便于行为和性能对比。
