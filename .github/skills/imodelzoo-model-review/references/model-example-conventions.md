# Model Example Conventions

本 reference 定义 iModelzoo 当前并存的两套模型示例约定。评审 `models/**`、相关 `tests/models_tests/**`、README 或聚合配置时，先识别示例所属体系，再应用对应规则。

## 目录

- [分类原则](#分类原则)
- [大模型脚本体系](#大模型脚本体系)
- [CV 与 HMATC 体系](#cv-与-hmatc-体系)
- [test.sh 评审规则](#testsh-评审规则)
- [测试配置耦合](#测试配置耦合)
- [跨体系与迁移规则](#跨体系与迁移规则)

## 分类原则

按文件结构和执行入口分类，不要只根据模型名称判断：

- 大模型脚本体系主要位于 `models/llm/**`、`models/tts/**`、`models/vlm/**`、`models/diffusion/**`、`models/asr/**`、`models/embedding/**`、`models/omni/**`、`models/reranker/**`、`models/ocr/**`。
- CV 与 HMATC 体系主要位于 `models/backbone/**`、`models/detection/**`、`models/segmentation/**`、`models/estimation/**`、`models/autodrive/**`、`models/ocr/**`。
- 目录出现 `config.yaml`、`ptq.py`、`build.py`、`demo.py` 且阶段由模型脚本实现时，按大模型脚本体系评审。
- 目录出现 `config.yml`、`model_impl.py`、`dataset.py` 且阶段由 `hmatc` 执行时，按 CV 与 HMATC 体系评审。
- 混合或迁移中的示例按实际入口逐段检查，不要为了统一外观要求无关重构。

## 大模型脚本体系

### 文件命名和职责

常见稳定文件为：

```text
README.MD
config.yaml
get_model.py
ptq.py
build.py
demo.py
test.sh
requirements.txt
```

按模型能力允许缺少某个阶段，也允许存在 `demo_base.py`、`demo_mtp.py`、`demo_multibatch.py`、`quant_pipeline.py`、`cpp/` 等明确命名的变体。不要要求不支持量化的模型补空 `ptq.py`，也不要将专项 Demo 逻辑硬塞回 `demo.py`。

评审文件命名时检查：

- 使用 `config.yaml`，不要在同一大模型示例中同时引入 `config.yml` 作为第二套真值。
- 保持阶段脚本名为 `get_model.py`、`ptq.py`、`build.py`、`demo.py`，除非专项脚本确实表达独立运行模式。
- 专项脚本名应表达差异，如 `demo_base.py`、`demo_mtp.py`、`demo_multibatch.py`，并在 `test.sh`、测试 JSON 和 README 中使用同一名称。
- 不提交日志、临时副本、下载模型、编译产物或手工输出图片作为实现文件；示例基准资产确有必要时要求有明确用途和文档。

### config.yaml 结构

当前大模型配置使用以下层级：

```yaml
default_model_name: <name>
default_model_size: <size>
model_configs:
  <name>:
    <size>:
      model_name: <name>
      model_size: <size>
      modelscope_repo: [...]
      ncore: ...
      ndevice: ...
      batch: ...
      quant_type: ...
      context_length: ...
      prefill_length: ...
```

模型类型可增加 `max_size_w/max_size_h/max_size_t`、`cpp_backend`、子模型选项等字段。评审时检查：

- `default_model_name/default_model_size` 能在 `model_configs` 中定位到真实条目。
- 条目内重复的 `model_name/model_size` 与外层 key 一致。
- `get_model.py`、`ptq.py`、`build.py` 和 `demo.py` 读取同一个默认配置路径。
- 公共字段含义在各阶段一致，不把 `context_length`、`prefill_length`、batch 或 device 数解释成不同单位。
- 模型特有字段只由需要它的阶段消费，且 CLI override 不会被配置值重新覆盖。

### Python 参数解析

大模型脚本通常使用 `argparse`，并通过 `get_model_configs`、`first_not_none` 解析配置。应用以下优先级：

```text
显式 CLI 参数
    > 选中 model_name/model_size 对应的 config.yaml 字段
    > 脚本中与模型无关的安全默认值
```

评审时检查：

- `--config` 默认指向脚本同目录的 `config.yaml`，从其他工作目录启动时仍可定位。
- `--model_name` 和 `--model_size` 未提供时使用配置默认值；显式提供时选择对应 `model_configs` 条目。
- 不支持的 name/size 组合应尽早报错，并列出可用值；不要让空字典延迟到模型加载阶段失败。
- CLI 的 `0`、`false`、空 list 等合法值不会因 truthy/falsy 写法被错误替换。
- `get_model.py` 的 `--type raw|hmm`、下载目录、解压目录和模型变体共同决定正确资源。
- `ptq.py` 的输入模型、输出目录、quant type、context/prefill length、校准参数和子模型选择正确透传。
- `build.py` 的 model dir、output dir、batch、ncore、ndevice、context/prefill length 和 stage 正确透传。
- `demo.py` 的 HMM、embedding/tokenizer/processor、device、sampling 或任务参数从 CLI 到运行对象保持一致。
- 保留已经公开的 option 拼写。部分脚本使用 `--model_name`，部分量化脚本存在 `--model-name`；测试、`test.sh` 和 README 必须匹配实际 parser，不要假定下划线和连字符可互换。

### 产物命名

检查生产者和消费者一致，不强制所有模型只有一种 suffix：

- 量化产物通常位于 `output/${HOUMO_TARGET}/hmquant/`。
- 编译产物通常位于 `output/${HOUMO_TARGET}/`。
- LLM/VLM 常使用 `<model>-<size>_prefill.hmm`、`<model>-<size>_decode.hmm`。
- VLM 可增加 `_visual_<width>x<height>x<count>.hmm`。
- TTS/Diffusion 等多子图模型使用 `<model>-<size>_<role>.hmm`，role 必须与 build、Demo、测试 JSON 一致。
- 多设备产物可能使用 `.hmms`；根据 `ndevice` 选择 suffix，不要让单卡和多卡路径混用。
- embedding、processor、scheduler 或其他 Demo 依赖的路径必须跟随实际量化/下载产物目录。

## CV 与 HMATC 体系

### 文件命名和职责

当前 CV 示例的稳定文件为：

```text
README.MD
config.yml
get_model.py
model_impl.py
dataset.py
test.sh
```

量化、编译、Compare、Demo、Eval 和 Perf 由 HMATC 公共入口执行，因此通常不需要模型目录中的 `ptq.py`、`build.py`、`demo.py` 或 `eval.py`。评审时检查：

- 使用 `config.yml` 作为 HMATC 单一配置入口，不额外复制一份 `config.yaml`。
- `model_impl.py` 暴露配置中 `model_impl_module/model_impl_cls` 指定的类。
- `dataset.py` 暴露配置中 `dataset_module/dataset_cls` 指定的类。
- class 名、module 名和大小写与动态 import 完全一致。
- 模型特有预处理、后处理和 metric 放在 `model_impl.py`/`dataset.py` 或既有公共基类扩展点，不复制 HMATC CLI 流程。

### config.yml 结构

当前 HMATC 模型配置通常包含：

```yaml
model:
  name: ...
  save_dir: output
  model_path: ...
  inputs: ...
  model_impl_module: model_impl
  model_impl_cls: ...
quant:
  calib_data: ...
  calib_num: ...
build:
  ncore: ...
  opt_level: ...
demo:
  data_dir: ...
  num: ...
eval:
  data_dir: ...
  num: ...
  dataset_module: dataset
  dataset_cls: Dataset
```

评审时检查：

- `model.model_path` 与 `get_model.py --type raw` 实际生成的 ONNX 文件一致。
- input name、shape、layout、RGB/YUV、mean/std、resize、padding 与模型导出和 `model_impl.py` 一致。
- `quant.calib_data` 和 `demo/eval.data_dir` 使用正确数据集及目录层级。
- `build.ncore/opt_level` 与预编译 HMM 的下载命名和支持平台一致。
- `model.name`、HMM 产物名和 `get_model.py` 中的归档路径一致。
- plugin class 满足 HMATC 基类 contract，且 Demo/Eval 对同一预处理元数据的解释一致。

### 参数解析和 get_model.py

CV 示例通常只在 `get_model.py` 中提供少量参数：

```text
--type raw|hmm
--build_model_dir
--model_dir
```

评审时检查：

- `--type` 只接受 `raw` 和 `hmm`，默认行为与 README、`test.sh` 一致。
- raw 模式把 ONNX 保存到 `config.yml` 可见的位置；若提取子图，生成文件名必须等于 `model.model_path`。
- hmm 模式把解压产物放入 `--build_model_dir`，与 `config.yml` 的 `save_dir` 和 HMATC 默认查找规则一致。
- 下载归档名中的 target、Houmo version、batch、ncore 和 opt level 与该预编译模型真实配置一致。
- 不在 `model_impl.py` 或 `dataset.py` 重复实现 HMATC 的通用 CLI parser。
- 量化、编译、Demo、Eval、Perf 的参数由 `hmatc <command> -c config.yml` 解析；修改 HMATC 参数时同时使用 `imodelzoo-hmatc-review`。

## test.sh 评审规则

### 大模型 test.sh

大模型 `test.sh` 通常：

1. 使用 `set -e`。
2. 从当前目录向上查找并 `source models/test_common.sh`。
3. 设置 `STEP=demo`、`SKIP_DOWNLOAD=false`、`MODEL_NAME`、`MODEL_SIZE`，按需设置 `NDEVICE` 等默认值。
4. 调用 `parse_args "$@"`，或先解析模型专有参数再将剩余参数交给 `parse_args`。
5. 通过 `should_run_step quant/build/demo` 执行阶段。
6. 在 quant 阶段下载 raw model，在默认 demo 阶段下载预编译 HMM，并遵守 `--skip_download`。
7. 按需创建 Python venv，结束时仅清理本脚本创建的环境。

检查公共参数契约：

- `-s/--step` 支持 `demo`、`build`、`quant`、`all`，并支持逗号分隔或重复 flag。
- `-name/--model_name`、`-size/--model_size`、`-b/--batch`、`--ndevice`、`--context_length`、`--prefill_length`、`--quant_type` 等值参数缺值时必须报错。
- `--multi_batch`、`--mtp`、`--lora`、`--dflash`、`--skip_download` 等 boolean flag 不应消费下一个参数。
- 未知参数必须失败；模型专有参数应先被精确剥离，再把其余参数原样传给公共 `parse_args`。
- parse 后验证 model size、device 数、mode 等枚举，错误信息列出支持值。
- 使用数组传递可选参数，保持每个值的 quoting；不要拼接可能产生 word splitting 的命令字符串。
- quant/build/demo 接收的 model name、size 和 override 必须一致；不能只在某一阶段透传。
- 默认 `STEP=demo` 不应意外触发量化；`all` 应按依赖顺序完成可支持阶段。
- `--skip_download` 只跳过下载，不应跳过后续阶段或掩盖缺失产物。
- GPU、Python package、target 和外部工具检查应与实际阶段匹配，不要让未选择的阶段阻塞执行。
- 新增系统安装、网络下载或特权命令时，检查是否符合仓库约束，并确保依赖和失败行为有文档说明。

### CV test.sh

CV `test.sh` 通常不解析阶段参数，而是执行完整 HMATC workflow：

```text
get_model.py --type raw
    -> hmatc quant -c config.yml
    -> hmatc build -c config.yml
    -> hmatc compare -c config.yml --data_path ...
    -> hmatc perf -c config.yml ...
    -> hmatc demo -c config.yml
    -> hmatc demo -c config.yml --onnx
    -> hmatc eval -c config.yml
    -> hmatc eval -c config.yml --onnx
```

检查：

- 使用 `set -e`、定位并 source `models/test_common.sh`，然后切换到脚本目录。
- 总是准备 raw ONNX；HMQuant 不可用时下载预编译 HMM，可用时执行 quant/build/compare。
- 正确处理 `check_python_package hmquant` 的返回状态；明确 skip 与 fallback，不能把依赖错误伪装为测试通过。
- 所有 HMATC 命令使用同一 `config.yml`。
- compare 的 `data_path` 与任务数据集匹配；classification、detection、face、segmentation、pose 不应机械复制同一图片路径。
- Perf 的 warmup/sample/thread 应符合模型成本；不要仅因相邻脚本使用 `10/1000/1` 就覆盖模型已有合理值。
- 同时验证 chip Demo/Eval 和 `--onnx` 路径时，确保它们使用等价输入和 metric。
- 若为 CV `test.sh` 新增 CLI 参数，必须实现显式解析并同步测试 JSON；当前无 parser 的脚本不能配置无效的 `test_sh_params`。

## 测试配置耦合

大模型测试配置通常使用：

```text
get_model_params
quant_params
compile_params
demo_params
perf_params
test_sh_params
enable_demo_test
```

CV/HMATC 测试配置通常使用：

```text
get_model_params
support_hmatc
hmquant_params
hmbuild_params
hmdemo_params
hmcompare_params
hmperf_params
hmeval_params
```

评审时检查：

- JSON key 与实际 argparse/HMATC option 名一致；连字符和下划线按 runner 生成规则精确匹配。
- column-oriented 参数数组长度表达正确用例组合，不能因长度错位把不同模型变体的参数拼在一起。
- `null`、`default`、boolean 和 script 字段符合测试 runner 的跳过/生成语义。
- `test_sh_params` 中每组参数都能被目标 `test.sh` 消费，并覆盖默认和重要非默认分支。
- `support_flow`、`support_hmatc`、backend、platform、core/device/memory marker 与真实能力一致。
- 生产阶段输出路径与后续 Demo/Compare/Eval 配置中的 `cached_results` 路径一致。

## 跨体系与迁移规则

- 不要求大模型示例改名为 `config.yml`，也不要求 CV 示例新增 `config.yaml` 和独立阶段脚本。
- 不因个人偏好在同一变更中统一 underscore/hyphen option；只报告会破坏现有调用方或造成跨阶段不一致的问题。
- 新模型应选择与最接近的同类模型一致的体系，而不是混合复制两套模板。
- 明确要求迁移体系时，同时更新所有阶段脚本、`test.sh`、测试 JSON、聚合配置和 README，避免保留两个有效入口。
- finding 应指出违反的是哪套体系、哪个调用链以及具体失败后果；不要只写“命名不统一”。
