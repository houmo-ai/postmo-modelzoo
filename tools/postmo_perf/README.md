# postmo_perf

`postmo_perf` 是基于 Python PostMo Engine 的固定长度设备性能测试工具。首版仅支持
Qwen3.5 Text-only、Batch 1、单设备和固定 Decode 次数。

调用关系：

```text
postmo_perf -> Qwen35Process -> Qwen35Module -> TcimBackend -> TCIM Lite
```

固定长度模式不调用 `Qwen35Engine`。Prefill Chunk、Cache 和 `context_length` 均由
`Qwen35Module` 管理。

## 配置

单 Case：

```yaml
model_dir: ../../utils/python/postmo_engine/models/qwen3.5-0.8b
input_tokens: 512
output_tokens: 128
warmup: 2
loop: 10
seed: 1234
model_name: qwen3.5-0.8b
dump_file: result.yaml
```

多 Case：

```yaml
dump_file: result.yaml
cases:
  - model_dir: ../../utils/python/postmo_engine/models/qwen3.5-0.8b
    input_tokens: 128
    output_tokens: 32
    warmup: 2
    loop: 10
    seed: 1
    model_name: qwen3.5-0.8b
  - model_dir: ../../utils/python/postmo_engine/models/qwen3.5-0.8b
    input_tokens: 512
    output_tokens: 128
    warmup: 2
    loop: 10
    seed: 2
    model_name: qwen3.5-0.8b
```

相对路径以配置文件所在目录为基准。`input_tokens + output_tokens` 不能超过模型 Context。

## 运行

直接指定 Prefill 和 Decode 模型：

```bash
python cli.py \
  --prefill /path/to/qwen3.5_prefill.hmm \
  --decode /path/to/qwen3.5_decode.hmm \
  --input-tokens 512 \
  --output-tokens 128 \
  --warmup 2 \
  --loop 10 \
  --dump-file result.yaml
```

交互式终端默认会在 stderr 显示 Warmup、正式 Loop、Prefill Chunk 和 Decode Token 进度。
Prefill 按实际 Chunk 数计算，例如输入 512 Token、图 Prefill 长度 256 时显示 `2` 个 Chunk。
关闭进度显示：

```bash
python cli.py path/to/perf.yaml --no-progress
```

进度输出不写入 YAML，也不会使用 Python `tqdm` 等额外依赖；重定向到非交互终端时默认不显示。

如果两个 HMM 位于模型目录根目录，工具会从同一目录下的
`hmquant/hf_config` 和 `hmquant/quant_embedding.pt` 加载 Tokenizer 与 Embedding。模型文件
不在同一目录时，必须使用 `--model-dir` 指定资产根目录。

从仓库根目录执行模块：

```bash
python -m tools.postmo_perf path/to/perf.yaml
```

在任意目录均可直接执行脚本：

```bash
python /hmdd/imodelzoo/tools/postmo_perf/cli.py path/to/perf.yaml
```

如果当前目录已经是 `tools/postmo_perf`：

```bash
python cli.py path/to/perf.yaml
```

`python -m tools.postmo_perf` 依赖当前 Python 搜索路径包含仓库根目录，因此不能直接在
`tools/postmo_perf` 目录中运行；该场景请使用 `python cli.py`。

工具会打印每个 Case 的 Average 摘要。设置 `dump_file` 后，同一个 YAML 文件使用与 C++
`tools/llm_perf` 一致的 `PerfMetrics` Schema 保存所有 Case 的正式平均结果，不写入 Warmup
或逐 Loop 明细：

```yaml
PerfMetrics:
  - PerfSettings:
      ModelName: qwen3.5-0.8b
      input: 512
      output: 128
      loop: 10
      perf_case_index: 1
      perf_case_total: 1
    PerfResults:
      input_token: 512
      output_token: 128
      prefill_time: "12.34"
      decode_time: "56.78"
      e2e_latency: "0.07"
```

与 C++ 一致，普通时间和吞吐字段是固定两位小数字符串；`e2e_latency` 的单位为秒，其他
时间字段为毫秒。当前未采集的 Vision、Embedding 和 KV Cache 性能字段保留为 `"0.00"`，
Host/Device Monitor 和 ModelLoadMemory 字段保留但值为空。

## 指标

- Runtime Operation：Prefill/Decode 的 Load、SetInput、Run+Sync 和 GetOutput。
- Stage：`postmo.prefill`、`postmo.decode`、`postmo.e2e`。
- 派生指标：Prefill TPS、Prefill Runtime TPS、Decode TPS、Decode Runtime TPS、TTFT、
  TPOT 和 E2E TPS。
- Prefill Chunk Count：`llm.prefill.run.count`。
- Decode Step Count：`llm.decode.run.count`。

这些指标是 device-oriented 口径，不包含真实 Prompt Tokenize、流式消费和外部服务开销。

## 首版限制

- 不支持数据集输入、VLM、ASR 和 TTS。
- 不支持 Multi-Batch 和 Multi-Device。
- 不支持 Host Monitor 和 Device Monitor。
- 固定 Token 会排除 EOS 和 Tokenizer 已知特殊 Token。
