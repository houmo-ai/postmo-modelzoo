# MiniCPM-V 4.5

本目录提供 MiniCPM-V 4.5 的模型下载、W8A8 量化和 HMONNX 导出流程。MiniCPM-V 4.5 使用独立的 Qwen3 文本底座 workflow，不与 MiniCPM-V 4.6 的 Qwen3.5 workflow 混用。

## 模型下载

```bash
python3 get_model.py --type raw --source_type modelscope
```

原始模型默认保存为 `MiniCPM-V-4_5/`。

## 量化

```bash
python3 ptq.py
```

默认使用 xh2modelzoo 的 `minicpm_v_4_5_xh2a_w8a8.yaml` workflow，输出到 `output/${HOUMO_TARGET}/hmquant/`。

当前 workflow 会直接在 `hmquant` 根目录生成 `prefill`、`decode`、
`vision_1x` 和 `vision_6x`，无需由 `ptq.py` 额外搬移、重命名产物或修改
`export_meta_info.json`。`ptq.py` 会在导出后校验该目录布局，确保产物可直接交给
`build.py` 编译。

`ptq.py` 还会读取最终保留的 ONNX 图中的 `external_data.location`，自动删除
不再被任何 ONNX 引用的原始导出权重文件，并保留 `_with_act.onnx` 实际引用的
`_with_act_external_data`。

## 推理

下载预编译模型并执行 Python Demo：

```bash
bash test.sh --step demo
```

也可以直接执行：

```bash
python3 get_model.py --type hmm
python3 demo.py
```

## 编译

量化完成后执行：

```bash
python3 build.py
```

默认读取 `output/${HOUMO_TARGET}/hmquant/`，并在
`output/${HOUMO_TARGET}/` 生成：

- `minicpm-v-4.5_prefill.hmm`
- `minicpm-v-4.5_decode.hmm`
- `minicpm-v-4.5_visual_1x.hmm`
- `minicpm-v-4.5_visual_6x.hmm`

默认上下文长度为 `40k`、prefill 长度为 `256`，可通过
`--context_length`、`--prefill_length` 覆盖。
