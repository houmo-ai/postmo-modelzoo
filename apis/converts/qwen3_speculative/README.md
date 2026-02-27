# Qwen3 投机解码模型编译指南

本指南介绍如何将 Qwen3 模型编译为支持投机解码的格式，以便部署到后摩鸿途芯片设备上。

---

## 1. 获取量化模型

在 `models` 目录下提供了量化脚本，可分别对以下模型进行量化：
- **Draft 模型**（如 Qwen3-4b 等）
- **Target 模型**（如 Qwen3-14b 等）

量化完成后，将生成的文件用于后续步骤。

---

## 2. 准备量化模型

将量化后的模型文件放置到指定目录中，确保目录结构如下：

```bash
qwen3-speculative/
└── output/xh2/
    ├── draft/hmquant/
    │   ├── decoder/          # Draft 模型的decode文件
    │   ├── prefill/          # Draft 模型的prefill文件
    │   └── quant_embedding.pt # Draft 模型的量化嵌入文件
    └── target/hmquant/
        ├── prefill/          # Target 模型的prefill文件
        └── quant_embedding.pt # Target 模型的量化嵌入文件
```

确保文件放置正确后，继续下一步。

---

## 3. 执行编译

使用以下命令编译投机解码所需的模型：

```bash
python build.py --verify_length 5
```

### 参数说明
- `--verify_length`：配置步长参数（默认为 5）。

您可以根据需求调整命令行参数，以生成所需的模型。

---

## 4. 编译结果

编译完成后，生成的模型目录结构应如下所示：

```bash
qwen3/
└── output/xh2/
    ├── draft/
    │   └── hmquant/
    │       ├── decoder/          # Draft 模型的decode文件
    │       ├── prefill/          # Draft 模型的prefill文件
    │       └── quant_embedding.pt # Draft 模型的量化嵌入文件
    ├── target/
    │   └── hmquant/
    │       ├── decoder/          # Target 模型的decode文件
    │       ├── prefill/          # Target 模型的prefill文件
    │       └── quant_embedding.pt # Target 模型的量化嵌入文件
    ├── qwen3_decode_draft.hmm    # Draft 模型的解码文件
    ├── qwen3_prefill_draft.hmm   # Draft 模型的prefill文件
    ├── qwen3_prefill.hmm         # Target 模型的prefill文件
    └── qwen3_verify.hmm          # verify文件
```

---

如有问题，请参考相关文档或联系技术支持。