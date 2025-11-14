# ViT-B-16

本示例展示如何把量化后的`ViT-B-16`模型量化、编译，部署到后摩芯片的设备上。

[TOC]

## 1.模型说明

本例使用的`ViT-B-16`模型地址：[https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

下载模型后可使用`transformers`库导出为`onnx`模型，导出代码如下：

```python
import torch
from transformers import ViTForImageClassification


class ExportONNX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


m = ExportONNX().cpu()
m.eval()

x = torch.randn(1, 3, 224, 224, dtype=torch.float32).cpu()
torch.onnx.export(
    m,
    x,
    "ViT-B-16.onnx",
    opset_version=17,
    verbose=True,
    input_names=["pixel_values"],
    output_names=["logits"],
)
```

建议导出用onnxsim工具简化后使用。


## 2.快速开始

通过hmatc工具执行性能测试和精度测试。

### 2.1 精度评估

### 2.1 量化

```bash
hmatc quant -c config.yml -t xh1       # 芯片
```

### 2.2 编译

```bash
hmatc build -c config.yml -t xh1       # 芯片
```

### 2.3 精度评估

```bash
hmatc eval -c config.yml -t xh1          # 芯片
hmatc eval -c config.yml -t xh1  --onnx  # onnx
```

执行完成后会打印模型的数据集精度信息。

### 2.4 性能评估

```bash
hmatc perf -c config.yml -t xh1 -wn 10 -sn 1000 -tn 8
```

执行完成后会打印模型推理延迟、吞吐量等信息。

> **注意：** 
> 
> 性能测试的线程数建议是芯片核数的两倍，核数可通过环境变量HOUMO_CORE_NUM查看
>


### 2.5 结果演示

本示例依赖hmatc演示结果，在编译完成的基础上执行：

```bash
# xh1
hmatc demo -c config.yml -t xh1
# onnx
hmatc demo -c config.yml -t xh1 --onnx
```

执行完成后会打印模型的检测结果信息，图片结果将保存在vis_xh1/vis_onnx目录。

## 3.参考结果

### 3.1 精度结果

```bash
# onnx
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 10000, 'top1_acc': '0.801100'
# xh1
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 10000, 'top1_acc': '0.797000'
# xh2
'input_size': [1, 3, 224, 224], 'dataset': 'ILSVRC_2012Val', 'num': 10000, 'top1_acc': '0.800900'
```

### 3.2 性能结果

```bash
# xh1
[Latency] Inference  avg:  69.300 ms, max:  70.738 ms, min:  34.094 ms, tp99:  70.452 ms, tp999:  70.738 ms
[Latency] Input      avg:   0.102 ms, max:   0.193 ms, min:   0.061 ms, tp99:   0.151 ms, tp999:   0.193 ms
[Latency] Output     avg:   0.091 ms, max:   0.254 ms, min:   0.049 ms, tp99:   0.187 ms, tp999:   0.254 ms
[Latency] End2end    avg:  69.492 ms, max:  71.006 ms, min:  34.258 ms, tp99:  70.660 ms, tp999:  71.006 ms
[Throughput] total: 4417.507 ms, avg: 8.835 ms, repeat: 500, rounds: 1
[Throughput] qps: 113.186

# xh2
[Latency] Inference  avg:  47.034 ms, max:  47.873 ms, min:  35.020 ms, tp99:  47.712 ms, tp999:  47.873 ms
[Latency] Input      avg:   0.275 ms, max:   0.352 ms, min:   0.080 ms, tp99:   0.341 ms, tp999:   0.352 ms
[Latency] Output     avg:   0.176 ms, max:   0.276 ms, min:   0.010 ms, tp99:   0.236 ms, tp999:   0.276 ms
[Latency] End2end    avg:  47.485 ms, max:  48.196 ms, min:  35.503 ms, tp99:  48.090 ms, tp999:  48.196 ms
[Throughput] total: 5986.108 ms, avg: 11.972 ms, repeat: 500, rounds: 1
[Throughput] qps: 83.527
```

>**注意：**
>
> 结果均是以芯片核数两倍的线程数来测试，这里分别是8和4线程，仅供参考，具体结果以实际测试为准。

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- ViT-B-16 模型：[https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- LSVRC_2012 验证集链接: [https://image-net.org/challenges/LSVRC](https://image-net.org/challenges/LSVRC)
