# YoloP

本示例展示如何把yolop模型量化，编译，最后部署到鸿途H30芯片的设备上。

[TOC]

## 1.模型说明

本例使用的 yolop 实现来源 github 开源项目[https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)的master分支。模型代码做一定调整后导出，细节见量化工具文档。

本例模型的训练数据集为bdd100k

## 2.快速开始

请参考houmo-modelzoo根目录下README.md文件的快速上手章节。

yolop模型PTQ量化精度下降较多，建议使用QAT方式进行量化。

将bdd100k的数据集数据放到$DATASETS_PATH/bdd100k目录下，包含bdd100k/images/100k，det_annotations，da_seg_annotations，ll_seg_annotations这几个子目录。

可通过qat.sh脚本直接运行，可通过修改train_yolop.py脚本中的cfg.TRAIN.BATCH_SIZE_PER_GPU参数修改batch等参数，目前默认值BATCH_SIZE_PER_GPU=8。

```bash
./qat.sh
```

注：如果遇到报错RuntimeError: result type Float can't be cast to the desired output type long int
修改/hmodel/thirdparty/YOLOP/lib/core/postprocess.py第30行加上.long()，修改后为：

```python
gain = torch.ones(7, device=targets.device).long()
```

QAT训练参考结果：

```bash
...
=> loaded checkpoint '/root/.cache/torch/hub/checkpoints/End-to-end.pth' 
building database...
...
...
Results saved to runs/BddDataset/_2023-09-06-20-33/visualization
Epoch: [10]    Loss(0.497)
Driving area Segment: Acc(0.974)    IOU (0.859)    mIOU(0.914)
Lane line Segment: Acc(0.643)    IOU (0.281)  mIOU(0.633)
Detect: P(0.084)  R(0.893)  mAP@0.5(0.767)  mAP@0.5:0.95(0.438)
```

## 3.参考结果

### 3.1性能结果

```bash
Input images: (1, 192, 320, 3), uint8
Inference time cost total = 115161us
Inference time cost per frame = 115161us
Average Throughput(QPS): 8.6835fps
```

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- yolop 模型：[https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx](https://github.com/hustvl/YOLOP/blob/main/weights/yolop-320-320.onnx)
- BDD100K 验证集链接: [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)
