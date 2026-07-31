# 数据集说明

除内置的少量 COCO 2017 采样（CC BY 4.0，已署名）外，本目录不包含其他第三方数据集文件，仅用于存放用户自行获取的数据集。本项目只提供数据加载逻辑，不持有相关数据集的版权。

使用数据集前，请从官方渠道获取数据并遵守其许可协议、使用条款、隐私政策及商用限制。因下载、使用或分发第三方数据产生的风险和责任由使用者自行承担。COCO 采样的来源与署名信息见根目录 `DATASET_NOTICE.md`。

默认情况下，示例通过环境变量 `HOUMO_DATASETS_PATH` 定位本目录。用户也可以将该变量设置为仓库外的数据集目录：

```bash
export HOUMO_DATASETS_PATH=/path/to/datasets
```

## 数据集许可信息

### COCO 数据集（本项目已内置少量采样）
```bash
- 许可：CC BY 4.0
- 许可协议原文：https://creativecommons.org/licenses/by/4.0/legalcode
- 数据来源：https://cocodataset.org/#home
- 使用说明：https://cocodataset.org/#termsofuse
- 随包情况：内置 10 张 val2017 图片及 val2017 标注，采样来源与署名见根目录 DATASET_NOTICE.md
- 图片版权：COCO 图片来自 Flickr，版权归各原始拍摄者所有
```

### ImageNet 2012
```bash
- 许可：Terms of access
- 许可协议原文：https://www.image-net.org/download?
- 数据来源：https://www.image-net.org/about.php
```

ImageNet 数据需由用户按官方流程申请和下载，本项目不提供自动下载或镜像。

### VOC 2012 数据集
```bash
- 许可：CC BY 4.0
- 许可协议原文：https://creativecommons.org/licenses/by/4.0/
- 数据来源：https://public.roboflow.com/object-detection/pascal-voc-2012/1
```

### BDD100K 数据集
```bash
- 许可：CC BY-NC-SA 4.0
- 许可协议原文：https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- 数据来源：https://bair.berkeley.edu/blog/2018/05/30/bdd/
```

BDD100K 包含非商用限制，本项目不提供自动下载或镜像。商用前需另行确认并取得必要授权。

### nuScenes 数据集
```bash
- 许可：CC BY-NC-SA 4.0
- 许可协议原文：https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- 数据来源：https://www.nuscenes.org/
- 使用说明：https://www.nuscenes.org/terms-of-use
```

nuScenes 包含非商用限制，本项目不提供自动下载或镜像。商用前需另行确认并取得必要授权。

### WIDER FACE 数据集
```bash
- 许可：CC BY-NC-ND-4.0
- 许可协议原文：https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
- 数据来源：http://shuoyang1213.me/WIDERFACE/
```

WIDER FACE 包含非商用和禁止演绎限制，本项目不提供自动下载、镜像或内置样本。

### CCPD 数据集
```bash
- 许可：MIT
- 许可协议原文：https://github.com/detectRecog/CCPD/blob/master/LICENSE
- 数据来源：https://github.com/detectRecog/CCPD
- 使用说明：https://github.com/detectRecog/CCPD/blob/master/README.md
```

## 其它说明

项目中提及的其它第三方数据集同样不随项目分发。用户需通过官方渠道自行获取，并严格遵守原始许可协议和使用条款。
