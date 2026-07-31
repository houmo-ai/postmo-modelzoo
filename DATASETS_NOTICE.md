# 数据集版权说明

> **重要提示：除少量 COCO 2017 采样（CC BY 4.0，已署名）用于快速演示外，本项目不随代码分发其他第三方数据集。**
> 示例代码仅提供数据加载、预处理和评估逻辑。除上述 COCO 采样外，不内置其他完整数据集文件。用户需自行从官方渠道获取数据，并严格遵守其原始许可协议、使用条款、隐私政策及商用限制。

## 1. 基本原则

1. 本项目主体代码遵循 Apache License 2.0，该许可不适用于任何第三方数据集，仅覆盖我们提供的代码和项目文件。
2. 除少量 COCO 2017 采样外，示例代码不内置其他数据集文件，仅保留数据加载逻辑、目录占位（`.gitkeep`）和下载脚本。
3. 本项目不持有任何第三方数据集的版权，使用数据集产生的版权责任由使用者自行承担。
4. 禁止将本项目搭配受限数据集用于不符合其许可的商业产品，违者自行承担法律风险。

## 2. 数据目录约定

除 `datasets/` 内置少量 COCO 采样外，数据目录仅保留结构：

```text
data/
├── datasets/   # 内置少量 COCO 2017 采样；其他数据集用户自行放置或通过下载脚本获取
├── pic/        # 图像样本
└── audio/      # 音频样本
```

示例默认通过环境变量定位数据，可指向仓库内目录或外部目录：

```bash
export HOUMO_DATASETS_PATH=/path/to/datasets
```

## 3. 数据集分类与处理方式

### 3.1 开源可商用（Apache 2.0 / MIT / CC BY 等）

可通过官方公开渠道获取，文档标注协议原文链接。

| 数据集 | 许可 | 来源 | 随包情况 |
| --- | --- | --- | --- |
| COCO 2017 | CC BY 4.0 | https://cocodataset.org/#termsofuse | 内置少量采样，见 3.5 |
| VOC 2012 | CC BY 4.0 | https://public.roboflow.com/object-detection/pascal-voc-2012 | 不随包，用户自行获取 |
| CCPD | MIT | https://github.com/detectRecog/CCPD | 不随包，用户自行获取 |

### 3.2 学术非商用（CC BY-NC / CC BY-NC-SA / CC BY-NC-ND）

> 以下数据集仅限学术研究使用，禁止企业商用和二次分发。本项目不提供自动下载、镜像或内置样本，请自行向数据集作者申请授权后下载。商用需向数据集作者单独获取书面授权。

| 数据集 | 许可 | 来源 |
| --- | --- | --- |
| BDD100K | CC BY-NC-SA 4.0 | https://bair.berkeley.edu/blog/2018/05/30/bdd/ |
| nuScenes | CC BY-NC-SA 4.0 | https://www.nuscenes.org/ |
| WIDER FACE | CC BY-NC-ND 4.0 | https://shuoyang1213.me/WIDERFACE/ |

### 3.3 需注册获取（Terms of Access）

> 以下数据集需按官方流程申请和下载，本项目不提供自动下载或镜像。

| 数据集 | 许可 | 来源 |
| --- | --- | --- |
| ImageNet 2012 | Terms of access | https://www.image-net.org/download |

### 3.4 第三方私有 / 付费版权数据集

业务完整测试数据集为第三方版权资源，本项目不内置、不镜像、不提供下载链接。如需获取用于商用测试，请联系商务对接授权流程。示例仅支持用户自备数据或仿真数据验证代码逻辑。

### 3.5 内置 COCO 2017 采样来源与署名

为便于快速跑通示例，本项目在 `data/datasets/coco2017/` 内置了少量 COCO 2017 采样。相关版权与署名信息如下：

- 数据集：COCO 2017（Common Objects in Context）
- 许可：Creative Commons Attribution 4.0 International (CC BY 4.0)
- 许可原文：https://creativecommons.org/licenses/by/4.0/legalcode
- 数据来源：https://cocodataset.org/#home
- 使用条款：https://cocodataset.org/#termsofuse
- 标注版权：© COCO Consortium，标注数据以 CC BY 4.0 授权
- 图片版权：COCO 图片来自 Flickr，版权归各原始拍摄者所有，COCO 仅提供标注，不拥有图片版权

内置内容：

- 图片（`val2017/`，共 10 张）：
  - `000000000139.jpg`
  - `000000000285.jpg`
  - `000000000632.jpg`
  - `000000000724.jpg`
  - `000000000776.jpg`
  - `000000000785.jpg`
  - `000000000802.jpg`
  - `000000000872.jpg`
  - `000000000885.jpg`
  - `000000001000.jpg`
- 标注（`annotations/`）：`instances_val2017.json`、`person_keypoints_val2017.json`、`coco.names`（COCO 2017 val 完整标注，CC BY 4.0）

使用上述采样须遵守 CC BY 4.0 的署名要求，保留本节的来源与许可信息。图片的进一步使用需自行确认其原始 Flickr 版权与许可。

## 4. 免责声明

本项目仅提供代码工具，不持有任何数据集版权。用户自行下载、使用、分发各类第三方数据集需遵守其独立版权协议，本项目作者不承担因数据集侵权引发的任何法律责任。

更多第三方组件说明参见 [NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。