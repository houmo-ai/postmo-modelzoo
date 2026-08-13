---
name: open-source-compliance
description: "Review or implement iModelzoo changes involving third-party software, copied or adapted code, vendored sources, PyPI dependencies, C/C++ extensions, model weights, datasets, license headers, LICENSE/NOTICE/THIRD_PARTY_NOTICES updates, redistribution, packaging, containers, or open-source license compatibility. Use for 开源合规、许可证审查、版权头、第三方依赖、模型/数据集分发、GPL/LGPL/MPL/Apache/MIT/BSD/CC 合规问题。"
---

# 🛡️ 开源合规 Skill (Open Source Compliance Skill)

## 定位、适用范围与权威来源

本 Skill 用于 iModelzoo 的 AI 辅助编程、代码审查、依赖引入、源码移植、模型与数据集接入、构建打包和发布前合规检查。

开始执行时说明：`我正在使用 open-source-compliance skill 检查开源合规。`

按以下优先级读取和执行仓库内事实来源：

1. 目标第三方组件随附的原始 `LICENSE`、`COPYING`、`NOTICE`、文件头和上游发布页。
2. 仓库根 `THIRD_PARTY_NOTICES` 中该组件的详细条目。
3. 仓库根 `NOTICE`、`DATASET_NOTICE.md` 和目标模块 README 中的来源、许可与使用限制。
4. 仓库根 `LICENSE` 和 `licenses/` 中保存的许可证文本。
5. `.github/guidance/coding-style.md` 中的 first-party source file header 规范。

若上述来源互相冲突、许可证缺失、模型卡与源码许可证不一致、权重许可证不明确，或者商业使用/再分发条件无法确定：**停止合并、打包或发布建议，标记为 `Legal review required`，不得由 AI 猜测。**

发现明确或潜在合规风险时，AI 必须先以聊天文字说明风险事实、涉及文件、可能影响和建议动作，并等待开发者明确确认。未经确认，AI 不得自行向源码、README、`NOTICE`、`THIRD_PARTY_NOTICES`、`DATASET_NOTICE.md` 或 `licenses/` 添加风险结论、许可证判断、责任声明或整改内容。

> 本 Skill 是工程合规控制，不替代正式法律意见。不得把“常见做法”“部署隔离”或 SPDX 名称推断成已经获得授权。

---

## 1. 核心合规原则 (Core Principles)

### 1.1 主工程与第三方资产边界

- iModelzoo 主体 first-party 代码采用 Apache License 2.0，版权归属按文件头记录为 `HOUMO AI`。
- 主工程 Apache-2.0 **不覆盖**第三方软件、模型权重、数据集、工具链、SDK、运行时、预训练参数或复制/修改的上游代码。
- 直接放入 Git 跟踪的开源源码范围，或随源码包、二进制包、SDK、Docker、wheel、离线包、公开模型包等发布物分发的第三方库、算法实现、模型、数据集及上游源码片段，必须继续遵守各自许可并完成归属登记。
- 仅在用户环境中运行时下载、安装或调用的模型、PyPI 包、系统库和远程服务，不因“运行时依赖”而自动成为本仓库的内置或分发资产；本仓库不要求仅为此新增根 `NOTICE` 条目，但 README 仍应在必要时说明获取方式和用户责任。
- 仅由 `test.sh`、pytest、量化、编译、推理或本地调试流程在用户环境中产生，且被 `.gitignore` 排除、不进入任何对外发布物的模型缓存、虚拟环境、中间模型、编译产物、日志、图片和压缩包，不属于本次源码开源审查的分发对象。典型路径或文件包括模型目录下的 `work_dirs/`、`output/`、`*_venv/`、运行时下载的模型目录、`__pycache__/`、生成的 `*.png`、`*.zip`、ONNX、HMONNX 和 HMM。
- 内部 CI cache、开发机共享缓存和仅供构建流水线临时传递的 artifact，在确认不会进入公开源码包或对外发布物时，按运行时/构建时中间产物处理；不能仅因磁盘上存在或曾上传到内部制品服务，就自动认定为本仓库对外分发。
- 合规审查必须先确认实际开源/发布清单，不得仅凭工作区中存在 ignored 文件、构建输出或本地下载目录推断本项目正在分发这些内容。
- 上述登记豁免不是对上游许可证的豁免。用户或分发者自行安装、打包、镜像、缓存或再分发这些运行时资产时，仍应遵守其原始许可证、模型条款和服务条款。
- 除 `data/datasets/coco2017/` 内少量、已署名的 COCO 2017 样本外，仓库声明不分发其他第三方数据集。不得通过提交、测试夹具、压缩包、CI cache、对象存储镜像或 Docker 镜像绕过该边界。
- 模型“可下载”“代码为 Apache-2.0”不等于模型权重、训练数据或输出内容可自由商用或再分发。模型代码许可、权重许可、模型卡限制和服务条款必须分别核验。

### 1.2 改动前的来源分类

AI 在新增或修改资产前必须将其归入一种类别：

1. **First-party original code**：由本项目原创，可使用标准 HOUMO AI Apache-2.0 文件头。
2. **Third-party unmodified**：完整或部分 vendored、镜像、复制的上游文件；保留原始文件头和许可证，不替换为 HOUMO AI 文件头。
3. **Third-party modified/adapted**：基于上游文件修改；保留原始声明，并添加明确修改声明、来源和修改日期/范围。
4. **Runtime dependency only**：仅在运行时由用户安装、下载或从外部环境提供；不视为本仓库直接包含的资产，不因该依赖单独要求更新根 `NOTICE`，但不得删除或改写其原始许可信息。
5. **Directly included/distributed dependency**：源码、头文件、库、wheel、镜像层、离线包或其他发布物实际包含该依赖；必须按直接包含规则登记来源、许可证和分发义务。
6. **Model or dataset reference**：仅提供下载或加载逻辑；不默认随仓库分发。若确实内置样本、权重或数据，按直接包含/分发资产处理。
7. **Generated artifact**：ONNX、HMONNX、HMM、量化权重、编译产物、缓存或代码生成文件。先判断其是否进入 Git 跟踪或对外发布物：仅本地/CI 临时产生且不发布时，排除在源码开源通知审查之外；实际对外分发时，再核验输入模型、工具、嵌入资产及分发条款。由本公司原创重构代码和自有工具链形成的文件格式、图优化、量化配置与编译适配可属于本公司工程成果，但这一事实本身不替第三方输入资产作再许可，也不得在未确认发布边界时推定存在分发风险。

### 1.3 合规审查触发条件

以下变更必须加载本 Skill：

- 将 PyPI、Conda、系统库、CMake FetchContent、git submodule、下载脚本或二进制依赖直接包含到仓库或发布物，或改变其分发方式。
- 将上游 Python/C/C++/CUDA/Shell 代码复制、翻译、改写或让 AI 仿写进仓库。
- 新增 `3rdparty/` 内容、C++/CUDA extension、静态库、共享库、预编译库或 header-only 库。
- 新增模型、tokenizer、processor、配置、权重、数据集、样本、媒体文件或下载地址。
- 修改 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES`、`DATASET_NOTICE.md`、模块 README 的许可说明。
- 构建 Docker/SDK/离线包/安装包/二进制发布物，或改变静态/动态链接方式。

### 1.4 License Compatibility Index

下表是本仓库的工程准入规则，不代表对所有许可证情形的法律结论。

| 许可证/资产类型 | 当前仓库事实与典型作用域 | 默认准入 | 必须履行 | 禁止或升级审批情形 |
| --- | --- | --- | --- | --- |
| Apache-2.0 | 主工程；OpenCV、tokenizer.cpp、jinja.hpp、LibrosaCpp、kaldi-native-fbank、Transformers/Qwen 衍生代码及多个模型条目 | 允许 | 保留 copyright、LICENSE、NOTICE；修改上游文件时显著声明 changed；关注专利终止条款 | 删除 NOTICE、移除原作者声明、把修改后的第三方文件伪装成纯 HOUMO 原创 |
| MIT / MIT-0 / Public Domain option | `half`、nlohmann/json、spdlog、yaml-cpp、utf8proc、miniaudio、stb_image、llama.cpp/whisper.cpp 片段等 | 允许 | MIT 必须随副本或实质性部分保留版权及许可；MIT-0/Public Domain 仍应保留上游说明和来源记录 | 无来源复制、删除嵌入式许可、未记录修改后的 vendored 代码 |
| BSD-2-Clause / BSD-3-Clause / ISC | libsamplerate、Oniguruma、mingw/NetBSD 兼容头、RapidJSON 部分等 | 允许 | 源码和二进制分发均保留 copyright、条件和 disclaimer；BSD-3 不得暗示上游背书 | 从二进制材料中漏发 disclaimer；营销中使用上游名称背书 |
| MPL-2.0 | `3rdparty/eigen3` 修改后的 Eigen 子集；部分文件另为 Apache/BSD | 禁止新增包含或分发；仅保留仓库现有已审查组件 | 不得扩大使用范围；维护现有文件级声明、许可证文本和 `EIGEN_MPL2_ONLY`；任何变更先人工审查 | 新增、复制、修改、重新打包或扩大分发 MPL 文件 |
| LGPL-2.1-or-later | 当前明确条目为 `libsndfile`，用于 `models/tts/cosyvoice3/cpp/` | 禁止新增包含或分发；现有已审查链路不得擅自扩展 | 仅维护现有合规方案；不得改变链接和分发边界；后续变更必须人工审查 | 新增 LGPL 组件、直接合并源码、静态链接、隐藏用户替换能力或扩大现有分发范围 |
| GPL / AGPL 及其他强 copyleft/传染性协议 | 当前根 `NOTICE`、`THIRD_PARTY_NOTICES` 和 `licenses/` 未声明实际采用组件 | 明确禁止进入主工程或发布物 | 不得复制、改写、链接、打包、镜像或分发；如业务确需使用，必须先取得法务/OSPO书面批准并设计独立边界 | 任何未经书面批准的包含、组合、发布、自动下载后再打包或以服务化规避义务的做法 |
| MulanPSL | 当前未发现仓库实际组件或许可证文本 | 未知许可证，默认阻断 | 获取准确版本和原文，完成法务/OSPO兼容性审查，补齐许可证文本与 NOTICE | 仅凭名称或中文“宽松许可”判断可兼容 Apache-2.0 |
| CC BY 4.0 | 内置少量 COCO 2017 样本；VOC 仅引用 | 条件允许 | 署名、来源、许可链接、修改说明；COCO 图片仍需核验原 Flickr 权利 | 删除 `DATASET_NOTICE.md` 署名；将完整 COCO 或来源不明图片直接打包 |
| CC BY-NC-SA 4.0 | BDD100K、nuScenes，仅引用、不分发 | 禁止商用与仓库/镜像再分发 | 用户自行从官方来源获取；保持非商用、署名、SA 说明 | 企业商用、自动下载镜像、提交数据或衍生数据到发布物 |
| CC BY-NC-ND 4.0 | WIDER FACE，仅引用、不分发 | 禁止商用和衍生数据再分发 | 用户自行获取并遵守署名、NC、ND | 修改后数据集/标注再分发；企业商用；内置样本 |
| ImageNet Terms of Use | 需注册，非商用研究/教育，仓库不分发 | 仅外部用户自备 | 官方注册获取，遵守 Terms of Use | 镜像、自动下载、内置样本、商业数据包 |
| 未知/自定义模型许可证 | 例如 NOTICE 中要求回原仓库确认的 CosyVoice3 | 默认阻断再分发与商用结论 | 核验代码许可、权重许可、模型卡、输出限制、商用与地域限制 | 许可证缺失时上传权重、编译模型、量化模型或声称可商用 |

### 1.5 Copyleft 隔离边界

- **动态链接不是“无义务”**：对 LGPL，动态链接通常有利于保持主程序许可证独立，但仍需保留许可证、通知、库源码/源码获取方式及用户替换/重新链接能力，具体依许可证版本与分发方式核验。
- **静态链接必须升级审批**：LGPL 静态链接通常要求提供可重新链接的目标文件或采用其他合规机制；本仓库不得由 AI 自动改成静态链接。
- **Docker 不是许可证隔离墙**：把 GPL/LGPL/AGPL 组件放进同一镜像不会消除分发义务；镜像本身属于发布组合，仍需逐层清单和许可证材料。
- **虚拟环境不是许可证隔离墙**：独立 venv 只解决依赖冲突，不改变复制、链接、组合或分发关系。
- **进程/RPC 仅是工程降低耦合手段**：只有组件保持独立程序、独立包、独立生命周期，通过通用协议和可替换接口交互，并分别分发/告知许可证时，才可能降低衍生作品风险；不得把 IPC/RPC 当作自动法律豁免。
- **AGPL 网络条款不能靠服务化规避**：若服务本身包含或修改 AGPL 程序，向网络用户提供服务可能触发对应源代码提供义务。
- 对 GPL/AGPL、静态链接 LGPL、自定义模型许可证、权利不清的训练数据，必须标记 `Legal review required`。

---

## 2. 文件 Header 标准模板 (License Header Templates)

### 2.1 First-party 通用规则

- 新建 first-party `.py`、`.pyi`、`.cpp`、`.cc`、`.c`、`.h`、`.hpp`、`.cu`、`.cuh` 等源码必须在第一段代码前包含标准 HOUMO AI Apache-2.0 文件头。
- 创建年份使用文件首次创建年份，后续普通修改不得自动更新年份。
- `File:` 必须与实际 basename 完全一致；`Description:` 必须非空且准确，禁止复制其他模型描述。
- Python 可将 shebang 和编码声明置于许可证头之前。
- Shell 脚本可不添加 copyright/license header；若项目或具体目录已有统一 header 要求，则 shebang 放第一行，许可证头紧随其后。不得仅因 first-party `test.sh` 缺少 header 报告开源合规 finding。
- YAML、JSON、二进制和不支持注释的格式不得插入破坏语法的 header；在相邻 README、NOTICE 或清单中记录归属。

### 2.2 Python 模板

```python
#!/usr/bin/env python3
# Copyright (c) <creation-year> HOUMO AI
#
# File: <basename>.py
# Description:
#   <accurate non-empty description>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
```

无执行需求的 Python 模块可省略 shebang，不得省略其余必需字段。

### 2.3 C / C++ / CUDA 模板

```cpp
/*
 * Copyright (c) <creation-year> HOUMO AI
 *
 * File: <basename>.cpp
 * Description:
 *   <accurate non-empty description>.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
```

`.h`、`.hpp`、`.cu`、`.cuh` 仅替换 `File:` basename；header 必须位于 include、pragma 或 header guard 之前。

### 2.4 Shell 模板（可选）

Shell header 不是本仓库开源合规的强制项。需要主动添加或目标目录已有统一要求时，可使用以下模板：

```bash
#!/usr/bin/env bash
# Copyright (c) <creation-year> HOUMO AI
#
# File: <basename>.sh
# Description:
#   <accurate non-empty description>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
```

### 2.5 第三方修改文件模板

不得删除、替换或缩减原始作者 header 和 copyright。原始署名只能保留；项目确有可主张的修改时，只能在原始声明之后新增修改说明，例如：

```text
Modifications Copyright (c) <year> HOUMO AI
Modified by HOUMO AI on <YYYY-MM-DD>.
Changes: <hardware adaptation / API integration / preprocessing changes>.
Original source: <upstream URL and commit/tag>.
```

若上游许可证规定其他格式，以上游要求为准。Apache-2.0 衍生文件必须有显著 changed notice；MPL 文件继续按原文件许可证发布；MIT/BSD 文件保留完整原始 copyright、条件和 disclaimer。

禁止对 third-party 文件仅添加 HOUMO Apache header 后删除上游 header，因为这会造成错误权属声明。

若上游文件或代码片段原本没有 copyright 声明：

- 必须记录官方来源 URL、版本/commit、原始文件路径和实际修改点。
- 不得仅因代码被复制、格式化、集成或由 AI 辅助调整，就新增虚假的 HOUMO AI copyright。
- 只有项目实际新增且具备独立原创性的修改部分，才可按事实增加 `Modifications Copyright`；无法区分时仅写来源和修改说明。

---

## 3. 第三方代码引用与修改指引 (Third-Party Code Reference)

### 3.1 引入前必做清单

直接包含、复制或随发布物分发第三方代码前必须收集：

- Component name、准确版本/tag/commit、上游 URL。
- SPDX license expression 和官方许可证链接；仅当上游协议或实际分发方式明确要求时，再收集并附带原始许可证文本。
- Copyright holder、NOTICE、专利或商标声明。
- 使用路径、使用方式：源码复制、header-only、静态链接、动态链接、Python import、子进程、RPC、工具期依赖或仅下载引用。
- 是否修改；若修改，列出文件、修改原因和差异范围。
- 是否进入源码包、wheel、Docker 镜像、SDK、离线包、二进制、模型包或测试数据。
- 商用、再分发、署名、同许可证发布、源码提供、用户替换、网络服务等义务。

任一项未知时不得直接合入 vendored 源码或发布物。AI 只能以聊天形式报告风险和建议，不得在开发者确认前自行添加合规声明或修改通知文件。

若第三方算法或仓库采用允许开源且不具传染性的许可证，并被直接包含进本仓库或发布物，必须同时：

1. 在对应模块 README 声明组件名称、官方来源、版本/commit、许可证和使用位置。
2. 在根 `NOTICE` 增加摘要引用。
3. 在根 `THIRD_PARTY_NOTICES` 增加详细条目。
4. 若存在明确修改，简要、准确地记录修改文件和修改点；不得复制示例占位内容冒充实际修改。

本仓库的 Apache-2.0 不替第三方依赖授予、扩展、收回或解释任何许可权限。对允许开源且不具传染性的依赖，本仓库仅按原始协议使用和声明；使用者必须自行严格遵守原始许可证、NOTICE、商标、专利及再分发条件。

仅由用户在运行时下载的模型、运行时安装的 PyPI 包或从用户系统加载的依赖，且未被提交、缓存、镜像、打入 wheel/Docker/SDK/离线包或随发布物提供时，可不更新根 `NOTICE`、`THIRD_PARTY_NOTICES` 和模块许可清单。若实际分发边界发生变化，立即改按直接包含资产处理。

### 3.2 Apache-2.0 代码

- 可用于主工程，但必须保留原许可证和归属。
- 复制或修改 Transformers、qwen_vl_utils、ONNX/PyTorch 示例代码时，记录原始文件路径、上游版本/commit 和 changed notice。
- 若上游带 `NOTICE`，分发衍生作品时将适用 attribution 合并到根 `NOTICE` 或分发材料中。
- 不得声称 Apache-2.0 覆盖第三方模型权重，除非权重仓库明确采用该许可证。

### 3.3 MIT / MIT-0 / BSD / ISC 代码

- 可复制、修改、静态或动态链接，但必须满足各许可证的保留条件。
- MIT 和 BSD 的“宽松”不等于可以删除文件头或许可证文本。
- 二进制发行时 BSD 条件和 disclaimer 必须进入文档或第三方通知。
- 对 header-only 或直接编译进目标的代码，视为进入分发物，必须在 `THIRD_PARTY_NOTICES` 中记录。

### 3.4 MPL-2.0 代码

- MPL 是文件级 copyleft。除维护仓库已经登记并审查的 Eigen3 子集外，禁止新增包含或分发 MPL 代码。
- 维护现有 MPL 文件时，修改后的该文件仍需按 MPL 提供源码和许可；任何修改、升级或扩大分发范围必须人工审查。
- 禁止把 MPL 文件内容复制粘贴进 first-party Apache 文件后删除 MPL 声明。
- Eigen 当前为裁剪后的多许可证文件集合，必须逐文件保留声明，并保持 `EIGEN_MPL2_ONLY`。

### 3.5 LGPL / GPL / AGPL 代码

- 具有传染性或 copyleft 要求的许可证默认禁止新增包含和分发，包括 GPL、AGPL、LGPL、MPL 及其他会对本仓库源码、链接产物、文件或网络服务施加开源义务的协议。
- `libsndfile` 和 Eigen3 仅属于仓库已有、已登记的历史合规边界，不得作为允许新增同类依赖的先例；只能维持现状，任何升级、修改、重新链接、重新打包或扩大分发范围必须人工审批。
- 不得从 GPL/AGPL 项目复制函数、核心算法实现、测试向量或大段结构到 Apache 主工程，即使变量改名、翻译语言或由 AI “重写”。
- 若只参考思想，必须基于公开规范、论文或 clean-room 描述独立实现，并保存来源与独立实现记录；不得让生成模型以受限源码为唯一输入进行近似复刻。
- GPL/AGPL 及其他传染性协议的新依赖必须拒绝包含和分发；确需例外时，必须先取得 OSPO/法务书面批准，AI 不得自行设计规避方案。

### 3.6 PyPI / Python 依赖

- 仅在 `requirements.txt` 中声明、由用户运行时自行安装且不随仓库或发布物提供的 PyPI 包，不要求仅因依赖声明而更新根 `NOTICE` 或 `THIRD_PARTY_NOTICES`。
- 本仓库不替这些运行时包授予或解释许可权限；用户安装、使用、打包或再分发时应自行遵守包的原始许可证和传递依赖条款。
- 若 wheel、sdist、site-packages、venv、依赖缓存或其代码被放入 Docker、SDK、离线包、安装包或其他发布物，则不再属于纯运行时依赖，必须核验 package metadata、许可证和传递依赖，并维护第三方清单。
- 示例不得在运行时静默 `pip install`、升级或下载未知依赖；依赖应声明在相应 requirements/安装文档中。

### 3.7 C++ / CUDA Extensions

- 检查 `.so`/`.a`/`.dll`、CUDA kernel、第三方头文件和编译生成源码的许可。
- 记录静态/动态链接、是否将源码编译进 wheel、是否依赖 NVIDIA/CUDA 或其他 SDK EULA。
- CPP extension 若包含 PyTorch/ONNX/Transformers 上游源码，应同时遵守上游文件许可证和 NOTICE，不得仅以扩展模块整体标注 Apache-2.0。
- 仅从用户系统环境运行时加载且不随发布物分发的库，不要求新增根 NOTICE；若库被链接、复制或打入发布物，则必须按直接包含资产处理。

### 3.8 ONNX / PyTorch / Transformers 衍生代码

- ONNX 模型格式本身不决定模型内容许可证；导出、量化、简化或编译不会自动解除原模型权重和处理器代码的许可义务。
- 对量化模型、HMONNX/HMM、external data 和 tokenizer 文件，先确认是否属于实际对外发布物。仅在 `test.sh`、pytest 或本地工具流程中生成，并由 `.gitignore` 排除、不进入源码包或其他对外发布物时，不要求更新根 `NOTICE`、`THIRD_PARTY_NOTICES` 或 `licenses/`，也不作为代码开源 blocker。
- 实际对外分发量化模型、HMONNX/HMM、external data 或 tokenizer 时，才核验第三方输入资产是否允许转换与再分发，并区分本公司原创导出、图重构、量化和编译实现与第三方模型资产本身的权利边界。
- 从 Transformers/PyTorch/ONNX 示例复制处理逻辑时保留上游版权、许可证、原始路径和修改说明。
- 对通过 Python monkey patch 在运行时替换第三方包方法的本公司原创兼容实现，不能仅因目标符号来自 Transformers/PyTorch/ONNX，就认定为复制或修改了第三方源码文件。若补丁代码由本公司基于公开 API、张量契约、算子语义和导出需求独立实现，未复制或近似改写第三方受版权保护的表达，可使用 HOUMO AI Apache-2.0 header，不要求添加 third-party modified header。
- monkey patch 的判定依据是补丁函数自身的来源与表达，而不是“替换了第三方方法”这一技术形式。调用第三方对象、沿用公开方法签名、访问公开成员、保持必要输入输出顺序或返回兼容类型，均不能单独证明源码复制。
- 若补丁函数确实从第三方实现复制、翻译或进行可识别的近似改写，则仍按 third-party modified/adapted 处理。审查者必须给出具体表达、注释或独特控制流的对应证据；不得只凭功能等价、API 相同或控制流程完成相同任务就要求改变版权归属。
- 前处理、后处理、tokenization、sampling、图像/音频变换、解码、NMS、评测或数据转换逻辑若引用或改编自开源代码，必须在文件注释或相邻 README 中注明官方来源、版本/commit、原始文件/函数和实际修改点。
- 后处理逻辑若仅参考公开思想、论文、算法描述或接口行为，并进行了大面积、独立的结构重构，没有复制或近似改写原代码、注释、表达结构或可识别实现细节，可视为独立实现，不要求在代码或 README 中添加开源代码引用。无法确认是否属于独立实现时，先向开发者报告，由开发者决定是否补充引用。
- 模型仓库中的 `README.md`、`LICENSE`、`config.json`、tokenizer 和 processor 文件也可能携带独立归属，不得在下载后批量删除。

### 3.9 模型与数据集

- 对仓库直接包含或随发布物提供的模型/数据集，README 至少记录：名称、来源 URL、版本/revision、许可证、权重/数据是否随仓库分发、商用与再分发限制。
- 对仅在运行时从官方渠道下载且不由本仓库分发的模型，可不新增根 NOTICE 条目；README 应尽量说明下载行为和责任边界，不得暗示本仓库授予该模型的使用、商用或再分发权限。
- 默认策略是用户从官方渠道自行获取；不得新增非官方镜像、内网转存或自动下载受限数据集。
- 对 CC BY 资产保留署名、许可链接和修改说明；对 NC/ND/SA 资产执行相应非商用、禁止衍生分发或同许可要求。
- 发现个人信息、生物识别、车牌、人脸、语音等数据时，除版权外还必须升级隐私与数据治理审查。

### 3.10 `test.sh` 与运行时产物边界

审查模型示例时必须先梳理 `test.sh`、pytest 配置和各阶段脚本的实际数据流，再判断哪些内容属于开源发布：

1. 读取 `test.sh` 的 `get_model`、quant、build/compile、demo/eval/perf 分支，以及 `--skip_download`、cache 和输出目录参数。
2. 将仓库跟踪的脚本、配置、README 和测试配置视为源码审查对象。
3. 将运行时下载的模型目录、临时 venv、cache、`work_dirs/`、`output/`、生成图片、日志、压缩包、ONNX、HMONNX、HMM 先标记为运行时或构建产物。
4. 结合 `.gitignore`、打包脚本、release manifest、Dockerfile、上传/发布脚本和用户明确说明，确认这些产物是否进入对外发布物。
5. 仅本地生成且不发布的产物不触发根通知文件更新；只有被 Git 跟踪、复制进发布目录、打入镜像/SDK/离线包或明确提供给外部用户时，才转为直接包含/分发资产。
6. `test.sh` 本身可不包含 copyright/license header，不得因此报告合规 finding；仍需检查脚本是否把第三方资产静默打包、镜像或发布。

不得把以下事实单独作为“正在对外分发”的证据：

- 工作区中存在 ignored 文件或目录；
- CI cache 或开发机共享目录中存在模型；
- 流程生成了 ZIP/HMM/ONNX/HMONNX；
- 内部测试上传或内部制品暂存；
- README 描述了生成文件名。

只有明确的对外发布边界证据才能触发直接分发结论。

---

## 4. NOTICE 与 THIRD_PARTY_NOTICES 更新规范

### 4.1 文件职责

- `LICENSE`：主工程 Apache-2.0 完整文本，不用于汇总第三方许可证。
- `NOTICE`：项目归属、主工程说明和第三方组件/模型/数据集摘要索引。
- `THIRD_PARTY_NOTICES`：详细记录组件版本、来源、版权、许可证、使用路径、修改、分发和特殊义务。
- `licenses/`：可保存仓库实际使用或分发所需的标准许可证文本。一般引用可仅记录上游官方许可证链接，不要求自动复制许可证原文；只有上游协议或发布方式明确要求随分发附带完整文本时，才需在开发者确认后补充准确原文。
- `DATASET_NOTICE.md`：数据集是否随包、来源、许可、署名、NC/ND/SA/注册限制和样本清单。
- 模块 README：记录当前示例直接包含或直接引用的模型、数据集和关键第三方实现，不替代根 NOTICE；对纯运行时下载/安装项可仅说明来源和用户责任。README 列出本地生成产物路径不等于这些产物被项目分发。

### 4.2 何时更新

本节仅适用于本次变更新增、修改、删除或改变分发边界的第三方资产。已有且不属于本次新增或修改点的 `NOTICE`、`THIRD_PARTY_NOTICES` 条目不得由 AI 修改、整理、补全、删除、纠错或顺带重写。即使 AI 发现既有条目存在风险，也只能在聊天中列出证据和建议，由开发者自行确认是否另行修改。

以下情况必须同步更新 `NOTICE` 和/或 `THIRD_PARTY_NOTICES`：

- 新增、升级、替换或删除实际 vendored、编译、链接、打包或随发布物分发的库、模型或数据集。
- 新增复制/修改的上游文件，或改变已登记组件的使用路径、修改范围或分发方式。
- 改变静态/动态链接方式、打包范围、Docker 镜像内容或发布物构成，使原本运行时获取的资产进入发布物。
- 上游许可证、NOTICE、版权年份、版本、来源 URL 或商用限制发生变化。
- 开始分发此前仅引用的模型、tokenizer、数据样本或预编译库。

以下情况不因其本身是依赖而强制更新根通知文件：

- 用户在运行时从官方渠道自行下载且仓库不缓存、不镜像、不打包的模型或数据。
- 用户在运行时自行安装且仓库不随包提供的 PyPI、Conda 或系统依赖。
- README、requirements 或下载脚本仅描述获取方式，未把第三方资产复制进仓库或发布物。
- `test.sh`、pytest、量化、编译或 demo 在本地/CI 中生成且被忽略的 ONNX、HMONNX、HMM、图片、日志、压缩包、虚拟环境、模型 cache 和中间目录，确认不进入对外发布物。

上述情形仍应避免删除上游许可信息；一旦仓库开始缓存、镜像、打包或再分发，必须转入直接包含/分发规则。

删除本次变更涉及的依赖时，可建议移除对应失效条目，但必须先确认所有源码、二进制、生成产物、文档和传递依赖均已移除，并取得开发者明确确认。不得借此修改其他历史条目。

### 4.3 `NOTICE` 摘要格式

对于需要登记的直接包含或随发布物分发的第三方组件、算法、模型和数据集，在相应分类下添加一行：

```text
<Component Name> (<SPDX license expression>) - <official upstream URL>
```

若是数据集或模型，补充“不随包/内置样本/用户自行获取”等关键限制，不得把未知模型许可简化成 Apache-2.0。

### 4.4 `THIRD_PARTY_NOTICES` 标准条目

以下模板仅用于直接包含或随发布物分发的资产；纯运行时下载、安装或外部加载的模型和依赖不要求仅因运行时使用而建立根条目。

```text
----------------------------------------------------------------------
<Component Name> (<short purpose>)
----------------------------------------------------------------------

Component Name: <name>
Version / Revision: <version, tag, or commit>
Usage Location:
    <repo/path>
Usage Type: <vendored source | header-only | static link | dynamic link |
             Python dependency | build-time tool | model reference | dataset sample>
Distribution: <source package / binary / wheel / Docker / not redistributed>

Source:
<official upstream URL>

Copyright Statement:
<verbatim upstream copyright>

License:
<SPDX identifier and full license name>

License Text:
<official upstream license link; optionally licenses/<file>.txt when required>

Modifications:
<None, or exact modified files and prominent change summary>

Compliance Notes:
<required attribution, NOTICE, source offer, relinking, non-commercial,
 no-derivatives, share-alike, model-weight restrictions, etc.>
```

许可证引用默认可只填写稳定的官方许可证 URL，无需把原文直接复制到仓库。若上游协议明确要求分发时附带许可证文本、版权声明或 disclaimer，则必须满足该原始要求；AI 应先告知开发者并取得确认，不得自行新增许可证文件。

### 4.5 直接分发的 PyPI 依赖记录

仅对随 Docker、SDK、离线包、wheel、sdist、venv 或其他发布物实际分发的 PyPI 依赖，附加或生成依赖清单，至少包含：

```text
Package: <distribution name>
Version: <resolved version>
License: <verified SPDX expression>
Source: <project URL>
Artifact: <wheel/sdist filename and hash when available>
Scope: <runtime/dev/test/model-specific>
Transitive Dependencies Reviewed: <yes/no>
```

仅在 `requirements.txt` 或安装说明中声明、由用户自行运行时安装且不随发布物提供的 PyPI 依赖，可不建立上述根通知条目；README 可注明“依赖由使用者自行安装并负责遵守其原始许可”。

### 4.6 修改第三方文件时的同步动作

以下动作只处理本次实际新增或修改的第三方文件和对应条目，不得触碰无关历史条目。涉及不确定许可、传染性协议、权属冲突或其他明确风险时，先以聊天文字告知开发者并等待确认，不得由 AI 自动添加风险性合规内容。

1. 保留原文件 header 和许可证。
2. 添加 prominent modification notice。
3. 在 `THIRD_PARTY_NOTICES` 的 `Modifications` 中列出文件和改动性质。
4. Apache-2.0 上游如含 NOTICE，合并适用 attribution。
5. 更新模块 README 的来源和许可说明。
6. 对直接包含的非传染性依赖，确认 README、根 `NOTICE` 和 `THIRD_PARTY_NOTICES` 均已登记，并写明实际修改点；对 MPL/LGPL/GPL/AGPL 或自定义许可证，停止并要求人工合规审查。

### 4.7 README 责任划分

涉及第三方算法、模型、数据集或运行时依赖的模块 README，应尽量明确：

- **本仓库责任**：仅对本仓库原创代码、实际提交的第三方副本/修改、随发布物提供的模型/数据/依赖及已声明的适配修改负责，并提供准确来源和许可证说明。
- **使用者责任**：对运行时自行下载或安装的模型、数据集、PyPI/系统依赖、权重、输出内容和商业部署负责；使用者不得将本仓库 README、NOTICE 或 Apache-2.0 解释为上游资产的再许可、商业授权或再分发许可。
- **修改责任**：使用者若自行修改、打包、镜像或再分发第三方资产，应保留原始署名和许可证，并自行完成相应通知、源码提供、非商用、禁止衍生或其他义务。

推荐使用类似表述：

```text
This example repository provides integration code and does not grant or extend
rights to third-party models, datasets, runtime packages, or services. Users
must obtain runtime assets from their official sources and comply with the
applicable original licenses and usage terms. Any bundled third-party asset is
listed in the module README, root NOTICE, and THIRD_PARTY_NOTICES.
```

---

## 5. AI 代码生成禁忌清单 (Forbidden Actions)

以下为硬性禁止项；命中任一项必须停止修改并报告合规风险：

1. **严禁**未经 OSPO/法务审批，将 GPL、AGPL 或许可证不明的源代码复制、翻译、改名、AI 仿写后提交到 Apache-2.0 主工程。
2. **严禁**删除、覆盖或缩减第三方文件中的 Copyright、LICENSE、NOTICE、SPDX、作者归属、专利或 disclaimer。
3. **严禁**把 third-party modified 文件重新标记成仅有 `Copyright (c) HOUMO AI` 的原创文件；必须保留原声明并记录修改。原始无 copyright 时，不得虚构 HOUMO copyright，只记录来源和修改点。
4. **严禁**仅凭 Docker、独立 venv、插件、动态链接、子进程或 RPC 宣称 GPL/LGPL/AGPL 义务已隔离或消失。
5. **严禁**未经核验上传、镜像或随仓库/发布物分发模型权重、tokenizer、数据集、图片、音频、量化模型、HMONNX/HMM 或预编译库。
6. **严禁**将 BDD100K、nuScenes、WIDER FACE、ImageNet 等受限或需注册数据用于企业商用包、自动下载、CI fixture 或二次分发。
7. **严禁**将直接包含或随发布物分发的 PyPI/C++/CUDA 依赖伪装成纯运行时依赖，或不核验其许可证、版本、传递依赖和实际打包范围；纯运行时依赖不要求根 NOTICE 登记，但不等于用户免除上游许可义务。
8. **严禁**在许可证未知、模型卡冲突、上游 NOTICE 缺失、权重与代码许可不一致时给出“可商用”“可闭源”“可再分发”的确定结论。
9. **严禁**修改本次任务范围外的既有 `NOTICE` 或 `THIRD_PARTY_NOTICES` 条目；发现历史风险只能在聊天中提出，由开发者确认是否另行处理。
10. **严禁**在存在明确合规风险时，未经开发者文字确认便自动向代码、README、通知文件或许可证目录添加风险结论、许可判断或整改声明。

附加禁止项：

- 不得修改 `3rdparty/` vendored 源码来“统一格式”或替换 header，除非任务明确要求且已完成上游许可核验。
- 不得把许可证文本当作普通注释进行格式化、翻译、截断或自动换行改写。
- 不得使用非官方来源或个人网盘作为受限模型/数据集的默认下载源。
- 不得在日志、脚本、NOTICE 或 README 中提交访问 token、账号、密码或私有制品库凭据。

---

## 6. AI 编程与代码审查工作流

### 6.1 实现前

1. 列出所有新增/修改资产及其来源类别。
2. 搜索目标文件现有 header、上游 URL、`NOTICE`、`THIRD_PARTY_NOTICES`、`licenses/` 和模块 README。
3. 建立 license BOM：直接依赖、传递依赖、模型、数据集、生成产物。
4. 根据兼容性矩阵判定：`允许`、`条件允许`、`默认禁止`、`Legal review required`。
5. 只有 `允许` 或义务已落实的 `条件允许` 才能继续修改；发现明确风险时先向开发者文字报告并等待确认。
6. 确定本次变更范围；既有且与本次新增/修改点无关的 `NOTICE`、`THIRD_PARTY_NOTICES` 和许可证记录只读，不得自动修订。

### 6.2 代码审查检查项

- 新增 first-party Python/C/C++/CUDA 源文件是否使用正确 creation year、basename、Description、Apache-2.0 和 SPDX header；Shell 脚本 header 可选，不因 `test.sh` 缺少 header 报告 finding。
- 新增文件是否实际来自第三方，却错误套用 first-party header。
- Copy/adapted code 是否保留原作者声明、原始来源、commit/tag 和 modification notice。
- 直接包含/分发的新依赖是否同步 requirements/CMake、NOTICE、THIRD_PARTY_NOTICES、licenses 和 README；纯运行时依赖是否明确由用户自行获取并承担原始许可责任。
- 链接方式是否从动态改为静态；发布物是否新增 `.so`、`.a`、wheel、模型或数据。
- 模型转换/量化/编译产物是否实际进入 Git 或对外发布物；若仅为 ignored 的运行时/构建时中间产物，不将其作为源码开源 blocker。只有实际分发时，才继续核验原权重许可、工具嵌入资产和随附通知义务。
- 数据集是否符合仓库“不分发，COCO 小样本例外”的策略。
- Docker、venv、RPC 描述是否错误声称能自动隔离 copyleft。
- 前处理、后处理及数据转换代码若确实复制或改编第三方表达，是否注明开源来源、版本/commit 和实际修改点；仅基于公开 API、张量契约或算法语义独立实现时，不强制添加第三方归属。
- monkey patch 是否被错误地仅因替换第三方方法就判定为 third-party modified；应审查补丁代码自身是否存在复制或近似改写证据。
- 后处理若声称仅参考思想且无需引用，是否确属大面积独立重构，不包含复制、近似改写或可识别的上游实现细节。
- README 是否清楚划分本仓库责任与使用者对运行时资产、模型、数据和依赖的责任。
- 删除依赖时是否仍有被 Git 跟踪或进入对外发布物的源码、二进制、生成文件、CMake 路径或 NOTICE 残留；ignored 的本地运行时产物不作为发布残留。
- 建议的通知修改是否严格限于本次新增/修改点；历史风险是否仅通过聊天告知并等待开发者确认。

### 6.3 审查输出格式

```markdown
## Compliance Findings

- [BLOCKER] <issue> — `<path:line>`
  License/source evidence, affected distribution, violated obligation, and required remediation.

## License Inventory Delta

- Added: <component/version/license/scope>
- Modified: <component/use or distribution change>
- Removed: <component and evidence of complete removal>

## Required Notice Updates

- `NOTICE`: <changes or None>
- `THIRD_PARTY_NOTICES`: <changes or None>
- `licenses/`: <changes or None>
- Module README / `DATASET_NOTICE.md`: <changes or None>

## Legal Review Required

- <specific unresolved question or None>
```

没有问题时写：`No actionable open-source compliance findings.`，但不得据此宣称已经获得法律意见。

---

## 7. 当前仓库合规基线摘要

- 主工程：Apache License 2.0。
- 已收录许可证文本：Apache-2.0、MIT、MIT-0、BSD-2-Clause、BSD-3-Clause、MPL-2.0、LGPL-2.1-or-later、CC-BY-4.0、CC-BY-SA-3.0。
- 当前明确 copyleft 组件：
  - Eigen3 子集：主要 MPL-2.0，另有 Apache/BSD 文件级许可；位于 `3rdparty/eigen3`，保持 `EIGEN_MPL2_ONLY`。
  - libsndfile：LGPL-2.1-or-later；位于 `models/tts/cosyvoice3/cpp/` 使用链，按现有 NOTICE 采用动态链接。
- 当前未在根合规文件中声明实际采用 GPL、AGPL 或 MulanPSL 组件；任何新增均默认阻断并升级审批。
- 数据集：仅少量 COCO 2017 样本随仓库分发并按 CC BY 4.0 署名；其他数据集默认用户自行从官方来源获取。
- 学术/非商用或禁止衍生数据：BDD100K、nuScenes、WIDER FACE；ImageNet 需注册并遵守 Terms of Use。
- 现有 Transformers 和 qwen_vl_utils 衍生脚本必须保留原始 Apache-2.0 归属并显著说明修改。
