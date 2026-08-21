# Instructions for iModelzoo

## 基本规则

- 默认使用中文说明；不要翻译代码、路径、命令、标识符、配置键或工具输出。
- 先阅读变更目录最近的 README、配置、入口脚本、测试和构建文件。
- 实现遵循 `.github/guidance/coding-style.md`；代码审查遵循 `.github/guidance/review-guidelines.md` 及对应 `.github/skills/`。
- 保持 diff 小而聚焦，保留用户已有修改，不做无关重构或格式化。
- 不新增依赖，不随意改变 CLI、配置键、默认值、产物名、输出格式或公共 API/ABI。

## 目录职责

- `models/<category>/<model>/`：模型获取、转换、量化、编译、推理、评测和性能示例；通常以 `test.sh` 组织阶段。
- `apis/converts/`、`apis/inferences/`：转换和 Runtime API 示例。
- `hmatc/hmatc/`：`hmatc` 公共 CLI 及量化、编译、推理、比较、评测和性能流程。
- `utils/python/houmo_engine/`、`utils/cpp/houmo_engine/`：跨模型复用的 Python/C++ Engine；模型专属逻辑留在模型目录。
- `tests/`：模型、API、HMATC 集成测试、共享测试工具和显式选择的 unit tests。
- `config/imodelExampleConfig.yaml`、`run_all.py`：changed-path 到测试用例的映射和批量测试编排。
- 详细结构见 `.github/guidance/repo-layout.md`。

## 外部仓库

- `hmodel/xh2` 仅引用独立仓库 `houmoquantization/xh2modelzoo`。
- `hmodel/gptqmodel` 仅引用独立仓库 `houmoquantization/gptqmodel`。
- iModelzoo 不包含这两个仓库的实现代码。分析其行为时，必须直接查看对应仓库及其 instructions；不能只依据 gitlink、软链接或 iModelzoo 调用点推断。
- 外部仓库或 revision 不可用时，明确说明限制，不虚构外部 contract；外部仓库修改须单独说明。

## 环境与工作流

- Linux 通常先执行 `source env.sh`；Windows 使用 `env.bat` 或 `tools/win_envs/`。
- 判断环境变量是否缺失前，沿初始化脚本、平台入口、父级脚本和文档命令追踪定义。
- 模型流程按实际实现检查 `get_model.py`、`ptq.py`/转换、`build.py`/HMATC、`demo.py`/C++、compare/eval/perf、tests 和 README；不要假设每个模型都有全部阶段。
- 修改模型示例时同步检查 `tests/models_tests/model_configs/`、测试 flow、`config/imodelExampleConfig.yaml`、必要的 packaging manifest，以及 README 的 `## 模型示例`。新增或修改模型未列出时属于 P0 review issue；删除、移动或重命名时同步更新条目。
- 修改 API 示例时同步检查 API tests、配置和 README 的 `## API 示例`。新增或修改 API 未列出时属于 P0 review issue；删除、移动或重命名时同步更新条目。
- 修改 HMATC 的 CLI、schema、默认值、产物或结果结构前，搜索所有模型/API 调用方、脚本、测试和 README。

## 测试与保护边界

- 只运行与变更相关的最小检查；Python 优先语法检查或聚焦 pytest，C/C++ 使用最近的 formatter/build/test，文档至少检查路径和 `git diff --check`。
- `tests/unit_tests/` 默认不参与普通集成收集，需显式路径或 `-m unit`。
- 不修改 `3rdparty/`、`apis/3rdparty/`、`hmatc/3rdparty/` 中的 vendored code；`apis/common/` 和 `tools/common/` 先确认所有权。
- 不编辑 build/dist/output/work_dir、缓存、虚拟环境、下载模型/数据集、日志、二进制和生成文件；应修改其源文件或生成器。
- 未经明确批准，不执行远程/历史 Git 操作、网络安装、提权、容器变更或大范围删除覆盖操作。

## 参考

- `.github/guidance/repo-layout.md`
- `.github/guidance/coding-style.md`
- `.github/guidance/review-guidelines.md`
- `README.md`
- `DATASET_NOTICE.md`
