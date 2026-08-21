# iModelzoo Repository Layout

This document is the repository navigation and ownership guide. It describes stable, checked-in source, configuration, tests, and entrypoints. Local model downloads, datasets, caches, virtual environments, build trees, logs, and generated artifacts are not part of the source layout unless explicitly noted.

## Repository purpose and execution model

iModelzoo (also presented as `houmo-examples` in the root README) provides end-to-end examples for moving models onto Houmo platforms. The main workflows cover model acquisition or conversion, quantization, compilation, deployment and inference, accuracy comparison or evaluation, and performance measurement.

The repository has four main implementation layers:

1. `models/` contains user-facing, model-specific workflows.
2. `apis/` demonstrates lower-level conversion and runtime APIs.
3. `hmatc/`, `utils/`, and `tools/` provide reusable workflow, engine, evaluation, and performance functionality.
4. `tests/`, `config/imodelExampleConfig.yaml`, and `run_all.py` provide regression coverage and changed-path test selection.

Before following a workflow, use `env.sh` on Linux or `env.bat` on Windows. These scripts establish repository-wide path contracts such as `HOUMO_EXAMPLES_PATH`, `HOUMO_PATH`, `HOUMO_SDK_PATH`, `TCIM_RUNTIME_PATH`, `HOUMO_DATASETS_PATH`, `HOUMO_MODEL_PATH`, `PATH`, `PYTHONPATH`, and the runtime library path.

## Top-level directories

- `.github/`: Repository guidance, prompts, and reusable agent skills.
  - `guidance/`: Repository layout, coding style, review routing, and small reference implementations.
  - `skills/`: Task-specific and code-review skills for models, APIs, HMATC, test generation, README generation, compliance, and review publication.
  - `prompts/`: Reusable task prompts.
- `.claude/` and `.codex/`: Assistant-specific repository configuration. Treat these as repository maintenance files, not runtime source.
- `3rdparty/`: Repository-wide vendored C/C++ dependencies, including Eigen, yaml-cpp, tokenizer.cpp, spdlog, nlohmann, and related libraries. Do not modify these trees unless explicitly requested.
- `apis/`: Conversion and inference API examples, their example assets, and API-local dependencies.
- `cmake/`: Shared CMake integration for TCIM Runtime, Houmo Engine, tokenizer.cpp, yaml-cpp, and supported platforms.
- `config/`: Repository-wide change-to-test selection configuration.
- `data/`: Checked-in demonstration media, dataset placeholders, dataset metadata, and the dataset preparation helper.
- `hmatc/`: Installable Houmo Model Assist Toolkit package and its package-local dependencies.
- `hmodel/`: References to independent large-model/quantization repositories plus lightweight shared registry and pretrained-model helpers owned by iModelzoo. The external repositories' source code is not included in iModelzoo.
- `licenses/`: License texts for third-party assets and examples.
- `models/`: End-to-end model examples grouped by task.
- `tests/`: API, HMATC, model-workflow, shared, and unit tests.
- `tools/`: Standalone hardware, evaluation, environment, and performance tools.
- `utils/`: Shared first-party Python and C++ engine code and packaged runtime libraries.

## API examples

`apis/` is organized by user-visible API workflow:

- `apis/converts/<example>/`: Model conversion, quantization, and compilation API examples. Current examples include Qwen pipeline/speculative compilation and ResNet50/YOLOv5s conversion flows.
- `apis/inferences/<example>/`: Python and C++ runtime inference examples, including single-model, multibatch, multistream, pipeline, and speculative-decoding variants.
- `apis/common/`: Shared API-example helpers when present. This directory may mix first-party wrappers with vendored headers; determine ownership before editing.
- `apis/3rdparty/`: API-local third-party source. Do not edit by default.
- `apis/data/` and `apis/models/`: Example inputs and model assets. Treat them as data, not implementation source.

API examples normally expose `run.sh`, `run.bat`, a Python entrypoint, a CMake project, or an equivalent one-command entry. When changing an example, keep the entry script, implementation, CMake, assets, tests under `tests/apis_tests/`, and the root README `## API 示例` table consistent. A new or modified API example must be represented in that README table; deleting or moving an example requires updating or removing the corresponding row.

## Model examples

Model examples live under `models/<category>/<model-name>/`. Current categories are:

- `asr/`: Speech recognition, forced alignment, audio classification, and audio understanding.
- `autodrive/`: Autonomous-driving perception workflows.
- `backbone/`: Classification and feature-extraction backbones.
- `detection/`: Object and face detection.
- `diffusion/`: Image and video generation.
- `embedding/`: Text, vision, and multimodal embeddings.
- `estimation/`: Pose and related estimation tasks.
- `lalm/`: Large audio-language models.
- `llm/`: Large language models.
- `ocr/`: Text detection, recognition, and document OCR.
- `omni/`: Multimodal omni models.
- `reranker/`: Ranking and reranking models.
- `segmentation/`: Image segmentation.
- `tts/`: Text-to-speech.
- `vlm/`: Vision-language models.

A model directory can contain:

- `README.md` or `README.MD`: Supported variants, prerequisites, commands, and limitations.
- `config.yaml` or `config.yml`: Model, preprocessing, quantization, build, inference, evaluation, and output configuration.
- `get_model.py`: Model/tokenizer/processor acquisition or conversion.
- `ptq.py`: Quantization or quantization-export logic.
- `build.py`: Compilation and artifact production.
- `demo.py` and optional `python/` or `cpp/`: End-to-end inference implementations.
- `eval.py`, `perf.py`, or HMATC/tool invocations: Accuracy and performance stages.
- `test.sh`: The model's supported stage dispatcher, commonly exposing some subset of `get`, `quant`, `build`/`compile`, `demo`, `compare`, `eval`, `perf`, and `all`.
- `requirements*.txt`: Model-specific Python dependencies.

Not every model implements every stage. Preserve the contract of the touched model family instead of imposing a single template. Small CV models often delegate several stages directly to HMATC using `config.yml`; large or multimodal models more often keep explicit Python/C++ orchestration alongside `config.yaml`.

When adding, changing, moving, or deleting a model example, check all coupled registration points:

- The root README `## 模型示例` table. A new or modified example must be listed; deleting or moving an example requires updating or removing the row.
- `tests/models_tests/model_configs/model_cfg_<model>.json` when the model participates in automated workflows.
- The generated or maintained pytest stage files under `tests/models_tests/`.
- `config/imodelExampleConfig.yaml` for changed-path selection.
- `models/benchmark.yml` when the model participates in benchmark matrices.
- `imodelzoo.yaml` and `imodelzoo_xh2.yaml` when packaging contents change.

## HMATC

`hmatc/` is an installable Python package with the console entrypoint `hmatc = hmatc:main`.

- `hmatc/hmatc/cli/`: CLI parser and request resolution.
- `hmatc/hmatc/base/`: Shared dataset, model, inference, and execution abstractions.
- `hmatc/hmatc/exec/`: Platform execution, quantization, compilation, comparison, and artifact orchestration.
- `hmatc/hmatc/infer/`: ONNX, HMONNX, and device/runtime inference backends.
- `hmatc/hmatc/dataloaders/` and `datasets/`: Dataset construction and task-specific loading.
- `hmatc/hmatc/models/`: Model backend registry and model-specific adapters.
- `hmatc/hmatc/optimizer/`: ONNX optimization and platform-specific graph transformations.
- `hmatc/hmatc/utils/`: Configuration validation, preprocessing/postprocessing, metrics, monitoring, result management, and benchmark helpers.
- `hmatc/hmatc/python/`: Native Python extensions for performance and optional device monitoring.
- `hmatc/3rdparty/`: HMATC-local vendored dependencies.
- `hmatc/build/` and `hmatc/dist/`: Generated packaging outputs; never edit directly.

The public CLI currently includes `quant`, `build`, `compare`, `perf`, `demo`, `eval`, `benchmark`, `check`, `gen`, and `golden`. Some commands accept either a model config or a direct artifact such as HMONNX/HMM. Configuration version 1 drives conventional ONNX workflows; version 2 routes large-model workflows through `lm_runner.py`. Treat CLI options, config schema, default artifact names, and result structures as public cross-repository contracts: model examples, API examples, tests, and README commands may depend on them.

## Shared engines and utilities

`utils/` contains reusable first-party implementation shared by examples:

- `utils/python/houmo_engine/`: Python Demo/Engine/Process/Module framework, sampling, performance helpers, and model engine components.
- `utils/python/image/`: Shared image preprocessing and visualization helpers.
- `utils/python/common/` and `utils/python/get_resources/`: Common and resource-acquisition helpers when present in the checkout; confirm whether files are tracked before treating local additions as repository APIs.
- `utils/cpp/houmo_engine/`: Cross-platform C++ inference engine with core model abstractions, processing modules, sampling, performance support, build scripts, and adaptation documentation.
- `utils/cpp/include/`: Shared C++ headers, including dataset helpers.
- `utils/lib/`: Packaged runtime libraries, not a place for handwritten source changes.

Put behavior here only when it is genuinely shared. Keep model-specific preprocessing, tensor semantics, and artifact assumptions in the model directory until a stable reusable contract exists.

## External large-model and quantization repositories

iModelzoo does not contain the implementation source of the repositories represented by `hmodel/xh2` and `hmodel/gptqmodel`. Their canonical ownership is:

- `hmodel/xh2` -> the independent `houmoquantization/xh2modelzoo` repository.
- `hmodel/gptqmodel` -> the independent `houmoquantization/gptqmodel` repository.

The paths are recorded by Git as gitlinks. A developer checkout may materialize them as submodules, local symlinks, linked sibling worktrees, or another integration, but that local representation does not make their code part of iModelzoo. `hmodel/utils/` is different: it contains lightweight first-party pretrained-model, registry, and helper code owned by iModelzoo.

When an iModelzoo model, HMATC flow, configuration, or test depends on behavior implemented by one of these repositories, inspect the corresponding repository directly at the revision used by the environment:

- Search `houmoquantization/xh2modelzoo` for XH2 model definitions, large-model workflow configuration, conversion/quantization implementation, tools, and backend behavior.
- Search `houmoquantization/gptqmodel` for GPTQModel quantization implementation, integrations, configuration, and related backend behavior.

Do not infer external implementation details from the `hmodel/xh2` or `hmodel/gptqmodel` path name, gitlink metadata, iModelzoo call site, or a possibly stale local symlink. Follow the target repository's own instructions and source. If the corresponding repository is unavailable, state that the external implementation could not be inspected instead of treating it as part of iModelzoo. Changes to either external repository are separate changes and are not included in an iModelzoo patch unless the gitlink revision itself is intentionally updated.

## Tests and test selection

- `tests/conftest.py`: Repository-wide environment setup, logging fixture, shared markers, and collection policy. Unit tests are skipped during ordinary integration collection unless explicitly selected by path or the `unit` marker.
- `tests/apis_tests/`: API example integration tests and `apis_configs/`.
- `tests/hmatc_tests/`: HMATC functional/performance tests and `hmatc_configs/`.
- `tests/models_tests/`: Generated or maintained pytest entrypoints for get, quant, compile, demo, compare, eval, and perf stages.
  - `model_configs/`: Per-model JSON configuration and supported workflow metadata.
  - `model_workflow/`: Reusable configuration, artifact cache/publication, workspace, parameter matrix, Python environment, backend policy, and metric-validation primitives.
  - `test_flows/`: Stage orchestration for get, quant, compile, demo, compare, eval, and perf.
  - `update_test_py.py`: Updates stage-specific pytest cases from model configuration.
- `tests/tests_utils/`: Shared command execution, platform/device, pytest, locking, environment, runtime-context, and workspace helpers.
- `tests/unit_tests/`: Device- and model-independent tests for API/HMATC runners and model workflow architecture. Select this directory explicitly or use `pytest -m unit`.

`config/imodelExampleConfig.yaml` maps changed paths to test modules/cases. `run_all.py --type diff_file --diff_file <path-list>` consumes that mapping, prepares required tools, marks selected pytest cases, and runs the selected suite. Because this is CI orchestration, changes to its path matching or config schema must remain synchronized with the YAML and tests.

## Tools

- `tools/bandwidth_perf/`: Memory-bandwidth measurement script and runner.
- `tools/computing_perf/`: Compute-throughput measurement script and runner.
- `tools/hm_check/`: Cross-platform C++ hardware/environment checker.
- `tools/hmeval/`: Installable evaluation package and example evaluation workflows.
- `tools/llm_perf/`: Cross-platform C++ ASR/LLM/TTS/VLM performance runner and YAML configuration.
- `tools/tcim_perf/`: Generic TCIM model performance executable and build/run scripts.
- `tools/win_envs/`: Windows environment configuration utility.
- `tools/common/`: Shared tool implementation when present. It may contain both first-party components such as `houmo-llm-engine` and vendored libraries; verify ownership before editing.
- `tools/bin/`: Built or packaged helper executables and wrappers. Prefer changing their source project instead of editing binaries or generated copies.

## Data, CMake, and packaging metadata

- `data/audio/`, `data/pic/`, and `data/video/`: Small demonstration media.
- `data/datasets/`: Dataset placeholders, metadata, a small distributable COCO sample, and `get_datasets.py`. Follow `DATASET_NOTICE.md`; do not commit restricted or downloaded datasets.
- `cmake/platforms/`: Platform toolchain/configuration for Linux, Windows, Android, and OpenHarmony/OHOS.
- `cmake/*.cmake`: Shared dependency and runtime integration used by C++ examples and tools.
- `imodelzoo.yaml` and `imodelzoo_xh2.yaml`: Packaging/export manifests that select repository contents for deliveries.
- `requirements.txt`: Top-level Python dependencies.
- `.pre-commit-config.yaml`: Formatting, linting, static analysis, and source checks.
- `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES`, `licenses/`, and `DATASET_NOTICE.md`: Licensing, attribution, and dataset-distribution policy.

## Navigation and change boundaries

- Start model work in `models/<category>/<model>/`, then inspect its `test.sh`, config, tests, aggregate selection config, packaging manifests, and root README row.
- Start API work in `apis/converts/<example>/` or `apis/inferences/<example>/`, then inspect its runner/CMake, API tests, aggregate selection config, and root README row.
- Start HMATC work at the CLI/parser or execution layer that owns the behavior, then search all model/API callers before changing an option, default, config key, artifact name, or result format.
- Start shared engine work in `utils/python/houmo_engine/` or `utils/cpp/houmo_engine/`; verify all model consumers and supported platforms.
- Use `tools/` for reusable standalone diagnostics, evaluation, and performance programs rather than model-specific orchestration.
- Do not treat `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, virtual environments, downloaded model trees, dataset downloads, device logs, `output*/`, `work_dir*/`, `build*/`, `dist/`, compiled binaries, or model artifacts as source.
- Preserve unrelated local changes. In particular, do not normalize or delete downloaded assets merely because they appear in the working tree.
