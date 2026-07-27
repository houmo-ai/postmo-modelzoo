# iModelzoo Repository Layout

This document is a navigation guide for the current repository. It describes stable source and configuration locations rather than generated outputs, caches, virtual environments, or every model instance.

## Top-level source and data directories

- `apis/`: HAL and runtime API examples plus API-side assets.
  - `3rdparty/`: Third-party dependencies used by API examples.
  - `converts/`: Model conversion and compilation API examples.
  - `inferences/`: Runtime inference API examples.
  - `data/`: API demonstration data.
  - `models/`: Sample models and assets used by API examples.
- `utils/`: Repository-wide first-party shared code and runtime libraries.
  - `python/houmo_engine/`: Shared Python Demo/Engine/Process/Module framework and model-specific engine components.
- `config/`: Shared configuration entrypoints.
  - `imodelExampleConfig.yaml`: Aggregated example and test configuration.
- `data/`: Repository data assets.
  - `audio/`: Audio samples.
  - `datasets/`: Evaluation datasets and dataset metadata.
  - `pic/`: Image samples.
- `hmatc/`: HMATC CLI and Python package for quantization, compilation, validation, inference, performance testing, demos, and evaluation.
  - `hmatc/`: Installable Python package and command implementation.
  - `3rdparty/`: HMATC-specific third-party dependencies.
  - `build/` and `dist/`: Generated packaging outputs; do not edit directly.
- `hmodel/`: Large-model projects, integrations, and shared utilities.
  - `gptqmodel/`: GPTQModel-based quantization project and integrations.
  - `utils/`: Shared pretrained-model, registry, and helper utilities.
  - `xh2/`: XH2 large-model repository with its own instructions, configurations, examples, tools, tests, and packaging rules.
- `licenses/`: Texts for licenses referenced by repository assets and examples.
- `models/`: End-to-end model examples, typically covering some combination of model acquisition, quantization, compilation, demo, performance testing, and evaluation.
- `tests/`: Pytest-based unit and integration tests.
- `tools/`: Performance, evaluation, runtime, environment, and platform utilities.

## Model categories

Model examples are grouped under `models/<category>/<model-name>/`:

- `asr/`: Automatic speech recognition and audio understanding models.
- `autodrive/`: Autonomous-driving models.
- `backbone/`: Classification and feature-extraction backbones such as ResNet, EfficientNet, MobileNet, and ViT.
- `detection/`: Object and face detection models.
- `diffusion/`: Diffusion and image-generation models.
- `embedding/`: Text, vision, and multimodal embedding models.
- `estimation/`: Pose and related estimation models.
- `llm/`: Large language models.
- `ocr/`: Optical character recognition models.
- `omni/`: Multimodal omni models.
- `reranker/`: Reranking models.
- `segmentation/`: Image segmentation models.
- `tts/`: Text-to-speech models.
- `vlm/`: Vision-language models.

A model directory may contain files such as `config.yaml`, `get_model.py`, `ptq.py`, `build.py`, `demo.py`, `eval.py`, `test.sh`, and `README.MD`. Do not assume every model implements every stage; preserve the workflow and naming conventions of the touched model family.

## Tests

- `tests/conftest.py`: Repository-wide pytest configuration and shared setup.
- `tests/apis_tests/`: API conversion, inference, and scene tests.
  - `apis_configs/`: API test configurations.
- `tests/hmatc_tests/`: HMATC CLI and utility tests.
  - `hmatc_configs/`: HMATC test configurations.
- `tests/models_tests/`: Model workflow tests for get, quant, compile, demo, performance, evaluation, and comparison stages.
  - `model_configs/`: Per-model test configuration JSON files.
- `tests/tools_tests/`: Tool-level tests and test scripts.
- `tests/tests_utils/`: Shared test utilities and fixtures.

When changing a model workflow, check both the model directory and the corresponding configuration or test entry under `tests/models_tests/` and `config/imodelExampleConfig.yaml`.

## Tools

- `tools/bandwidth_perf/`: Memory-bandwidth performance tools.
- `tools/bin/`: Helper executables and wrapper scripts.
- `tools/computing_perf/`: Compute-performance tools.
- `tools/hm_check/`: Hardware and environment checking tools.
- `tools/hmeval/`: Evaluation CLI and examples.
- `tools/llm_perf/`: Large-language-model performance tools.
- `tools/tcim_perf/`: TCIM model performance tools.
- `tools/win_envs/`: Windows environment setup documentation and utilities.
- `cmake/tcim_runtime.cmake`: Shared TCIM runtime CMake integration.

## Top-level entrypoints and metadata

- `env.sh` / `env.bat`: Linux and Windows environment setup scripts.
- `enter_docker.sh`: Docker entry helper.
- `imodelzoo.yaml` / `imodelzoo_xh2.yaml`: Top-level model and example aggregation manifests.
- `run_all.py`: Batch runner for repository examples and tests.
- `.clang-format`: Root C/C++ formatting configuration; more-specific subtree configurations may override it.
- `.pre-commit-config.yaml`: Repository formatting, linting, and static-check hooks.
- `requirements.txt`: Top-level Python dependencies.
- `README.md`: Repository overview, platform requirements, model matrix, and API example documentation.
- `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES`: Licensing and attribution files.

## Navigation and change boundaries

- Start model-specific work in `models/<category>/<model-name>/`, then check the related tests, aggregate configuration, and README.
- Put reusable Python engine behavior in `utils/python/houmo_engine/` only when it is shared across model examples; keep model-specific logic with the model when reuse is not established.
- Use `hmatc/` for shared quantization, build, validation, demo, evaluation, and CLI behavior rather than duplicating HMATC logic in individual examples.
- Use `tools/` for reusable standalone utilities and performance tooling.
- Respect nested project instructions and local formatter or build configuration, especially under `hmodel/xh2/`, `hmodel/gptqmodel/`, `apis/`, and `hmatc/`.
- Do not treat caches, virtual environments, downloaded models, generated packages, or build directories as source. Common examples include `__pycache__/`, `.pytest_cache/`, `*_venv/`, `build/`, `builds/`, `dist/`, and model output directories.
