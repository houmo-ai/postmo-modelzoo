# Copyright (c) 2026 HOUMO AI
#
# File: _flow_contract_support.py
# Description:
#  Shared builders, paths, fixtures, and source-inspection helpers used by
#    multiple model-flow unit-test modules.
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

"""Shared builders and source-tree paths for model-flow unit tests."""

import ast
from pathlib import Path
from types import SimpleNamespace

from tests.models_tests.model_workflow.flow_contracts import (
    DiagnosticContext,
    FlowContext,
    FlowRequest,
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import ModelConfig
from tests.tests_utils.runtime_context import TCaseType

__all__ = [
    "CONFIG_DIR",
    "MODELS_TESTS_DIR",
    "TESTS_DIR",
    "_demo_artifact_config",
    "_demo_artifact_request",
    "_hmatc_inference_test_request",
    "_module_imports",
    "_write_hmatc_inference_artifacts",
    "_write_hmatc_inference_config",
]


TESTS_DIR = Path(__file__).resolve().parents[2]
MODELS_TESTS_DIR = TESTS_DIR / "models_tests"
CONFIG_DIR = MODELS_TESTS_DIR / "model_configs"


def _demo_artifact_config(tmp_path: Path) -> ModelConfig:
    """Build a config whose demo references an artifact compile never produces."""
    return ModelConfig(
        model_name="aux-demo",
        path=tmp_path / "model_cfg_aux-demo.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/aux-demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "compile", "demo")},
        raw={
            "get_model_params": {
                "xh2": {
                    "type": ["hmm", "hmm"],
                    "extract_dir": [
                        "cached_models/hmm_xh2_main",
                        "cached_models/hmm_xh2_aux",
                    ],
                }
            },
            "compile_params": {
                "xh2": {
                    "model_dir": ["cached_results/hmquant_xh2"],
                    "output_dir": ["cached_results/hmm_xh2_main"],
                }
            },
            "demo_params": {
                "xh2": {
                    "script": ["demo.py"],
                    "main_hmm": ["cached_results/hmm_xh2_main/main.hmm"],
                    "aux_hmm": ["cached_results/hmm_xh2_aux/aux.hmm"],
                }
            },
        },
    )


def _demo_artifact_request(config: ModelConfig, tmp_path: Path) -> FlowRequest:
    """Build the common request used by demo artifact preparation tests."""
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "aux", config.model_name, config.family, "xh2", ModelFlow.DEMO
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "demo.log",
        source_dir=tmp_path / "source",
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="16g",
    )
    return FlowRequest(context, config)


def _hmatc_inference_test_request(tmp_path: Path) -> FlowRequest:
    """Build the common request used by HMATC inference preparation tests."""
    config = ModelConfig(
        model_name="resnet50",
        path=tmp_path / "model_cfg_resnet50.json",
        family=ModelFamily.CV,
        model_dir=Path("models/backbone/resnet50"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("demo", "compare", "eval", "perf")},
        raw={"hmbuild_params": {}},
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "prepare", config.model_name, ModelFamily.CV, "xh2", ModelFlow.DEMO
        ),
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        log_file=tmp_path / "prepare.log",
    )
    return FlowRequest(context, config)


def _write_hmatc_inference_config(workspace: Path, *, opt_level: int = 2) -> None:
    """Write a minimal HMATC config used by inference preparation tests."""
    (workspace / "config.yml").write_text(
        "model:\n"
        "  name: resnet50\n"
        "  save_dir: output\n"
        "  model_path: raw.onnx\n"
        f"build:\n  opt_level: {opt_level}\n",
        encoding="utf-8",
    )


def _write_hmatc_inference_artifacts(directory: Path) -> None:
    """Write the minimal HMM and quantized ONNX artifact bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resnet50.hmm").write_bytes(b"hmm")
    hmquant = directory / "hmquant"
    hmquant.mkdir(exist_ok=True)
    (hmquant / "hmquant_resnet50_with_act.onnx").write_bytes(b"onnx")


def _module_imports(path: Path) -> set[str]:
    """Return modules imported by one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
