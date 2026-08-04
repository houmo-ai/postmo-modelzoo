# Copyright (c) 2026 HOUMO AI
#
# File: test_update_test_py.py
# Description:
#  Unit tests for generated pytest functions, dependency markers, obsolete
#    models, and disabled-backend behavior.
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

"""Unit tests extracted from the former model-flow contract suite: test_update_test_py.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfig,
    ModelConfigRepository,
)
from tests.models_tests.update_test_py import (
    convert_model_name,
    generated_outputs,
    supported_flows,
)
from ._flow_contract_support import (
    CONFIG_DIR,
)

pytestmark = pytest.mark.unit


def test_codegen_dependency_names_match_cross_file_targets() -> None:
    outputs = generated_outputs()
    get_model = next(
        content
        for path, content in outputs.items()
        if path.name == "test_get_models.py"
    )
    quant = next(
        content
        for path, content in outputs.items()
        if path.name == "test_quant_models.py"
    )
    compile_output = next(
        content
        for path, content in outputs.items()
        if path.name == "test_compile_models.py"
    )
    assert 'name="test_get_models.py::test_' in get_model
    assert 'depends_on=["test_get_models.py::test_' in quant
    assert 'depends_on=["test_quant_models.py::test_' in compile_output


def test_codegen_keeps_obsolete_markers_and_cases() -> None:
    repository = ModelConfigRepository(CONFIG_DIR)
    configs = tuple(repository.iter_configs(include_obsolete=True))
    obsolete_markers = {
        convert_model_name(config.model_name) for config in configs if config.obsolete
    }
    assert obsolete_markers

    outputs = generated_outputs()
    marker_output = next(
        content for path, content in outputs.items() if path.name == "model_names.txt"
    )
    assert obsolete_markers <= set(marker_output.splitlines())

    generated_tests = "\n".join(
        content for path, content in outputs.items() if path.name.startswith("test_")
    )
    for marker in obsolete_markers:
        assert f"@pytest.mark.{marker}\n" in generated_tests


def test_codegen_ignores_flow_sections_for_disabled_backends(tmp_path: Path) -> None:
    config = ModelConfig(
        model_name="backend-toggle",
        path=tmp_path / "model_cfg_backend-toggle.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/backend-toggle"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh1": ("eval",), "xh2": ("demo",)},
        raw={},
    )
    assert supported_flows(config) == frozenset({ModelFlow.DEMO})
