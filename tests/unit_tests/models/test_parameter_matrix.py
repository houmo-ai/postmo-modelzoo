# Copyright (c) 2026 HOUMO AI
#
# File: test_parameter_matrix.py
# Description:
#  Unit tests for parameter-matrix expansion, command rendering, and logical
#    cache-path resolution.
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

"""Unit tests extracted from the former model-flow contract suite: test_parameter_matrix.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.cache_path_resolver import (
    cache_case_reference,
    resolve_case_paths,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ConfigError,
)
from tests.models_tests.model_workflow.parameter_matrix import (
    ParameterCase,
    ParameterMatrix,
    render_case_options,
)

pytestmark = pytest.mark.unit


def test_parameter_matrix_requires_equal_columns() -> None:
    with pytest.raises(ConfigError):
        ParameterMatrix.from_columns({"a": [1], "b": [1, 2]}, location="test")


def test_parameter_renderer_normalizes_values() -> None:
    matrix = ParameterMatrix.from_columns(
        {
            "script": ["demo.py"],
            "batch": [4],
            "verbose": [True],
            "unused": [None],
            "defaulted": ["default"],
        }
    )
    assert render_case_options(matrix.cases[0], positional_keys={"script"}) == (
        "demo.py",
        "--batch",
        "4",
        "--verbose",
    )


def test_shared_case_path_resolution_replaces_both_cache_roots(
    tmp_path: Path,
) -> None:
    case = ParameterCase(
        3,
        {
            "model_dir": "cached_models/raw/model",
            "output_dir": "cached_results/hmm_xh2",
            "batch": 2,
        },
    )
    resolved = resolve_case_paths(case, tmp_path / "models", tmp_path / "results")
    assert resolved.index == case.index
    assert resolved.values == {
        "model_dir": str(tmp_path / "models" / "raw" / "model"),
        "output_dir": str(tmp_path / "results" / "hmm_xh2"),
        "batch": 2,
    }


def test_cache_case_reference_reads_both_cache_roots() -> None:
    assert cache_case_reference("cached_results/hmm_xh2_2k/hmquant/quant.pt") == (
        "cached_results",
        "hmm_xh2_2k",
    )
    assert cache_case_reference("cached_models/raw_xh2/model.onnx") == (
        "cached_models",
        "raw_xh2",
    )
    assert cache_case_reference("cached_results") is None
    assert cache_case_reference("data/pic/dog.jpg") is None
