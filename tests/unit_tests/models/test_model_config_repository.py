# Copyright (c) 2026 HOUMO AI
#
# File: test_model_config_repository.py
# Description:
#  Unit tests for model JSON discovery, normalization, schema validation, and
#    flow configuration contracts.
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

"""Unit tests extracted from the former model-flow contract suite: test_model_config_repository.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    BACKEND_POLICIES,
    FLOW_DEPENDENCY_RULES,
    FLOW_ORDER,
    PARTIAL_DATASET_THRESHOLD_FACTOR,
    filter_xh2_ncore4,
    hmatc_build_header,
    release_source_allowed,
    should_check_output_failure,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ConfigError,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfigRepository,
)
from tests.models_tests.model_workflow.perf_metric_validation import (
    PERF_BEHAVIOR_OVERRIDES,
)
from tests.models_tests.test_flows.flow_registry import (
    FLOW_REGISTRY,
)
from ._flow_contract_support import (
    CONFIG_DIR,
)

pytestmark = pytest.mark.unit


def test_all_model_configs_load() -> None:
    repository = ModelConfigRepository(CONFIG_DIR)
    configs = tuple(repository.iter_configs(include_obsolete=True))
    assert configs
    assert any(not config.obsolete for config in configs)


def test_every_active_model_flow_has_a_registered_handler() -> None:
    repository = ModelConfigRepository(CONFIG_DIR)
    for config in repository.iter_configs():
        for backend in config.support_backend:
            for flow_name in config.support_flow[backend]:
                if flow_name == "demo_multibatch":
                    continue
                handler = FLOW_REGISTRY.resolve(
                    config.family, backend, ModelFlow(flow_name)
                )
                assert handler is not None


def test_fixed_framework_policies(monkeypatch) -> None:
    assert PARTIAL_DATASET_THRESHOLD_FACTOR == 0.5
    assert BACKEND_POLICIES["xh1"].compile_cosine_threshold == 0.99
    assert BACKEND_POLICIES["xh2"].compare_cosine_threshold == 0.90
    assert FLOW_ORDER[ModelFlow.GET_MODEL] < FLOW_ORDER[ModelFlow.QUANT]
    assert FLOW_ORDER[ModelFlow.QUANT] < FLOW_ORDER[ModelFlow.COMPILE]
    assert FLOW_DEPENDENCY_RULES[ModelFlow.DEMO] == ()
    assert not release_source_allowed("raw", None)
    assert not release_source_allowed("hmm", "modelscope")
    monkeypatch.delenv("IMODELZOO_ALLOW_XH2_NCORE4", raising=False)
    cases = ({"ncore": 2}, {"ncore": 4}, {"ncore": "4"})
    assert filter_xh2_ncore4(cases, lambda case: case["ncore"]) == (cases[0],)
    monkeypatch.setenv("IMODELZOO_ALLOW_XH2_NCORE4", "ON")
    assert filter_xh2_ncore4(cases, lambda case: case["ncore"]) == cases
    assert "--skip_check" in hmatc_build_header("xh2", asic=False)
    assert "--skip_check" not in hmatc_build_header("xh2", asic=True)
    assert not should_check_output_failure("qwen2.5-vl", ModelFlow.DEMO)
    assert not should_check_output_failure("qwen2.5-vl", ModelFlow.PERF)
    assert should_check_output_failure("resnet50", ModelFlow.DEMO)


def test_model_specific_threshold_overrides_backend_default(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_cfg_demo.json").write_text(
        """{
          "obsolete": false,
          "model_dir": "models/cv/demo",
          "model_type": "cv",
          "dependencies": {},
          "support_platform": ["x86_64"],
              "support_backend": ["xh2"],
              "support_flow": {"xh2": ["compile"]},
              "compile_params": {
                "xh2": {
                  "model_dir": ["cached_results/hmquant_xh2"],
                  "output_dir": ["cached_results/hmm_xh2"]
                }
              },
              "validation": {"xh2": {"compile_cosine_threshold": 0.76}}
        }""",
        encoding="utf-8",
    )
    config = ModelConfigRepository(config_dir).load("demo")
    resolved = config.validation_threshold("xh2", "compile")
    assert resolved.value == 0.76
    assert resolved.source == "model.validation.xh2.compile_cosine_threshold"


def test_supported_flow_requires_its_parameter_section(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_cfg_invalid.json").write_text(
        """{
          "obsolete": false,
          "model_dir": "models/cv/invalid",
          "model_type": "cv",
          "dependencies": {},
          "support_platform": ["x86_64"],
          "support_backend": ["xh2"],
          "support_flow": {"xh2": ["eval"]}
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing parameter sections"):
        ModelConfigRepository(config_dir).load("invalid")


def test_model_json_does_not_control_perf_extraction_rules() -> None:
    repository = ModelConfigRepository(CONFIG_DIR)
    assert all(
        "perf_behavior" not in config.raw
        for config in repository.iter_configs(include_obsolete=True)
    )
    assert set(PERF_BEHAVIOR_OVERRIDES) == {("sdxl", "xh1"), ("wenet", "xh1")}
