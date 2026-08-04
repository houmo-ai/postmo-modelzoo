# Copyright (c) 2026 HOUMO AI
#
# File: test_artifact_preparer.py
# Description:
#  Unit tests for high-level artifact dependency preparation, reuse, and
#    release-model fallback behavior.
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

"""Unit tests extracted from the former model-flow contract suite: test_artifact_preparer.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
    LLM_FLOW_POLICY,
)
from tests.models_tests.model_workflow.flow_contracts import (
    DiagnosticContext,
    FlowContext,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
    ValidationResult,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfig,
)
from tests.models_tests.test_flows.artifact_preparation import (
    ArtifactNeed,
    ArtifactPreparer,
    PreparationReport,
)
from tests.models_tests.test_flows.compile_flow import (
    CompileFlowHandler,
)
from tests.models_tests.test_flows.get_model_flow import (
    GetModelFlowHandler,
)
from tests.models_tests.test_flows.inference_flow_support import (
    prepare_inference_workspace,
)
from tests.models_tests.test_flows.quant_flow import (
    QuantFlowHandler,
)
from tests.tests_utils.runtime_context import (
    TCaseType,
)
from types import (
    SimpleNamespace,
)

pytestmark = pytest.mark.unit


def test_inference_compile_failure_can_fall_back_to_downloaded_hmm(
    tmp_path: Path, monkeypatch
) -> None:
    config = ModelConfig(
        model_name="download-fallback",
        path=tmp_path / "model_cfg_download-fallback.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "compile", "demo")},
        raw={
            "compile_params": {
                "xh2": {
                    "model_dir": ["cached_results/hmquant_xh2"],
                    "output_dir": ["cached_results/hmm_xh2"],
                }
            }
        },
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "fallback", config.model_name, ModelFamily.LLM, "xh2", ModelFlow.DEMO
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        log_file=tmp_path / "demo.log",
    )
    compile_failure = FlowResult(
        FlowDisposition.EXECUTED,
        "compile failed",
        validation=ValidationResult(
            False, "compile failed", failures=("compile failed",)
        ),
    )
    download_success = FlowResult(
        FlowDisposition.EXECUTED,
        "downloaded",
        validation=ValidationResult(True, "downloaded"),
    )
    validations = iter((["missing local HMM"], ["compile still missing HMM"], []))
    monkeypatch.setattr(CompileFlowHandler, "run", lambda *args: compile_failure)
    monkeypatch.setattr(
        ArtifactPreparer,
        "_download_release_hmms",
        lambda *args: PreparationReport(commands=download_success.commands),
    )
    monkeypatch.setattr(
        "tests.models_tests.test_flows.inference_flow_support."
        "validate_python_compiled_artifacts",
        lambda *args: next(validations),
    )
    commands, failures = prepare_inference_workspace(
        FlowRequest(context, config),
        SimpleNamespace(),
        tmp_path,
        LLM_FLOW_POLICY,
    )
    assert commands == []
    assert failures == []


def test_artifact_preparer_restores_raw_workspace_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_cache = tmp_path / "models"
    (model_cache / "assets").mkdir(parents=True)
    (model_cache / "assets" / "audio.mp3").write_bytes(b"audio")
    (model_cache / "raw-model.bin").write_bytes(b"raw")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = ModelConfig(
        model_name="artifact-preparer-raw",
        path=tmp_path / "model_cfg_artifact-preparer-raw.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "quant")},
        raw={"model_type": "llm"},
    )
    context = FlowContext(
        DiagnosticContext(
            "artifact-raw",
            config.model_name,
            ModelFamily.LLM,
            "xh2",
            ModelFlow.QUANT,
        ),
        "x86_64",
        TCaseType.DEFAULT,
        False,
        tmp_path / "quant.log",
        source,
        model_cache,
        tmp_path / "results",
        "1",
        "12g",
    )
    calls = []

    def fake_get_model(handler, request, services):
        calls.append((handler.file_types, handler.case_ids))
        return FlowResult(
            FlowDisposition.EXECUTED,
            "raw prepared",
            validation=ValidationResult(True, "raw prepared"),
            workspace_outputs=(Path("assets/audio.mp3"),),
        )

    monkeypatch.setattr(GetModelFlowHandler, "run", fake_get_model)
    report = ArtifactPreparer().ensure(
        FlowRequest(context, config),
        SimpleNamespace(),
        (ArtifactNeed.raw_model(frozenset({"raw-case"})),),
        workspace=workspace,
        policy=LLM_FLOW_POLICY,
    )

    assert not report.failures
    assert calls == [(frozenset({"raw"}), frozenset({"raw-case"}))]
    assert (workspace / "assets" / "audio.mp3").read_bytes() == b"audio"
    assert not (workspace / "raw-model.bin").exists()


def test_artifact_preparer_reuses_existing_quant_input_without_running_quant(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    quant_dir = tmp_path / "results" / "hmquant_xh2"
    quant_dir.mkdir(parents=True)
    (quant_dir / "model.onnx").write_bytes(b"quant")
    config = ModelConfig(
        model_name="artifact-preparer-quant",
        path=tmp_path / "model_cfg_artifact-preparer-quant.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("quant", "compile")},
        raw={
            "model_type": "cv",
            "compile_params": {
                "xh2": {
                    "model_dir": ["cached_results/hmquant_xh2"],
                    "output_dir": ["cached_results/hmm_xh2"],
                }
            },
        },
    )
    context = FlowContext(
        DiagnosticContext(
            "artifact-quant",
            config.model_name,
            ModelFamily.CV,
            "xh2",
            ModelFlow.COMPILE,
        ),
        "x86_64",
        TCaseType.DEFAULT,
        False,
        tmp_path / "compile.log",
        source,
        tmp_path / "models",
        tmp_path / "results",
        "1",
        "12g",
    )

    def fail_quant(*args):
        raise AssertionError("existing quant input must not rerun quant")

    monkeypatch.setattr(QuantFlowHandler, "run", fail_quant)
    report = ArtifactPreparer().ensure(
        FlowRequest(context, config),
        SimpleNamespace(),
        (ArtifactNeed.quant_model(),),
        policy=CV_FLOW_POLICY,
    )

    assert not report.failures
    assert report.message == "quant inputs already exist"
