# Copyright (c) 2026 HOUMO AI
#
# File: test_get_model_flow.py
# Description:
#  Unit tests for get-model case filtering, artifact publication, and
#    workspace side-effect persistence.
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

"""Unit tests extracted from the former model-flow contract suite: test_get_model_flow.py."""

import pytest
from dataclasses import (
    replace,
)
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
    ArtifactRequirement,
    ArtifactType,
    CacheStatus,
)
from tests.models_tests.model_workflow.artifact_workspace import (
    restore_workspace_outputs,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    GET_MODEL_COMMAND_TIMEOUT_SECONDS,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandResult,
    DiagnosticContext,
    FlowContext,
    FlowRequest,
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfig,
)
from tests.models_tests.test_flows.get_model_flow import (
    GetModelFlowHandler,
)
from tests.tests_utils.runtime_context import (
    TCaseType,
)
from tests.tests_utils.workspace import (
    WorkspaceManager,
)
from types import (
    SimpleNamespace,
)

pytestmark = pytest.mark.unit


def test_llm_get_model_flow_filters_release_cases_and_publishes_hmm(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    model_cache = tmp_path / "cache"
    config = ModelConfig(
        model_name="demo",
        path=tmp_path / "model_cfg_demo.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model",)},
        raw={
            "model_dir": "models/llm/demo",
            "model_type": "llm",
            "get_model_params": {
                "xh2": {
                    "type": ["raw", "hmm", "hmm"],
                    "download_dir": [
                        "cached_models",
                        "cached_models",
                        "cached_models",
                    ],
                    "extract_dir": [
                        None,
                        "cached_models/modelscope-hmm",
                        "cached_models/release-hmm",
                    ],
                    "source_type": [None, "modelscope", None],
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            run_id="test",
            model_name="demo",
            family=ModelFamily.LLM,
            backend="xh2",
            flow=ModelFlow.GET_MODEL,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=True,
        log_file=tmp_path / "flow.log",
        source_dir=source,
        model_cache_dir=model_cache,
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            assert command.timeout_seconds == GET_MODEL_COMMAND_TIMEOUT_SECONDS
            argv = list(command.argv)
            extract_dir = Path(argv[argv.index("--extract_dir") + 1])
            extract_dir.mkdir(parents=True, exist_ok=True)
            (extract_dir / "model.hmm").write_bytes(b"hmm")
            return CommandResult(command, 0, "downloaded\n", "", 0.01)

    services = SimpleNamespace(
        command_runner=FakeRunner(),
        workspace_manager=WorkspaceManager(),
        artifact_cache=ArtifactCache(),
    )
    result = GetModelFlowHandler(ModelFamily.LLM).run(
        FlowRequest(context=context, config=config), services
    )
    assert result.validation is not None and result.validation.passed
    assert len(result.commands) == 1
    artifact_dir = model_cache / "release-hmm"
    inspection = ArtifactCache().inspect(
        artifact_dir,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            "demo",
            "xh2",
            "release-hmm",
        ),
    )
    assert inspection.status == CacheStatus.VALID
    assert not (model_cache / "modelscope-hmm").exists()


def test_cv_get_model_workspace_outputs_follow_parent_flow_lifetime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    model_cache = tmp_path / "cache"
    config = ModelConfig(
        model_name="demo-cv",
        path=tmp_path / "model_cfg_demo-cv.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model",)},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "get_model_params": {
                "xh2": {"type": ["hmm"], "model_dir": ["cached_models"]}
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            run_id="get-model-test",
            model_name="demo-cv",
            family=ModelFamily.CV,
            backend="xh2",
            flow=ModelFlow.GET_MODEL,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "get-model.log",
        source_dir=source,
        model_cache_dir=model_cache,
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            output = command.cwd / "output" / "xh2"
            output.mkdir(parents=True)
            (output / "model.hmm").write_bytes(b"hmm")
            dataset = command.cwd / "calibration_data"
            dataset.mkdir()
            (dataset / "sample.bin").write_bytes(b"sample")
            (command.cwd / "classifier.pt").write_bytes(b"classifier")
            return CommandResult(command, 0, "downloaded\n", "", 0.01)

    result = GetModelFlowHandler(ModelFamily.CV).run(
        FlowRequest(context=context, config=config),
        SimpleNamespace(
            command_runner=FakeRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )
    assert result.validation is not None and result.validation.passed
    assert not (model_cache / "output").exists()
    assert not (model_cache / "calibration_data").exists()
    assert not (model_cache / "classifier.pt").exists()

    quant_context = replace(
        context,
        diagnostic=replace(context.diagnostic, flow=ModelFlow.QUANT),
        model_cache_dir=tmp_path / "quant-cache",
    )
    quant_result = GetModelFlowHandler(ModelFamily.CV).run(
        FlowRequest(context=quant_context, config=config),
        SimpleNamespace(
            command_runner=FakeRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )
    assert quant_result.validation is not None and quant_result.validation.passed
    quant_cache = quant_context.model_cache_dir
    assert (quant_cache / "output" / "xh2" / "model.hmm").read_bytes() == b"hmm"
    assert (quant_cache / "calibration_data" / "sample.bin").read_bytes() == b"sample"
    assert (quant_cache / "classifier.pt").read_bytes() == b"classifier"

    inspection = ArtifactCache().inspect(
        quant_cache / "output" / "xh2",
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            "demo-cv",
            "xh2",
            "xh2",
        ),
    )
    assert inspection.status == CacheStatus.VALID


def test_llm_get_model_restores_side_effects_without_copying_raw_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    model_cache = tmp_path / "cache"
    config = ModelConfig(
        model_name="demo-llm",
        path=tmp_path / "model_cfg_demo-llm.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model",)},
        raw={
            "model_dir": "models/llm/demo",
            "model_type": "llm",
            "get_model_params": {
                "xh2": {
                    "type": ["raw"],
                    "download_dir": ["cached_models/raw-model"],
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "llm-assets",
            config.model_name,
            ModelFamily.LLM,
            "xh2",
            ModelFlow.QUANT,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "get-model.log",
        source_dir=source,
        model_cache_dir=model_cache,
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            argv = list(command.argv)
            raw_model = Path(argv[argv.index("--download_dir") + 1])
            raw_model.mkdir(parents=True)
            (raw_model / "model.bin").write_bytes(b"model")
            (command.cwd / "audio.mp3").write_bytes(b"audio")
            return CommandResult(command, 0, "downloaded\n", "", 0.01)

    result = GetModelFlowHandler(ModelFamily.LLM, frozenset({"raw"})).run(
        FlowRequest(context, config),
        SimpleNamespace(
            command_runner=FakeRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )

    assert result.validation is not None and result.validation.passed
    assert result.workspace_outputs == (Path("audio.mp3"),)
    quant_workspace = tmp_path / "quant-workspace"
    quant_workspace.mkdir()
    restore_workspace_outputs(
        model_cache,
        quant_workspace,
        result.workspace_outputs,
    )
    assert (quant_workspace / "audio.mp3").read_bytes() == b"audio"
    assert not (quant_workspace / "raw-model").exists()
