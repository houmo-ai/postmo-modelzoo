# Copyright (c) 2026 HOUMO AI
#
# File: test_compile_flow.py
# Description:
#  Unit tests for Python and HMATC compile flow artifact handling, filtering,
#    publication, and diagnostics.
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

"""Unit tests extracted from the former model-flow contract suite: test_compile_flow.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
    ArtifactRequirement,
    ArtifactType,
    AtomicArtifactWriter,
    CacheStatus,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
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
    ModelConfigRepository,
)
from tests.models_tests.test_flows.compile_flow import (
    CompileFlowHandler,
    _artifact_owned_root,
)
from tests.models_tests.test_flows.inference_flow_support import (
    mirror_local_compile_outputs,
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
from ._flow_contract_support import (
    CONFIG_DIR,
)

pytestmark = pytest.mark.unit


def test_python_compile_flow_publishes_hmm_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "build.py").write_text("# fake\n", encoding="utf-8")
    result_cache = tmp_path / "results"
    quant_dir = result_cache / "hmquant_xh2"
    quant_dir.mkdir(parents=True)
    (quant_dir / "quant.onnx").write_bytes(b"quant")
    (quant_dir / "quant_embedding.pt").write_bytes(b"embedding")
    (quant_dir / "quant_embedding_code_predictor.pt").write_bytes(b"predictor")
    config = ModelConfig(
        model_name="demo-compile",
        path=tmp_path / "model_cfg_demo-compile.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("compile",)},
        raw={
            "model_dir": "models/cv/demo",
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
        diagnostic=DiagnosticContext(
            run_id="compile-test",
            model_name="demo-compile",
            family=ModelFamily.CV,
            backend="xh2",
            flow=ModelFlow.COMPILE,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "compile.log",
        source_dir=source,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=result_cache,
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    command_contexts = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            command_contexts.append(diagnostic_fields)
            argv = list(command.argv)
            output_dir = Path(argv[argv.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.hmm").write_bytes(b"hmm")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    services = SimpleNamespace(
        command_runner=FakeRunner(),
        workspace_manager=WorkspaceManager(),
        artifact_cache=ArtifactCache(),
    )
    result = CompileFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context=context, config=config), services
    )
    assert result.validation is not None and result.validation.passed
    assert len(result.commands) == 1
    assert command_contexts[0]["case_id"] == "hmm_xh2"
    assert command_contexts[0]["phase"] == "python-build"
    compiled_dir = result_cache / "hmm_xh2"
    inspection = ArtifactCache().inspect(
        compiled_dir,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            "demo-compile",
            "xh2",
            "hmm_xh2",
        ),
    )
    assert inspection.status == CacheStatus.VALID
    assert inspection.manifest is not None
    assert inspection.manifest.producer_flow == ModelFlow.COMPILE.value
    assert (
        compiled_dir / "hmquant" / "quant_embedding.pt"
    ).read_bytes() == b"embedding"
    assert (
        compiled_dir / "hmquant" / "quant_embedding_code_predictor.pt"
    ).read_bytes() == b"predictor"

    reused = CompileFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context=context, config=config), services
    )
    assert reused.validation is not None and reused.validation.passed
    assert reused.commands == ()
    assert len(command_contexts) == 1


def test_python_compile_output_can_be_owned_by_model_cache(tmp_path: Path) -> None:
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "model-cache-output",
            "qwen3-tts",
            ModelFamily.LLM,
            "xh2",
            ModelFlow.COMPILE,
        ),
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
    )
    destination = context.model_cache_dir / "hmm_xh2"
    request = SimpleNamespace(context=context)
    assert (
        _artifact_owned_root(request, destination) == context.model_cache_dir.resolve()
    )
    writer = AtomicArtifactWriter(
        destination,
        root=_artifact_owned_root(request, destination),
        token="model-cache-output",
    )
    with writer as staging:
        (staging / "model.hmm").write_bytes(b"hmm")
        writer.commit()
    assert (destination / "model.hmm").read_bytes() == b"hmm"


def test_model_cache_compile_output_is_mirrored_for_demo(tmp_path: Path) -> None:
    config = ModelConfigRepository(CONFIG_DIR).load("qwen3-tts")
    model_cache = tmp_path / "models"
    source = model_cache / "hmm_xh2_1.7b_customvoice_2k"
    source.mkdir(parents=True)
    (source / "model.hmm").write_bytes(b"hmm")
    (source / "hmquant").mkdir()
    (source / "hmquant" / "quant_embedding_code_predictor.pt").write_bytes(b"embedding")
    context = SimpleNamespace(
        model_cache_dir=model_cache,
        result_cache_dir=tmp_path / "results",
        diagnostic=DiagnosticContext(
            "compile-mirror",
            "qwen3-tts",
            ModelFamily.LLM,
            "xh2",
            ModelFlow.DEMO,
        ),
    )
    mirror_local_compile_outputs(
        SimpleNamespace(config=config, context=context),
        SimpleNamespace(artifact_cache=ArtifactCache()),
    )
    destination = context.result_cache_dir / source.name
    assert (destination / "model.hmm").read_bytes() == b"hmm"
    assert (
        destination / "hmquant" / "quant_embedding_code_predictor.pt"
    ).read_bytes() == b"embedding"
    inspection = ArtifactCache().inspect(
        destination,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            "qwen3-tts",
            "xh2",
            source.name,
        ),
    )
    assert inspection.status == CacheStatus.VALID
    assert inspection.manifest is not None
    assert inspection.manifest.producer_flow == ModelFlow.COMPILE.value


def test_python_compile_missing_input_is_a_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "build.py").write_text("# fake\n", encoding="utf-8")
    config = ModelConfig(
        model_name="missing-compile-input",
        path=tmp_path / "model_cfg_missing-compile-input.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("compile",)},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "compile_params": {
                "xh2": {
                    "model_dir": ["cached_results/missing_quant"],
                    "output_dir": ["cached_results/hmm_xh2"],
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "missing-input",
            config.model_name,
            ModelFamily.CV,
            "xh2",
            ModelFlow.COMPILE,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "compile.log",
        source_dir=source,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class UnexpectedRunner:
        def run(self, command, *, diagnostic_fields=None):
            raise AssertionError(f"unexpected command: {command.argv}")

    result = CompileFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context, config),
        SimpleNamespace(
            command_runner=UnexpectedRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )
    assert result.validation is not None and not result.validation.passed
    assert "compile input artifact is missing" in result.validation.summary
    assert result.commands == ()


def test_xh2_compile_skips_when_all_ncore4_cases_are_filtered(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("IMODELZOO_ALLOW_XH2_NCORE4", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    config = ModelConfig(
        model_name="ncore4-only",
        path=tmp_path / "model_cfg_ncore4-only.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "quant", "compile")},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "get_model_params": {
                "xh2": {"type": ["raw"], "download_dir": ["cached_models"]}
            },
            "hmquant_params": {
                "params": {
                    "required": {"onnx": ["model.onnx"]},
                    "optional": {},
                }
            },
            "hmbuild_params": {
                "xh2": {
                    "required": {"model": ["model.onnx"], "ncore": [4]},
                    "optional": {},
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "ncore4-filter",
            config.model_name,
            ModelFamily.CV,
            "xh2",
            ModelFlow.COMPILE,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "compile.log",
        source_dir=source,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.argv[0] == "python3":
                argv = list(command.argv)
                target = Path(argv[argv.index("--download_dir") + 1])
                target.mkdir(parents=True, exist_ok=True)
                (target / "model.onnx").write_bytes(b"onnx")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    result = CompileFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context, config),
        SimpleNamespace(
            command_runner=FakeRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )
    assert result.disposition.value == "skipped"
    assert "IMODELZOO_ALLOW_XH2_NCORE4=ON" in result.message
    assert [command.argv[0:2] for command in commands] == [
        ("python3", "get_model.py"),
        ("hmatc", "quant"),
    ]
