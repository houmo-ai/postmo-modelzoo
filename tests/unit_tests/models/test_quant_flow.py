# Copyright (c) 2026 HOUMO AI
#
# File: test_quant_flow.py
# Description:
#  Unit tests for Python and HMATC quant flow command execution and atomic
#    artifact publication.
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

"""Unit tests extracted from the former model-flow contract suite: test_quant_flow.py."""

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
)
from tests.models_tests.test_flows.quant_flow import (
    QuantFlowHandler,
    run_hmatc_quant_cases,
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


def test_atomic_quant_writer_can_defer_staging_directory_creation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "results" / "quant"
    writer = AtomicArtifactWriter(
        destination,
        root=destination.parent,
        token="deferred",
        create_directory=False,
    )
    with writer as staging:
        assert not staging.exists()
        staging.mkdir()
        (staging / "quant.bin").write_bytes(b"quant")
        writer.commit()

    assert (destination / "quant.bin").read_bytes() == b"quant"


def test_cv_quant_flow_uses_structured_commands_and_publishes_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    (source / "ptq.py").write_text("# fake\n", encoding="utf-8")
    model_cache = tmp_path / "models"
    result_cache = tmp_path / "results"
    config = ModelConfig(
        model_name="demo-cv",
        path=tmp_path / "model_cfg_demo-cv.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "quant")},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "get_model_params": {
                "xh2": {"type": ["raw"], "model_dir": ["cached_models"]}
            },
            "quant_params": {
                "xh2": {
                    "model_path": ["cached_models/raw.onnx"],
                    "output_path": ["cached_results/hmquant_xh2"],
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            run_id="quant-test",
            model_name="demo-cv",
            family=ModelFamily.CV,
            backend="xh2",
            flow=ModelFlow.QUANT,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "quant.log",
        source_dir=source,
        model_cache_dir=model_cache,
        result_cache_dir=result_cache,
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            argv = list(command.argv)
            if argv[1] == "get_model.py":
                target = Path(argv[argv.index("--model_dir") + 1])
                target.mkdir(parents=True, exist_ok=True)
                (target / "raw.onnx").write_bytes(b"onnx")
                dataset = command.cwd / "calibration_data"
                dataset.mkdir()
                (dataset / "sample.bin").write_bytes(b"sample")
            else:
                assert (
                    command.cwd / "calibration_data" / "sample.bin"
                ).read_bytes() == b"sample"
                target = Path(argv[argv.index("--output_path") + 1])
                target.mkdir(parents=True, exist_ok=True)
                (target / "quant.bin").write_bytes(b"quant")
            output = (
                "failed calibration samples: 0\n[error] records: 0\n"
                if argv[1] == "ptq.py"
                else "ok\n"
            )
            return CommandResult(command, 0, output, "", 0.01)

    services = SimpleNamespace(
        command_runner=FakeRunner(),
        workspace_manager=WorkspaceManager(),
        artifact_cache=ArtifactCache(),
    )
    old_quant_dir = result_cache / "hmquant_xh2"
    old_quant_dir.mkdir(parents=True)
    (old_quant_dir / "old.bin").write_bytes(b"old")
    result = QuantFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context=context, config=config), services
    )
    assert result.validation is not None and result.validation.passed
    assert len(result.commands) == 2
    quant_dir = result_cache / "hmquant_xh2"
    assert not (quant_dir / "old.bin").exists()
    assert not tuple(result_cache.glob(".hmquant_xh2.partial-*"))
    inspection = ArtifactCache().inspect(
        quant_dir,
        ArtifactRequirement(
            ArtifactType.QUANT_MODEL,
            "demo-cv",
            "xh2",
            "hmquant_xh2",
        ),
    )
    assert inspection.status == CacheStatus.VALID

    class FailingRunner(FakeRunner):
        def run(self, command, *, diagnostic_fields=None):
            if command.argv[1] == "get_model.py":
                return super().run(command, diagnostic_fields=diagnostic_fields)
            argv = list(command.argv)
            target = Path(argv[argv.index("--output_path") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "partial.bin").write_bytes(b"partial")
            return CommandResult(command, 7, "failed\n", "", 0.01)

    failed = QuantFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context=context, config=config),
        SimpleNamespace(
            command_runner=FailingRunner(),
            workspace_manager=WorkspaceManager(),
            artifact_cache=ArtifactCache(),
        ),
    )
    assert failed.validation is not None and not failed.validation.passed
    assert (quant_dir / "quant.bin").read_bytes() == b"quant"
    assert not tuple(result_cache.glob(".hmquant_xh2.partial-*"))
    assert not tuple(result_cache.glob(".hmquant_xh2.partial-*.owner"))

    (quant_dir / "quant.bin").unlink()
    corrupted = ArtifactCache().inspect(
        quant_dir,
        ArtifactRequirement(
            ArtifactType.QUANT_MODEL,
            "demo-cv",
            "xh2",
            "hmquant_xh2",
        ),
    )
    assert corrupted.status == CacheStatus.CORRUPTED


def test_hmatc_quant_ignores_failure_words_when_return_code_is_zero(
    tmp_path: Path,
) -> None:
    config = ModelConfig(
        model_name="demo-hmatc-quant",
        path=tmp_path / "model_cfg_demo-hmatc-quant.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("quant",)},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "hmquant_params": {
                "params": {
                    "required": {"onnx": ["model.onnx"]},
                    "optional": {"out-dir": ["cached_results/hmquant_xh2"]},
                }
            },
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "hmatc-output-check",
            config.model_name,
            ModelFamily.CV,
            "xh2",
            ModelFlow.QUANT,
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "quant.log",
        source_dir=tmp_path,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )

    class SuccessfulRunner:
        def run(self, command, *, diagnostic_fields=None):
            return CommandResult(
                command,
                0,
                "Fail count: 0\n[error] records retained for diagnostics\n",
                "",
                0.01,
            )

    phase = run_hmatc_quant_cases(
        FlowRequest(context, config),
        SimpleNamespace(command_runner=SuccessfulRunner()),
        tmp_path,
    )

    assert phase.executed_cases == 1
    assert phase.failures == ()
