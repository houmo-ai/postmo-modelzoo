# Copyright (c) 2026 HOUMO AI
#
# File: test_hmatc_flow_support.py
# Description:
#  Unit tests for HMATC configuration discovery, inference bundle reuse, and
#    separate-stage persistence and restoration.
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

"""Unit tests extracted from the former model-flow contract suite: test_hmatc_flow_support.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    CacheStatus,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandResult,
    DiagnosticContext,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfig,
)
from tests.models_tests.test_flows.artifact_preparation import (
    ArtifactNeed,
    ArtifactPreparer,
)
from tests.models_tests.test_flows.get_model_flow import (
    GetModelFlowHandler,
)
from tests.models_tests.test_flows.hmatc_flow_support import (
    _hmatc_inference_fingerprint,
    persist_separate_workspace,
    run_hmatc_inference_preparation,
)
from tests.models_tests.test_flows.inference_flow_support import (
    prepare_inference_workspace,
)
from tests.tests_utils.runtime_context import (
    TCaseType,
)
from types import (
    SimpleNamespace,
)
from ._flow_contract_support import (
    _hmatc_inference_test_request,
    _write_hmatc_inference_artifacts,
    _write_hmatc_inference_config,
)

pytestmark = pytest.mark.unit


def test_cv_separate_workspace_artifacts_are_persisted_and_restored(
    tmp_path: Path,
) -> None:
    source_workspace = tmp_path / "source-workspace"
    source_workspace.mkdir()
    _write_hmatc_inference_config(source_workspace)
    output = source_workspace / "output" / "xh2"
    _write_hmatc_inference_artifacts(output)
    (source_workspace / "derived_opset13.onnx").write_bytes(b"derived")
    (source_workspace / "get_model.py").write_text("pass\n", encoding="utf-8")
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    (model_cache / "raw.onnx").write_bytes(b"onnx")
    dataset = model_cache / "calibration_data"
    dataset.mkdir()
    (dataset / "sample.bin").write_bytes(b"sample")
    (model_cache / "classifier.pt").write_bytes(b"classifier")
    result_cache = tmp_path / "results"
    config = ModelConfig(
        model_name="separate-cv",
        path=tmp_path / "model_cfg_separate-cv.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/separate"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("demo",)},
        raw={"hmbuild_params": {}},
    )
    no_infer_context = SimpleNamespace(
        result_cache_dir=result_cache,
        test_type=TCaseType.SEPARATE_NO_INFER,
        diagnostic=SimpleNamespace(backend="xh2"),
        release=False,
    )
    no_infer_request = FlowRequest(no_infer_context, config)
    persist_separate_workspace(no_infer_request, source_workspace, CV_FLOW_POLICY)
    assert (result_cache / "config.yml").is_file()
    assert (result_cache / "output" / "xh2" / "resnet50.hmm").read_bytes() == b"hmm"
    assert not (result_cache / "derived_opset13.onnx").exists()
    assert not (result_cache / "get_model.py").exists()

    restored_workspace = tmp_path / "restored-workspace"
    restored_workspace.mkdir()
    _write_hmatc_inference_config(restored_workspace)
    infer_context = SimpleNamespace(
        model_cache_dir=model_cache,
        result_cache_dir=result_cache,
        test_type=TCaseType.SEPARATE_INFER,
        diagnostic=SimpleNamespace(backend="xh2"),
        release=False,
    )
    commands, failures = prepare_inference_workspace(
        FlowRequest(infer_context, config),
        SimpleNamespace(),
        restored_workspace,
        CV_FLOW_POLICY,
    )
    assert commands == []
    assert failures == []
    assert (
        restored_workspace / "output" / "xh2" / "resnet50.hmm"
    ).read_bytes() == b"hmm"
    assert (restored_workspace / "raw.onnx").read_bytes() == b"onnx"
    assert (
        restored_workspace / "calibration_data" / "sample.bin"
    ).read_bytes() == b"sample"
    assert (restored_workspace / "classifier.pt").read_bytes() == b"classifier"


def test_hmatc_inference_preparation_uses_only_default_config_case(
    tmp_path: Path,
) -> None:
    request = _hmatc_inference_test_request(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hmatc_inference_config(workspace)
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.name == "hmatc-inference-build":
                _write_hmatc_inference_artifacts(workspace / "output" / "xh2")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    results, failures = run_hmatc_inference_preparation(
        request,
        SimpleNamespace(command_runner=FakeRunner(), artifact_cache=ArtifactCache()),
        workspace,
    )
    assert not failures
    assert len(results) == 2
    assert [command.argv for command in commands] == [
        (
            "hmatc",
            "quant",
            "--target",
            "xh2",
            "--config",
            "./config.yml",
        ),
        (
            "hmatc",
            "build",
            "--skip_check",
            "--target",
            "xh2",
            "--config",
            "./config.yml",
        ),
    ]


def test_hmatc_inference_preparation_supports_multiple_component_configs(
    tmp_path: Path,
) -> None:
    """SAM2-style encoder/decoder configs share one reusable bundle."""
    config = ModelConfig(
        model_name="sam2",
        path=tmp_path / "model_cfg_sam2.json",
        family=ModelFamily.CV,
        model_dir=Path("models/segmentation/sam2"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("get_model", "quant", "compile", "demo")},
        raw={
            "hmquant_params": {
                "params": {
                    "required": {"config": ["./encoder.yml", "./decoder.yml"]},
                    "optional": {"target": [None, None]},
                }
            },
            "hmbuild_params": {
                "xh2": {
                    "required": {"config": ["./encoder.yml", "./decoder.yml"]},
                    "optional": {"ncore": ["1", "1"]},
                }
            },
        },
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "prepare", "sam2", ModelFamily.CV, "xh2", ModelFlow.DEMO
        ),
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        log_file=tmp_path / "prepare.log",
    )
    request = FlowRequest(context, config)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, model_name in (
        ("encoder.yml", "sam2.1s_encoder"),
        ("decoder.yml", "sam2.1s_decoder"),
    ):
        (workspace / name).write_text(
            f"model:\n  name: {model_name}\n  save_dir: output\n",
            encoding="utf-8",
        )
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.name.startswith("hmatc-inference-build"):
                output = workspace / "output" / "xh2"
                output.mkdir(parents=True, exist_ok=True)
                suffix = (
                    "encoder"
                    if any("encoder.yml" in value for value in command.argv)
                    else "decoder"
                )
                (output / f"sam2.1s_{suffix}.hmm").write_bytes(b"hmm")
                hmquant = output / "hmquant"
                hmquant.mkdir(exist_ok=True)
                (hmquant / f"sam2.1s_{suffix}_with_act.onnx").write_bytes(b"onnx")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    results, failures = run_hmatc_inference_preparation(
        request,
        SimpleNamespace(command_runner=FakeRunner(), artifact_cache=ArtifactCache()),
        workspace,
    )
    assert failures == []
    assert len(results) == 4
    assert [command.argv[-1] for command in commands] == [
        "./encoder.yml",
        "./decoder.yml",
        "./encoder.yml",
        "./decoder.yml",
    ]
    cached = request.context.result_cache_dir / "output" / "xh2"
    assert (cached / "sam2.1s_encoder.hmm").is_file()
    assert (cached / "sam2.1s_decoder.hmm").is_file()


def test_hmatc_inference_preparation_reuses_valid_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    request = _hmatc_inference_test_request(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hmatc_inference_config(workspace)
    cached = request.context.result_cache_dir / "output" / "xh2"
    _write_hmatc_inference_artifacts(cached)
    fingerprint = _hmatc_inference_fingerprint(request, workspace / "config.yml")
    ArtifactCache().write_manifest(
        cached,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend="xh2",
            case_id="inference-default",
            producer_flow=ModelFlow.DEMO.value,
            source_type="local_hmatc_inference",
            config_fingerprint=fingerprint,
            required_files={
                "hmm_0": "resnet50.hmm",
                "quant_onnx_0": "hmquant/hmquant_resnet50_with_act.onnx",
            },
        ),
    )

    class FailingRunner:
        def run(self, command, *, diagnostic_fields=None):
            raise AssertionError("valid artifact must skip HMATC commands")

    results, failures = run_hmatc_inference_preparation(
        request,
        SimpleNamespace(command_runner=FailingRunner(), artifact_cache=ArtifactCache()),
        workspace,
    )

    assert results == []
    assert failures == []
    assert (workspace / "output" / "xh2" / "resnet50.hmm").read_bytes() == b"hmm"

    # HMATC still requires its raw model input even when the compiled artifact
    # is reusable; only the quant/build preparation commands should be skipped.
    source = tmp_path / "source"
    source.mkdir()
    cached_workspace = tmp_path / "preparer-workspace"
    cached_workspace.mkdir()
    _write_hmatc_inference_config(cached_workspace)
    preparer_context = SimpleNamespace(
        **vars(request.context),
        test_type=TCaseType.DEFAULT,
        platform="x86_64",
        release=False,
        source_dir=source,
    )

    get_model_calls = []

    def prepare_raw_model(*args):
        get_model_calls.append(args)
        return FlowResult(FlowDisposition.EXECUTED, "raw model prepared")

    monkeypatch.setattr(GetModelFlowHandler, "run", prepare_raw_model)
    report = ArtifactPreparer().ensure(
        FlowRequest(preparer_context, request.config),
        SimpleNamespace(artifact_cache=ArtifactCache()),
        (ArtifactNeed.inference_compiled_model(),),
        workspace=cached_workspace,
        policy=CV_FLOW_POLICY,
    )
    assert len(get_model_calls) == 1
    assert not report.failures
    assert report.message == "raw model prepared and compiled artifact already exists"
    assert (cached_workspace / "output" / "xh2" / "resnet50.hmm").read_bytes() == b"hmm"


def test_hmatc_inference_preparation_adopts_legacy_artifacts(
    tmp_path: Path,
) -> None:
    request = _hmatc_inference_test_request(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hmatc_inference_config(workspace)
    cached = request.context.result_cache_dir / "output" / "xh2"
    _write_hmatc_inference_artifacts(cached)
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    (request.context.result_cache_dir / "config.yml").write_bytes(
        (workspace / "config.yml").read_bytes()
    )

    class FailingRunner:
        def run(self, command, *, diagnostic_fields=None):
            raise AssertionError("legacy artifact must be adopted without rebuilding")

    results, failures = run_hmatc_inference_preparation(
        request,
        SimpleNamespace(command_runner=FailingRunner(), artifact_cache=ArtifactCache()),
        workspace,
    )

    assert results == []
    assert failures == []
    manifest = cached / ArtifactCache.typed_manifest_filename(
        ArtifactType.COMPILED_MODEL.value, "inference-default"
    )
    assert manifest.is_file()
    assert (workspace / "output" / "xh2" / "resnet50.hmm").is_file()


def test_hmatc_inference_preparation_rebuilds_stale_config(
    tmp_path: Path,
) -> None:
    request = _hmatc_inference_test_request(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hmatc_inference_config(workspace, opt_level=1)
    cached = request.context.result_cache_dir / "output" / "xh2"
    _write_hmatc_inference_artifacts(cached)
    old_fingerprint = _hmatc_inference_fingerprint(request, workspace / "config.yml")
    ArtifactCache().write_manifest(
        cached,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend="xh2",
            case_id="inference-default",
            producer_flow=ModelFlow.DEMO.value,
            source_type="local_hmatc_inference",
            config_fingerprint=old_fingerprint,
            required_files={
                "hmm_0": "resnet50.hmm",
                "quant_onnx_0": "hmquant/hmquant_resnet50_with_act.onnx",
            },
        ),
    )
    _write_hmatc_inference_config(workspace, opt_level=2)
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.name == "hmatc-inference-build":
                artifact = workspace / "output" / "xh2"
                _write_hmatc_inference_artifacts(artifact)
                (artifact / "resnet50.hmm").write_bytes(b"rebuilt")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    results, failures = run_hmatc_inference_preparation(
        request,
        SimpleNamespace(command_runner=FakeRunner(), artifact_cache=ArtifactCache()),
        workspace,
    )

    assert len(results) == 2
    assert failures == []
    assert len(commands) == 2
    assert (cached / "resnet50.hmm").read_bytes() == b"rebuilt"
    inspection = ArtifactCache().inspect(
        cached,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            request.config.model_name,
            "xh2",
            "inference-default",
        ),
        expected_fingerprint=_hmatc_inference_fingerprint(
            request, workspace / "config.yml"
        ),
    )
    assert inspection.status == CacheStatus.VALID
