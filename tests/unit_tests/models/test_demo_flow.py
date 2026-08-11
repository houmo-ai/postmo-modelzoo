# Copyright (c) 2026 HOUMO AI
#
# File: test_demo_flow.py
# Description:
#  Unit tests for demo artifact preparation, test.sh execution, release HMM
#    reuse, and workspace management.
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

"""Unit tests extracted from the former model-flow contract suite: test_demo_flow.py."""

import pytest
from dataclasses import (
    replace,
)
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
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
    ModelConfigRepository,
)
from tests.models_tests.model_workflow.parameter_matrix import (
    ParameterCase,
)
from tests.models_tests.test_flows.artifact_preparation import (
    ArtifactPreparer,
)
from tests.models_tests.test_flows.demo_flow import (
    DemoFlowHandler,
    _run_test_sh,
    _test_sh_cases,
)
from tests.models_tests.test_flows.get_model_flow import (
    GetModelFlowHandler,
)
from tests.models_tests.test_flows.hmatc_flow_support import (
    run_hmatc_cases,
)
from tests.models_tests.test_flows.inference_flow_support import (
    backfill_referenced_demo_artifacts,
    best_matching_download_case,
    release_hmm_case_ids,
    unproduced_demo_artifact_refs,
)
from tests.tests_utils.python_environment import (
    PythonEnvironment,
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
    _demo_artifact_config,
    _demo_artifact_request,
)

pytestmark = pytest.mark.unit


def test_release_hmm_selection_uses_matching_case_ids() -> None:
    config = ModelConfigRepository(CONFIG_DIR).load("gte")
    request = FlowRequest(
        config=config,
        context=SimpleNamespace(
            diagnostic=SimpleNamespace(backend="xh2"), release=True
        ),
    )
    assert release_hmm_case_ids(request) == frozenset({"hmm_xh2_2k"})


def test_hmm_download_selection_rejects_ambiguous_cases() -> None:
    compile_case = ParameterCase(0, {"output_dir": "cached_results/hmm_xh2"})
    source_cases = {
        "hmm_7b": ParameterCase(0, {"extract_dir": "cached_models/hmm_7b"}),
        "hmm_14b": ParameterCase(1, {"extract_dir": "cached_models/hmm_14b"}),
    }
    with pytest.raises(ArtifactValidationError, match="Ambiguous HMM download mapping"):
        best_matching_download_case(compile_case, source_cases)


def test_demo_artifact_references_exclude_compile_outputs(tmp_path: Path) -> None:
    config = _demo_artifact_config(tmp_path)
    request = _demo_artifact_request(config, tmp_path)
    assert unproduced_demo_artifact_refs(request, "xh2") == {
        ("cached_results", "hmm_xh2_aux")
    }


def test_demo_artifact_not_produced_by_compile_is_downloaded(
    tmp_path: Path, monkeypatch
) -> None:
    config = _demo_artifact_config(tmp_path)
    request = _demo_artifact_request(config, tmp_path)
    requested: list[frozenset[str]] = []

    def fake_run(self, download_request, services):
        requested.append(self.case_ids)
        destination = download_request.context.result_cache_dir / "hmm_xh2_aux"
        destination.mkdir(parents=True)
        (destination / "aux.hmm").write_bytes(b"aux")
        return FlowResult(
            FlowDisposition.EXECUTED,
            "downloaded",
            validation=ValidationResult(True, "downloaded"),
        )

    monkeypatch.setattr(GetModelFlowHandler, "run", fake_run)
    prepared = backfill_referenced_demo_artifacts(
        request, SimpleNamespace(artifact_cache=ArtifactCache())
    )

    assert requested == [frozenset({"hmm_xh2_aux"})]
    assert prepared == [request.context.result_cache_dir / "hmm_xh2_aux"]
    assert (prepared[0] / "aux.hmm").read_bytes() == b"aux"


def test_existing_demo_artifact_directory_is_not_downloaded_again(
    tmp_path: Path, monkeypatch
) -> None:
    config = _demo_artifact_config(tmp_path)
    request = _demo_artifact_request(config, tmp_path)
    existing = request.context.result_cache_dir / "hmm_xh2_aux"
    existing.mkdir(parents=True)
    (existing / "aux.hmm").write_bytes(b"aux")

    def fail_run(self, download_request, services):
        raise AssertionError("existing demo artifact must not be downloaded again")

    monkeypatch.setattr(GetModelFlowHandler, "run", fail_run)
    assert backfill_referenced_demo_artifacts(request, SimpleNamespace()) == []


def test_release_hmm_download_requests_json_referenced_cases(
    tmp_path: Path, monkeypatch
) -> None:
    """Release preparation delegates caching to get_model for referenced cases."""
    config = _demo_artifact_config(tmp_path)
    request = _demo_artifact_request(config, tmp_path)
    existing = request.context.result_cache_dir / "hmm_xh2_main"
    existing.mkdir(parents=True)
    (existing / "main.hmm").write_bytes(b"hmm")
    requested: list[frozenset[str] | None] = []

    def fake_run(self, download_request, services):
        requested.append(self.case_ids)
        return FlowResult(
            FlowDisposition.EXECUTED,
            "downloaded",
            validation=ValidationResult(True, "downloaded"),
        )

    monkeypatch.setattr(GetModelFlowHandler, "run", fake_run)
    report = ArtifactPreparer()._download_release_hmms(
        request, SimpleNamespace(artifact_cache=ArtifactCache())
    )

    assert requested == [frozenset({"hmm_xh2_main", "hmm_xh2_aux"})]
    assert report.failures == ()


def test_demo_artifact_download_failure_does_not_fail_the_flow(
    tmp_path: Path, monkeypatch
) -> None:
    config = _demo_artifact_config(tmp_path)
    request = _demo_artifact_request(config, tmp_path)

    def failing_run(self, download_request, services):
        return FlowResult(
            FlowDisposition.EXECUTED,
            "download failed",
            validation=ValidationResult(
                False, "download failed", failures=("download failed",)
            ),
        )

    monkeypatch.setattr(GetModelFlowHandler, "run", failing_run)
    prepared = backfill_referenced_demo_artifacts(
        request, SimpleNamespace(artifact_cache=ArtifactCache())
    )
    assert prepared == [request.context.result_cache_dir / "hmm_xh2_aux"]


def test_test_sh_params_keep_legacy_column_format() -> None:
    diagnostic = DiagnosticContext(
        "test-sh", "demo", ModelFamily.CV, "xh2", ModelFlow.DEMO
    )
    assert _test_sh_cases(
        {
            "xh2": {
                "model_size": ["7b", "14b"],
                "use_cache": [True, False],
                "ndevice": [1, 2],
            }
        },
        "xh2",
        diagnostic,
    ) == [
        ("--model_size", "7b", "--use_cache", "--ndevice", "1"),
        ("--model_size", "14b", "--ndevice", "2"),
    ]


def test_qwen25_vl_test_sh_ignores_legacy_failure_words(tmp_path: Path) -> None:
    config = SimpleNamespace(
        model_name="qwen2.5-vl",
        value=lambda name: None,
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "qwen-vl", "qwen2.5-vl", ModelFamily.LLM, "xh2", ModelFlow.DEMO
        ),
        log_file=tmp_path / "demo.log",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            assert command.environment == {
                "VIRTUAL_ENV": "/workspace/venv",
                "PATH": "/workspace/venv/bin:/usr/bin",
            }
            return CommandResult(command, 0, "Fail is a normal label\n", "", 0.01)

    _, failures = _run_test_sh(
        FlowRequest(config=config, context=context),
        SimpleNamespace(command_runner=FakeRunner()),
        tmp_path,
        environment={
            "VIRTUAL_ENV": "/workspace/venv",
            "PATH": "/workspace/venv/bin:/usr/bin",
        },
    )
    assert failures == []


def test_hmatc_demo_receives_virtualenv_pythonpath(tmp_path: Path) -> None:
    config = SimpleNamespace(
        model_name="hmdemo",
        section=lambda name: {
            "params": {
                "required": {"config": ["config.yml"]},
                "optional": {},
            }
        },
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "hmdemo", "hmdemo", ModelFamily.CV, "xh2", ModelFlow.DEMO
        ),
        log_file=tmp_path / "demo.log",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            assert command.environment == {"PYTHONPATH": "/venv:/system"}
            return CommandResult(command, 0, "ok\n", "", 0.01)

    _, failures = run_hmatc_cases(
        SimpleNamespace(config=config, context=context),
        SimpleNamespace(command_runner=FakeRunner()),
        tmp_path,
        section_name="hmdemo_params",
        subcommand="demo",
        environment=PythonEnvironment(
            "/venv/bin/python3", {"PYTHONPATH": "/venv:/system"}
        ).environment,
    )
    assert failures == []


def test_demo_workspaces_are_isolated_and_cleaned(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (tmp_path / "test_common.sh").write_text(
        "# shared test.sh functions\n", encoding="utf-8"
    )
    (source / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    (source / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    config = ModelConfig(
        model_name="demo",
        path=tmp_path / "model_cfg_demo.json",
        family=ModelFamily.CV,
        model_dir=Path("models/cv/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": ("demo",)},
        raw={
            "model_dir": "models/cv/demo",
            "model_type": "cv",
            "enable_demo_test": False,
            "support_flow": {"xh2": ["demo"]},
            "demo_params": {"xh2": {"script": ["demo.py"]}},
        },
    )
    context = FlowContext(
        diagnostic=DiagnosticContext(
            "demo-test", "demo", ModelFamily.CV, "xh2", ModelFlow.DEMO
        ),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "demo.log",
        source_dir=source,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="1",
        device_mem_marker="12g",
    )
    workspaces = []
    command_names = []
    shell_should_fail = False

    class TrackingWorkspaceManager(WorkspaceManager):
        def open(self, source_dir, *, phase):
            handle = super().open(source_dir, phase=phase)

            class RecordingHandle:
                def __enter__(self):
                    path = handle.__enter__()
                    workspaces.append((phase, path))
                    return path

                def __exit__(self, *args):
                    return handle.__exit__(*args)

            return RecordingHandle()

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            command_names.append(command.name)
            if command.name.startswith("test-sh"):
                assert (command.cwd.parent / "test_common.sh").read_text(
                    encoding="utf-8"
                ) == "# shared test.sh functions\n"
                assert not (command.cwd / "test_common.sh").exists()
                (command.cwd / "test-sh-output").write_text("x", encoding="utf-8")
            if command.name.startswith("demo["):
                assert not (command.cwd / "test-sh-output").exists()
                assert not (command.cwd / "test_common.sh").exists()
            if command.name.startswith("test-sh") and shell_should_fail:
                return CommandResult(command, 7, "failed\n", "", 0.01)
            return CommandResult(command, 0, "ok\n", "", 0.01)

    monkeypatch.setattr(
        "tests.models_tests.test_flows.demo_flow.is_asic_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        "tests.models_tests.test_flows.demo_flow.check_device_info",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "tests.models_tests.test_flows.demo_flow.ensure_inference_artifacts",
        lambda *args: SimpleNamespace(commands=(), failures=()),
    )
    monkeypatch.setattr(
        "tests.models_tests.test_flows.demo_flow.prepare_python_environment",
        lambda *args, **kwargs: PythonEnvironment("python3", {}),
    )
    services = SimpleNamespace(
        command_runner=FakeRunner(),
        workspace_manager=TrackingWorkspaceManager(),
        artifact_cache=ArtifactCache(),
    )
    result = DemoFlowHandler(CV_FLOW_POLICY).run(FlowRequest(context, config), services)
    assert result.validation is not None and result.validation.passed
    assert [phase for phase, _ in workspaces] == [
        "demo_test_sh",
        "demo",
    ]
    assert workspaces[0][1] != workspaces[1][1]
    assert all(not path.exists() for _, path in workspaces)
    assert command_names == ["test-sh[0]", "demo[0]"]

    workspaces.clear()
    command_names.clear()
    shell_should_fail = True
    failed_shell_result = DemoFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(context, config), services
    )
    assert failed_shell_result.validation is not None
    assert not failed_shell_result.validation.passed
    assert command_names == ["test-sh[0]", "demo[0]"]

    workspaces.clear()
    command_names.clear()
    shell_should_fail = False
    release_result = DemoFlowHandler(CV_FLOW_POLICY).run(
        FlowRequest(replace(context, release=True), config), services
    )
    assert release_result.validation is not None
    assert release_result.validation.passed
    assert command_names == ["test-sh[0]"]
    assert [phase for phase, _ in workspaces] == ["demo_test_sh"]
