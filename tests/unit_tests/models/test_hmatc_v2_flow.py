# Copyright (c) 2026 HOUMO AI
#
# File: test_hmatc_v2_flow.py
# Description:
#  Unit Tests for HMATC v2 Schema, YAML Materialization, Quant/Build Artifact
#  Reuse, Environment Injection, and Runtime Sidecar Publication.
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

"""Verify the independent HMATC v2 configuration and execution protocol."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
    ArtifactRequirement,
    ArtifactType,
    CacheStatus,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandResult,
    DiagnosticContext,
    FlowContext,
    FlowRequest,
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.backend_flow_policies import LLM_FLOW_POLICY
from tests.models_tests.model_workflow.hmatc_v2_config import (
    deep_merge,
    hmatc_v2_case_id,
    materialize_hmatc_v2_config,
    resolve_nested_cache_paths,
)
from tests.models_tests.model_workflow.model_config_repository import ModelConfig
from tests.models_tests.test_flows.hmatc_v2_flow_support import (
    run_hmatc_v2_build_cases,
    run_hmatc_v2_quant_cases,
)
from tests.models_tests.test_flows.artifact_preparation import (
    ArtifactNeed,
    ArtifactPreparer,
)
from tests.models_tests.test_flows.get_model_flow import GetModelFlowHandler
from tests.tests_utils.runtime_context import TCaseType
from tests.tests_utils.workspace import WorkspaceManager

pytestmark = pytest.mark.unit


def _v2_request(tmp_path: Path, *, include_get_model: bool = False) -> FlowRequest:
    """Build a small HMATC v2 request with one quant/build pair."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "configs").mkdir()
    (source / "configs" / "demo.yml").write_text(
        "save_dir: original\n" "model:\n" "  model_dir: original-model\n" "build:\n" "  ndevice: 2\n",
        encoding="utf-8",
    )
    (source / "get_model.py").write_text("# fake\n", encoding="utf-8")
    raw = {
        "model_dir": "models/llm/demo",
        "model_type": "llm",
        "hmatc_flow_version": 2,
        "hmquant_params": [
            {
                "config": "./configs/demo.yml",
                "override": {
                    "save_dir": "cached_results/hmquant_demo",
                    "model": {"model_dir": "cached_models/demo-model"},
                },
            }
        ],
        "hmbuild_params": [
            {
                "config": "./configs/demo.yml",
                "ENV_HMATC_BUILD_OUTPUT_DIR": "cached_results/hmm_demo",
                "override": {
                    "save_dir": "cached_results/hmquant_demo",
                    "build": {"ndevice": 1},
                },
            }
        ],
    }
    flows = ["quant", "compile", "demo"]
    if include_get_model:
        flows.insert(0, "get_model")
        raw["get_model_params"] = {
            "xh2": {
                "config": ["./configs/demo.yml", "./configs/unused.yml"],
                "type": ["raw", "raw"],
                "download_dir": ["cached_models", "cached_models"],
            }
        }
    config = ModelConfig(
        model_name="demo-v2",
        path=tmp_path / "model_cfg_demo-v2.json",
        family=ModelFamily.LLM,
        model_dir=Path("models/llm/demo"),
        obsolete=False,
        dependencies={},
        support_platform=("x86_64",),
        support_backend=("xh2",),
        support_flow={"xh2": tuple(flows)},
        raw=raw,
    )
    context = FlowContext(
        diagnostic=DiagnosticContext("hmatc-v2", config.model_name, config.family, "xh2", ModelFlow.QUANT),
        platform="x86_64",
        test_type=TCaseType.DEFAULT,
        release=False,
        log_file=tmp_path / "hmatc-v2.log",
        source_dir=source,
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        ndevice_marker="2",
        device_mem_marker="48g",
    )
    return FlowRequest(context, config)


def test_hmatc_v2_materializer_deep_merges_without_mutating_source(
    tmp_path: Path,
) -> None:
    request = _v2_request(tmp_path)
    case = request.config.hmatc_v2_cases("hmbuild_params")[0]
    source = request.context.source_dir / "configs" / "demo.yml"
    original = source.read_text(encoding="utf-8")
    original_mtime = source.stat().st_mtime_ns
    staging = tmp_path / "staging"
    staging.mkdir()
    execution_override = resolve_nested_cache_paths(
        case.override,
        model_cache_dir=request.context.model_cache_dir,
        result_cache_dir=request.context.result_cache_dir,
    )
    target = materialize_hmatc_v2_config(
        case,
        request.context.source_dir,
        staging,
        flow=ModelFlow.COMPILE,
        fingerprint="sha256:1234567890",
        execution_override=execution_override,
    )

    effective = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert target.parent == staging / ".imodelzoo_configs"
    assert effective["save_dir"] == str(request.context.result_cache_dir / "hmquant_demo")
    assert effective["model"]["model_dir"] == "original-model"
    assert effective["build"] == {"ndevice": 1}
    assert source.read_text(encoding="utf-8") == original
    assert source.stat().st_mtime_ns == original_mtime
    assert deep_merge({"a": {"b": 1}, "items": [1]}, {"a": {"c": 2}, "items": [3]}) == {
        "a": {"b": 1, "c": 2},
        "items": [3],
    }
    assert hmatc_v2_case_id(case).startswith("demo-")


def test_hmatc_v2_quant_and_build_publish_and_reuse_artifacts(tmp_path: Path) -> None:
    request = _v2_request(tmp_path, include_get_model=True)
    model_dir = request.context.model_cache_dir / "demo-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.argv[1] == "get_model.py":
                return CommandResult(command, 0, "cached\n", "", 0.01)
            effective = yaml.safe_load(
                Path(command.argv[command.argv.index("--config") + 1]).read_text(encoding="utf-8")
            )
            if command.argv[1] == "quant":
                staging = Path(effective["save_dir"])
                hmquant = staging / "xh2" / "hmquant"
                (hmquant / "nested").mkdir(parents=True)
                (hmquant / "embedding.pt").write_bytes(b"pt")
                (hmquant / "nested" / "predictor.pt").write_bytes(b"nested")
                (hmquant / "intermediate.onnx").write_bytes(b"onnx")
                (hmquant / "hf_config").mkdir()
                (hmquant / "hf_config" / "config.json").write_text("{}\n", encoding="utf-8")
            else:
                assert effective["save_dir"] == str(request.context.result_cache_dir / "hmquant_demo")
                staging = Path(command.environment["HMATC_BUILD_OUTPUT_DIR"])
                assert "ENV_HMATC_BUILD_OUTPUT_DIR" not in command.environment
                (staging / "xh2").mkdir(parents=True, exist_ok=True)
                (staging / "xh2" / "model.hmm").write_bytes(b"hmm")
            return CommandResult(command, 0, "ok\n", "", 0.01)

    services = SimpleNamespace(
        command_runner=FakeRunner(),
        artifact_cache=ArtifactCache(),
        workspace_manager=WorkspaceManager(),
    )
    quant = run_hmatc_v2_quant_cases(request, services, request.context.source_dir)
    assert quant.failures == ()
    assert quant.executed_cases == 1
    assert len(commands) == 2

    quant_reused = run_hmatc_v2_quant_cases(request, services, request.context.source_dir)
    assert quant_reused.reused_cases == 1
    assert len(commands) == 2
    quant_dir = request.context.result_cache_dir / "hmquant_demo"
    quant_case = request.config.hmatc_v2_cases("hmquant_params")[0]
    assert (
        ArtifactCache()
        .inspect(
            quant_dir,
            ArtifactRequirement(
                ArtifactType.QUANT_MODEL,
                request.config.model_name,
                "xh2",
                hmatc_v2_case_id(quant_case),
                required_roles=("effective_config",),
            ),
        )
        .status
        == CacheStatus.VALID
    )

    build_request = FlowRequest(
        SimpleNamespace(
            **{
                **vars(request.context),
                "diagnostic": DiagnosticContext(
                    "hmatc-v2-build",
                    request.config.model_name,
                    request.config.family,
                    "xh2",
                    ModelFlow.COMPILE,
                ),
            }
        ),
        request.config,
    )
    build = run_hmatc_v2_build_cases(build_request, services, request.context.source_dir)
    assert build.failures == ()
    assert build.executed_cases == 1
    assert len(commands) == 3
    build_staging = Path(commands[-1].environment["HMATC_BUILD_OUTPUT_DIR"])
    assert build_staging.parent == request.context.result_cache_dir
    assert build_staging.name.startswith(".hmm_demo.partial-")
    build_dir = request.context.result_cache_dir / "hmm_demo"
    assert (build_dir / "xh2" / "hmquant" / "embedding.pt").read_bytes() == b"pt"
    assert (build_dir / "xh2" / "hmquant" / "nested" / "predictor.pt").read_bytes() == b"nested"
    assert (build_dir / "xh2" / "hmquant" / "hf_config" / "config.json").is_file()
    assert not (build_dir / "xh2" / "hmquant" / "intermediate.onnx").exists()

    build_reused = run_hmatc_v2_build_cases(build_request, services, request.context.source_dir)
    assert build_reused.reused_cases == 1
    assert len(commands) == 3


def test_hmatc_v2_quant_downloads_matching_raw_when_cache_is_nonempty(
    tmp_path: Path,
) -> None:
    request = _v2_request(tmp_path, include_get_model=True)
    model_dir = request.context.model_cache_dir / "demo-model"
    model_dir.mkdir(parents=True)
    (model_dir / "stale.bin").write_bytes(b"stale")
    commands = []

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            commands.append(command)
            if command.argv[1] == "get_model.py":
                assert command.argv[command.argv.index("--config") + 1] == "./configs/demo.yml"
                (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            else:
                effective = yaml.safe_load(
                    Path(command.argv[command.argv.index("--config") + 1]).read_text(encoding="utf-8")
                )
                hmquant = Path(effective["save_dir"]) / "xh2" / "hmquant"
                hmquant.mkdir(parents=True)
            return CommandResult(command, 0, "ok\n", "", 0.01)

    phase = run_hmatc_v2_quant_cases(
        request,
        SimpleNamespace(
            command_runner=FakeRunner(),
            artifact_cache=ArtifactCache(),
            workspace_manager=WorkspaceManager(),
        ),
        request.context.source_dir,
    )
    assert phase.failures == ()
    assert [command.argv[1] for command in commands] == ["get_model.py", "quant"]


def test_hmatc_v2_separate_infer_reuses_nonempty_compiled_cache(tmp_path: Path, monkeypatch) -> None:
    request = _v2_request(tmp_path)
    raw = {
        **request.config.raw,
        "get_model_params": {
            "xh2": {
                "type": ["hmm"],
                "extract_dir": ["cached_models/hmm_demo"],
            }
        },
        "demo_params": {"xh2": {"model": ["cached_results/hmm_demo/xh2"]}},
    }
    config = replace(request.config, raw=raw)
    compiled = request.context.result_cache_dir / "hmm_demo"
    compiled.mkdir(parents=True)
    (compiled / "artifact.bin").write_bytes(b"compiled")
    workspace = tmp_path / "infer-workspace"
    workspace.mkdir()
    context = replace(
        request.context,
        test_type=TCaseType.SEPARATE_INFER,
        diagnostic=DiagnosticContext(
            "hmatc-v2-infer",
            config.model_name,
            config.family,
            "xh2",
            ModelFlow.DEMO,
        ),
    )

    def unexpected_download(*args, **kwargs):
        raise AssertionError("non-empty HMATC v2 compiled cache must be reused")

    monkeypatch.setattr(GetModelFlowHandler, "run", unexpected_download)
    report = ArtifactPreparer().ensure(
        FlowRequest(context, config),
        SimpleNamespace(artifact_cache=ArtifactCache()),
        (ArtifactNeed.inference_compiled_model(),),
        workspace=workspace,
        policy=LLM_FLOW_POLICY,
    )
    assert report.failures == ()
    assert report.commands == ()
