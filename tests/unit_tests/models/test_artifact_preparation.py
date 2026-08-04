# Copyright (c) 2026 HOUMO AI
#
# File: test_artifact_preparation.py
# Description:
#  Unit tests for raw-model preparation and restoration during separate
#    inference execution.
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

"""Contract tests for separate-infer HMATC raw model preparation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.models_tests.model_workflow.backend_flow_policies import CV_FLOW_POLICY
from tests.models_tests.model_workflow.flow_contracts import FlowDisposition, FlowResult
from tests.models_tests.test_flows.artifact_preparation import ArtifactPreparer

pytestmark = pytest.mark.unit


class _HmatcConfig:
    """Provide the minimal config contract required by HMATC path resolution."""

    model_name = "contract-model"

    @staticmethod
    def has_section(name: str) -> bool:
        return name in {"hmquant_params", "hmbuild_params"}

    @staticmethod
    def hmatc_columns(name: str, *, backend: str | None = None):
        del name, backend
        return {"config": ["./config.yml"]}

    @staticmethod
    def backend_section(name: str, backend: str):
        del backend
        return {} if name == "hmbuild_params" else None


def _request(model_cache_dir: Path):
    """Build a minimal flow request for one HMATC config."""
    diagnostic = SimpleNamespace(backend="xh2")
    context = SimpleNamespace(
        diagnostic=diagnostic,
        model_cache_dir=model_cache_dir,
    )
    return SimpleNamespace(context=context, config=_HmatcConfig())


def _write_config(workspace: Path) -> None:
    """Write a source-style HMATC config with a relative raw model path."""
    (workspace / "config.yml").write_text(
        "model:\n  name: yolov5s\n  save_dir: output\n  model_path: yolov5s.onnx\n",
        encoding="utf-8",
    )


def test_separate_infer_restores_existing_hmatc_raw_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reuse a non-empty raw cache without invoking get_model again."""
    workspace = tmp_path / "workspace"
    model_cache = tmp_path / "cached_models"
    workspace.mkdir()
    model_cache.mkdir()
    _write_config(workspace)
    (model_cache / "yolov5s.onnx").write_bytes(b"onnx")

    def unexpected_download(*args, **kwargs):
        del args, kwargs
        raise AssertionError("raw get_model must not run for a valid cache")

    monkeypatch.setattr(
        ArtifactPreparer,
        "_run_separate_infer_raw_get_model",
        staticmethod(unexpected_download),
    )
    report = ArtifactPreparer()._prepare_separate_infer_hmatc_raw_models(
        _request(model_cache),
        SimpleNamespace(),
        workspace,
        CV_FLOW_POLICY,
    )

    assert not report.failures
    assert (workspace / "yolov5s.onnx").read_bytes() == b"onnx"


def test_separate_infer_downloads_missing_hmatc_raw_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run only the raw producer after the required cache file is missing."""
    workspace = tmp_path / "workspace"
    model_cache = tmp_path / "cached_models"
    workspace.mkdir()
    model_cache.mkdir()
    _write_config(workspace)
    calls = []

    def download_raw(request, services, policy):
        del services, policy
        calls.append(True)
        (request.context.model_cache_dir / "yolov5s.onnx").write_bytes(b"downloaded")
        return FlowResult(FlowDisposition.EXECUTED, "raw downloaded")

    monkeypatch.setattr(
        ArtifactPreparer,
        "_run_separate_infer_raw_get_model",
        staticmethod(download_raw),
    )
    report = ArtifactPreparer()._prepare_separate_infer_hmatc_raw_models(
        _request(model_cache),
        SimpleNamespace(),
        workspace,
        CV_FLOW_POLICY,
    )

    assert calls == [True]
    assert not report.failures
    assert (workspace / "yolov5s.onnx").read_bytes() == b"downloaded"


def test_separate_infer_reports_raw_model_still_missing_after_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Stop before HMATC execution when get_model creates no required ONNX."""
    workspace = tmp_path / "workspace"
    model_cache = tmp_path / "cached_models"
    workspace.mkdir()
    model_cache.mkdir()
    _write_config(workspace)

    monkeypatch.setattr(
        ArtifactPreparer,
        "_run_separate_infer_raw_get_model",
        staticmethod(
            lambda request, services, policy: FlowResult(
                FlowDisposition.EXECUTED,
                "raw command returned without output",
            )
        ),
    )
    report = ArtifactPreparer()._prepare_separate_infer_hmatc_raw_models(
        _request(model_cache),
        SimpleNamespace(),
        workspace,
        CV_FLOW_POLICY,
    )

    assert len(report.failures) == 1
    assert "HMATC raw model is missing for separate infer" in report.failures[0]
    assert "model_path=yolov5s.onnx" in report.failures[0]
    assert str(model_cache / "yolov5s.onnx") in report.failures[0]
