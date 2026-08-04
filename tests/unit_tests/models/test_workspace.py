# Copyright (c) 2026 HOUMO AI
#
# File: test_workspace.py
# Description:
#  Unit tests for isolated workspace creation, ownership validation, cleanup,
#    and failure recovery.
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

"""Unit tests extracted from the former model-flow contract suite: test_workspace.py."""

import pytest
import re
from pathlib import (
    Path,
)
from tests.tests_utils.workspace import (
    WorkspaceManager,
    WorkspaceOwnershipError,
)

pytestmark = pytest.mark.unit


def test_workspace_is_created_beside_model_and_keeps_relative_layout(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    source = models_dir / "autodrive" / "yolop"
    source.mkdir(parents=True)
    (source / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (models_dir / "test_common.sh").write_text(
        "# shared test.sh functions\n", encoding="utf-8"
    )

    with WorkspaceManager().open(source, phase="demo") as workspace:
        assert workspace.parent == source.parent
        assert re.fullmatch(
            r"yolop_demo_\d{8}_\d{6}_\d{6}(?:_\d{2})?",
            workspace.name,
        )
        assert (workspace / "test.sh").is_file()
        assert not (workspace / "test_common.sh").exists()
        assert (workspace.parent.parent / "test_common.sh").is_file()

    assert not workspace.exists()


def test_failed_workspace_copy_does_not_leave_sibling_directory(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "models" / "autodrive" / "yolop"
    source.mkdir(parents=True)

    def fail_copy(*args, **kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(
        "tests.tests_utils.workspace.shutil.copytree",
        fail_copy,
    )

    with pytest.raises(OSError, match="copy failed"):
        with WorkspaceManager().open(source, phase="demo"):
            pass

    assert list(source.parent.glob("yolop_demo_*")) == []


def test_workspace_refuses_cleanup_without_ownership_marker(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    handle = WorkspaceManager().open(source, phase="demo")
    workspace = handle.__enter__()
    sentinel = workspace / ".imodelzoo-workspace"
    sentinel.unlink()
    with pytest.raises(WorkspaceOwnershipError, match="unowned workspace"):
        handle.__exit__(None, None, None)
    assert workspace.is_dir()
    sentinel.write_text("owned\n", encoding="utf-8")
    handle.__exit__(None, None, None)
    assert not workspace.exists()
