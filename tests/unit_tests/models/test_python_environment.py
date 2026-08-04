# Copyright (c) 2026 HOUMO AI
#
# File: test_python_environment.py
# Description:
#  Unit tests for model virtual-environment activation, requirements
#    discovery, installation order, and inherited process state.
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

"""Unit tests extracted from the former model-flow contract suite: test_python_environment.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandResult,
)
from tests.models_tests.model_workflow.python_environment import (
    prepare_python_environment,
)
from tests.tests_utils import (
    python_environment as cross_suite_python_environment,
)
from tests.tests_utils.python_environment import (
    PythonEnvironment,
    build_venv_environment,
)

pytestmark = pytest.mark.unit


def test_venv_environment_keeps_existing_and_system_packages(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(
        "tests.tests_utils.python_environment._get_site_packages",
        lambda executable: "/venv" if executable == "/venv/bin/python3" else "/system",
    )
    assert build_venv_environment("/venv/bin/python3", activated=False) == {
        "PYTHONPATH": "/venv:/existing:/system"
    }


def test_quant_virtualenv_installs_ptq_then_runtime_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "requirements_ptq.txt").write_text("ptq-package\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("runtime-package\n", encoding="utf-8")
    commands = []

    class FakeRunner:
        def run(self, command, **_kwargs):
            commands.append(command)
            return CommandResult(command, 0, "", "", 0.01)

    monkeypatch.setattr(
        cross_suite_python_environment, "_get_python_executable", lambda: "python3"
    )
    monkeypatch.setattr(
        cross_suite_python_environment, "_get_site_packages", lambda _: "/system"
    )
    monkeypatch.setattr(
        cross_suite_python_environment,
        "_create_virtualenv",
        lambda **_kwargs: "/venv-site",
    )

    result = cross_suite_python_environment.prepare_python_environment(
        tmp_path,
        (tmp_path / "requirements_ptq.txt", tmp_path / "requirements.txt"),
        base_environment={"PATH": "/usr/bin", "PYTHONPATH": "/original-site"},
        runner=FakeRunner(),
    )
    assert result.executable.endswith("/imodelzoo_test/bin/python3")
    assert [command.name for command in commands] == [
        "pip-install-requirements_ptq.txt",
        "pip-install-requirements.txt",
    ]
    for command in commands:
        assert command.environment == {
            "PATH": f"{tmp_path / 'imodelzoo_test' / 'bin'}:/usr/bin",
            "PYTHONPATH": "/venv-site:/original-site:/system",
            "VIRTUAL_ENV": str(tmp_path / "imodelzoo_test"),
        }
    assert result.environment == commands[0].environment


@pytest.mark.parametrize(
    ("requirement_name", "requirement_option"),
    (("requirements.txt", "-r"), ("package.whl", None)),
)
def test_pip_install_uses_network_timeout_and_retries(
    tmp_path: Path,
    requirement_name: str,
    requirement_option: str | None,
) -> None:
    requirement_path = tmp_path / requirement_name
    command = list(
        cross_suite_python_environment._pip_install_command(
            tmp_path / "venv", requirement_path
        )
    )

    assert command[0] == str(tmp_path / "venv" / "bin" / "pip3")
    assert command[1] == "install"
    assert command[command.index("--timeout") + 1] == "60"
    assert command[command.index("--retries") + 1] == "15"
    if requirement_option is None:
        assert "-r" not in command
        assert str(requirement_path) in command
    else:
        assert command[command.index(requirement_option) + 1] == str(requirement_path)


def test_activated_python_environment_exposes_venv_commands(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "requirements.txt").write_text("runtime\n", encoding="utf-8")
    monkeypatch.setattr(
        "tests.models_tests.model_workflow.python_environment."
        "prepare_cross_suite_python_environment",
        lambda workspace, requirements, **kwargs: PythonEnvironment(
            str(
                workspace / cross_suite_python_environment.VENV_NAME / "bin" / "python3"
            ),
            {
                "VIRTUAL_ENV": str(
                    workspace / cross_suite_python_environment.VENV_NAME
                ),
                "PATH": (
                    f"{workspace / cross_suite_python_environment.VENV_NAME / 'bin'}"
                    ":/usr/bin"
                ),
                "PYTHONPATH": "/venv:/system",
            },
        ),
    )
    python = prepare_python_environment(tmp_path, tmp_path / "flow.log", activated=True)
    venv_dir = str(tmp_path / cross_suite_python_environment.VENV_NAME)
    assert python.executable == str(
        tmp_path / cross_suite_python_environment.VENV_NAME / "bin" / "python3"
    )
    assert python.environment["VIRTUAL_ENV"] == venv_dir
    assert python.environment["PATH"].split(":", 1)[0] == f"{venv_dir}/bin"
    assert python.environment["PYTHONPATH"] == "/venv:/system"


def test_shared_python_environment_forwards_quant_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HOUMO_EXAMPLES_PATH", raising=False)
    (tmp_path / "requirements_ptq.txt").write_text("ptq\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("runtime\n", encoding="utf-8")
    calls = []

    def prepare(workspace, requirements, **kwargs):
        calls.append((workspace, tuple(path.name for path in requirements), kwargs))
        return PythonEnvironment("python3", {})

    monkeypatch.setattr(
        "tests.models_tests.model_workflow.python_environment."
        "prepare_cross_suite_python_environment",
        prepare,
    )
    python = prepare_python_environment(
        tmp_path,
        tmp_path / "quant.log",
        flow_type="quant",
        other_requirements={"hm_gptq": True},
    )
    assert python == PythonEnvironment("python3", {})
    assert calls[0][0] == tmp_path
    assert calls[0][1] == ("requirements_ptq.txt", "requirements.txt")
    assert calls[0][2]["activated"] is False


def test_model_py_reqs_prefer_workspace_over_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    datasets = tmp_path / "datasets"
    workspace.mkdir()
    datasets.mkdir()
    requirement_name = "model-dependency.whl"
    workspace_requirement = workspace / requirement_name
    workspace_requirement.write_bytes(b"workspace")
    (datasets / requirement_name).write_bytes(b"datasets")
    monkeypatch.setenv("HOUMO_DATASETS_PATH", str(datasets))
    calls = []

    def prepare(received_workspace, requirements, **kwargs):
        calls.append((received_workspace, tuple(requirements), kwargs))
        return PythonEnvironment("python3", {})

    monkeypatch.setattr(
        "tests.models_tests.model_workflow.python_environment."
        "prepare_cross_suite_python_environment",
        prepare,
    )

    python = prepare_python_environment(
        workspace,
        workspace / "quant.log",
        flow_type="quant",
        other_requirements={"py_reqs": [requirement_name]},
    )

    assert python == PythonEnvironment("python3", {})
    assert calls[0][0] == workspace
    assert calls[0][1] == (workspace_requirement,)


def test_model_py_reqs_fall_back_to_datasets_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    datasets = tmp_path / "datasets"
    workspace.mkdir()
    datasets.mkdir()
    requirement_name = "model-dependency.whl"
    dataset_requirement = datasets / requirement_name
    dataset_requirement.write_bytes(b"datasets")
    monkeypatch.setenv("HOUMO_DATASETS_PATH", str(datasets))
    calls = []

    def prepare(received_workspace, requirements, **kwargs):
        calls.append((received_workspace, tuple(requirements), kwargs))
        return PythonEnvironment("python3", {})

    monkeypatch.setattr(
        "tests.models_tests.model_workflow.python_environment."
        "prepare_cross_suite_python_environment",
        prepare,
    )

    prepare_python_environment(
        workspace,
        workspace / "quant.log",
        flow_type="quant",
        other_requirements={"py_reqs": [requirement_name]},
    )

    assert calls[0][1] == (dataset_requirement,)
