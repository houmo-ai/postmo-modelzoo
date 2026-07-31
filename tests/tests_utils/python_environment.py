# Copyright (c) 2025 HOUMO AI
#
# File: python_environment.py
# Description:
#  Cross-Suite Python Virtual Environment Preparation.
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

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .command_execution import CommandRunner, CommandSpec

VENV_NAME = "imodelzoo_test"
PIP_NETWORK_TIMEOUT_SECONDS = 60
PIP_NETWORK_RETRIES = 15
PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUSTED_HOST = "pypi.tuna.tsinghua.edu.cn"


def _imodelzoo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_python_executable() -> str:
    executable = shutil.which("python3")
    if executable is None:
        raise RuntimeError("Python3 executable not found")
    return executable


def _get_site_packages(python_executable: str) -> str:
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "stdout", "") or getattr(error, "stderr", "")
        raise RuntimeError(f"Failed to get site-packages: {output}") from error
    return result.stdout.strip()


def _append_activate_environment(
    activate_path: Path,
    *,
    venv_site: str,
    system_site: str,
) -> None:
    try:
        with activate_path.open("a", encoding="utf-8") as stream:
            stream.write("export ORIGINAL_PYTHONPATH=$PYTHONPATH\n")
            stream.write(f"export PYTHONPATH={venv_site}:$ORIGINAL_PYTHONPATH:{system_site}\n")
    except OSError as error:
        raise RuntimeError(f"Failed to modify activate script: {error}") from error


def _append_deactivate_environment(deactivate_path: Path) -> None:
    try:
        with deactivate_path.open("a", encoding="utf-8") as stream:
            stream.write("export PYTHONPATH=$ORIGINAL_PYTHONPATH\n")
            stream.write("unset ORIGINAL_PYTHONPATH\n")
    except OSError as error:
        raise RuntimeError(f"Failed to modify deactivate script: {error}") from error


def _disable_system_site_packages(config_path: Path) -> None:
    try:
        content = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            content.replace(
                "include-system-site-packages = true",
                "include-system-site-packages = false",
            ),
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Failed to modify pyvenv.cfg: {error}") from error


def _write_project_paths(venv_site: str) -> None:
    repo_root = _imodelzoo_root()
    project_paths = (repo_root, repo_root / "hmodel" / "xh2")
    missing_paths = tuple(path for path in project_paths if not path.exists())
    if missing_paths:
        raise RuntimeError(f"Failed to configure virtualenv import paths, missing: {missing_paths}")
    try:
        (Path(venv_site) / "imodelzoo_paths.pth").write_text(
            "\n".join(str(path) for path in project_paths) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Failed to write virtualenv .pth file: {error}") from error


def _create_distutils_symlink(venv_dir: Path, system_site: str) -> None:
    try:
        result = subprocess.run(
            [
                str(venv_dir / "bin" / "python3"),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "stdout", "") or getattr(error, "stderr", "")
        raise RuntimeError(f"Failed to get virtualenv Python version: {output}") from error
    python_version = f"python{result.stdout.strip()}"
    destination = venv_dir / "lib" / python_version / "site-packages" / "distutils"
    source = Path(system_site) / "setuptools" / "_distutils"
    if destination.exists():
        return
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError as error:
        raise RuntimeError(f"Failed to create distutils symlink: {error}") from error


def _environment_command(
    name: str,
    argv: Sequence[str],
    *,
    workspace: Path,
    log_file: Path | None,
    environment: Mapping[str, str] | None = None,
) -> CommandSpec:
    return CommandSpec(
        name=name,
        argv=tuple(argv),
        cwd=workspace,
        environment=environment or {},
        log_file=log_file,
        mirror_to_console=True,
        timestamp_log_lines=log_file is not None,
    )


def _create_virtualenv(
    *,
    workspace: Path,
    venv_dir: Path,
    python_executable: str,
    system_site: str,
    runner: CommandRunner,
    log_file: Path | None,
) -> str:
    venv_dir.mkdir(parents=True, exist_ok=True)
    argv = ["virtualenv", f"--python={python_executable}"]
    if "/opt/venv" in python_executable:
        argv.append(f"--extra-search-dir={system_site}")
    else:
        argv.append("--system-site-packages")
    argv.append(str(venv_dir))
    runner.run(
        _environment_command(
            "create-python-environment",
            argv,
            workspace=workspace,
            log_file=log_file,
        )
    )

    venv_python = str(venv_dir / "bin" / "python3")
    venv_site = _get_site_packages(venv_python)
    _write_project_paths(venv_site)
    if "/opt/venv" in python_executable:
        _append_activate_environment(
            venv_dir / "bin" / "activate",
            venv_site=venv_site,
            system_site=system_site,
        )
        _append_deactivate_environment(venv_dir / "bin" / "deactivate")
        _disable_system_site_packages(venv_dir / "pyvenv.cfg")
        _create_distutils_symlink(venv_dir, system_site)
    return venv_site


def _pip_install_command(venv_dir: Path, requirement_path: Path) -> tuple[str, ...]:
    argv = [
        str(venv_dir / "bin" / "pip3"),
        "install",
        "--timeout",
        str(PIP_NETWORK_TIMEOUT_SECONDS),
        "--retries",
        str(PIP_NETWORK_RETRIES),
    ]
    if requirement_path.suffix == ".whl":
        argv.append(str(requirement_path))
    else:
        argv.extend(("-r", str(requirement_path)))
    argv.extend(
        (
            "-i",
            PIP_INDEX_URL,
            "--trusted-host",
            PIP_TRUSTED_HOST,
        )
    )
    return tuple(argv)


@dataclass(frozen=True)
class PythonEnvironment:
    """Describe the executable and environment for subsequent commands."""

    executable: str
    environment: Mapping[str, str]


def _compose_venv_environment(
    venv_dir: Path,
    *,
    venv_site: str,
    system_site: str,
    base_environment: Mapping[str, str] | None,
    activated: bool,
) -> dict[str, str]:
    """Expose venv packages first while retaining original Python packages."""
    base = dict(os.environ if base_environment is None else base_environment)
    pythonpath = os.pathsep.join(value for value in (venv_site, base.get("PYTHONPATH", ""), system_site) if value)
    environment = {"PYTHONPATH": pythonpath}
    if activated:
        environment.update(
            {
                "VIRTUAL_ENV": str(venv_dir),
                "PATH": os.pathsep.join(value for value in (str(venv_dir / "bin"), base.get("PATH", "")) if value),
            }
        )
    return environment


def build_venv_environment(
    python_executable: str,
    *,
    base_environment: Mapping[str, str] | None = None,
    activated: bool = True,
) -> dict[str, str]:
    """Build subprocess environment without mutating the parent process."""
    venv_python = Path(python_executable).resolve()
    venv_dir = venv_python.parents[1]
    venv_site = _get_site_packages(str(venv_python))
    system_site = _get_site_packages("python3")
    return _compose_venv_environment(
        venv_dir,
        venv_site=venv_site,
        system_site=system_site,
        base_environment=base_environment,
        activated=activated,
    )


def prepare_python_environment(
    workspace: Path,
    requirements: Sequence[Path],
    *,
    base_environment: Mapping[str, str] | None = None,
    activated: bool = True,
    log_file: Path | None = None,
    runner: CommandRunner | None = None,
) -> PythonEnvironment:
    """Create a venv, reuse original packages, and install ordered differences."""
    resolved_requirements = tuple(path.resolve() for path in requirements if path.is_file())
    if not resolved_requirements:
        return PythonEnvironment("python3", {})

    command_runner = runner or CommandRunner()
    python_executable = _get_python_executable()
    system_site_packages = _get_site_packages(python_executable)
    venv_dir = workspace / VENV_NAME
    venv_site = _create_virtualenv(
        workspace=workspace,
        venv_dir=venv_dir,
        python_executable=python_executable,
        system_site=system_site_packages,
        runner=command_runner,
        log_file=log_file,
    )
    install_environment = _compose_venv_environment(
        venv_dir,
        venv_site=venv_site,
        system_site=system_site_packages,
        base_environment=base_environment,
        activated=True,
    )
    for requirement_path in resolved_requirements:
        command_runner.run(
            _environment_command(
                f"pip-install-{requirement_path.name}",
                _pip_install_command(venv_dir, requirement_path),
                workspace=workspace,
                log_file=log_file,
                environment=install_environment,
            )
        )

    venv_python = str(venv_dir / "bin" / "python3")
    return PythonEnvironment(
        executable=venv_python,
        environment=_compose_venv_environment(
            venv_dir,
            venv_site=venv_site,
            system_site=system_site_packages,
            base_environment=base_environment,
            activated=activated,
        ),
    )


def as_python_environment(value: object) -> PythonEnvironment:
    if isinstance(value, PythonEnvironment):
        return value
    return PythonEnvironment(str(value), {})


__all__ = [
    "PythonEnvironment",
    "PIP_NETWORK_RETRIES",
    "PIP_NETWORK_TIMEOUT_SECONDS",
    "VENV_NAME",
    "as_python_environment",
    "build_venv_environment",
    "prepare_python_environment",
]
