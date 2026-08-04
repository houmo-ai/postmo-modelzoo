# Copyright 2025 HOUMO AI
#
# File: test_apis_utils.py
# Description:
#  API Example Orchestration Using Shared Runtime Infrastructure.
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

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from ..tests_utils.command_execution import (
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    CommandSpec,
    output_reports_failure,
)
from ..tests_utils.platform_device import (
    check_device_info,
    check_vpu_status,
    get_platform,
)
from ..tests_utils.pytest_support import (
    MarkerConfigurationError,
    device_markers_from_request,
)
from ..tests_utils.python_environment import prepare_python_environment
from ..tests_utils.resource_lock import ModelResourceLock
from ..tests_utils.runtime_context import TCaseType, TestRuntimeContext
from ..tests_utils.workspace import WorkspaceManager, WorkspaceOwnershipError

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_example_cfg(example_name: str) -> dict | None:
    """Load one API example configuration locally to the API suite."""
    path = SCRIPT_DIR / "apis_configs" / f"apis_cfg_{example_name}.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _generate_cmds(
    cmd_header: list,
    params_dict: dict,
    max_core_num: int = 0,
    start_idx: int = 0,
    name_prefix: str = "",
) -> list[list[str]]:
    """Preserve the API suite's column-oriented parameter expansion."""
    cmd_list = [] if start_idx == 0 else [cmd_header]
    idx = start_idx
    active = True
    while active:
        options, active = _generate_case_options(params_dict, idx, max_core_num=max_core_num, name_prefix=name_prefix)
        if options or active:
            cmd_list.append([*cmd_header, *options])
        idx += 1
    return cmd_list


def _generate_case_options(
    params_dict: dict,
    index: int,
    *,
    max_core_num: int,
    name_prefix: str,
) -> tuple[list[str], bool]:
    """Render one parameter column index and report whether it was active."""
    options: list[str] = []
    active = False
    for param_name, param_list in params_dict.items():
        value = _parameter_value(param_name, param_list, index, max_core_num)
        if value is _MISSING:
            continue
        rendered = _render_api_value(param_name, value, name_prefix)
        if rendered:
            options.extend(rendered)
            active = True
    return options, active


_MISSING = object()


def _parameter_value(param_name: str, values, index: int, max_core_num: int):
    """Return one active parameter value or the internal missing sentinel."""
    if param_name in {"defines", "envs"} or len(values) <= index:
        return _MISSING
    value = values[index]
    if value is None or value == "default":
        return _MISSING
    if max_core_num > 0 and param_name == "ncore" and int(value) > max_core_num:
        return _MISSING
    return value


def _render_api_value(param_name: str, value, name_prefix: str) -> list[str]:
    """Render one API parameter value as command-line tokens."""
    if param_name == "name":
        return [f"{name_prefix}{value}" if name_prefix else str(value)]
    if param_name.startswith("#"):
        return [str(value)]
    option = param_name if param_name.startswith("-") else f"--{param_name}"
    if isinstance(value, bool):
        return [option] if value else []
    return [option, str(value)]


def _api_command(
    name: str,
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path,
    log_file: Path | None,
) -> CommandSpec:
    return CommandSpec(
        name,
        tuple(str(value) for value in argv),
        cwd=cwd,
        allow_nonzero_exit=True,
        log_file=log_file,
        mirror_to_console=True,
        timestamp_log_lines=True,
        suppressed_display_substrings=("MB/s",),
    )


def _command_succeeded(result: CommandResult, *, check_output: bool = True) -> bool:
    return result.succeeded and (not check_output or not output_reports_failure(result.combined_output))


def _require_success(result: CommandResult, *, check_output: bool = True) -> None:
    if _command_succeeded(result, check_output=check_output):
        return
    raise CommandExecutionError(
        f"API command failed: {result.command.name}",
        details={
            "argv": result.command.argv,
            "return_code": result.return_code,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        },
    )


def _compile_cpp_exec(
    workspace: Path,
    runner: CommandRunner,
    log_file: Path,
    defines: list,
) -> None:
    """Compile one C++ example case without changing process cwd."""
    build_dir = workspace / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake = [
        "cmake",
        f"-DCMAKE_INSTALL_PREFIX={workspace}",
        "-DCMAKE_BUILD_TYPE=Release",
        *defines,
        "..",
    ]
    for name, argv in (
        ("api-cmake", cmake),
        ("api-make", ["make", "-j"]),
        ("api-make-install", ["make", "install"]),
    ):
        _require_success(runner.run(_api_command(name, argv, cwd=build_dir, log_file=log_file)))


def _test_get_model(
    example_info: dict,
    platform_name: str,
    model_set_dir: Path,
    *,
    backend: str,
    workspace: Path,
    runner: CommandRunner,
) -> bool:
    """Run all parameterized API get-model cases under one resource lock."""
    max_core_num = 2 if platform_name == "aarch64" else 0
    params_dict = example_info["get_model_params"][backend]
    commands = _generate_cmds(
        ["python3", "get_model.py", "--model_dir", str(model_set_dir)],
        params_dict,
        max_core_num,
        start_idx=1,
    )
    logger.info("Get model cmds: %s", commands)
    succeeded = True
    with ModelResourceLock(
        model_set_dir / "lock.lock",
        ModelResourceLock.LockMode.WRITE,
        "model downloading",
    ):
        for index, argv in enumerate(commands):
            result = runner.run(_api_command(f"api-get-model[{index}]", argv, cwd=workspace, log_file=None))
            if not _command_succeeded(result):
                succeeded = False
                logger.error("Get Model Test Failed, test cmd: %s", argv)
    return succeeded


def _execute_run_sh_if_present(
    source_dir: Path,
    *,
    runtime: TestRuntimeContext,
    runner: CommandRunner,
    manager: WorkspaceManager,
    log_file: Path,
) -> None:
    """Run the mutating run.sh stage in its own disposable workspace."""
    if not runtime.is_asic or not (source_dir / "run.sh").is_file():
        return
    with manager.open(source_dir, phase="apis_run_sh") as workspace:
        result = runner.run(
            _api_command(
                "api-run-sh",
                ["bash", "run.sh"],
                cwd=workspace,
                log_file=log_file,
            )
        )
        _require_success(result)


def execute_apis_examples(example_name: str, setup_logging) -> None:
    """Execute one API example with explicit runtime and owned workspaces."""
    log_file_value, pytest_request = setup_logging
    log_file = Path(log_file_value)
    try:
        markers = device_markers_from_request(pytest_request)
    except MarkerConfigurationError as error:
        pytest.skip(str(error))

    runtime = TestRuntimeContext.from_environment()
    runner = CommandRunner()
    manager = WorkspaceManager()
    logger.info(
        "log_file: %s, dev_res_dir: %s",
        log_file,
        markers.result_directory_name,
    )

    example_info = _load_example_cfg(example_name)
    platform_name = _validate_example_support(example_name, example_info, runtime, runner)

    source_dir = (SCRIPT_DIR / "../.." / example_info["example_dir"]).resolve()
    if not source_dir.is_dir():
        raise AssertionError(f"The {example_name} example folder doesn't exist.")

    try:
        _execute_run_sh_if_present(source_dir, runtime=runtime, runner=runner, manager=manager, log_file=log_file)
        _execute_api_workspace(
            example_name,
            example_info,
            platform_name,
            runtime=runtime,
            runner=runner,
            manager=manager,
            source_dir=source_dir,
            log_file=log_file,
        )
    except (CommandExecutionError, WorkspaceOwnershipError) as error:
        pytest.fail(error.format_diagnostic(), pytrace=False)

    logger.info("Apis Example Test Success!")


def _validate_example_support(example_name, example_info, runtime, runner):
    """Validate config, platform, device, and dependency prerequisites."""
    if (
        example_info is None
        or example_info["obsolete"] is True
        or runtime.backend not in example_info["support_backend"]
        or example_info["support_backend"][runtime.backend] is None
    ):
        pytest.skip("This testcase is not supported.")
    if runtime.test_type == TCaseType.SEPARATE_NO_INFER:
        pytest.skip(f"Skip apis testcase {example_name} in the SEPARATE NO INFER stage.")
    platform_name = get_platform(example_info["support_platform"])
    if platform_name is None:
        pytest.skip("This testcase is not supported on the current platform.")
    if (
        runtime.is_asic
        and platform_name == "aarch64"
        and not check_device_info(
            example_info["support_core_num"].get(runtime.backend),
            backend=runtime.backend,
            runner=runner,
        )
    ):
        pytest.skip("This testcase is not supported on the current core count.")
    dependencies = example_info.get("dependency") or ()
    if isinstance(dependencies, str):
        dependencies = (dependencies,)
    if "vpu" in dependencies and (not runtime.is_asic or not check_vpu_status(backend=runtime.backend, runner=runner)):
        pytest.skip("This testcase needs the VPU driver.")
    return platform_name


def _execute_api_workspace(
    example_name,
    example_info,
    platform_name,
    *,
    runtime,
    runner,
    manager,
    source_dir,
    log_file,
) -> None:
    """Run get-model and all configured API demo implementations."""
    with manager.open(source_dir, phase="apis_example") as workspace:
        logger.info("workspace: %s", workspace)
        model_set_dir = runtime.models_path / example_info["example_dir"]
        get_model_ok = _execute_api_get_model(
            example_name, example_info, platform_name, runtime, runner, workspace, model_set_dir
        )
        demo_types = example_info["support_backend"][runtime.backend]
        py_ok = _execute_python_api_demo(example_info, runtime, runner, workspace, log_file, demo_types)
        cpp_ok = _execute_cpp_api_demo(example_info, runtime, runner, workspace, log_file, demo_types)
        assert get_model_ok and py_ok and cpp_ok, "Apis Example Test Failed!"


def _execute_api_get_model(
    example_name,
    example_info,
    platform_name,
    runtime,
    runner,
    workspace,
    model_set_dir,
) -> bool:
    """Run the configured or default get-model stage."""
    backend_params = (example_info.get("get_model_params") or {}).get(runtime.backend)
    if backend_params is not None:
        return _test_get_model(
            example_info,
            platform_name,
            model_set_dir,
            backend=runtime.backend,
            workspace=workspace,
            runner=runner,
        )
    argv = ["python3", "get_model.py"]
    if example_name not in {"qwen3", "qwen3_multibatch", "qwen3_speculative"}:
        argv.extend(("--model_dir", str(model_set_dir)))
    with ModelResourceLock(
        model_set_dir / "lock.lock",
        ModelResourceLock.LockMode.WRITE,
        "model downloading",
    ):
        _require_success(runner.run(_api_command("api-get-model-default", argv, cwd=workspace, log_file=None)))
    return True


def _execute_python_api_demo(example_info, runtime, runner, workspace, log_file, demo_types):
    """Run all configured Python API demo cases."""
    if "python" not in demo_types:
        return True
    python = prepare_python_environment(
        workspace,
        (workspace / "requirements.txt",),
        base_environment=runtime.environment,
        log_file=log_file,
    )
    commands = _generate_cmds([python.executable], example_info["py_example_params"])
    logger.info("python exe cmd_list: %s", commands)
    succeeded = True
    for index, argv in enumerate(commands):
        spec = replace(
            _api_command(f"api-python[{index}]", argv, cwd=workspace, log_file=log_file),
            environment=python.environment,
        )
        succeeded &= _command_succeeded(runner.run(spec))
    return succeeded


def _execute_cpp_api_demo(example_info, runtime, runner, workspace, log_file, demo_types):
    """Compile and run all configured C++ API demo cases."""
    if "cpp" not in demo_types:
        return True
    params = example_info["cpp_example_params"]
    defines = params.get("defines", [])
    commands = _generate_cmds([], params, name_prefix="./")
    logger.info("cpp exe cmd_list: %s", commands)
    succeeded = True
    for index, argv in enumerate(commands):
        _compile_cpp_exec(workspace, runner, log_file, defines[index] if defines else [])
        result = runner.run(_api_command(f"api-cpp[{index}]", argv, cwd=workspace, log_file=log_file))
        succeeded &= _command_succeeded(result)
    return succeeded


__all__ = ["execute_apis_examples"]
