# Copyright 2025 HOUMO AI
#
# File: test_hmatc_utils.py
# Description:
#  HMATC Test Orchestration Using Shared Runtime Infrastructure.
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

import logging
from glob import glob
from pathlib import Path

import pytest

from ..tests_utils.command_execution import (
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    CommandSpec,
    output_reports_failure,
)
from ..tests_utils.pytest_support import (
    MarkerConfigurationError,
    device_markers_from_request,
)
from ..tests_utils.runtime_context import TCaseType, TestRuntimeContext
from ..tests_utils.workspace import WorkspaceManager, WorkspaceOwnershipError

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def _hmatc_command(
    name: str,
    argv: tuple[str, ...],
    *,
    cwd: Path,
    log_file: Path | None,
) -> CommandSpec:
    """Build a command retaining legacy HMATC console and log presentation."""
    return CommandSpec(
        name,
        argv,
        cwd=cwd,
        allow_nonzero_exit=True,
        log_file=log_file,
        mirror_to_console=True,
        timestamp_log_lines=True,
        suppressed_display_substrings=("MB/s",),
    )


def _command_succeeded(result: CommandResult) -> bool:
    """Preserve the legacy return-code plus stdout marker validation."""
    return result.succeeded and not output_reports_failure(result.stdout)


def _run_hmatc(
    runner: CommandRunner,
    workspace: Path,
    model_info: dict,
    config_yml: str,
    hmatc_type: str,
    log_file: Path,
) -> bool:
    argv = ["hmatc", hmatc_type, "--config", config_yml]
    if hmatc_type == "compare":
        argv.extend(("--data_path", model_info["data_path"]))
    elif hmatc_type == "perf":
        argv.extend(("-wn", "10", "-sn", "500", "-tn", "8"))
    result = runner.run(_hmatc_command(f"hmatc-{hmatc_type}", tuple(argv), cwd=workspace, log_file=log_file))
    succeeded = _command_succeeded(result)
    if not succeeded:
        logger.error("Execute hmatc %s %s failed!", hmatc_type, config_yml)
    return succeeded


def _perf_models(
    runner: CommandRunner,
    workspace: Path,
    config_yml: str,
    log_file: Path,
    *,
    backend: str,
) -> bool:
    quant = runner.run(
        _hmatc_command(
            "hmatc-perf-quant",
            ("hmatc", "quant", "--config", config_yml),
            cwd=workspace,
            log_file=log_file,
        )
    )
    if not _command_succeeded(quant):
        logger.error("Perf test quant: %s failed!", config_yml)
        return False

    ncore = "2" if backend == "xh2" else "4"
    build = runner.run(
        _hmatc_command(
            "hmatc-perf-build",
            ("hmatc", "build", "--config", config_yml, "--ncore", ncore),
            cwd=workspace,
            log_file=log_file,
        )
    )
    if not _command_succeeded(build):
        logger.error("Perf test build: %s failed!", config_yml)
        return False

    perf = runner.run(
        _hmatc_command(
            "hmatc-perf",
            (
                "hmatc",
                "perf",
                "--config",
                config_yml,
                "-wn",
                "10",
                "-sn",
                "1000",
                "-tn",
                "8",
            ),
            cwd=workspace,
            log_file=log_file,
        )
    )
    succeeded = _command_succeeded(perf)
    if not succeeded:
        logger.error("Perf test: %s failed!", config_yml)
    return succeeded


def execute_hmatc_cmd(model_name: str, setup_logging) -> None:
    """Execute the complete HMATC matrix in one owned workspace."""
    log_file_value, pytest_request = setup_logging
    log_file = Path(log_file_value)
    try:
        markers = device_markers_from_request(pytest_request)
    except MarkerConfigurationError as error:
        pytest.skip(str(error))

    runtime = TestRuntimeContext.from_environment()
    logger.info(
        "log_file: %s, dev_res_dir: %s",
        log_file,
        markers.result_directory_name,
    )
    if runtime.test_type == TCaseType.SEPARATE_NO_INFER:
        pytest.skip(f"Skip hmatc testcase {model_name} in the SEPARATE NO INFER stage.")

    model_dict = {
        "resnet50": {
            "model_dir": (SCRIPT_DIR / "../../models/backbone/resnet50").resolve(),
            "data_path": "./imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG",
        },
        "yolov5s": {
            "model_dir": (SCRIPT_DIR / "../../models/detection/yolov5s").resolve(),
            "data_path": "./coco2017/val2017/000000000139.jpg",
        },
    }
    runner = CommandRunner()
    manager = WorkspaceManager()

    try:
        with manager.open(model_dict[model_name]["model_dir"], phase="hmatc") as workspace:
            logger.info("workspace: %s", workspace)
            get_model = runner.run(
                _hmatc_command(
                    "hmatc-get-model",
                    ("python3", "get_model.py", "--type", "raw"),
                    cwd=workspace,
                    log_file=None,
                )
            )
            if not _command_succeeded(get_model):
                pytest.fail("HMATC get-model failed", pytrace=False)
            final_flag = _run_hmatc_matrix(model_name, model_dict[model_name], runtime, runner, workspace, log_file)
    except (CommandExecutionError, WorkspaceOwnershipError) as error:
        pytest.fail(error.format_diagnostic(), pytrace=False)

    assert final_flag is True, "Hmatc Test Failed!"
    logger.info("Hmatc Test Success!")


def _run_hmatc_matrix(model_name, model_info, runtime, runner, workspace, log_file) -> bool:
    """Run functional HMATC cases and optional xh1 performance cases."""
    test_configs = SCRIPT_DIR / "hmatc_configs" / model_name
    hmatc_types = ("quant", "build", "demo", "compare", "eval", "perf")
    passed = _run_hmatc_functional_cases(
        runner,
        workspace,
        model_info,
        test_configs,
        hmatc_types,
        log_file,
    )
    if runtime.backend == "xh1":
        passed &= _run_hmatc_perf_cases(runner, workspace, test_configs, log_file, runtime.backend)
    return passed


def _run_hmatc_functional_cases(runner, workspace, model_info, test_configs, hmatc_types, log_file) -> bool:
    """Run each functional config through all supported HMATC subcommands."""
    passed = True
    for config_yml in sorted(glob(str(test_configs / "func_test" / "*.yml"))):
        logger.info("test config file: %s", config_yml)
        for hmatc_type in hmatc_types:
            passed &= _run_hmatc(runner, workspace, model_info, config_yml, hmatc_type, log_file)
    return passed


def _run_hmatc_perf_cases(runner, workspace, test_configs, log_file, backend) -> bool:
    """Run optional xh1 performance configurations."""
    passed = True
    for config_yml in sorted(glob(str(test_configs / "perf_test" / "*.yml"))):
        passed &= _perf_models(runner, workspace, config_yml, log_file, backend=backend)
    return passed


__all__ = ["execute_hmatc_cmd"]
