# Copyright (c) 2025 HOUMO AI
#
# File: test_models_utils.py
# Description:
#  Pytest-Facing Model-Flow Orchestration Utilities.
#  Keep the main flow in this file so maintainers can understand the complete test
#    path from one place. Detailed CV/LLM behavior, policies, cache management,
#    workspace handling and result parsing live in :mod:`tests.models_tests.test_flows`.
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
from pathlib import Path
from uuid import uuid4

import pytest

from ..tests_utils.command_execution import CommandExecutionError
from ..tests_utils.platform_device import get_platform
from ..tests_utils.pytest_support import (
    MarkerConfigurationError,
    device_markers_from_request,
)
from ..tests_utils.runtime_context import TestRuntimeContext
from ..tests_utils.workspace import WorkspaceOwnershipError
from .model_workflow.flow_contracts import (
    DiagnosticContext,
    FlowDisposition,
    FlowContext,
    FlowRequest,
    ModelFlow,
    ModelTestError,
)
from .test_flows.flow_registry import (
    FLOW_REGISTRY,
    FlowServices,
)
from .model_workflow.model_config_repository import ModelConfigRepository

logger = logging.getLogger(__name__)
MODELS_TESTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODELS_TESTS_DIR.parents[1]
CONFIG_REPOSITORY = ModelConfigRepository(MODELS_TESTS_DIR / "model_configs")


def create_flow_context(config, flow: ModelFlow, setup_logging) -> FlowContext:
    """Create immutable runtime context from pytest and model configuration."""
    log_file, pytest_request = setup_logging
    marker_values = device_markers_from_request(pytest_request)
    runtime = TestRuntimeContext.from_environment()
    platform = get_platform(config.support_platform)
    diagnostic = DiagnosticContext(
        run_id=uuid4().hex,
        model_name=config.model_name,
        family=config.family,
        backend=runtime.backend,
        flow=flow,
    )
    return FlowContext(
        diagnostic=diagnostic,
        platform=platform,
        test_type=runtime.test_type,
        release=runtime.release,
        log_file=Path(log_file),
        source_dir=REPOSITORY_ROOT / config.model_dir,
        model_cache_dir=runtime.models_path / config.model_dir,
        result_cache_dir=(runtime.results_path / marker_values.result_directory_name / config.model_dir),
        ndevice_marker=marker_values.ndevice_token,
        device_mem_marker=marker_values.device_mem_token,
    )


def execute_model_flow(model_name: str, setup_logging, flow: ModelFlow) -> None:
    """Run the common pytest orchestration for one model and one flow.

    Main path:

    ``config -> context -> family/backend handler -> result -> pytest outcome``
    """
    context = None
    try:
        # 1. Load and validate the model configuration once.
        config = CONFIG_REPOSITORY.load(model_name)

        # 2. Resolve execution facts such as platform, test stage, device markers,
        #    cache locations and diagnostic identifiers.
        context = create_flow_context(config, flow, setup_logging)

        # 3. Dispatch by model family, backend and flow. CV/LLM and xh1/xh2
        #    differences are implemented by their own handlers/policies.
        handler = FLOW_REGISTRY.resolve(config.family, context.diagnostic.backend, flow)
        logger.info("model flow start: %s", context.diagnostic.as_fields())

        # 4. Execute the selected flow. Runtime services are passed explicitly so
        #    shared helpers do not need to discover pytest fixtures globally.
        result = handler.run(
            FlowRequest(context=context, config=config),
            FlowServices(),
        )

        # 5. Convert the structured handler result into the final pytest outcome.
        if result.disposition in (
            FlowDisposition.SKIPPED,
            FlowDisposition.PREPARED_ONLY,
        ):
            logger.warning(
                "model flow skipped: %s disposition=%s reason=%s",
                context.diagnostic.as_fields(),
                result.disposition.value,
                result.message,
            )
            pytest.skip(result.message)
        if result.validation is not None and not result.validation.passed:
            logger.error(
                "model flow failed: %s summary=%s failures=%s",
                context.diagnostic.as_fields(),
                result.validation.summary,
                result.validation.failures,
            )
            pytest.fail(result.validation.summary, pytrace=False)

        logger.info(
            "model flow finish: %s disposition=%s",
            context.diagnostic.as_fields(),
            result.disposition.value,
        )
    except ModelTestError as error:
        logger.error("model flow error:\n%s", error.format_diagnostic())
        pytest.fail(error.format_diagnostic(), pytrace=False)
    except (CommandExecutionError, WorkspaceOwnershipError) as error:
        logger.error("model infrastructure error:\n%s", error.format_diagnostic())
        pytest.fail(error.format_diagnostic(), pytrace=False)
    except MarkerConfigurationError as error:
        logger.warning("model flow marker configuration: %s", error)
        pytest.skip(str(error))
    except AssertionError as error:
        diagnostic = context.diagnostic.as_fields() if context is not None else ""
        logger.exception("model flow assertion failed: %s", diagnostic or model_name)
        pytest.fail(f"{error}\n{diagnostic}".strip(), pytrace=False)
    except Exception:
        diagnostic = context.diagnostic.as_fields() if context is not None else model_name
        logger.exception("unexpected model flow failure: %s", diagnostic)
        raise


def execute_get_model_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.GET_MODEL)


def execute_quant_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.QUANT)


def execute_compile_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.COMPILE)


def execute_demo_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.DEMO)


def execute_compare_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.COMPARE)


def execute_eval_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.EVAL)


def execute_perf_flow(model_name: str, setup_logging) -> None:
    execute_model_flow(model_name, setup_logging, ModelFlow.PERF)


__all__ = [
    "execute_get_model_flow",
    "execute_quant_flow",
    "execute_compile_flow",
    "execute_demo_flow",
    "execute_compare_flow",
    "execute_eval_flow",
    "execute_perf_flow",
]
