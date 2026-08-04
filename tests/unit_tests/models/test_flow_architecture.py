# Copyright (c) 2026 HOUMO AI
#
# File: test_flow_architecture.py
# Description:
#  Architecture contract tests for flow registration, dependency boundaries,
#    public interfaces, and execution policies.
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

"""Unit tests extracted from the former model-flow contract suite: test_flow_architecture.py."""

import ast
import inspect
import pytest
from tests.models_tests import (
    test_models_utils,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
    LLM_FLOW_POLICY,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.test_flows import (
    flow_registry,
)
from tests.models_tests.test_flows.compile_flow import (
    CompileFlowHandler,
)
from tests.models_tests.test_flows.demo_flow import (
    DemoFlowHandler,
)
from tests.models_tests.test_flows.flow_registry import (
    FLOW_REGISTRY,
)
from tests.models_tests.test_flows.quant_flow import (
    QuantFlowHandler,
)
from ._flow_contract_support import (
    MODELS_TESTS_DIR,
    TESTS_DIR,
    _module_imports,
)

pytestmark = pytest.mark.unit


def test_xh1_and_xh2_registry_entries_use_backend_policy_seams() -> None:
    for family, expected_policy in (
        (ModelFamily.CV, CV_FLOW_POLICY),
        (ModelFamily.LLM, LLM_FLOW_POLICY),
    ):
        for backend in ("xh1", "xh2"):
            compile_handler = FLOW_REGISTRY.resolve(family, backend, ModelFlow.COMPILE)
            demo_handler = FLOW_REGISTRY.resolve(family, backend, ModelFlow.DEMO)
            perf_handler = FLOW_REGISTRY.resolve(family, backend, ModelFlow.PERF)
            assert type(compile_handler) is CompileFlowHandler
            assert type(demo_handler) is DemoFlowHandler
            assert compile_handler.policy is expected_policy
            assert demo_handler.policy is expected_policy
            assert perf_handler.policy is expected_policy


def test_model_workflow_does_not_import_flow_implementations() -> None:
    workflow_dir = MODELS_TESTS_DIR / "model_workflow"
    for path in workflow_dir.glob("*.py"):
        imports = _module_imports(path)
        assert not any("test_flows" in name for name in imports), path


def test_inference_flows_do_not_import_private_support_apis() -> None:
    flow_dir = MODELS_TESTS_DIR / "test_flows"
    for filename in ("demo_flow.py", "compare_flow.py", "eval_flow.py", "perf_flow.py"):
        tree = ast.parse((flow_dir / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "inference_flow_support":
                continue
            assert all(not alias.name.startswith("_") for alias in node.names), filename


def test_parameterized_command_sites_use_shared_renderer() -> None:
    flow_dir = MODELS_TESTS_DIR / "test_flows"
    expected_counts = {
        "compile_flow.py": 2,
        "demo_flow.py": 1,
        "get_model_flow.py": 1,
        "hmatc_flow_support.py": 2,
        "inference_flow_support.py": 0,
        "perf_flow.py": 2,
        "quant_flow.py": 1,
    }
    for filename, expected_count in expected_counts.items():
        tree = ast.parse((flow_dir / filename).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_case_options"
        ]
        assert len(calls) == expected_count, filename


def test_compiler_pruning_precedes_manifest_publication() -> None:
    flow_dir = MODELS_TESTS_DIR / "test_flows"
    compile_source = (flow_dir / "compile_flow.py").read_text(encoding="utf-8")
    assert compile_source.index("prune_compiler_intermediates(staging_dir)") < (
        compile_source.index(
            "publish_compiled_artifact(", compile_source.index("staging_dir")
        )
    )
    hmatc_source = (flow_dir / "hmatc_flow_support.py").read_text(encoding="utf-8")
    prune_index = hmatc_source.index("prune_compiler_intermediates(source)")
    assert prune_index < hmatc_source.index(
        "_write_hmatc_inference_manifest(", prune_index
    )


def test_quant_and_compile_implementations_are_split() -> None:
    assert QuantFlowHandler.__module__.endswith(".quant_flow")
    assert CompileFlowHandler.__module__.endswith(".compile_flow")


def test_pytest_flow_entry_signatures_remain_compatible() -> None:
    for name in (
        "execute_get_model_flow",
        "execute_quant_flow",
        "execute_compile_flow",
        "execute_demo_flow",
        "execute_compare_flow",
        "execute_eval_flow",
        "execute_perf_flow",
    ):
        signature = inspect.signature(getattr(test_models_utils, name))
        assert tuple(signature.parameters) == ("model_name", "setup_logging")


def test_flow_registry_has_no_pytest_entrypoint_dependency() -> None:
    assert not hasattr(flow_registry, "create_flow_context")
    assert not hasattr(flow_registry, "execute_model_flow")
    assert not any(
        hasattr(flow_registry, f"execute_{flow.value}_flow") for flow in ModelFlow
    )


def test_only_artifact_preparer_or_registry_may_reference_other_handlers() -> None:
    flow_dir = MODELS_TESTS_DIR / "test_flows"
    allowed = {"artifact_preparation.py", "flow_registry.py"}
    handler_modules = {
        "get_model_flow",
        "quant_flow",
        "compile_flow",
        "demo_flow",
        "compare_flow",
        "eval_flow",
        "perf_flow",
    }
    for path in flow_dir.glob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.rsplit(".", 1)[-1] not in handler_modules, path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert not node.func.id.endswith("FlowHandler"), path


def test_model_workflow_has_no_execution_resource_compatibility_facades() -> None:
    models_root = MODELS_TESTS_DIR
    removed_modules = {"execution_resources", "test_workspace_manager"}
    for path in models_root.rglob("*.py"):
        assert path.stem not in removed_modules, path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.rsplit(".", 1)[-1] not in removed_modules, path


def test_model_workflow_modules_declare_explicit_exports() -> None:
    models_root = MODELS_TESTS_DIR
    module_roots = (
        models_root / "model_workflow",
        models_root / "test_flows",
    )
    for module_root in module_roots:
        for path in module_root.glob("*.py"):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            exports = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            ]
            assert len(exports) == 1, path
            assert isinstance(exports[0].value, (ast.List, ast.Tuple)), path
            assert exports[0].value.elts, path


def test_hmatc_inference_support_is_owned_by_hmatc_module() -> None:
    flow_root = MODELS_TESTS_DIR / "test_flows"
    inference_source = (flow_root / "inference_flow_support.py").read_text(
        encoding="utf-8"
    )
    hmatc_source = (flow_root / "hmatc_flow_support.py").read_text(encoding="utf-8")
    assert len(inference_source.splitlines()) < 800
    for symbol in (
        "run_hmatc_cases",
        "run_hmatc_inference_preparation",
        "restore_reusable_hmatc_inference_artifact",
        "hmatc_inference_artifact_relative_path",
    ):
        assert f"def {symbol}" not in inference_source
        assert f"def {symbol}" in hmatc_source


def test_active_test_code_does_not_reintroduce_legacy_execution_patterns() -> None:
    tests_root = TESTS_DIR
    forbidden_calls = {"execute_test_cmd", "prepare_test_folder"}
    for path in tests_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "*" for alias in node.names), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, path
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "chdir"
            ):
                raise AssertionError(f"process cwd mutation is forbidden: {path}")
