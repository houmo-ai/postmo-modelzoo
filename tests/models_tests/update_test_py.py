# Copyright (c) 2025 HOUMO AI
#
# File: update_test_py.py
# Description:
#  Deterministic Pytest Entry-File Generator for Active Model Configurations.
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

"""Deterministically generate model-flow pytest entry files from active configs."""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    # Support direct execution (``python3 update_test_py.py``) despite the
    # package-relative imports below: establish the package context and
    # repository root on ``sys.path`` so they resolve.
    # ``python3 -m tests.models_tests.update_test_py`` already sets
    # ``__package__`` and skips this branch.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "tests.models_tests"

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

from .model_workflow.flow_contracts import ModelFlow
from .model_workflow.backend_flow_policies import FLOW_DEPENDENCY_RULES
from .model_workflow.backend_flow_policies import FLOW_ORDER as FLOW_ORDER_POLICY
from .model_workflow.model_config_repository import ModelConfig, ModelConfigRepository

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_REPOSITORY = ModelConfigRepository(SCRIPT_DIR / "model_configs")
FLOW_ORDER = tuple(sorted(FLOW_ORDER_POLICY, key=FLOW_ORDER_POLICY.__getitem__))

FLOW_DESCRIPTION = {
    ModelFlow.GET_MODEL: ("downloading", "download"),
    ModelFlow.QUANT: ("quantization", "quantize"),
    ModelFlow.COMPILE: ("compilation", "compile"),
    ModelFlow.DEMO: ("demo", "demo"),
    ModelFlow.COMPARE: ("comparison", "compare"),
    ModelFlow.EVAL: ("evaluation", "evaluate"),
    ModelFlow.PERF: ("performance", "performance test"),
}


@dataclass(frozen=True)
class GeneratedCase:
    """Immutable description of one pytest test function to be generated."""

    config: ModelConfig
    category: str
    marker: str
    function_name: str
    flow: ModelFlow
    supported_flows: frozenset[ModelFlow]


def convert_model_name(model_name: str) -> str:
    """Normalize a model name into a valid Python/pytest marker identifier."""
    return model_name.replace("-", "_").replace(".", "dot")


def flow_test_filename(flow: ModelFlow) -> str:
    """Return the generated test-module filename for a flow.

    ``GET_MODEL`` maps to ``test_get_models.py``; every other flow maps to
    ``test_<flow>_models.py``.
    """
    if flow == ModelFlow.GET_MODEL:
        return "test_get_models.py"
    return f"test_{flow.value}_models.py"


def model_category(config: ModelConfig) -> str:
    """Derive the model category (e.g. ``llm``, ``asr``) from its source directory."""
    parts = config.model_dir.parts
    if len(parts) < 2 or parts[0] != "models":
        raise ValueError(f"Unexpected model_dir for {config.model_name}: {config.model_dir}")
    return parts[1].lower()


def supported_flows(config: ModelConfig) -> frozenset[ModelFlow]:
    """Return the set of flows a model config declares across all its backends.

    The synthetic ``demo_multibatch`` entry is excluded because it is a demo
    variant, not a standalone flow. Unknown flow names (which can appear in
    stale obsolete configs that skip active validation) are skipped rather than
    raising, so one stale config cannot abort the whole generation.
    """
    flows: set[ModelFlow] = set()
    for backend in config.support_backend:
        for flow_name in config.support_flow.get(backend, ()):
            if flow_name == "demo_multibatch":
                continue
            try:
                flows.add(ModelFlow(flow_name))
            except ValueError:
                # Stale/unknown flow name in an obsolete config; ignore it
                # instead of crashing generation for every model.
                continue
    return frozenset(flows)


def build_cases(
    configs: tuple[ModelConfig, ...],
) -> dict[ModelFlow, list[GeneratedCase]]:
    """Build one :class:`GeneratedCase` per (model, supported flow) pair.

    Returns a mapping from flow to the ordered list of cases for that flow.
    Raises ``ValueError`` if two cases would produce the same function name.
    """
    result = {flow: [] for flow in FLOW_ORDER}
    seen_names: set[str] = set()
    for config in configs:
        marker = convert_model_name(config.model_name)
        try:
            category = model_category(config)
        except ValueError as error:
            # Obsolete configs skip active validation, so a stale model_dir
            # shape must not abort generation for every other model.
            print(f"warning: skipping {config.model_name}: {error}")
            continue
        flows = supported_flows(config)
        for flow in FLOW_ORDER:
            if flow not in flows:
                continue
            function_name = f"test_{category}_{marker}_{flow.value}"
            if function_name in seen_names:
                raise ValueError(f"Duplicate generated test function: {function_name}")
            seen_names.add(function_name)
            result[flow].append(GeneratedCase(config, category, marker, function_name, flow, flows))
    return result


def render_header(flow: ModelFlow) -> str:
    """Render the fixed header (license, imports, helper function) for a flow file."""
    noun, verb = FLOW_DESCRIPTION[flow]
    filename = flow_test_filename(flow)
    helper = f"_{flow.value}_func"
    executor = f"execute_{flow.value}_flow"
    return f'''# Copyright (c) 2025 HOUMO AI
#
# File: {filename}
# Description:
#  Model {noun.title()} Tests Module.
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

# Generated by update_test_py.py. Do not edit manually.

import logging

import pytest

from .test_models_utils import {executor}


logger = logging.getLogger(__name__)


def {helper}(model_name: str, setup_logging) -> None:
    """Execute model {noun} test for a specific model."""
    logger.info("===> TEST START: test_%s_{flow.value}", model_name)
    {executor}(model_name, setup_logging)
'''


def dependency_target(case: GeneratedCase) -> str | None:
    """Return the ``file::function`` prerequisite this case should depend on, if any.

    Picks the first declared prerequisite flow that the model actually supports.
    """
    for prerequisite in FLOW_DEPENDENCY_RULES[case.flow]:
        if prerequisite not in case.supported_flows:
            continue
        function_name = f"test_{case.category}_{case.marker}_{prerequisite.value}"
        return f"{flow_test_filename(prerequisite)}::{function_name}"
    return None


def dependency_name(case: GeneratedCase) -> str:
    """Return the ``file::function`` identity used as this case's dependency name."""
    return f"{flow_test_filename(case.flow)}::{case.function_name}"


def render_case(case: GeneratedCase) -> str:
    """Render a single test function (markers, signature, helper call) as text."""
    markers = [f"@pytest.mark.{case.marker}"]
    dependencies = case.config.dependencies
    ndevice = dependencies.get("ndevice") if isinstance(dependencies, dict) else None
    dev_mem = dependencies.get("dev_mem") if isinstance(dependencies, dict) else None
    if isinstance(ndevice, list) and ndevice:
        markers.append(f"@pytest.mark.ndevice_{ndevice[0]}")
    if isinstance(dev_mem, list) and dev_mem:
        markers.append(f"@pytest.mark.dev_mem_{str(dev_mem[0]).lower()}")
    markers.append(f"@pytest.mark.{case.flow.value}")

    target = dependency_target(case)
    if case.flow == ModelFlow.GET_MODEL:
        markers.append(f'@pytest.mark.dependency(name="{dependency_name(case)}")')
    elif target is not None:
        markers.append(
            "@pytest.mark.dependency(\n" f'    name="{dependency_name(case)}",\n' f'    depends_on=["{target}"],\n' ")"
        )

    helper = f"_{case.flow.value}_func"
    body = [
        *markers,
        f"def {case.function_name}(setup_logging) -> None:",
        f'    """{case.function_name}"""',
        f'    {helper}("{case.config.model_name}", setup_logging)',
    ]
    return "\n".join(body)


def render_flow_file(flow: ModelFlow, cases: list[GeneratedCase]) -> str:
    """Assemble the full text of one flow's test file from header and cases."""
    sections = [render_header(flow).rstrip()]
    sections.extend(render_case(case) for case in cases)
    return "\n\n\n".join(sections) + "\n"


def generated_outputs() -> dict[Path, str]:
    """Generate the full set of output files and their expected contents.

    Obsolete configs are included so toggling the obsolete flag does not rewrite
    the test surface; runtime handlers turn obsolete cases into pytest skips.
    """
    all_configs = tuple(CONFIG_REPOSITORY.iter_configs(include_obsolete=True))
    # Keep generated cases stable when a model is temporarily marked obsolete.
    # Runtime handlers already turn obsolete cases into pytest skips; removing
    # them here would make toggling the flag rewrite the test surface.
    cases = build_cases(all_configs)
    outputs = {SCRIPT_DIR / flow_test_filename(flow): render_flow_file(flow, cases[flow]) for flow in FLOW_ORDER}
    # Keep obsolete model markers registered as well. Both the marker list and
    # the generated cases are derived from the complete configuration set.
    markers = sorted({convert_model_name(config.model_name) for config in all_configs})
    outputs[SCRIPT_DIR / "model_names.txt"] = "\n".join(markers) + "\n"
    return outputs


def _test_function_names(text: str) -> set[str]:
    """Extract the set of ``test_*`` function names defined in a file's text."""
    import re

    return set(re.findall(r"^def (test_\w+)\(", text, re.M))


def _summarize_outdated(path: Path, actual: str, expected: str) -> dict:
    """Summarize a single out-of-date generated file without dumping its diff."""
    diff = list(
        difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(path),
            tofile=f"{path} (generated)",
            lineterm="",
        )
    )
    hunks = sum(1 for line in diff if line.startswith("@@"))
    added, removed = 0, 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    actual_names = _test_function_names(actual)
    expected_names = _test_function_names(expected)
    return {
        "hunks": hunks,
        "added_lines": added,
        "removed_lines": removed,
        "new_funcs": sorted(expected_names - actual_names),
        "gone_funcs": sorted(actual_names - expected_names),
    }


def check_outputs(outputs: dict[Path, str]) -> bool:
    """Compare generated outputs against files on disk and print a summary.

    Returns ``True`` when every output matches its file. Prints a compact
    per-file report (hunk counts, line deltas, added/removed test functions)
    instead of a full unified diff.
    """
    up_to_date, outdated = _classify_outputs(outputs)
    _print_output_summary(up_to_date, outdated)
    return not outdated


def _classify_outputs(outputs: dict[Path, str]):
    """Split generated outputs into matching and stale files."""
    up_to_date = []
    outdated = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if _output_matches(actual, expected):
            up_to_date.append(path)
        else:
            outdated.append((path, actual, expected))
    return up_to_date, outdated


def _print_output_summary(up_to_date, outdated) -> None:
    """Print compact status information for generated outputs."""
    if up_to_date:
        print(f"up-to-date ({len(up_to_date)}):")
        for path in up_to_date:
            print(f"  = {path.name}")
    if not outdated:
        return
    print(f"\nout-of-date ({len(outdated)}):")
    for path, actual, expected in outdated:
        _print_outdated_file(path, actual, expected)
    print("\nRe-run without --check to apply the changes above.")


def _print_outdated_file(path: Path, actual: str, expected: str) -> None:
    """Print one stale generated file and its compact diff summary."""
    info = _summarize_outdated(path, actual, expected)
    print(f"  ~ {path.name}: {info['hunks']} hunks, " f"+{info['added_lines']}/-{info['removed_lines']} lines")
    if info["new_funcs"]:
        print(f"      to add: {', '.join(info['new_funcs'])}")
    if info["gone_funcs"]:
        print(f"      to remove: {', '.join(info['gone_funcs'])}")


def _output_matches(actual: str, expected: str) -> bool:
    """Return whether one generated file already contains expected content."""
    return actual == expected


def write_outputs(outputs: dict[Path, str]) -> None:
    """Write every generated output to disk, logging each path written."""
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"generated {path}")


def main() -> int:
    """Entry point: ``--check`` verifies outputs are up to date, otherwise writes them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are out of date")
    args = parser.parse_args()
    outputs = generated_outputs()
    if args.check:
        return 0 if check_outputs(outputs) else 1
    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
