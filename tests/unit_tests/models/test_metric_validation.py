# Copyright (c) 2026 HOUMO AI
#
# File: test_metric_validation.py
# Description:
#  Unit tests for compile, compare, eval, and performance metric extraction
#    and threshold validation.
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

"""Unit tests extracted from the former model-flow contract suite: test_metric_validation.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandResult,
    CommandSpec,
    DiagnosticContext,
    FlowRequest,
    ModelFamily,
    ModelFlow,
)
from tests.models_tests.model_workflow.model_config_repository import (
    ModelConfigRepository,
)
from tests.models_tests.model_workflow.perf_metric_validation import (
    _extract_overall_metric,
    extract_perf_metrics,
    resolve_perf_behavior,
    validate_perf_metrics,
)
from tests.models_tests.test_flows.compare_flow import (
    _compare_output_passed,
)
from tests.models_tests.test_flows.compile_flow import (
    _compile_output_passed,
)
from tests.models_tests.test_flows.eval_flow import (
    _validate_eval_outputs,
)
from tests.models_tests.test_flows.perf_flow import (
    _run_custom_perf_case,
    _run_python_perf_case,
    _select_perf_text,
)
from types import (
    SimpleNamespace,
)
from ._flow_contract_support import (
    CONFIG_DIR,
)

pytestmark = pytest.mark.unit


def test_default_perf_rules_extract_common_demo_metrics() -> None:
    behavior = resolve_perf_behavior(
        "default-demo",
        backend="xh2",
        baseline_keys=("prefill", "decode", "end2end"),
        default_runner="demo",
        default_source="stdout",
    )
    actual = extract_perf_metrics(
        "Prefill Speed: 120.5 tokens/s\n"
        "Decode Speed: 47.8 tokens/s\n"
        "E2E TPS: 11.2 tokens/s\n",
        behavior,
    )
    assert actual == {"prefill": 120.5, "decode": 47.8, "end2end": 11.2}
    assert validate_perf_metrics(
        actual,
        {"prefill": 100.0, "decode": 45.0, "end2end": 10.0},
        behavior,
        minimum_ratio=0.95,
    ).passed


def test_default_perf_rules_extract_timing_and_overall_metrics() -> None:
    behavior = resolve_perf_behavior(
        "structured-demo",
        backend="xh2",
        baseline_keys=("prefill", "decode", "end2end"),
        default_runner="demo",
        default_source="stdout",
    )
    text = """
Timing
prefill_load       1    7669.925  7669.925  7669.925  7669.925                -
prefill             1   1378.053  1378.053  1378.053  1378.053  587.06 tokens/s
  infer             4   1317.884   329.471   237.329   362.842  613.86 tokens/s
decode            420  18055.749    42.990    40.627    43.687   23.26 tokens/s
  infer             420  15118.332    35.996    35.411    36.587  27.78 tokens/s

Overall Performance Metrics
E2E TPS (Throughput): 20.56 tokens/s
"""
    assert extract_perf_metrics(text, behavior) == {
        "prefill": 613.86,
        "decode": 27.78,
        "end2end": 20.56,
    }


def test_overall_metric_accepts_parenthesized_end2end_label() -> None:
    text = "E2E TPS (Throughput): 20.56 tokens/s\n"
    assert _extract_overall_metric(text, "end2end") == [20.56]


def test_timing_perf_rules_match_configured_vision_scope() -> None:
    behavior = resolve_perf_behavior(
        "structured-vlm",
        backend="xh2",
        baseline_keys=("vision",),
        default_runner="demo",
    )
    text = """
Timing
vision              1    769.335   769.335   769.335   769.335    1.30 images/s
  infer             1    751.099   751.099   751.099   751.099    1.33 images/s
"""
    assert extract_perf_metrics(text, behavior) == {"vision": 1.33}


def test_structured_perf_metrics_can_report_missing_sections() -> None:
    behavior = resolve_perf_behavior(
        "structured-demo",
        backend="xh2",
        baseline_keys=("prefill", "decode", "end2end"),
        default_runner="demo",
    )
    with pytest.raises(Exception, match="end2end"):
        extract_perf_metrics("prefill 1 1 1 1 1 10 tokens/s\n", behavior)


def test_sdxl_code_override_extracts_lower_is_better_latency() -> None:
    behavior = resolve_perf_behavior(
        "sdxl",
        backend="xh1",
        baseline_keys=("avg_cost",),
        default_runner="demo",
    )
    assert extract_perf_metrics("18.2 ms, average", behavior) == {"avg_cost": 18.2}
    with pytest.raises(Exception, match="Failed to extract"):
        extract_perf_metrics("no metrics here", behavior)


def test_custom_script_perf_is_maintained_by_code_override(tmp_path: Path) -> None:
    wenet = ModelConfigRepository(CONFIG_DIR).load("wenet")
    _, wenet_behavior = wenet.perf_contract("xh1", "x86_64")
    assert wenet_behavior.runner == "custom_script"
    assert wenet_behavior.custom_script is not None
    assert extract_perf_metrics("infer completed: 61.2 qps", wenet_behavior) == {
        "qps": 61.2
    }

    assert wenet_behavior.custom_script is not None
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "wenet-perf-run",
            "wenet",
            ModelFamily.CV,
            "xh1",
            ModelFlow.PERF,
        ),
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        log_file=tmp_path / "perf.log",
    )
    (tmp_path / "build.py").write_text("# root perf\n", encoding="utf-8")
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "build.py").write_text("# perf\n", encoding="utf-8")

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            assert command.argv == (
                "python3",
                "python/build.py",
                "--model_dir",
                str(tmp_path / "results" / "hmquant_xh1"),
            )
            return CommandResult(command, 0, "infer completed: 61.2 qps\n", "", 0.1)

    results, failures = _run_custom_perf_case(
        SimpleNamespace(config=wenet, context=context),
        SimpleNamespace(command_runner=FakeRunner()),
        tmp_path,
        "python3",
        wenet_behavior.custom_script,
    )
    assert not failures
    actual = extract_perf_metrics(results[0].stdout, wenet_behavior)
    assert actual == {"qps": 61.2}


def test_demo_perf_prefers_script_in_python_directory(tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("# root demo\n", encoding="utf-8")
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "demo.py").write_text("# python demo\n", encoding="utf-8")
    config = SimpleNamespace(
        model_name="demo-perf",
        backend_section=lambda name, backend: {"script": ["demo.py"]},
    )
    context = SimpleNamespace(
        diagnostic=DiagnosticContext(
            "demo-perf", "demo-perf", ModelFamily.LLM, "xh2", ModelFlow.PERF
        ),
        model_cache_dir=tmp_path / "models",
        result_cache_dir=tmp_path / "results",
        log_file=tmp_path / "perf.log",
    )

    class FakeRunner:
        def run(self, command, *, diagnostic_fields=None):
            assert command.argv[:2] == ("python3", "python/demo.py")
            return CommandResult(command, 0, "ok\n", "", 0.1)

    results, failures = _run_python_perf_case(
        FlowRequest(context, config),
        SimpleNamespace(command_runner=FakeRunner()),
        tmp_path,
        "python3",
    )

    assert len(results) == 1
    assert failures == []


def test_hmatc_perf_aggregates_max_qps_across_all_cases(tmp_path: Path) -> None:
    behavior = resolve_perf_behavior(
        "default-hmatc",
        backend="xh2",
        baseline_keys=("qps",),
        default_runner="hmatc",
        default_source="stdout",
    )
    first = CommandResult(
        CommandSpec("perf[0]", ("hmatc", "perf")),
        0,
        "[Throughput] qps: 91.5\n",
        "",
        0.1,
    )
    second = CommandResult(
        CommandSpec("perf[1]", ("hmatc", "perf")),
        0,
        "[Throughput] qps: 113.2\n",
        "",
        0.1,
    )
    text = _select_perf_text("stdout", [first, second], tmp_path / "unused.log")
    assert extract_perf_metrics(text, behavior) == {"qps": 113.2}


def test_eval_keeps_partial_dataset_factor_and_requires_all_metrics(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HOUMO_FULL_DATASET", raising=False)
    onnx = CommandResult(
        CommandSpec("onnx", ("hmatc", "eval")),
        0,
        "{'top1_acc': '0.8'}\n",
        "",
        0.1,
    )
    hm = CommandResult(
        CommandSpec("hm", ("hmatc", "eval")),
        0,
        "{'top1_acc': '0.39'}\n",
        "",
        0.1,
    )
    _, failures = _validate_eval_outputs([onnx], [hm], {"top1_acc": 0.95})
    assert not failures  # 0.39 >= 0.8 * 0.95 * 0.5

    missing = CommandResult(
        CommandSpec("missing", ("hmatc", "eval")), 0, "{}\n", "", 0.1
    )
    _, failures = _validate_eval_outputs([onnx], [missing], {"top1_acc": 0.95})
    assert failures == ["HM eval case 0 missing metric top1_acc"]


def test_compile_and_compare_parsers_use_supplied_threshold() -> None:
    compile_output = "\n".join(
        [
            "| model | cosine_dist |",
            "| demo.onnx | 0.75 |",
        ]
    )
    assert _compile_output_passed(compile_output, "xh2", 0.76) is False
    assert _compile_output_passed(compile_output, "xh2", 0.75) is True

    compare_output = "\n".join(
        [
            "| Cosine Distance |",
            "| name | onnx vs hmquant | onnx vs hmm | hmquant vs hmm |",
            "| output | 0.99 | 0.95 | 0.89 |",
        ]
    )
    assert _compare_output_passed(compare_output, "xh2", 0.90) is False
    assert _compare_output_passed(compare_output, "xh2", 0.89) is True

    xh1_compile_output = "\n".join(
        [
            "| model | cosine_dist | golden | cosine_dist_2 | golden_2 |",
            "| demo.onnx | 0.99 | Pass | 0.98 | Pass |",
        ]
    )
    assert _compile_output_passed(xh1_compile_output, "xh1", 0.99) is False
    assert _compile_output_passed(xh1_compile_output, "xh1", 0.98) is True
