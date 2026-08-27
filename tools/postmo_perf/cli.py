# Copyright (c) 2026 HOUMO AI
#
# File: cli.py
# Description:
#   Command-line interface for fixed-length PostMo performance tests.
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

"""Command line entry point for fixed-length PostMo performance tests."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .config import PerfCase, PerfSettings, load_config
    from .dumper import dump_llm_perf_results
    from .formatter import format_case
    from .runner import PerfRunner
else:
    # Direct script execution does not provide a package context. Add the
    # repository roots before importing through the canonical package name.
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    _UTILS_PYTHON = _REPOSITORY_ROOT / "utils" / "python"
    for path in (_REPOSITORY_ROOT, _UTILS_PYTHON):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from tools.postmo_perf.config import PerfCase, PerfSettings, load_config
    from tools.postmo_perf.dumper import dump_llm_perf_results
    from tools.postmo_perf.formatter import format_case
    from tools.postmo_perf.runner import PerfRunner


def _settings_from_args(args, parser: argparse.ArgumentParser) -> PerfSettings:
    direct_values = (
        args.prefill,
        args.decode,
        args.model_dir,
        args.input_tokens,
        args.output_tokens,
        args.dump_file,
    )
    if args.config:
        if any(value is not None for value in direct_values):
            parser.error("config cannot be combined with direct model or case arguments")
        return load_config(args.config)
    missing = [
        option
        for option, value in (
            ("--prefill", args.prefill),
            ("--decode", args.decode),
            ("--input-tokens", args.input_tokens),
            ("--output-tokens", args.output_tokens),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "direct mode requires " + ", ".join(missing) + "; alternatively provide a YAML config"
        )
    prefill = Path(args.prefill).resolve()
    decode = Path(args.decode).resolve()
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    elif prefill.parent == decode.parent:
        model_dir = prefill.parent
    else:
        parser.error("--model-dir is required when Prefill and Decode are in different directories")
    return PerfSettings(
        cases=(
            PerfCase(
                model_dir=model_dir,
                prefill=prefill,
                decode=decode,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                loop=args.loop,
                warmup=args.warmup,
                seed=args.seed,
                model_name=args.model_name,
            ),
        ),
        dump_file=Path(args.dump_file).resolve() if args.dump_file else None,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-length PostMo device performance tests")
    parser.add_argument("config", nargs="?", help="YAML performance configuration")
    parser.add_argument("--prefill", help="Prefill HMM model path")
    parser.add_argument("--decode", help="Decode HMM model path")
    parser.add_argument(
        "--model-dir",
        help="Model asset root containing hmquant/hf_config and quant_embedding.pt",
    )
    parser.add_argument("--input-tokens", "--input_tokens", type=int, help="Fixed Prefill token count")
    parser.add_argument("--output-tokens", "--output_tokens", type=int, help="Fixed Decode step count")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default: 1)")
    parser.add_argument("--loop", type=int, default=1, help="Formal iterations (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Fixed Token ID seed (default: 0)")
    parser.add_argument("--model-name", default="qwen3.5", help="Model name in reports")
    parser.add_argument("--dump-file", help="YAML result path")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress output",
    )
    args = parser.parse_args(argv)
    settings = _settings_from_args(args, parser)
    results = []
    for case in settings.cases:
        result = PerfRunner(case).run(progress=False if args.no_progress else None)
        results.append(result)
        print(format_case(result))
    if settings.dump_file:
        dump_llm_perf_results(tuple(results), settings.dump_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
