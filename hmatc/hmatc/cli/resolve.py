# Copyright 2025 HOUMO AI
#
# File: resolve.py
# Description:
#     Resolve parsed HMATC CLI arguments into explicit command modes.
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
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandRequest:
    kind: str
    args: Namespace


def resolve_command_request(args: Namespace, parser: ArgumentParser) -> CommandRequest:
    command = args.command
    if command == "build":
        return CommandRequest(
            "build.hmonnx" if args.hmonnx is not None else "build.config", args
        )
    if command == "perf":
        return CommandRequest(
            "perf.model" if args.model is not None else "perf.config", args
        )
    if command == "check":
        if args.hmm is not None and not args.golden:
            parser.error("hmatc check --hmm requires --golden")
        return CommandRequest(
            "check.hmm" if args.hmm is not None else "check.config", args
        )
    if command == "eval":
        return _resolve_eval_request(args, parser)
    if command == "gen":
        return CommandRequest("gen.onnx", args)
    if command == "golden":
        return CommandRequest("golden.hmonnx", args)
    if command in {"quant", "compare", "demo", "benchmark"}:
        if not args.config:
            parser.error(f"hmatc {command} requires -c/--config")
        return CommandRequest(f"{command}.config", args)
    return CommandRequest(command, args)


def _resolve_eval_request(args: Namespace, parser: ArgumentParser) -> CommandRequest:
    llm_values = [args.model_name, args.model_size, args.model, args.dataset]
    has_llm_arg = any(value is not None for value in llm_values)
    has_all_llm_args = all(value is not None for value in llm_values)

    required_args = "--model-name, --model-size, --model, and --dataset"
    if args.config and has_llm_arg:
        parser.error(
            "hmatc eval uses either -c/--config for config-driven ONNX/small-model "
            f"evaluation or {required_args} for large-model evaluation, but not both."
        )

    if has_llm_arg:
        if not has_all_llm_args:
            parser.error(
                f"hmatc eval without -c/--config requires {required_args}"
            )
        return CommandRequest("eval.llm", args)

    if args.config:
        return CommandRequest("eval.config", args)

    parser.error(
        f"hmatc eval requires either -c/--config or {required_args}"
    )
