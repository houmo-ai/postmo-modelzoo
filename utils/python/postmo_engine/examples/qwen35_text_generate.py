# Copyright (c) 2026 HOUMO AI
#
# File: qwen35_text_generate.py
# Description:
#   Command-line example for Qwen3.5 Text-only generation.
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

"""Minimal Qwen3.5 Text-only generate example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from postmo_engine.core import EngineRequest
from postmo_engine.engine import Qwen35Engine
from postmo_engine.perf import dump_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 Text-only generation")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parents[1] / "models" / "qwen3.5-0.8b"),
        help="Directory containing *_prefill.hmm, *_decode.hmm and hmquant/",
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument(
        "--dump",
        type=Path,
        help="Write the performance report to the specified YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    engine = Qwen35Engine.from_model_dir(
        model_dir,
        perf=args.perf or args.dump is not None,
        aggregate_parents=True,
    )
    request = EngineRequest(request_id="example-1", prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    for chunk in engine.generate(request):
        if chunk.text_delta:
            print(chunk.text_delta, end="", flush=True)
        if chunk.is_final:
            print()
            print(f"stop_reason={chunk.stop_reason}")
    if engine.last_result is not None:
        print(engine.last_result)
    if args.perf:
        engine.perf.print_summary(time_unit="min")
    if args.dump is not None:
        dump_yaml(
            engine.perf.summary(),
            args.dump,
            model_name=model_dir.name,
        )
        print(f"perf_yaml={args.dump}")


if __name__ == "__main__":
    main()
