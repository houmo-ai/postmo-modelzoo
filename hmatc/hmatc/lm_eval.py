# Copyright 2025 HOUMO AI
#
# File: lm_eval.py
# Description:
#     EvalScope based large-model evaluation helpers for HMATC.
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
import copy
import logging
import os
from typing import Any, Dict, List

from .models import (
    get_model_spec,
    register_model_api,
    resolve_backend,
    validate_model_size,
)
from .utils import logger


class _EvalScopeHandler(logging.StreamHandler):
    def __init__(self) -> None:
        from .utils import console_handler

        super().__init__(console_handler.stream)
        super().setFormatter(console_handler.formatter)

    def setFormatter(self, formatter: logging.Formatter) -> None:
        from .utils import console_handler

        super().setFormatter(console_handler.formatter)

    def emit(self, record: logging.LogRecord) -> None:
        from .utils import console_handler

        console_handler.handle(copy.copy(record))


def parse_scalar_value(raw: str) -> Any:
    text = raw.strip()
    lower_text = text.lower()

    if lower_text == "true":
        return True
    if lower_text == "false":
        return False
    if lower_text in ("none", "null"):
        return None

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return text


def parse_model_args(items: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --model-args '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --model-args '{item}', key cannot be empty")
        parsed[key] = parse_scalar_value(value)
    return parsed


def _load_evalscope():
    from evalscope import TaskConfig, run_task
    from evalscope.utils.logger import configure_logging, get_logger

    configure_logging(debug=False)
    evalscope_logger = get_logger()
    for handler in evalscope_logger.handlers[:]:
        evalscope_logger.removeHandler(handler)
        handler.close()
    evalscope_logger.addHandler(_EvalScopeHandler())
    return TaskConfig, run_task


def run_lm_eval(args: Any) -> int:
    try:
        TaskConfig, run_task = _load_evalscope()
    except ImportError as e:
        logger.error(
            "Failed to import evalscope. Please install hmatc dependencies including "
            "evalscope==1.9.1 and pydantic==2.12.5."
        )
        logger.error(str(e))
        return -1

    try:
        model_args = parse_model_args(args.model_args)
        dataset_args = args.dataset_args
        if not isinstance(dataset_args, dict):
            raise ValueError("--dataset-args must be a JSON object")

        reserved = {"model_size", "backend"}.intersection(model_args)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"--model-args cannot override reserved arguments: {names}"
            )

        model_spec = get_model_spec(args.model_name)
        model_size = validate_model_size(model_spec, args.model_size)
        backend = resolve_backend(args.model, args.backend)
        register_model_api(model_spec)
        model_args["model_size"] = model_size
        model_args["backend"] = backend

        os.makedirs(args.output, exist_ok=True)
        task_config = TaskConfig(
            model=os.path.normpath(args.model),
            model_id=f"{backend}-{args.model_name}-{model_size}",
            eval_type=model_spec.api_name,
            datasets=args.dataset,
            dataset_args=dataset_args,
            model_args=model_args,
            limit=None if args.limit == 0 else args.limit,
            work_dir=args.output,
        )
        run_task(task_cfg=task_config)
    except Exception as e:
        logger.error(f"Large-model evaluation failed: {e}")
        return -1
    return 0
