# Copyright 2025 HOUMO AI
#
# File: llm_eval.py
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
import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any, Dict, List

from .utils import logger


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


def import_model_module(model_ref: str) -> ModuleType:
    """Import the user-provided model script/module and trigger API registration."""
    is_py_file = model_ref.endswith(".py") or os.path.sep in model_ref
    if is_py_file:
        module_path = os.path.abspath(model_ref)
        if not os.path.isfile(module_path):
            raise FileNotFoundError(
                f"Model implementation script not found: {module_path}"
            )

        module_basename = os.path.splitext(os.path.basename(module_path))[0]
        module_name = f"hmatc_llm_eval_user_model_{module_basename}"

        script_dir = os.path.dirname(module_path)
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_items = [p for p in current_pythonpath.split(os.pathsep) if p]
        if script_dir and script_dir not in pythonpath_items:
            os.environ["PYTHONPATH"] = (
                script_dir
                if not current_pythonpath
                else f"{script_dir}{os.pathsep}{current_pythonpath}"
            )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load model script: {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(model_ref)


def run_llm_eval(args: Any) -> int:
    try:
        from evalscope import TaskConfig, run_task
    except ImportError as e:
        logger.error(
            "Failed to import evalscope. Please install hmatc dependencies including "
            "evalscope==1.5.2 and pydantic==2.12.5."
        )
        logger.error(str(e))
        return -1

    os.makedirs(args.output, exist_ok=True)

    try:
        custom_module = import_model_module(args.model)
    except Exception as e:
        logger.error(f"Failed to import model script: {e}")
        return -1

    if not hasattr(custom_module, "API_NAME"):
        logger.error(f"Model script {args.model} does not define API_NAME")
        return -1

    try:
        model_args = parse_model_args(args.model_args)
    except ValueError as e:
        logger.error(str(e))
        return -1

    model_dir = os.path.normpath(args.model_dir)
    model_args["model_dir"] = model_dir

    task_config = TaskConfig(
        model=model_dir,
        eval_type=getattr(custom_module, "API_NAME"),
        datasets=args.dataset,
        model_args=model_args,
        limit=None if args.limit == 0 else args.limit,
        work_dir=args.output,
    )
    run_task(task_cfg=task_config)
    return 0
