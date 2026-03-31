import argparse
import importlib
import importlib.util
import json
import os
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional
from evalscope import run_task, TaskConfig
from evalscope.api.tool.utils import logger

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    # fmt: off
    parser = argparse.ArgumentParser(prog="hmeval", description="Large model evaluation command line tool")
    parser.add_argument("--model", type=str, required=True, help="Model implementation script path (.py) or module name")
    parser.add_argument("--model-dir", type=str, required=True, help="Model artifact directory")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names/paths, e.g. --dataset mmlu gsm8k",
    )
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of samples to evaluate, 0 means full dataset")
    parser.add_argument(
        "--model-args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra model args passed to TaskConfig.model_args. "
            "Repeat this option for multiple values, e.g. --model-args temperature=0.7"
        ),
    )
    parser.add_argument("--version", "-v", action="version", version=__version__, help="Show version")
    # fmt: on
    return parser


def _parse_scalar_value(raw: str) -> Any:
    text = raw.strip()
    lower_text = text.lower()

    if lower_text == "true":
        return True
    if lower_text == "false":
        return False
    if lower_text == "none" or lower_text == "null":
        return None

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _parse_model_args(items: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --model-arg '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --model-arg '{item}', key cannot be empty")
        parsed[key] = _parse_scalar_value(value)
    return parsed


def _import_model_module(model_ref: str) -> ModuleType:
    """Import the user-provided model script/module and trigger register_model_api registration."""
    is_py_file = model_ref.endswith(".py") or os.path.sep in model_ref
    if is_py_file:
        module_path = os.path.abspath(model_ref)
        if not os.path.isfile(module_path):
            raise FileNotFoundError(
                f"Model implementation script not found: {module_path}"
            )

        module_basename = os.path.splitext(os.path.basename(module_path))[0]
        module_name = f"hmeval_user_model_{module_basename}"

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


def _run_eval(args: argparse.Namespace) -> int:
    os.makedirs(args.output, exist_ok=True)

    try:
        custom_module = _import_model_module(args.model)
    except Exception as e:
        logger.error(f"Failed to import model script: {e}")
        return -1

    if not hasattr(custom_module, "API_NAME"):
        logger.error(f"Model script {args.model} does not define API_NAME")
        return -1

    eval_type = getattr(custom_module, "API_NAME")
    try:
        model_args = _parse_model_args(args.model_args)
    except ValueError as e:
        logger.error(str(e))
        return -1

    if args.model_dir:
        args.model_dir = os.path.normpath(args.model_dir)
        model_args["model_dir"] = args.model_dir

    task_config = TaskConfig(
        model=args.model_dir,
        eval_type=eval_type,
        datasets=args.dataset,
        model_args=model_args,
        limit=None if args.limit == 0 else args.limit,
    )
    _ = run_task(task_cfg=task_config)
    return 0


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_eval(args)
