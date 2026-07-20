# Copyright 2025 HOUMO AI
#
# File: factory.py
# Description:
#   Factory functions for eval Dataset instances.
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
import importlib.util
import inspect
import os
import sys

from ..utils import logger


def create_eval_dataset(eval_cfg, data_dir=None, num=0, config_dir=None):
    """Create the eval Dataset configured by eval.dataset_module/dataset_cls."""
    eval_cfg = eval_cfg or {}
    module_path = eval_cfg.get("dataset_module")
    class_name = eval_cfg.get("dataset_cls")
    if module_path is None and class_name is None:
        return None
    if module_path is None or class_name is None:
        logger.fatal("eval.dataset_module and eval.dataset_cls must be configured together")

    module_path = resolve_dataset_module_path(module_path, config_dir)
    Dataset = _import_dataset_cls(module_path, class_name)
    logger.info(f"Use eval Dataset: {class_name} ({module_path})")
    return _instantiate_dataset(Dataset, data_dir, num)


def resolve_dataset_module_path(module_path, config_dir=None):
    """Resolve dataset module path relative to config dir, then cwd."""
    if not module_path.endswith(".py"):
        module_path = f"{module_path}.py"

    candidates = []
    if os.path.isabs(module_path):
        candidates.append(module_path)
    else:
        if config_dir:
            candidates.append(os.path.join(config_dir, module_path))
        candidates.append(module_path)

    for path in candidates:
        if os.path.exists(path):
            return path

    logger.fatal(f"dataset_module not exists -> candidates: {candidates}")


def _import_dataset_cls(module_path, class_name):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        logger.fatal(f"module spec is None -> {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        logger.fatal(f"dataset_cls not found -> {class_name}")
    return getattr(module, class_name)


def _instantiate_dataset(Dataset, data_dir, num):
    """Instantiate Dataset using constructor signature.

    Supports:
    - ``data_dir`` / ``num`` constructors
    - ``root_path``-only constructors (built-in datasets)
    - ``**kwargs`` wrappers that forward to a parent constructor
    """
    signature = inspect.signature(Dataset)
    params = signature.parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs = {}
    path_param = _get_dataset_path_param(Dataset, params, has_var_keyword)
    if path_param is None:
        logger.fatal(
            f"Dataset {Dataset.__name__} constructor must accept "
            "data_dir, root_path, or **kwargs"
        )
    kwargs[path_param] = data_dir

    if "num" in params:
        kwargs["num"] = num

    try:
        return Dataset(**kwargs)
    except TypeError as err:
        logger.fatal(f"Failed to create Dataset {Dataset.__name__}: {err}")


def _get_dataset_path_param(Dataset, params, has_var_keyword):
    if "data_dir" in params:
        return "data_dir"
    if "root_path" in params:
        return "root_path"
    if not has_var_keyword:
        return None

    for cls in Dataset.__mro__[1:]:
        if cls is object:
            break
        parent_params = inspect.signature(cls).parameters
        if "data_dir" in parent_params:
            return "data_dir"
        if "root_path" in parent_params:
            return "root_path"
    return "root_path"
