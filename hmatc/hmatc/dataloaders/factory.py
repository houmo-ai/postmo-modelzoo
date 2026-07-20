# Copyright 2025 HOUMO AI
#
# File: factory.py
# Description:
#   Factory function for creating DataLoader instances.
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
from ..utils.utils import SUPPORT_IMAGE_FORMATS
from .loaders import ImageDataLoader, NpzDataLoader, RandomDataLoader
from .imagenet import ImageNetDataLoader
from .coco import CocoDataLoader
from .widerface import WiderFaceDataLoader
from .ccpd import CCPDImageDataLoader
from .sintel import SintelDataLoader


def create_dataloader(model_cfg, data_dir=None, stage=None, num=0, dataset=None):
    """Create a custom or built-in DataLoader for one stage."""
    model_cfg = model_cfg or {}
    inputs_cfg = model_cfg.get("inputs") or {}

    dataloader_cls = _get_custom_dataloader_cls(model_cfg)
    if dataloader_cls is not None:
        logger.info(f"Use custom DataLoader: {dataloader_cls.__name__} ({stage})")
        kwargs = {
            "data_dir": data_dir,
            "model_cfg": model_cfg,
            "inputs_cfg": inputs_cfg,
            "stage": stage,
            "num": num,
        }
        if "dataset" in inspect.signature(dataloader_cls).parameters:
            kwargs["dataset"] = dataset
        return dataloader_cls(**kwargs)

    if stage == "quant" and data_dir is None:
        logger.info("Use RandomDataLoader (quant)")
        return RandomDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if _has_npz(data_dir):
        logger.info(f"Use NpzDataLoader ({stage}): {data_dir}")
        return NpzDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if stage == "eval" and ImageNetDataLoader.matches(data_dir):
        logger.info(f"Use ImageNetDataLoader ({stage}): {data_dir}")
        return ImageNetDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if stage == "eval" and CocoDataLoader.matches(data_dir):
        logger.info(f"Use CocoDataLoader ({stage}): {data_dir}")
        return CocoDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if stage == "eval" and WiderFaceDataLoader.matches(data_dir):
        logger.info(f"Use WiderFaceDataLoader ({stage}): {data_dir}")
        return WiderFaceDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if stage == "eval" and CCPDImageDataLoader.matches(data_dir):
        logger.info(f"Use CCPDImageDataLoader ({stage}): {data_dir}")
        return CCPDImageDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if SintelDataLoader.matches(data_dir):
        logger.info(f"Use SintelDataLoader ({stage}): {data_dir}")
        return SintelDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if stage == "demo" and _is_two_image_model(inputs_cfg) and _has_image(data_dir):
        logger.info(f"Use SintelDataLoader ({stage}): {data_dir}")
        return SintelDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    if _is_single_image_model(inputs_cfg) and _has_image(data_dir):
        logger.info(f"Use ImageDataLoader ({stage}): {data_dir}")
        return ImageDataLoader(data_dir, model_cfg, inputs_cfg, stage, num, dataset)

    logger.fatal(
        "No built-in DataLoader matches the data. "
        "Please configure model.dataloader_module and model.dataloader_cls."
    )


def _get_custom_dataloader_cls(model_cfg):
    module_path = model_cfg.get("dataloader_module")
    class_name = model_cfg.get("dataloader_cls")
    if module_path is None and class_name is None:
        return None
    if module_path is None or class_name is None:
        logger.fatal(
            "model.dataloader_module and model.dataloader_cls must be configured together"
        )

    if not module_path.endswith(".py"):
        module_path = f"{module_path}.py"
    if not os.path.exists(module_path):
        logger.fatal(f"dataloader_module not exists -> {module_path}")

    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        logger.fatal(f"module spec is None -> {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        logger.fatal(f"dataloader_cls not found -> {class_name}")
    logger.info(f"from {module_path} import {class_name} successfully")
    return getattr(module, class_name)


def _has_npz(data_dir):
    return _has_file(data_dir, [".npz"])


def _has_image(data_dir):
    return _has_file(data_dir, SUPPORT_IMAGE_FORMATS)


def _has_file(data_dir, exts):
    if data_dir is None:
        return False
    if os.path.isfile(data_dir):
        return os.path.splitext(data_dir)[1] in exts
    if not os.path.isdir(data_dir):
        return False
    return any(os.path.splitext(name)[1] in exts for name in os.listdir(data_dir))


def _is_single_image_model(inputs_cfg):
    if len(inputs_cfg) != 1:
        return False
    input_cfg = next(iter(inputs_cfg.values()))
    return input_cfg.get("data_format") is not None


def _is_two_image_model(inputs_cfg):
    if len(inputs_cfg) != 2:
        return False
    return all(
        input_cfg.get("data_format") is not None for input_cfg in inputs_cfg.values()
    )
