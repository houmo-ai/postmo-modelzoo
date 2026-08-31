# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Execution support for logical model package configurations."""

import importlib.util
import os
import sys

from ..utils import logger
from ..utils.check import check_cfg
from ..utils.utils import read_yaml_to_dict


class PackageExec:
    """Load package components and delegate end-to-end evaluation to a pipeline."""

    def __init__(self, cfg, config_path, target):
        self.cfg = cfg
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        self.target = target
        self.package_cfg = cfg.get("package") or {}
        self.eval_cfg = cfg.get("eval") or {}
        self.pipeline_cfg = cfg.get("pipeline") or {}
        self.component_cfgs = self._load_components(cfg.get("components"))

    def _load_components(self, components):
        if not isinstance(components, list) or not components:
            logger.fatal("[components] must be a non-empty list")

        loaded = {}
        for component in components:
            if not isinstance(component, dict):
                logger.fatal("Each package component must be a mapping")
            name = component.get("name")
            config_name = component.get("config")
            if not isinstance(name, str) or not name:
                logger.fatal("[components.name] must be a non-empty string")
            if name in loaded:
                logger.fatal(f"Duplicate package component: {name}")
            if not isinstance(config_name, str) or not config_name:
                logger.fatal(f"Component '{name}' is missing config")

            config_path = os.path.abspath(os.path.join(self.config_dir, config_name))
            if os.path.commonpath([self.config_dir, config_path]) != self.config_dir:
                logger.fatal(f"Component config escapes package directory: {config_name}")
            if not os.path.isfile(config_path):
                logger.fatal(f"Component config not found: {config_path}")

            component_cfg = read_yaml_to_dict(config_path)
            if component_cfg is None or component_cfg.get("kind") == "package":
                logger.fatal(f"Component must be a single-model config: {config_path}")
            if not check_cfg(component_cfg):
                logger.fatal(f"Invalid component config: {config_path}")
            component_cfg["target"] = self.target
            component_cfg["_config_dir"] = os.path.dirname(config_path)
            component_cfg["_config_path"] = config_path
            model_cfg = component_cfg["model"]
            model_path = model_cfg.get("model_path", "")
            if model_path and not os.path.isabs(model_path):
                model_cfg["model_path"] = os.path.abspath(
                    os.path.join(component_cfg["_config_dir"], model_path)
                )
            save_dir = model_cfg.get("save_dir", "output")
            if not os.path.isabs(save_dir):
                model_cfg["save_dir"] = os.path.abspath(
                    os.path.join(component_cfg["_config_dir"], save_dir)
                )
            loaded[name] = component_cfg
        return loaded

    def _load_pipeline(self):
        module_name = self.pipeline_cfg.get("module")
        class_name = self.pipeline_cfg.get("class")
        if not module_name or not class_name:
            logger.fatal("[pipeline.module] and [pipeline.class] are required")
        module_path = module_name if module_name.endswith(".py") else f"{module_name}.py"
        if not os.path.isabs(module_path):
            module_path = os.path.join(self.config_dir, module_path)
        if not os.path.isfile(module_path):
            logger.fatal(f"Pipeline module not found: {module_path}")

        import_name = f"package_pipeline_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(import_name, module_path)
        if spec is None or spec.loader is None:
            logger.fatal(f"Failed to load pipeline module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[import_name] = module
        if self.config_dir not in sys.path:
            sys.path.insert(0, self.config_dir)
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            logger.fatal(f"Pipeline class not found: {class_name}")
        return getattr(module, class_name)

    def evaluate(self, backend, device_id=0):
        if not self.eval_cfg:
            logger.fatal("[eval] section is required for package evaluation")
        Pipeline = self._load_pipeline()
        pipeline = Pipeline(
            package_cfg=self.cfg,
            component_cfgs=self.component_cfgs,
            config_dir=self.config_dir,
            target=self.target,
        )
        try:
            results = pipeline.evaluate(backend=backend, device_id=device_id)
        finally:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
        return {
            "eval": {
                backend: {
                    "success": True,
                    "results": results,
                }
            }
        }
