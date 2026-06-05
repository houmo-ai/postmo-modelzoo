# Copyright 2025 HOUMO AI
#
# File: base_exec.py
# Description:
#   Base execution class
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
import abc
import os
import sys
import onnx
import json
import importlib
import numpy as np
import psutil
import torch
from onnx import StringStringEntryProto
from pathlib import Path
from datetime import datetime
from ..utils import logger
from ..utils.utils import get_onnx_inputs_info

QUANTIZATION_RANGES = {
    "int8": (-128, 127),  # 8-bit signed integer
    "uint8": (0, 255),  # 8-bit unsigned integer
    "int16": (-32768, 32767),  # 16-bit signed integer
    "uint16": (0, 65535),  # 16-bit unsigned integer
    "int32": (-2147483648, 2147483647),  # 32-bit signed integer
}


class BaseExec(object, metaclass=abc.ABCMeta):
    """Base execution class for model processing operations.

    This abstract class handles common operations for model quantization, compilation,
    and evaluation, including device configuration, model path management, input/output
    handling, and performance optimization parameters.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the execution instance with configuration.

        Args:
            cfg (dict): Configuration dictionary containing model, quantization,
                       build, demo, and evaluation parameters.
        """
        # Basic parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target = cfg["target"]
        self.enable_upload = False

        # Model parameters
        self.model_cfg = cfg.get("model")
        self.save_dir = self.model_cfg.get("save_dir")
        self.model_path = self.model_cfg.get("model_path", "")
        HOUMO_MODEL_PATH = os.environ.get("HOUMO_MODEL_PATH", "")
        if not os.path.isfile(self.model_path):
            new_model_path = os.path.join(HOUMO_MODEL_PATH, self.model_path)
            if not os.path.isfile(new_model_path):
                logger.fatal(f"Not found model_path: {self.model_path}")
            self.model_path = new_model_path
        logger.info(f"model_path: {self.model_path}")
        self.onnx_inputs_info, self.onnx_outputs_info = get_onnx_inputs_info(
            self.model_path
        )
        self.onnx_is_static = True  # Initially True
        self.model_name = self.model_cfg.get("name", "model")  # Compiled model name
        self.upload_dir_name = (
            Path.cwd().name
        )  # Upload directory name (defaults to current dir name)
        self.inputs_cfg = self.model_cfg.get("inputs")
        self.check_input_shape()
        self.model_inputs_batch = dict()
        self.inputs_name = list()
        self.data_formats = list()
        self.inputs_shape = list()
        # Resizer related parameters
        # Per-input resizer mode:
        # 0 - Non-image input or disabled resizer
        # 1 - Fully dynamic resizer (10 params: [y, x, height, width, h, w, top, left, bottom, right])
        # 2 - Crop partial dynamic resizer (4 params: [y, x, height, width])
        # 3 - Static resizer
        self.resizer_modes = {}  # Per-input resizer mode
        self.resizers_cfg = {}
        self.max_inputs_size = dict()
        self.has_resizer = False  # Whether any input has resizer
        self.has_dynamic_resizer = False
        self.has_dynamic_v1_resizer = False
        self.has_dynamic_v2_resizer = False
        self.has_static_resizer = False

        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            shape = input_cfg["shape"]
            self.inputs_shape.append(shape)
            model_batch = shape[0]
            data_format = input_cfg.get("data_format")
            self.model_inputs_batch[input_name] = model_batch
            self.inputs_name.append(input_name)
            self.data_formats.append(data_format)
            if data_format is None or "resizer" not in input_cfg:
                self.resizer_modes[input_name] = 0
                continue
            self.has_resizer = True
            resizer_cfg = input_cfg["resizer"]
            if resizer_cfg is None:
                resizer_cfg = dict()
            resizer_mode = resizer_cfg.get("resizer_mode", 3)
            self.resizer_modes[input_name] = resizer_mode
            if resizer_mode == 3:
                self.has_static_resizer = True
            elif resizer_mode == 2:
                self.has_dynamic_v1_resizer = True
                self.has_dynamic_resizer = True
            elif resizer_mode == 1:
                self.has_dynamic_v2_resizer = True
                self.has_dynamic_resizer = True
        logger.info(f"resizer_modes: {self.resizer_modes}")

        self.model_input_batch = self.model_inputs_batch[self.inputs_name[0]]
        # Whether multiple inputs
        self.is_multi_input_model = len(self.inputs_cfg) > 1
        # Whether single input and is image
        self.is_image_single_input = (
            not self.is_multi_input_model and self.data_formats[0] is not None
        )

        # Quantization parameters
        self.quant_cfg = cfg.get("quant", dict())
        self.quant_advance_cfg = self.quant_cfg.get("config", dict())
        self.calib_data = self.quant_cfg.get("calib_data")
        self.use_random_data = (
            self.calib_data is None
        )  # Use random data for quantization
        self.quant_output_dir = os.path.join(self.save_dir, self.target, "hmquant")
        self.hmm_save_dir = os.path.join(self.save_dir, self.target)
        if not os.path.exists(self.hmm_save_dir):
            os.makedirs(self.hmm_save_dir)
        self.quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"hmquant_{self.model_name}_with_act.onnx"
        )

        # Build parameters
        self.build_cfg = cfg.get("build", dict())
        self.build_parallel_jobs = self.build_cfg.get(
            "parallel_jobs", psutil.cpu_count(logical=False)
        )
        self.build_batch = self.build_cfg.get("batch", 1)
        self.build_ncore = self.build_cfg.get("ncore", 1)
        self.build_opt_level = self.build_cfg.get("opt_level", 2)
        self.build_opt_level = f"O{self.build_opt_level}"
        self.cpp_backend = self.build_cfg.get("cpp_backend", "v1")
        self.dump_compiled_mlir = self.build_cfg.get("dump_compiled_mlir", False)
        # Override upload_dir_name if provided via command line
        if self.build_cfg.get("upload_dir_name"):
            self.upload_dir_name = self.build_cfg.get("upload_dir_name")
        # File prefix for compressed file name (defaults to upload_dir_name)
        self.file_prefix = self.build_cfg.get("file_prefix", self.upload_dir_name)
        self.build_output_dir = os.path.join(self.save_dir, self.target, "tcim")
        self.hmm_batch = self.build_batch * self.model_input_batch
        # ROI number
        # Used for dynamic resizer mode, allowing multiple ROIs from one image
        # roi_num > 1: one input image -> multiple cropped ROIs -> multiple outputs
        # Constraints:
        #   - Only valid when hmm_batch == 1 (model_input_batch * build_batch == 1)
        #   - Only valid for image single input with dynamic resizer enabled
        self.roi_num = self.build_cfg.get("roi_num", 1)
        if not isinstance(self.roi_num, int) or self.roi_num < 1:
            logger.fatal("[build.roi_num] must be int and >= 1")
        if not self.has_dynamic_resizer:
            self.roi_num = 1
        if self.roi_num > 1 and self.hmm_batch > 1:
            logger.fatal(
                "[build.roi_num] > 1 requires model_input_batch * build_batch == 1"
            )

        # Determine HMM naming based on resizer configuration
        # For single-input: use that input's mode
        # For multi-input: use dynamic if any input is dynamic, static if all are static
        suffix = ""
        roi_tag = ""
        if (
            self.has_dynamic_v1_resizer
            and self.has_dynamic_v2_resizer
            and self.has_static_resizer
        ):
            suffix = "_static_dynamic_v1+v2"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_static_resizer and self.has_dynamic_v1_resizer:
            suffix = "_static_dynamic_v1"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_static_resizer and self.has_dynamic_v2_resizer:
            suffix = "_static_dynamic_v2"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_dynamic_v1_resizer and self.has_dynamic_v2_resizer:
            suffix = "_dynamic_v1+v2"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_dynamic_v2_resizer:
            suffix = "_dynamic_v2"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_dynamic_v1_resizer:
            suffix = "_dynamic_v1"
            roi_tag = f"_{self.roi_num}roi"
        elif self.has_static_resizer:
            suffix = "_static"
            roi_tag = "_1roi"
        self.resizer_mode: str = suffix[1:] if suffix else "No Resizer"
        self.resizer_mode = self.resizer_mode.upper()
        self.hmm_name = f"{self.model_name}_{self.target}_b{self.hmm_batch}{roi_tag}_{self.build_ncore}core_{self.build_opt_level}{suffix}"
        self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")

        # For passing preprocessing information to hmm
        self.custom_msg = dict()
        for name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[name]
            self.custom_msg[name] = dict(
                shape=input_cfg["shape"],
                resizer_mode=self.resizer_modes.get(name, 0),
                input_cfg=input_cfg,
            )

        # Model evaluation parameters
        self.demo_cfg = cfg.get("demo", dict())
        self.eval_cfg = cfg.get("eval", dict())

        # Graph optimization module
        if "app_onnx_opt" in cfg["model"]:
            from ..optimizer.onnx_opt_engine import HMAppOnnxOptConvert

            self.ApplicationOnnxOpt = HMAppOnnxOptConvert(cfg)

    def check_input_shape(self):
        """
        Check if the configured shape matches the ONNX model.
        """
        for input_name in self.inputs_cfg:
            onnx_shape = self.onnx_inputs_info[input_name]["shape"]
            cfg_shape = self.inputs_cfg[input_name]["shape"]
            for idx, val in enumerate(onnx_shape):
                if (
                    val < 0
                    or val is None
                    or isinstance(val, str)
                    or not isinstance(val, int)
                ):
                    self.onnx_is_static = False
                else:
                    # Check if the configured shape matches the ONNX model
                    if val != cfg_shape[idx]:
                        logger.fatal(
                            f"onnx shape {onnx_shape} is not equal to cfg shape {cfg_shape}"
                        )

    @staticmethod
    def dtype_transform(dtype):
        """Convert data type string to feature string.

        Args:
            dtype (str): Input data type string.

        Returns:
            str: Corresponding feature string.
        """
        if dtype == "float32":
            return "Float32Feature"
        elif dtype == "float16":
            return "Float16Feature"
        elif dtype == "float64":
            return "Float64Feature"
        elif dtype == "int8":
            return "Int8Feature"
        elif dtype == "uint8":
            return "Uint8Feature"
        elif dtype == "int16":
            return "Int16Feature"
        else:
            logger.fatal(f"Not support dtype: {dtype}")

    @abc.abstractmethod
    def quantize(self):
        """Perform model quantization."""
        pass

    @abc.abstractmethod
    def build(self):
        """Compile the model."""
        pass

    @abc.abstractmethod
    def compare(self):
        """Compare model similarity."""
        pass

    @staticmethod
    def import_py_module_from_file(module_path: str, module_cls: str):
        """Import a Python module from file.

        Args:
            module_path (str): Path to the module file.
            module_cls (str): Class name to import from the module.

        Returns:
            class: The imported class object.
        """
        module_name = os.path.splitext(os.path.basename(module_path))[1]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logger.fatal(f"module spec is None -> {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, module_cls):
            logger.fatal(f"module_cls not found -> {module_cls}")
        return getattr(module, module_cls)

    def get_model(self, backend):
        """Get model instance.

        Args:
            backend (str): Backend type for the model.

        Returns:
            object: Model instance or None if failed.
        """
        model_impl_module = self.model_cfg.get("model_impl_module")
        model_impl_cls = self.model_cfg.get("model_impl_cls")
        if model_impl_module is None or model_impl_cls is None:
            logger.error("model_impl_module or model_impl_cls is None")
            return None
        model_impl_module_path = f"{model_impl_module}.py"
        if not os.path.exists(model_impl_module_path):
            logger.error(f"model_impl_module not exists -> {model_impl_module_path}")
            return None
        Model = self.import_py_module_from_file(model_impl_module_path, model_impl_cls)
        logger.info(f"from {model_impl_module} import {model_impl_cls} successfully")
        return Model(
            inputs_cfg=self.inputs_cfg,
            is_image_single_input=self.is_image_single_input,
            resizer_modes=self.resizer_modes,
            roi_num=self.roi_num,
            backend=backend,
        )

    def get_dataset(self, data_dir):
        """Get dataset instance.

        Args:
            data_dir (str): Directory containing the dataset.

        Returns:
            object: Dataset instance or None if failed.
        """
        dataset_module = self.eval_cfg.get("dataset_module")
        dataset_cls = self.eval_cfg.get("dataset_cls")
        if dataset_module is None or dataset_cls is None:
            logger.error("dataset_module or dataset_cls is None")
            return None
        module_path = f"{dataset_module}.py"
        if not os.path.exists(module_path):
            logger.error(f"dataset_module not exists -> {dataset_module}")
            return None
        Dataset = self.import_py_module_from_file(module_path, dataset_cls)
        logger.info(f"from {dataset_module} import {dataset_cls} successfully")
        return Dataset(root_path=data_dir)

    def demo(self, backend, device_id=0):
        """Demo entry point.

        Args:
            backend (str): Backend type for the demo.
            device_id (int): Device ID to run the demo on.

        Returns:
            dict: Result dictionary with success status and backend info.
        """
        if not self.demo_cfg:
            logger.error("demo config not found")
            return {
                "demo": {
                    "success": False,
                    "backend": backend,
                    "error": "demo config not found",
                }
            }
        data_dir = self.demo_cfg.get("data_dir", "")
        HOUMO_DATASETS_PATH = os.environ.get("HOUMO_DATASETS_PATH", "")
        HM_data_dir = os.path.join(HOUMO_DATASETS_PATH, data_dir)
        if not os.path.isdir(data_dir) and not os.path.isdir(HM_data_dir):
            logger.error("data_dir must be a exist directory")
            return {
                "demo": {
                    "success": False,
                    "backend": backend,
                    "error": "data_dir not found",
                }
            }
        if not os.path.isdir(data_dir):
            data_dir = HM_data_dir
        logger.info(f"[demo] data_dir: {data_dir}")
        test_num = self.demo_cfg.get("num", 0)
        if not isinstance(test_num, int):
            logger.error(f"test_num must be int -> {test_num}")
            return {
                "demo": {
                    "success": False,
                    "backend": backend,
                    "error": "test_num must be int",
                }
            }
        if test_num < 0:
            logger.error(f"test_num must >= 0 -> {test_num}")
            return {
                "demo": {
                    "success": False,
                    "backend": backend,
                    "error": "test_num must >= 0",
                }
            }
        model = self.get_model(backend)
        if model is None:
            logger.error("Failed to get model")
            return {
                "demo": {
                    "success": False,
                    "backend": backend,
                    "error": "failed to get model",
                }
            }
        filenames = os.listdir(data_dir)
        data_num = len(filenames)
        if test_num > 0 and test_num < data_num:
            filenames = filenames[:test_num]
        filepaths = list()
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"filepath not exists -> {filepath}")
                continue
            filepaths.append(filepath)
        model.load(self.get_model_path(backend), device_id)
        model.demo(filepaths)
        model.unload()

        return {
            "demo": {
                "success": True,
                "backend": backend,
                "data_dir": data_dir,
                "num": len(filepaths),
            }
        }

    def evaluate(self, backend, device_id=0):
        """Evaluation entry point.

        Args:
            backend (str): Backend type for the evaluation.
            device_id (int): Device ID to run the evaluation on.

        Returns:
            dict: Evaluation results dictionary with success status and backend info.
        """
        if not self.eval_cfg:
            logger.error("eval config not found")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "eval config not found",
                    }
                }
            }
        data_dir = self.eval_cfg.get("data_dir", "")
        HOUMO_DATASETS_PATH = os.environ.get("HOUMO_DATASETS_PATH", "")
        HM_data_dir = os.path.join(HOUMO_DATASETS_PATH, data_dir)
        if not os.path.isdir(data_dir) and not os.path.isdir(HM_data_dir):
            logger.error("data_dir must be a exist directory")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "data_dir not found",
                    }
                }
            }
        if not os.path.isdir(data_dir):
            data_dir = HM_data_dir
        logger.info(f"[eval] data_dir: {data_dir}")
        num = self.eval_cfg.get("num", 0)
        if not isinstance(num, int):
            logger.error(f"eval test_num must be int -> {num}")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "num must be int",
                    }
                }
            }
        if num < 0:
            logger.error(f"eval test_num must >= 0 -> {num}")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "num must >= 0",
                    }
                }
            }
        # Get dataset
        dataset = self.get_dataset(data_dir)
        if dataset is None:
            logger.error("get_dataset failed")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "failed to get dataset",
                    }
                }
            }
        # Get model
        model = self.get_model(backend)
        if model is None:
            logger.error("Failed to get model")
            return {
                "eval": {
                    backend: {
                        "success": False,
                        "error": "failed to get model",
                    }
                }
            }
        model.load(self.get_model_path(backend), device_id)
        res = model.evaluate(dataset, num)
        model.unload()
        logger.info(f"{res}")

        return {
            "eval": {
                backend: {
                    "success": True,
                    "data_dir": data_dir,
                    "num": num,
                    "results": res,
                }
            }
        }

    def get_model_path(self, backend):
        """Get model path based on backend.

        Args:
            backend (str): Backend type.

        Returns:
            str: Model path.
        """
        if backend == "onnx":
            model_path = self.model_path
        elif backend == "hmonnx":
            model_path = self.quant_onnx_model_path
        else:
            model_path = self.hmm_path
        return model_path

    @staticmethod
    def model_perf(
        model_path,
        warmup_num,
        sample_num,
        loop_num=1,
        thread_num=1,
        stream_num=0,
        batch=1,
        infer_only=False,
        devices=[0],
    ):
        """Run model performance test.

        Args:
            model_path (str): Path to the model file.
            warmup_num (int): Number of warmup iterations.
            sample_num (int): Number of sample iterations.
            loop_num (int): Number of loops to run.
            thread_num (int): Number of threads to use.
            stream_num (int): Number of streams to use.
            batch (int): Batch size for inference.
            infer_only (bool): Whether to only perform inference.
            devices (list): List of device IDs to use.

        Returns:
            dict: Performance test results.
        """
        from ..python import perf

        try:
            import tcim_lite
        except ModuleNotFoundError:
            logger.warning("tcim_lite not found, skipping perf test.")
            exit(0)

        if tcim_lite.runtime.get_device_num() < 1:
            logger.warning("No available devices found, skipping perf test.")
            exit(0)

        # TODO Use golden data
        perf_info = perf.CModelRunner(
            model_path,
            warmup_num,
            sample_num,
            loop_num,
            thread_num,
            stream_num=stream_num,
            check_output=False,
            batch=batch,
            infer_only=infer_only,
            devices=devices,
        )
        res_info = {
            "perf": {
                "success": True,
                "params": {
                    "hmm_path": model_path,
                    "warmup_num": warmup_num,
                    "sample_num": sample_num,
                    "loop_num": loop_num,
                    "thread_num": thread_num,
                    "stream_num": stream_num,
                    "devices": devices,
                },
                "perf_info": {
                    "input_avg_latency": perf_info.input_avg_latency,
                    "input_max_latency": perf_info.input_max_latency,
                    "infer_avg_latency": perf_info.infer_avg_latency,
                    "infer_max_latency": perf_info.infer_max_latency,
                    "output_avg_latency": perf_info.output_avg_latency,
                    "output_max_latency": perf_info.output_max_latency,
                    "avg_cost": perf_info.avg_cost,
                    "qps": perf_info.qps,
                },
            }
        }
        return res_info

    def save_profile_data(self, outputs: dict):
        """Save profiling data to file.

        Args:
            outputs (dict): Dictionary containing profiling data.
        """
        profile_dir = os.path.join(self.build_output_dir, "profile")
        if "auto_profile_data.bin" in outputs:
            os.makedirs(profile_dir, exist_ok=True)
            outputs["auto_profile_data.bin"].tofile(
                os.path.join(profile_dir, "auto_profile_data.bin")
            )
        if "primitive_profile_data.bin" in outputs:
            os.makedirs(profile_dir, exist_ok=True)
            outputs["primitive_profile_data.bin"].tofile(
                os.path.join(profile_dir, "primitive_profile_data.bin")
            )

    @staticmethod
    def add_node_output_as_graph_output(model_path, target):
        """Add node outputs as graph outputs for debugging.

        Args:
            model_path (str): Path to the ONNX model file.
            target (str): Target device type.

        Returns:
            str: Path to the debug model file.
        """
        new_model_path = model_path.replace(".onnx", "_debug.onnx")
        # if os.path.exists(new_model_path):
        #     return new_model_path
        model = onnx.load(model_path)
        graph = model.graph
        for node in graph.node:
            if "constant" in str(node.op_type).lower():
                continue
            for output in node.output:
                if any(
                    output == existing_output.name for existing_output in graph.output
                ):
                    continue
                value_info = None
                for vi in graph.value_info:
                    if vi.name == output:
                        value_info = vi
                        break

                if value_info is None:
                    logger.warning(
                        f"Warning: Cannot determine type/shape for output {output}, creating basic output"
                    )
                    continue

                graph.output.append(value_info)

        model.graph.CopyFrom(graph)
        onnx.checker.check_model(model)
        model.opset_import[0].version = max(model.opset_import[0].version, 11)
        model.ir_version = max(model.ir_version, 6)
        onnx.save(model, new_model_path)
        logger.info(f"Saved debug hmonnx to {new_model_path}")
        return new_model_path
