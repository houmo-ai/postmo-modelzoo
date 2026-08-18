# Copyright (c) 2026 HOUMO AI
#
# File: runtime.py
# Description:
#   TCIM-Lite-compatible runtime adapter for HMONNX inference.
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

"""TCIM-Lite-compatible runtime adapter for HMONNX inference."""

import math
from pathlib import Path

import numpy as np


def _is_torch_tensor(value):
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


def _is_backend_cache_tensor(value):
    return value.__class__.__name__ in {"CacheTensor", "HybridCacheTensor"}


def _tensor_data(value):
    if isinstance(value, np.ndarray):
        return value
    return getattr(value, "data", value)


def _same_tensor_buffer(left, right):
    left = _tensor_data(left)
    right = _tensor_data(right)
    if left is right:
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.__array_interface__["data"][0]
            == right.__array_interface__["data"][0]
            and left.shape == right.shape
            and left.strides == right.strides
        )
    if hasattr(left, "data_ptr") and hasattr(right, "data_ptr"):
        return (
            left.data_ptr() == right.data_ptr()
            and tuple(left.shape) == tuple(right.shape)
            and tuple(left.stride()) == tuple(right.stride())
        )
    return False


def _copy_tensor_value(target, source, name):
    """Copy source into a bound output tensor in place."""
    target_value = _tensor_data(target)
    source_value = _tensor_data(source)
    target_shape = tuple(getattr(target_value, "shape", ()))
    source_shape = tuple(getattr(source_value, "shape", ()))
    if target_shape != source_shape:
        raise RuntimeError(
            f"bound output {name} shape mismatch: "
            f"expected {target_shape}, got {source_shape}"
        )
    target_dtype = getattr(target_value, "dtype", None)
    source_dtype = getattr(source_value, "dtype", None)
    if target_dtype is not None and source_dtype is not None:
        if _to_numpy_dtype(target_dtype) != _to_numpy_dtype(source_dtype):
            raise RuntimeError(
                f"bound output {name} dtype mismatch: "
                f"expected {target_dtype}, got {source_dtype}"
            )
    if _same_tensor_buffer(target_value, source_value):
        return

    if isinstance(target_value, np.ndarray):
        source_array = Tensor(source_value).numpy()
        np.copyto(target_value, source_array, casting="no")
        return
    if hasattr(target_value, "copy_"):
        if isinstance(source_value, np.ndarray) and _is_torch_tensor(target_value):
            import torch

            source_value = torch.as_tensor(
                source_value,
                device=target_value.device,
                dtype=target_value.dtype,
            )
        target_value.copy_(source_value)
        return
    if hasattr(target_value, "copy_to"):
        target_value.copy_to(source_value)
        return
    raise TypeError(f"bound output {name} does not support buffer writes")


def _to_numpy_dtype(dtype):
    try:
        return np.dtype(dtype)
    except TypeError:
        name = str(dtype).removeprefix("torch.")
        try:
            return np.dtype(name)
        except TypeError as exc:
            raise TypeError(f"unsupported tensor dtype: {dtype!r}") from exc


class TensorInfo:
    """Describe the shape and scalar type of a runtime tensor."""

    def __init__(
        self,
        name,
        shape,
        dtype,
        stride=None,
        format=None,
        device=None,
    ):
        self.name = str(name)
        self.shape = tuple(shape)
        self.dtype = _to_numpy_dtype(dtype)
        self.stride = None if stride is None else tuple(stride)
        self.format = format
        self.device = device

    @property
    def mem_size(self):
        if any(not isinstance(dim, (int, np.integer)) or dim < 0 for dim in self.shape):
            return None
        return math.prod(self.shape) * self.dtype.itemsize

    def is_match(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

    def astype(self, dtype):
        return TensorInfo(
            self.name,
            self.shape,
            dtype,
            self.stride,
            self.format,
            self.device,
        )

    def clone(self):
        return TensorInfo(
            self.name,
            self.shape,
            self.dtype,
            self.stride,
            self.format,
            self.device,
        )

    @classmethod
    def from_value(cls, value, name=""):
        shape = getattr(value, "shape", ())
        dtype = getattr(value, "dtype", np.float32)
        return cls(name, shape, dtype, device=getattr(value, "device", None))


class Tensor:
    """Lightweight wrapper that defers host conversion until requested."""

    def __init__(self, value, info=None):
        self._value = value
        self._info = info or TensorInfo.from_value(value)

    @property
    def value(self):
        return self._value

    @property
    def info(self):
        return self._info

    @property
    def shape(self):
        return self._info.shape

    @property
    def dtype(self):
        return self._info.dtype

    @property
    def device(self):
        return self._info.device

    @property
    def device_id(self):
        device = self.device
        return getattr(device, "idx", None) if device is not None else None

    @property
    def mem_size(self):
        return self._info.mem_size

    def numpy(self):
        if isinstance(self._value, np.ndarray):
            return self._value
        if _is_torch_tensor(self._value):
            return self._value.detach().cpu().numpy()
        if hasattr(self._value, "numpy"):
            return self._value.numpy()
        raise TypeError("tensor value does not support host conversion")

    def astype(self, dtype):
        return Tensor(self.numpy().astype(dtype), self._info.astype(dtype))

    def clone(self):
        if isinstance(self._value, np.ndarray):
            return Tensor(self._value.copy(), self._info.clone())
        if hasattr(self._value, "clone"):
            return Tensor(self._value.clone(), self._info.clone())
        raise TypeError("tensor value does not support cloning")

    def copy_to(self, source):
        source_value = source.value if isinstance(source, Tensor) else source
        if isinstance(self._value, np.ndarray):
            np.copyto(self._value, Tensor(source_value).numpy())
        elif hasattr(self._value, "copy_"):
            self._value.copy_(source_value)
        else:
            raise TypeError("tensor value does not support copying")
        return self

    def set_zero(self):
        value = _tensor_data(self._value)
        if isinstance(value, np.ndarray):
            value.fill(0)
        elif hasattr(value, "zero_"):
            value.zero_()
        else:
            raise TypeError("tensor value does not support zeroing")
        return self

    def to_host(self):
        return self

    def __array__(self, dtype=None):
        array = self.numpy()
        return array.astype(dtype) if dtype is not None else array


class DevManager:
    """Store validated device configuration for the HMONNX backend."""

    def __init__(self, device_ids=0, backend_name="hmonnx"):
        if isinstance(device_ids, bool):
            raise TypeError("device_ids must be an int or a sequence of ints")
        if isinstance(device_ids, int):
            device_ids = (device_ids,)
        elif isinstance(device_ids, (list, tuple)):
            device_ids = tuple(device_ids)
        else:
            raise TypeError("device_ids must be an int or a sequence of ints")
        if not device_ids or any(
            not isinstance(device_id, int)
            or isinstance(device_id, bool)
            or device_id < 0
            for device_id in device_ids
        ):
            raise ValueError("device_ids must contain non-negative integers")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError("backend_name must be a non-empty string")
        self.device_ids = device_ids
        self.backend_name = backend_name

    @property
    def dev_count(self):
        return len(self.device_ids)

    def verify(self):
        return True


class _SharedModelGroup:
    def __init__(self, loader, paths, graphs, config):
        self.loader = loader
        self.paths = paths
        self.graphs = graphs
        self.auto_offload = config["auto_offload"]
        self.cuda_graph = config["cuda_graph"]
        self.layer_infos = None


def _get_shared_hmonnx_types():
    try:
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
            HMONNXInferenceConfig,
            HMONNXInferenceV2,
        )
        from xhquant.xhonnxruntime.llm_hmonnx_loader import LLMHMONNXLoader
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("xhquant"):
            raise ImportError(
                "shared HMONNX runtime requires XHQuantTool V2"
            ) from exc
        raise
    return LLMHMONNXLoader, HMONNXInferenceConfig, HMONNXInferenceV2


class WeightManager:
    """Own shared HMONNX weights and create compatible runtime sessions."""

    def __init__(self, device=0):
        self.device_manager = (
            device if isinstance(device, DevManager) else DevManager(device)
        )
        self.shared = False
        self._groups = {}
        self._resources = []
        self._session_groups = {}

    @staticmethod
    def _model_key(path):
        return str(Path(path).resolve())

    def register_model_group(
        self,
        *,
        prefill,
        decode,
        auto_offload=False,
        cuda_graph=False,
    ):
        paths = {
            "prefill": self._model_key(prefill),
            "decode": self._model_key(decode),
        }
        duplicate = set(paths.values()) & set(self._groups)
        if duplicate:
            names = ", ".join(sorted(duplicate))
            raise ValueError(f"model path is already registered: {names}")

        loader_type, _, _ = _get_shared_hmonnx_types()
        loader = loader_type(str(prefill), str(decode))
        group = _SharedModelGroup(
            loader,
            paths,
            {
                paths["prefill"]: loader.prefill_graph,
                paths["decode"]: loader.decode_graph,
            },
            {
                "auto_offload": bool(auto_offload),
                "cuda_graph": bool(cuda_graph),
            },
        )
        self._resources.append(loader)
        for path in paths.values():
            self._groups[path] = group
        self.shared = True
        return self

    def is_registered(self, model_path):
        return self._model_key(model_path) in self._groups

    def get_input_names(self, model_path):
        key = self._model_key(model_path)
        try:
            graph = self._groups[key].graphs[key]
        except KeyError as exc:
            raise KeyError(f"model path is not registered: {model_path}") from exc
        return tuple(value.name for value in graph.inputs)

    def create_session(self, model_path):
        key = self._model_key(model_path)
        group = self._groups.get(key)
        if group is None:
            return None

        _, config_type, inference_type = _get_shared_hmonnx_types()
        if key == group.paths["decode"] and group.layer_infos is None:
            raise RuntimeError(
                "prefill session must be created before decode session"
            )
        config = config_type()
        config.enable_auto_offload = group.auto_offload
        config.enable_cuda_graph = group.cuda_graph
        config.exec_devices = list(self.device_manager.device_ids)
        if key == group.paths["decode"]:
            config.layers = group.layer_infos
        session = inference_type.from_onnx_graph(
            str(model_path),
            group.graphs[key],
            config,
        )
        if key == group.paths["prefill"]:
            get_layer_infos = getattr(session, "get_layer_infos", None)
            if callable(get_layer_infos):
                group.layer_infos = get_layer_infos()
        self._session_groups[id(session)] = group
        return session


class Option:
    """Store runtime options accepted by the compatibility layer."""

    def __init__(self, device_or_manager=0):
        if isinstance(device_or_manager, WeightManager):
            self.weight_manager = device_or_manager
            self.device_manager = device_or_manager.device_manager
        elif isinstance(device_or_manager, DevManager):
            self.weight_manager = None
            self.device_manager = device_or_manager
        else:
            self.weight_manager = None
            self.device_manager = DevManager(device_or_manager)
        self.dummy_tensor_names = ()

    def set_dummy_tensors(self, names):
        if isinstance(names, str):
            names = (names,)
        else:
            try:
                names = tuple(names)
            except TypeError as exc:
                raise TypeError("dummy tensor names must be strings") from exc
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("dummy tensor names must be non-empty strings")
        self.dummy_tensor_names = tuple(dict.fromkeys(names))
        return self


def _get_hmonnx_inference():
    try:
        from xhquant.api import HMONNXInference
    except ModuleNotFoundError as exc:
        if exc.name in {"xhquant", "xhquant.api"}:
            raise ImportError(
                "HMONNX runtime requires XHQuantTool (xhquant.api)"
            ) from exc
        raise
    return HMONNXInference


def _normalize_device(device, option):
    if device is None:
        device = option.device_manager.device_ids[0]
    if isinstance(device, int):
        device = f"cuda:{device}"
    if not isinstance(device, str):
        return device
    try:
        import torch
    except ImportError:
        return device
    return torch.device(device)


def _metadata_to_info(metadata, fallback_name):
    return TensorInfo(
        getattr(metadata, "name", fallback_name),
        getattr(metadata, "shape", ()),
        getattr(metadata, "dtype", np.float32),
        stride=getattr(metadata, "stride", None),
        format=getattr(metadata, "format", None),
        device=getattr(metadata, "device", None),
    )


def _is_static_shape(shape):
    return all(
        isinstance(dim, (int, np.integer))
        and not isinstance(dim, (bool, np.bool_))
        and dim >= 0
        for dim in shape
    )


def _allocate_hmonnx_input(metadata, device, cache=False):
    shape = tuple(getattr(metadata, "shape", ()))
    if not _is_static_shape(shape):
        return None

    import torch

    dtype = getattr(metadata, "dtype")
    value = torch.zeros(shape, dtype=dtype, device=device) if cache else torch.empty(
        shape, dtype=dtype, device=device
    )
    if not cache:
        return value

    from xhquant.api import CacheTensor

    return CacheTensor(value)


class Module:
    """Own an HMONNX session and expose tcim-lite-style metadata APIs."""

    def __init__(self, model_path, option=None, device=None):
        self.model_path = Path(model_path)
        self.option = option if isinstance(option, Option) else Option(option or 0)
        manager = self.option.weight_manager
        registered = manager is not None and manager.is_registered(self.model_path)
        if registered and device is not None:
            raise ValueError(
                "registered model uses the WeightManager device configuration"
            )
        self._session = self._create_session(manager)
        target_device = _normalize_device(device, self.option)
        if not registered and hasattr(self._session, "to"):
            self._session.to(target_device)

        self._initialize_metadata()
        self.inputs = self._allocate_inputs(target_device)
        self.outputs = {}
        self._bound_outputs = {}

    def _create_session(self, manager):
        session = (
            manager.create_session(self.model_path) if manager is not None else None
        )
        if session is None:
            inference_type = _get_hmonnx_inference()
            session = inference_type(str(self.model_path))
        return session

    def _initialize_metadata(self):
        self.input_names = tuple(self._session.get_input_names())
        self.output_names = tuple(self._session.get_output_names())
        self._input_metadata = {
            name: self._session.get_input(name) for name in self.input_names
        }
        self.inputs_info = {
            name: _metadata_to_info(self._input_metadata[name], name)
            for name in self.input_names
        }
        self.outputs_info = {
            name: _metadata_to_info(self._session.get_output(name), name)
            for name in self.output_names
        }

    def _allocate_inputs(self, target_device):
        dummy_names = set(self.option.dummy_tensor_names)
        unknown_dummy_names = dummy_names - set(self.input_names)
        if unknown_dummy_names:
            names = ", ".join(sorted(unknown_dummy_names))
            raise ValueError(f"unknown dummy tensor names: {names}")

        inputs = {}
        for name in self.input_names:
            value = _allocate_hmonnx_input(
                self._input_metadata[name],
                target_device,
                cache=name in dummy_names,
            )
            if value is None and name in dummy_names:
                raise ValueError(f"dummy tensor {name!r} must have a static shape")
            if value is not None:
                inputs[name] = value
        return inputs

    @classmethod
    def load(cls, model_path, option=None, device=None):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model path does not exist: {path}")
        return cls(path, option=option, device=device)

    def _ensure_loaded(self):
        if self._session is None:
            raise RuntimeError("module is not loaded")

    def get_num_inputs(self):
        self._ensure_loaded()
        return len(self.input_names)

    def get_num_outputs(self):
        self._ensure_loaded()
        return len(self.output_names)

    def get_input_name(self, idx):
        self._ensure_loaded()
        return self.input_names[idx]

    def get_output_name(self, idx):
        self._ensure_loaded()
        return self.output_names[idx]

    def get_input_info(self, name):
        self._ensure_loaded()
        return self.inputs_info[name]

    def get_output_info(self, name):
        self._ensure_loaded()
        return self.outputs_info[name]

    def unload(self):
        self._session = None
        self.inputs.clear()
        self.outputs.clear()
        self._bound_outputs.clear()

    def close(self):
        self.unload()

    def _check_input_name(self, name):
        self._ensure_loaded()
        if name not in self.inputs_info:
            raise KeyError(f"unknown input name: {name}")

    def _check_output_name(self, name):
        self._ensure_loaded()
        if name not in self.outputs_info:
            raise KeyError(f"unknown output name: {name}")

    @staticmethod
    def _unwrap_input(value):
        if isinstance(value, Tensor):
            return value.value
        if isinstance(value, np.ndarray):
            try:
                import torch
            except ImportError:
                return value
            return torch.from_numpy(value)
        if _is_torch_tensor(value) or value.__class__.__name__ in {
            "CacheTensor",
            "HybridCacheTensor",
        }:
            return value
        raise TypeError("input must be a Tensor, NumPy array, or backend tensor")

    def set_input(self, name, value):
        self._check_input_name(name)
        unwrapped = self._unwrap_input(value)
        if _is_backend_cache_tensor(unwrapped):
            raise TypeError(
                f"input {name!r} uses CacheTensor through the ordinary input API"
            )
        self.inputs[name] = unwrapped
        return self

    def set_dev_input(self, name, value):
        self._check_input_name(name)
        if isinstance(value, Tensor):
            value = value.value
        if not hasattr(value, "shape"):
            raise TypeError("device input must expose a tensor shape")
        if _is_backend_cache_tensor(value) and name not in self.option.dummy_tensor_names:
            raise TypeError(
                f"input {name!r} uses CacheTensor without being declared as dummy"
            )
        if name in self.option.dummy_tensor_names and not _is_backend_cache_tensor(value):
            raise TypeError(f"dummy input {name!r} requires a CacheTensor")
        self.inputs[name] = value
        return self

    def get_dev_input(self, name):
        self._check_input_name(name)
        if name not in self.inputs:
            raise RuntimeError(f"input has not been bound: {name}")
        return Tensor(self.inputs[name], self.inputs_info[name])

    def _normalize_outputs(self, result):
        if isinstance(result, dict):
            missing = [name for name in self.output_names if name not in result]
            if missing:
                raise RuntimeError(f"backend output is missing: {missing}")
            if set(result) != set(self.output_names):
                raise RuntimeError("backend output names do not match model outputs")
            return {name: result[name] for name in self.output_names}
        if len(self.output_names) == 1 and not isinstance(result, (list, tuple)):
            return {self.output_names[0]: result}
        if isinstance(result, (list, tuple)) and len(result) == len(self.output_names):
            return dict(zip(self.output_names, result))
        raise RuntimeError("backend output count does not match model outputs")

    def run(self, sync=False):
        self._ensure_loaded()
        missing = [name for name in self.input_names if name not in self.inputs]
        if missing:
            raise RuntimeError(f"missing required inputs: {', '.join(missing)}")
        feed = {name: self.inputs[name] for name in self.input_names}
        normalized = self._normalize_outputs(self._session.run(feed))
        if self._bound_outputs:
            # HMONNX does not accept external output buffers. Copy each result
            # back into the bound object while preserving its identity so that
            # another module can consume the same device buffer directly.
            for name, target in self._bound_outputs.items():
                _copy_tensor_value(target.value, normalized[name], name)
                normalized[name] = target.value
        self.outputs = normalized
        if sync:
            self.sync()
        return self

    def get_output(self, name):
        self._check_output_name(name)
        if name not in self.outputs:
            raise RuntimeError("module has not run successfully")
        return Tensor(self.outputs[name], self.outputs_info[name])

    def get_dev_output(self, name):
        return self.get_output(name)

    def set_dev_output(self, name, value):
        self._check_output_name(name)
        if not isinstance(value, Tensor):
            value = Tensor(value, self.outputs_info[name])
        self._bound_outputs[name] = value
        return self

    def sync(self):
        # HMONNXInference.run() is synchronous, so no backend wait is needed.
        return None


def load(model_path, option=None, device=None):
    return Module.load(model_path, option=option, device=device)
