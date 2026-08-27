# Copyright (c) 2026 HOUMO AI
#
# File: tcim_backend.py
# Description:
#   TCIM Lite implementation of the PostMo runtime backend contract.
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

"""TCIM Lite implementation of the PostMo runtime backend contract."""

from typing import Any

import numpy as np

from .base import PostMoBackend
from . import tcim_tensor
from .tensor import TensorInfo


def _load_tcim_runtime():
    try:
        import tcim_lite as tcim
    except ImportError as exc:
        raise ImportError("tcim_lite is required to use TcimBackend") from exc
    return tcim.runtime


class TcimBackend(PostMoBackend):
    """PostMo backend using TCIM Lite runtime models."""

    def __init__(
        self,
        *,
        perf=None,
        runtime=None,
    ) -> None:
        super().__init__(perf=perf)
        self._runtime = runtime if runtime is not None else _load_tcim_runtime()

    def create_weight_manager(self, device_id: int = 0) -> Any:
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        return self._runtime.WeightManager(device_id)

    def model_info(self, path) -> str:
        return self._runtime.load_model_info(str(path))

    def _input_names(self, model: Any) -> tuple[str, ...]:
        return tuple(model.get_input_name(index) for index in range(model.get_num_inputs()))

    def _output_names(self, model: Any) -> tuple[str, ...]:
        return tuple(model.get_output_name(index) for index in range(model.get_num_outputs()))

    def _input_info(self, model: Any, name: str) -> Any:
        return model.get_input_info(name)

    def _output_info(self, model: Any, name: str) -> Any:
        return model.get_output_info(name)

    def _get_device_input(self, model: Any, name: str) -> Any:
        return model.get_dev_input(name)

    def _bind_device_input(self, model: Any, name: str, tensor: Any) -> None:
        model.set_dev_input(name, tensor)

    def _bind_device_output(self, model: Any, name: str, tensor: Any) -> None:
        model.set_dev_output(name, tensor)

    def _tensor_info(self, tensor: Any, *, name: str = "") -> TensorInfo:
        return tcim_tensor.tensor_info(tensor, name=name)

    def _zero_tensor(self, tensor: Any) -> None:
        tcim_tensor.set_zero(tensor)

    def _copy_tensor(self, source: Any, destination: Any) -> None:
        tcim_tensor.copy_tensor(source, destination)

    def _clone_tensor(self, tensor: Any) -> Any:
        return tcim_tensor.clone_tensor(tensor)

    def _prepare_model_load(
        self,
        path: str,
        *,
        weight_manager: Any,
        dummy_inputs: tuple[str, ...],
    ) -> Any:
        del path
        option = self._runtime.Option(weight_manager)
        if dummy_inputs:
            option.set_dummy_tensors(list(dummy_inputs))
        return option

    def _load_prepared_model(self, path: str, prepared: Any) -> Any:
        return self._runtime.load(path, option=prepared)

    def _prepare_host_input(self, model: Any, name: str, value: Any) -> np.ndarray:
        array = np.asarray(value)
        info = model.get_input_info(name)
        shape = tuple(int(dimension) for dimension in info.shape)
        if array.shape != shape:
            expected_size = int(np.prod(shape, dtype=np.int64))
            if array.size != expected_size:
                raise ValueError(
                    f"input {name!r} expects shape {shape}, got {array.shape}"
                )
            array = array.reshape(shape)
        return np.ascontiguousarray(array)

    def _set_prepared_host_input(
        self,
        model: Any,
        name: str,
        value: np.ndarray,
    ) -> None:
        model.set_input(name, value)

    def _run_and_sync(self, model: Any) -> None:
        model.run()
        model.sync()

    def _copy_output_to_host(self, model: Any, name: str) -> Any:
        return model.get_output(name)

    def _normalize_host_output(self, output: Any) -> np.ndarray:
        value = output.numpy() if hasattr(output, "numpy") else output
        return np.array(value, copy=True)
