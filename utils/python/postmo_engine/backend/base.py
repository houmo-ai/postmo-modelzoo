# Copyright (c) 2026 HOUMO AI
#
# File: base.py
# Description:
#   Backend-independent runtime model execution contract.
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

"""Backend-independent runtime model execution contract."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from postmo_engine.perf import PerfTracker
from .tensor import TensorHandle, TensorInfo

_INPUT_NAME = "input name"


class ModelHandle:
    """Opaque model handle retaining its performance identity and owner."""

    __slots__ = ("__backend_token", "__raw_model", "model_category", "model_role")

    def __init__(
        self,
        backend_token: object,
        raw_model: Any,
        model_category: str,
        model_role: str,
    ) -> None:
        self.__backend_token = backend_token
        self.__raw_model = raw_model
        self.model_category = model_category
        self.model_role = model_role

    def _unwrap(self, backend_token: object) -> Any:
        if backend_token is not self.__backend_token:
            raise ValueError("model handle belongs to a different backend")
        return self.__raw_model


class PostMoBackend(ABC):
    """Template-method contract enforcing stable model timing boundaries.

    Public methods own validation, opaque-handle checks, and performance
    collection. Implementations provide only the protected runtime hooks.
    """

    def __init__(
        self,
        *,
        perf: PerfTracker | None = None,
    ) -> None:
        self._perf = perf if perf is not None else PerfTracker.create(enabled=False)
        self.__backend_token = object()

    @property
    def perf(self) -> PerfTracker:
        return self._perf

    def load_model(
        self,
        path: str | Path,
        *,
        model_category: str,
        model_role: str,
        weight_manager: Any,
        dummy_inputs: tuple[str, ...] = (),
    ) -> ModelHandle:
        """Load one runtime model and bind its stable performance identity.

        Runtime options are prepared before timing. Only the implementation's
        actual model-load hook is measured under ``<category>.load.<role>_model``.
        """
        model_category = self._normalize_name(model_category, name="model_category")
        model_role = self._normalize_name(model_role, name="model_role")
        normalized_path = str(Path(path))
        normalized_dummy_inputs = self._normalize_names(
            dummy_inputs,
            name="dummy_inputs",
        )
        prepared = self._prepare_model_load(
            normalized_path,
            weight_manager=weight_manager,
            dummy_inputs=normalized_dummy_inputs,
        )
        with self._perf.scope(
            self._scope_path_parts(
                model_category,
                "load",
                f"{model_role}_model",
            )
        ):
            raw_model = self._load_prepared_model(normalized_path, prepared)
        return ModelHandle(self.__backend_token, raw_model, model_category, model_role)

    def set_host_input(self, model: ModelHandle, name: str, value: Any) -> None:
        """Validate host input and measure only its runtime H2D API call."""
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name=_INPUT_NAME)
        prepared_value = self._prepare_host_input(raw_model, normalized_name, value)
        with self._perf.scope(self._scope_path(model, "set_input")):
            self._set_prepared_host_input(raw_model, normalized_name, prepared_value)

    def set_host_inputs(self, model: ModelHandle, values: dict[str, Any]) -> None:
        """Set all host inputs for one model execution in one measured stage."""
        raw_model = self._unwrap_model(model)
        if not isinstance(values, dict):
            raise TypeError("values must be a dict")
        if not values:
            raise ValueError("values must not be empty")
        prepared = []
        for name, value in values.items():
            normalized_name = self._normalize_name(name, name=_INPUT_NAME)
            prepared.append(
                (
                    normalized_name,
                    self._prepare_host_input(raw_model, normalized_name, value),
                )
            )
        with self._perf.scope(self._scope_path(model, "set_input")):
            for name, value in prepared:
                self._set_prepared_host_input(raw_model, name, value)

    def initialize_host_input(self, model: ModelHandle, name: str, value: Any) -> None:
        """Set initialization/reset data without recording request H2D timing."""
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name=_INPUT_NAME)
        prepared_value = self._prepare_host_input(raw_model, normalized_name, value)
        self._set_prepared_host_input(raw_model, normalized_name, prepared_value)

    def run(self, model: ModelHandle) -> None:
        """Run and synchronize a model as one measured runtime operation."""
        raw_model = self._unwrap_model(model)
        with self._perf.scope(self._scope_path(model, "run")):
            self._run_and_sync(raw_model)

    def get_output(self, model: ModelHandle, name: str) -> Any:
        """Measure only runtime D2H, then normalize the host result."""
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name="output name")
        with self._perf.scope(self._scope_path(model, "get_output")):
            host_output = self._copy_output_to_host(raw_model, normalized_name)
        return self._normalize_host_output(host_output)

    def input_names(self, model: ModelHandle) -> tuple[str, ...]:
        return self._input_names(self._unwrap_model(model))

    def output_names(self, model: ModelHandle) -> tuple[str, ...]:
        return self._output_names(self._unwrap_model(model))

    def input_info(self, model: ModelHandle, name: str) -> Any:
        raw_model = self._unwrap_model(model)
        return self._input_info(raw_model, self._normalize_name(name, name=_INPUT_NAME))

    def get_device_input(self, model: ModelHandle, name: str) -> Any:
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name=_INPUT_NAME)
        return self._wrap_tensor(
            self._get_device_input(raw_model, normalized_name),
            name=normalized_name,
        )

    def bind_device_input(self, model: ModelHandle, name: str, tensor: Any) -> None:
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name=_INPUT_NAME)
        raw_tensor = self._unwrap_tensor(tensor)
        self._require_compatible_tensor(
            raw_tensor,
            self._input_info(raw_model, normalized_name),
            name=normalized_name,
        )
        self._bind_device_input(raw_model, normalized_name, raw_tensor)

    def bind_device_output(self, model: ModelHandle, name: str, tensor: Any) -> None:
        raw_model = self._unwrap_model(model)
        normalized_name = self._normalize_name(name, name="output name")
        raw_tensor = self._unwrap_tensor(tensor)
        destination = self._output_info(raw_model, normalized_name)
        self._require_compatible_tensor(raw_tensor, destination, name=normalized_name)
        self._bind_device_output(raw_model, normalized_name, raw_tensor)

    def tensor_info(self, tensor: Any, *, name: str = "") -> TensorInfo:
        return self._tensor_info(self._unwrap_tensor(tensor), name=name)

    def zero_tensor(self, tensor: Any) -> None:
        self._zero_tensor(self._unwrap_tensor(tensor))

    def copy_tensor(self, source: Any, destination: Any) -> None:
        raw_source = self._unwrap_tensor(source)
        raw_destination = self._unwrap_tensor(destination)
        self._require_compatible_tensor(raw_source, raw_destination, name="copied tensor")
        self._copy_tensor(raw_source, raw_destination)

    def clone_tensor(self, tensor: Any) -> TensorHandle:
        raw_tensor = self._unwrap_tensor(tensor)
        cloned = self._clone_tensor(raw_tensor)
        return self._wrap_tensor(cloned, name=self.tensor_info(tensor).name)

    def _unwrap_model(self, model: ModelHandle) -> Any:
        if not isinstance(model, ModelHandle):
            raise TypeError("model must be a ModelHandle")
        return model._unwrap(self.__backend_token)

    def _wrap_tensor(self, tensor: Any, *, name: str = "") -> TensorHandle:
        if isinstance(tensor, TensorHandle):
            return tensor
        return TensorHandle(self.__backend_token, tensor, self._tensor_info(tensor, name=name))

    def _unwrap_tensor(self, tensor: Any) -> Any:
        if isinstance(tensor, TensorHandle):
            return tensor._unwrap(self.__backend_token)
        return tensor

    def _require_compatible_tensor(self, source: Any, destination: Any, *, name: str) -> None:
        source_info = self._tensor_info(source, name=name)
        destination_info = self._tensor_info(destination, name=name)
        if not source_info.matches(destination_info):
            raise ValueError(
                f"{name} tensor mismatch: source {source_info.shape}/{source_info.dtype} "
                f"vs destination {destination_info.shape}/{destination_info.dtype}"
            )

    @staticmethod
    def _scope_path_parts(category: str, role: str, operation: str) -> str:
        return f"{category}.{role}.{operation}"

    @staticmethod
    def _scope_path(model: ModelHandle, operation: str) -> str:
        return PostMoBackend._scope_path_parts(
            model.model_category, model.model_role, operation
        )

    @staticmethod
    def _normalize_name(value: str, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value or value.strip() != value:
            raise ValueError(f"{name} must be a non-empty trimmed string")
        return value

    @classmethod
    def _normalize_names(
        cls,
        values: tuple[str, ...],
        *,
        name: str,
    ) -> tuple[str, ...]:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple")
        normalized = tuple(
            cls._normalize_name(value, name=f"{name} item") for value in values
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must not contain duplicates")
        return normalized

    @abstractmethod
    def create_weight_manager(self, device_id: int = 0) -> Any:
        """Create the Runtime weight manager used by related models."""

    @abstractmethod
    def model_info(self, path: str | Path) -> str:
        """Return Runtime model metadata without loading the executable model."""

    @abstractmethod
    def _input_names(self, model: Any) -> tuple[str, ...]:
        """Return Runtime input port names."""

    @abstractmethod
    def _output_names(self, model: Any) -> tuple[str, ...]:
        """Return Runtime output port names."""

    @abstractmethod
    def _input_info(self, model: Any, name: str) -> Any:
        """Return Runtime input metadata."""

    @abstractmethod
    def _output_info(self, model: Any, name: str) -> Any:
        """Return Runtime output metadata."""

    @abstractmethod
    def _get_device_input(self, model: Any, name: str) -> Any:
        """Return a borrowed Runtime input tensor handle."""

    @abstractmethod
    def _bind_device_input(self, model: Any, name: str, tensor: Any) -> None:
        """Bind an existing device tensor as model input without copying."""

    @abstractmethod
    def _bind_device_output(self, model: Any, name: str, tensor: Any) -> None:
        """Bind an existing device tensor as model output without copying."""

    @abstractmethod
    def _tensor_info(self, tensor: Any, *, name: str = "") -> TensorInfo:
        """Normalize Runtime tensor metadata."""

    @abstractmethod
    def _zero_tensor(self, tensor: Any) -> None:
        """Zero a Runtime tensor in place."""

    @abstractmethod
    def _copy_tensor(self, source: Any, destination: Any) -> None:
        """Copy tensor data without changing destination identity."""

    @abstractmethod
    def _clone_tensor(self, tensor: Any) -> Any:
        """Allocate an independent Runtime tensor with copied data."""

    @abstractmethod
    def _prepare_model_load(
        self,
        path: str,
        *,
        weight_manager: Any,
        dummy_inputs: tuple[str, ...],
    ) -> Any:
        """Prepare runtime load options outside the measured load operation."""

    @abstractmethod
    def _load_prepared_model(self, path: str, prepared: Any) -> Any:
        """Call only the runtime model-load API."""

    @abstractmethod
    def _prepare_host_input(self, model: Any, name: str, value: Any) -> Any:
        """Normalize and validate host data outside measured H2D."""

    @abstractmethod
    def _set_prepared_host_input(self, model: Any, name: str, value: Any) -> None:
        """Call only the runtime Host-to-device input API."""

    @abstractmethod
    def _run_and_sync(self, model: Any) -> None:
        """Call runtime run followed by runtime synchronization."""

    @abstractmethod
    def _copy_output_to_host(self, model: Any, name: str) -> Any:
        """Call only the runtime Device-to-host output API."""

    @abstractmethod
    def _normalize_host_output(self, output: Any) -> Any:
        """Produce the public host result outside measured D2H."""
