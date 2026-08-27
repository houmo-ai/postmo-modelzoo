# Copyright (c) 2026 HOUMO AI
#
# File: tensor.py
# Description:
#   Backend-independent tensor metadata and opaque tensor handles.
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

"""Backend-independent tensor metadata and opaque handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def numpy_dtype(value: Any) -> np.dtype:
    try:
        return np.dtype(value)
    except TypeError:
        name = str(value).split(".")[-1].lower()
        try:
            return np.dtype(name)
        except TypeError as error:
            raise TypeError(f"unsupported tensor dtype: {value!r}") from error


@dataclass(frozen=True)
class TensorInfo:
    """Describe shape, dtype, and optional layout of a runtime tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    device: str | None = None
    stride: tuple[int, ...] | None = None
    format: str | None = None

    def matches(self, other: "TensorInfo") -> bool:
        return self.shape == other.shape and self.dtype == other.dtype

    @classmethod
    def from_runtime(cls, value: Any, *, name: str = "") -> "TensorInfo":
        info = getattr(value, "info", value)
        shape = tuple(int(dimension) for dimension in getattr(info, "shape", ()))
        dtype = numpy_dtype(getattr(info, "dtype", np.float32))
        stride = getattr(info, "stride", None)
        return cls(
            name=str(getattr(info, "name", name) or name),
            shape=shape,
            dtype=dtype,
            device=str(getattr(info, "device", None) or "") or None,
            stride=None if stride is None else tuple(int(value) for value in stride),
            format=getattr(info, "format", None),
        )


class TensorHandle:
    """Opaque tensor handle bound to the creating backend."""

    __slots__ = ("__backend_token", "__raw_tensor", "info")

    def __init__(self, backend_token: object, raw_tensor: Any, info: TensorInfo) -> None:
        self.__backend_token = backend_token
        self.__raw_tensor = raw_tensor
        self.info = info

    def _unwrap(self, backend_token: object) -> Any:
        if backend_token is not self.__backend_token:
            require_same_backend = "tensor handle belongs to a different backend"
            raise ValueError(require_same_backend)
        return self.__raw_tensor
