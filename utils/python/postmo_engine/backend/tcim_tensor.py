# Copyright (c) 2026 HOUMO AI
#
# File: tcim_tensor.py
# Description:
#   TCIM Lite helpers for opaque tensor metadata and tensor operations.
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

"""TCIM Lite helpers for opaque tensor metadata and operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor import TensorInfo, numpy_dtype


def tensor_info(value: Any, *, name: str = "") -> TensorInfo:
    return TensorInfo.from_runtime(value, name=name)


def require_compatible(source: Any, destination: Any, *, name: str) -> None:
    source_info = tensor_info(source, name=name)
    destination_info = tensor_info(destination, name=name)
    if not source_info.matches(destination_info):
        raise ValueError(
            f"{name} tensor mismatch: source {source_info.shape}/{source_info.dtype} "
            f"vs destination {destination_info.shape}/{destination_info.dtype}"
        )


def to_host(value: Any, *, copy: bool = True) -> np.ndarray:
    if hasattr(value, "to_host"):
        host = value.to_host(True)
        array = host.numpy() if hasattr(host, "numpy") else np.asarray(host)
    elif hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    return np.array(array, copy=True) if copy else np.ascontiguousarray(array)


def set_zero(value: Any) -> None:
    if hasattr(value, "set_zero"):
        value.set_zero()
        return
    array = np.asarray(value)
    array.fill(0)


def copy_tensor(source: Any, destination: Any, *, name: str = "tensor") -> None:
    require_compatible(source, destination, name=name)
    if hasattr(source, "copy_to"):
        source.copy_to(destination)
        return
    np.copyto(np.asarray(destination), to_host(source, copy=False), casting="no")


def clone_tensor(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone(True)
    return np.array(to_host(value, copy=True), copy=True)


def host_zeros(info: Any) -> np.ndarray:
    tensor = tensor_info(info)
    return np.zeros(tensor.shape, dtype=numpy_dtype(tensor.dtype))
