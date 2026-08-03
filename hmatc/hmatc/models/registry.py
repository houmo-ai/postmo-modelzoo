# Copyright 2025 HOUMO AI
#
# File: registry.py
# Description:
#   Registry for managing HMATC large models.
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
import importlib
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LmModelSpec:
    model_name: str
    api_name: str
    supported_sizes: Tuple[str, ...]
    module: str


_MODEL_SPECS = {
    "gemma4": LmModelSpec(
        model_name="gemma4",
        api_name="HmGemma4",
        supported_sizes=("26b-a4b", "31b", "e2b", "e4b"),
        module="hmatc.models.gemma4.gemma4",
    ),
}


def get_supported_models() -> Tuple[str, ...]:
    return tuple(_MODEL_SPECS)


def get_model_spec(model_name: str) -> LmModelSpec:
    normalized_name = model_name.strip().lower()
    try:
        return _MODEL_SPECS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(get_supported_models())
        raise ValueError(
            f"Unsupported model name {model_name!r}. Supported models: {supported}"
        ) from exc


def validate_model_size(spec: LmModelSpec, model_size: str) -> str:
    normalized_size = model_size.strip().lower()
    if normalized_size not in spec.supported_sizes:
        supported = ", ".join(spec.supported_sizes)
        raise ValueError(
            f"Unsupported model size {model_size!r} for {spec.model_name}. "
            f"Supported sizes: {supported}"
        )
    return normalized_size


def register_model_api(spec: LmModelSpec) -> None:
    importlib.import_module(spec.module)
