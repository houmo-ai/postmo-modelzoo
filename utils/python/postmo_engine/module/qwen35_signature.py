# Copyright (c) 2026 HOUMO AI
#
# File: qwen35_signature.py
# Description:
#   Qwen3.5 Prefill and Decode graph signature parsing and validation.
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

"""Name-based Qwen3.5 Prefill/Decode graph signatures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_DTYPE_MAP = {
    "FLOAT16": "float16",
    "FLOAT32": "float32",
    "INT32": "int32",
    "INT16": "int16",
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Qwen35GraphSignature:
    embeddings: TensorSpec
    time_positions: TensorSpec
    height_positions: TensorSpec
    width_positions: TensorSpec
    valid_length: TensorSpec
    current_length: TensorSpec
    attention_mask: TensorSpec
    logits: TensorSpec
    kv_inputs: tuple[str, ...]
    conv_inputs: tuple[str, ...]
    recurrent_inputs: tuple[str, ...]
    conv_outputs: tuple[str, ...]
    recurrent_outputs: tuple[str, ...]
    prefill_length: int
    context_max_length: int
    embedding_size: int


def _spec(item: dict[str, Any]) -> TensorSpec:
    dtype = str(item["dtype"]).upper()
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported tensor dtype: {item['dtype']!r}")
    return TensorSpec(
        name=str(item["name"]),
        shape=tuple(int(value) for value in item["shape"]),
        dtype=_DTYPE_MAP[dtype],
    )


def _require(inputs: dict[str, dict[str, Any]], *names: str) -> dict[str, Any]:
    for name in names:
        if name in inputs:
            return inputs[name]
    raise ValueError(f"graph is missing required input {names[0]!r}")


def parse_qwen35_signature(model_info: str) -> Qwen35GraphSignature:
    payload = json.loads(model_info)
    inputs = {item["name"]: item for item in payload["input"]}
    outputs = {item["name"]: item for item in payload["output"]}
    if "logits" not in outputs:
        raise ValueError("graph is missing logits output")
    embeddings = _spec(_require(inputs, "input_1", "embeddings"))
    if len(embeddings.shape) != 3:
        raise ValueError(f"embedding input must be rank 3, got {embeddings.shape}")
    kv_inputs = tuple(name for name in inputs if "model_layers" in name)
    conv_inputs = tuple(name for name in inputs if "conv_cache" in name)
    recurrent_inputs = tuple(name for name in inputs if "recurrent_state" in name)
    conv_outputs = tuple(name for name in outputs if name.startswith("conv_cache_out_"))
    recurrent_outputs = tuple(name for name in outputs if name.startswith("recurrent_state_out_"))
    custom = json.loads(payload.get("info", {}).get("custom_msg") or "{}")
    prefill_length = int(embeddings.shape[1])
    context_max_length = int(custom.get("context_length") or 0)
    if context_max_length <= 0 and kv_inputs:
        context_max_length = int(tuple(inputs[kv_inputs[0]]["shape"])[2])
    if context_max_length <= 0:
        raise ValueError("unable to determine context_max_length")
    return Qwen35GraphSignature(
        embeddings=embeddings,
        time_positions=_spec(_require(inputs, "time_position_ids")),
        height_positions=_spec(_require(inputs, "hight_position_ids", "height_position_ids")),
        width_positions=_spec(_require(inputs, "width_position_ids")),
        valid_length=_spec(_require(inputs, "valid_length")),
        current_length=_spec(_require(inputs, "current_length")),
        attention_mask=_spec(_require(inputs, "linear_attn_mask", "attention_mask")),
        logits=_spec(outputs["logits"]),
        kv_inputs=kv_inputs,
        conv_inputs=conv_inputs,
        recurrent_inputs=recurrent_inputs,
        conv_outputs=conv_outputs,
        recurrent_outputs=recurrent_outputs,
        prefill_length=prefill_length,
        context_max_length=context_max_length,
        embedding_size=int(embeddings.shape[-1]),
    )
