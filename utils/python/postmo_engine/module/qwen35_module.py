# Copyright (c) 2026 HOUMO AI
#
# File: qwen35_module.py
# Description:
#   Qwen3.5 Module for chunked Prefill, Decode, cache, and session state.
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

"""Qwen3.5 Text-only Module: chunked Prefill, Decode, and session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from postmo_engine.backend import PostMoBackend
from postmo_engine.core import (
    DecodeInputs,
    DecodeOutputs,
    ModuleSessionState,
    PostMoModule,
    PrefillInputs,
    PrefillOutputs,
    SessionStatus,
)
from postmo_engine.perf import PerfTracker

from .qwen35_signature import Qwen35GraphSignature, parse_qwen35_signature


def _as_array(value: Any, *, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value), dtype=dtype)


def _pad_positions(value: np.ndarray, length: int, dtype: str) -> np.ndarray:
    array = _as_array(value, dtype=dtype)
    if array.shape[-1] == length:
        return array
    if array.shape[-1] > length:
        raise ValueError("chunk exceeds the graph sequence length")
    missing = length - array.shape[-1]
    if array.shape[-1] == 0:
        raise ValueError("position input must not be empty")
    increments = np.arange(1, missing + 1, dtype=array.dtype)
    extension = array[..., -1:] + increments
    return np.concatenate((array, extension), axis=-1)


class Qwen35Module(PostMoModule):
    """Own Qwen3.5 Prefill/Decode execution and context_length."""

    def __init__(
        self,
        backend: PostMoBackend,
        prefill_path: str | Path,
        decode_path: str | Path,
        *,
        weight_manager: Any | None = None,
        perf: PerfTracker | None = None,
    ) -> None:
        self.backend = backend
        self.perf = perf if perf is not None else backend.perf
        self.session_state = ModuleSessionState()
        manager = weight_manager if weight_manager is not None else backend.create_weight_manager()
        self.prefill_signature = parse_qwen35_signature(backend.model_info(prefill_path))
        self.decode_signature = parse_qwen35_signature(backend.model_info(decode_path))
        self._validate_signatures()
        dummy_inputs = self.prefill_signature.kv_inputs
        self.prefill_model = backend.load_model(
            prefill_path,
            model_category="llm",
            model_role="prefill",
            weight_manager=manager,
        )
        self.decode_model = backend.load_model(
            decode_path,
            model_category="llm",
            model_role="decode",
            weight_manager=manager,
            dummy_inputs=dummy_inputs,
        )
        self._bind_caches()
        self.clear_session()

    def _validate_signatures(self) -> None:
        if self.prefill_signature.embedding_size != self.decode_signature.embedding_size:
            raise ValueError("Prefill and Decode embedding sizes differ")
        if self.prefill_signature.context_max_length != self.decode_signature.context_max_length:
            raise ValueError("Prefill and Decode context lengths differ")
        if self.decode_signature.embeddings.shape[1] != 1:
            raise ValueError("Decode embedding sequence length must be 1")

    @property
    def context_length(self) -> int:
        return self.session_state.context_length

    @property
    def context_max_length(self) -> int:
        return self.prefill_signature.context_max_length

    @property
    def prefill_length(self) -> int:
        return self.prefill_signature.prefill_length

    @property
    def embedding_size(self) -> int:
        return self.prefill_signature.embedding_size

    def _bind_caches(self) -> None:
        for name in self.prefill_signature.kv_inputs:
            cache = self.backend.get_device_input(self.prefill_model, name)
            self.backend.bind_device_input(self.decode_model, name, cache)
        for name in self.prefill_signature.conv_inputs:
            output = name.replace("past_conv_cache_", "conv_cache_out_")
            cache = self.backend.get_device_input(self.prefill_model, name)
            self.backend.bind_device_output(self.prefill_model, output, cache)
            self.backend.bind_device_input(self.decode_model, name, cache)
            self.backend.bind_device_output(self.decode_model, output, cache)
        for name in self.prefill_signature.recurrent_inputs:
            output = name.replace("past_recurrent_state_", "recurrent_state_out_")
            cache = self.backend.get_device_input(self.prefill_model, name)
            self.backend.bind_device_output(self.prefill_model, output, cache)
            self.backend.bind_device_input(self.decode_model, name, cache)
            self.backend.bind_device_output(self.decode_model, output, cache)

    def clear_session(self) -> None:
        self.session_state.status = SessionStatus.EXECUTING
        try:
            cache_names = tuple(
                dict.fromkeys(
                    self.prefill_signature.kv_inputs
                    + self.prefill_signature.conv_inputs
                    + self.prefill_signature.recurrent_inputs
                )
            )
            for name in cache_names:
                cache = self.backend.get_device_input(self.prefill_model, name)
                self.backend.zero_tensor(cache)
        except Exception:
            self._mark_invalid()
            raise
        self.session_state = ModuleSessionState()

    def _set_named_inputs(self, model, values: dict[str, Any]) -> None:
        self.backend.set_host_inputs(model, values)

    def _chunk_values(
        self,
        signature: Qwen35GraphSignature,
        embeddings: np.ndarray,
        positions: np.ndarray,
        attention_mask: np.ndarray,
        valid_length: int,
        current_length: int,
    ) -> dict[str, Any]:
        sequence = signature.prefill_length
        padded_embeddings = np.zeros((1, sequence, signature.embedding_size), dtype=signature.embeddings.dtype)
        padded_embeddings[:, :current_length] = embeddings
        padded_positions = _pad_positions(positions, sequence, signature.time_positions.dtype)
        if padded_positions.ndim == 1:
            padded_positions = np.broadcast_to(padded_positions, (3, sequence)).copy()
        mask = np.zeros((1, sequence), dtype=signature.attention_mask.dtype)
        mask[:, :current_length] = attention_mask
        return {
            signature.embeddings.name: padded_embeddings,
            signature.time_positions.name: padded_positions[0],
            signature.height_positions.name: padded_positions[1],
            signature.width_positions.name: padded_positions[2],
            signature.valid_length.name: np.array([valid_length], dtype=signature.valid_length.dtype),
            signature.current_length.name: np.array([current_length], dtype=signature.current_length.dtype),
            signature.attention_mask.name: mask,
        }

    def prefill(self, inputs: PrefillInputs, *, progress_callback=None) -> PrefillOutputs:
        if not isinstance(inputs, PrefillInputs):
            raise TypeError("inputs must be PrefillInputs")
        embeddings = np.asarray(inputs.embeddings)
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prefill embeddings must have shape (1, T, D)")
        input_length = int(inputs.input_length)
        if embeddings.shape[1] != input_length:
            raise ValueError("input_length does not match embedding sequence")
        if embeddings.shape[2] != self.embedding_size:
            raise ValueError("Prefill embedding width does not match the graph")
        if inputs.positions is not None:
            positions = np.asarray(inputs.positions)
            if positions.ndim not in (1, 2) or positions.shape[-1] != input_length:
                raise ValueError("positions must have shape (T,) or (3, T)")
            if positions.ndim == 2 and positions.shape[0] != 3:
                raise ValueError("rank-2 positions must have shape (3, T)")
        attention_mask = np.ones((1, input_length), dtype=np.float16)
        if inputs.attention_mask is not None:
            attention_mask = np.asarray(inputs.attention_mask)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.reshape(1, -1)
            if attention_mask.shape != (1, input_length):
                raise ValueError("attention_mask must have shape (1, T)")
        if self.context_length + input_length > self.context_max_length:
            raise ValueError("input exceeds model context length")
        self._begin_execution()
        logits = None
        try:
            chunk_count = (input_length + self.prefill_length - 1) // self.prefill_length
            for chunk_index, start in enumerate(
                range(0, input_length, self.prefill_length), start=1
            ):
                current_length = min(self.prefill_length, input_length - start)
                chunk_embeddings = embeddings[:, start : start + current_length]
                if inputs.positions is None:
                    chunk_positions = np.arange(
                        self.context_length + start,
                        self.context_length + start + current_length,
                        dtype=self.prefill_signature.time_positions.dtype,
                    )
                else:
                    chunk_positions = (
                        positions[..., start : start + current_length]
                        + self.context_length
                    )
                chunk_mask = attention_mask[:, start : start + current_length]
                values = self._chunk_values(
                    self.prefill_signature,
                    chunk_embeddings,
                    chunk_positions,
                    chunk_mask,
                    valid_length=self.context_length + start,
                    current_length=current_length,
                )
                self._set_named_inputs(self.prefill_model, values)
                self.backend.run(self.prefill_model)
                logits = self.backend.get_output(
                    self.prefill_model,
                    self.prefill_signature.logits.name,
                )
                if progress_callback is not None:
                    progress_callback(chunk_index, chunk_count)
            self.session_state.context_length += input_length
            self._mark_valid()
        except Exception:
            self._mark_invalid()
            raise
        return PrefillOutputs(logits=logits)

    def decode(self, inputs: DecodeInputs) -> DecodeOutputs:
        if not isinstance(inputs, DecodeInputs):
            raise TypeError("inputs must be DecodeInputs")
        if self.remaining_context <= 0:
            raise ValueError("input exceeds model context length")
        embedding = np.asarray(inputs.embedding)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, 1, -1)
        elif embedding.ndim == 2:
            embedding = embedding.reshape(1, 1, -1)
        if embedding.shape != (1, 1, self.embedding_size):
            raise ValueError("Decode embedding must have shape (1, 1, D)")
        position = self.context_length if inputs.position is None else int(np.asarray(inputs.position).reshape(-1)[0])
        self._begin_execution()
        try:
            signature = self.decode_signature
            values = {
                signature.embeddings.name: embedding,
                signature.time_positions.name: np.array([position], dtype=signature.time_positions.dtype),
                signature.height_positions.name: np.array([position], dtype=signature.height_positions.dtype),
                signature.width_positions.name: np.array([position], dtype=signature.width_positions.dtype),
                signature.valid_length.name: np.array([self.context_length], dtype=signature.valid_length.dtype),
                signature.current_length.name: np.array([1], dtype=signature.current_length.dtype),
                signature.attention_mask.name: np.ones((1, 1), dtype=signature.attention_mask.dtype),
            }
            self._set_named_inputs(self.decode_model, values)
            self.backend.run(self.decode_model)
            logits = self.backend.get_output(self.decode_model, signature.logits.name)
            self.session_state.context_length += 1
            self._mark_valid()
        except Exception:
            self._mark_invalid()
            raise
        return DecodeOutputs(logits=logits)
