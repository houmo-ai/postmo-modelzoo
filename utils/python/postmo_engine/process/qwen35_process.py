# Copyright (c) 2026 HOUMO AI
#
# File: qwen35_process.py
# Description:
#   Qwen3.5 Text-only request input and output processing.
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

"""Qwen3.5 Text-only request and output processing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from postmo_engine.core import (
    DecodeInputs,
    DecodeOutputs,
    PostMoProcessor,
    PrefillInputs,
    PrefillOutputs,
)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class Qwen35PreparedRequest:
    """Complete logical text request before Runtime-specific chunking."""

    input_ids: np.ndarray
    embeddings: np.ndarray
    positions: np.ndarray
    attention_mask: np.ndarray

    @property
    def input_length(self) -> int:
        return int(self.input_ids.shape[1])


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_tokenizer(path: str | Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required when tokenizer is not injected"
        ) from error
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _load_embedding(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required to load non-NPY embedding weights"
        ) from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    elif hasattr(value, "weight"):
        value = value.weight
    return _as_numpy(value)


def _is_stable_char(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
    )


class Qwen35Process(PostMoProcessor):
    """Prepare Qwen3.5 text inputs without owning Module cache state."""

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        embedding_path: str | Path | None = None,
        embedding_size: int | None = None,
        *,
        tokenizer: Any | None = None,
        embedding_weight: Any | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if tokenizer is None:
            if tokenizer_path is None:
                raise ValueError("tokenizer or tokenizer_path is required")
            tokenizer = _load_tokenizer(tokenizer_path)
        if embedding_weight is None:
            if embedding_path is None:
                raise ValueError("embedding_weight or embedding_path is required")
            embedding_weight = _load_embedding(embedding_path)
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")

        weight = _as_numpy(embedding_weight)
        if weight.ndim == 1:
            if embedding_size is None or embedding_size <= 0:
                raise ValueError("embedding_size is required for flat embedding weights")
            if weight.size % embedding_size:
                raise ValueError("embedding weight size is not divisible by embedding_size")
            weight = weight.reshape(-1, embedding_size)
        if weight.ndim != 2:
            raise ValueError("embedding weights must be a two-dimensional matrix")
        if embedding_size is not None and weight.shape[1] != embedding_size:
            raise ValueError(
                f"embedding weights expect width {embedding_size}, got {weight.shape[1]}"
            )

        self.tokenizer = tokenizer
        if not np.issubdtype(weight.dtype, np.floating):
            raise TypeError("embedding weights must use a floating dtype")
        self.embedding_weight = np.ascontiguousarray(weight)
        self.system_prompt = system_prompt
        eos = tokenizer.eos_token_id
        eos_values = eos if isinstance(eos, (list, tuple, set)) else (eos,)
        self.eos_token_ids = frozenset(int(token) for token in eos_values)

    @property
    def embedding_size(self) -> int:
        return int(self.embedding_weight.shape[1])

    @staticmethod
    def text_positions(length: int) -> np.ndarray:
        if isinstance(length, bool) or length <= 0:
            raise ValueError("length must be greater than zero")
        positions = np.arange(length, dtype=np.int32)
        return np.broadcast_to(positions, (3, length)).copy()

    def preprocess(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> Qwen35PreparedRequest:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        selected_system_prompt = self.system_prompt if system_prompt is None else system_prompt
        if not isinstance(selected_system_prompt, str):
            raise TypeError("system_prompt must be a string")
        messages = []
        if selected_system_prompt:
            messages.append({"role": "system", "content": selected_system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            text,
            return_tensors="np",
            add_special_tokens=False,
        )
        input_ids = _as_numpy(encoded["input_ids"]).astype(np.int64, copy=False)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("tokenizer must return non-empty batch-1 input_ids")
        minimum = int(input_ids.min())
        maximum = int(input_ids.max())
        if minimum < 0 or maximum >= self.embedding_weight.shape[0]:
            raise ValueError("tokenizer returned an ID outside the embedding vocabulary")
        embeddings = self.embedding_weight[input_ids]
        length = int(input_ids.shape[1])
        return Qwen35PreparedRequest(
            input_ids=np.array(input_ids, copy=True),
            embeddings=np.ascontiguousarray(embeddings),
            positions=self.text_positions(length),
            attention_mask=np.ones((1, length), dtype=np.float16),
        )

    def build_prefill_inputs(self, prepared: Qwen35PreparedRequest) -> PrefillInputs:
        if not isinstance(prepared, Qwen35PreparedRequest):
            raise TypeError("prepared must be a Qwen35PreparedRequest")
        return PrefillInputs(
            embeddings=prepared.embeddings,
            input_length=prepared.input_length,
            positions=prepared.positions,
            attention_mask=prepared.attention_mask,
        )

    def build_prefill_inputs_from_token_ids(self, token_ids: Any) -> PrefillInputs:
        """Build logical Prefill inputs for a caller-owned token sequence."""
        ids = _as_numpy(token_ids).astype(np.int64, copy=False)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("token_ids must be a non-empty batch-1 sequence")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding_weight.shape[0]:
            raise ValueError("token_ids contain an ID outside the embedding vocabulary")
        input_length = int(ids.shape[1])
        return PrefillInputs(
            embeddings=np.ascontiguousarray(self.embedding_weight[ids]),
            input_length=input_length,
            positions=self.text_positions(input_length),
            attention_mask=np.ones((1, input_length), dtype=np.float16),
        )

    def build_decode_inputs(self, token_id: int) -> DecodeInputs:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")
        token_id = int(token_id)
        if not 0 <= token_id < self.embedding_weight.shape[0]:
            raise ValueError("token_id is outside the embedding vocabulary")
        embedding = self.embedding_weight[token_id].reshape(1, 1, -1)
        return DecodeInputs(embedding=np.ascontiguousarray(embedding))

    def process_prefill_outputs(self, outputs: PrefillOutputs) -> np.ndarray:
        if not isinstance(outputs, PrefillOutputs):
            raise TypeError("outputs must be PrefillOutputs")
        return np.asarray(outputs.logits)

    def process_decode_outputs(self, outputs: DecodeOutputs) -> np.ndarray:
        if not isinstance(outputs, DecodeOutputs):
            raise TypeError("outputs must be DecodeOutputs")
        return np.asarray(outputs.logits)

    def decode_text(
        self,
        token_ids: tuple[int, ...],
        emitted_text: str,
        *,
        final: bool = False,
    ) -> str:
        if not isinstance(token_ids, tuple):
            raise TypeError("token_ids must be a tuple")
        if not isinstance(emitted_text, str):
            raise TypeError("emitted_text must be a string")
        text = self.tokenizer.decode(list(token_ids))
        if not text.startswith(emitted_text):
            raise ValueError("emitted_text is not a prefix of decoded text")
        delta = text[len(emitted_text) :]
        if final or (delta and _is_stable_char(delta[-1])):
            return delta
        return ""
