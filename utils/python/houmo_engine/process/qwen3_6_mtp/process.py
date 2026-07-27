# Copyright (c) 2026 HOUMO AI
#
# File: process.py
# Description:
#   Qwen3.6 MTP input and output Process implementation.
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

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ...core import ModelProcess
from ...core.types import GenerationState, StageInputs
from ...perf import PerfTracker


@dataclass
class Qwen36MtpPreparedRequest:
    """Tokenized single-turn Qwen3.6 MTP request."""

    input_ids: np.ndarray


@dataclass
class Qwen36MtpGenerationState(GenerationState):
    """Engine-owned CPU state specific to Qwen3.6 MTP generation."""

    mtp_context_length: int = 0
    pending_token: int | None = None
    draft_anchor_hidden: np.ndarray | torch.Tensor | None = None
    finish_reason: str | None = None


def _load_embedding(path, embedding_size: int) -> torch.Tensor:
    if embedding_size <= 0:
        raise ValueError("embedding_size must be greater than zero")

    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        if "weight" not in value:
            raise KeyError(f"embedding state_dict at {path} does not contain 'weight'")
        value = value["weight"]
    elif isinstance(value, torch.nn.Embedding):
        value = value.weight.detach()
    elif hasattr(value, "weight"):
        value = value.weight

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.numel() % embedding_size:
        raise ValueError(
            f"embedding with {value.numel()} values cannot be reshaped to "
            f"hidden size {embedding_size}"
        )
    value = value.reshape(-1, embedding_size)
    if value.ndim != 2 or value.shape[1] != embedding_size:
        raise ValueError("embedding hidden size does not match the model graph")
    return value.to(dtype=torch.float16, device="cpu").contiguous()


def _stop_token_ids(tokenizer) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        stop_ids: set[int] = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        stop_ids = {int(token_id) for token_id in eos_token_id}
    else:
        stop_ids = {int(eos_token_id)}

    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and int(token_id) >= 0:
            stop_ids.add(int(token_id))
    return stop_ids


def _is_stable_char(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
        or 0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
    )


class Qwen36MtpProcess(ModelProcess):
    """Qwen3.6 MTP request preprocessing and token postprocessing."""

    def __init__(
        self,
        tokenizer_path,
        embedding_path,
        embedding_size: int,
        *,
        perf: PerfTracker,
    ):
        self.perf = perf
        self.embedding_size = int(embedding_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.embedding_weight = _load_embedding(
            embedding_path, self.embedding_size
        )
        self.stop_token_ids = _stop_token_ids(self.tokenizer)

    @staticmethod
    def _token_ids(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        token_ids = np.asarray(value, dtype=np.int64).reshape(-1)
        if token_ids.size == 0:
            raise ValueError("token IDs must not be empty")
        return np.ascontiguousarray(token_ids)

    @staticmethod
    def _positions(start: int, length: int) -> tuple[np.ndarray, ...]:
        values = np.arange(start, start + length, dtype=np.int32).reshape(1, -1)
        return values, values.copy(), values.copy()

    @staticmethod
    def _attention_mask(length: int, current_length: int) -> np.ndarray:
        mask = np.zeros((1, length), dtype=np.float16)
        mask[:, :current_length] = 1
        return mask

    @staticmethod
    def _hidden_states(value, embedding_size: int) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        hidden = np.asarray(value, dtype=np.float16)
        if hidden.ndim == 2:
            hidden = hidden[np.newaxis, :, :]
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("hidden states must have shape [1, sequence, hidden]")
        if hidden.shape[2] != embedding_size:
            raise ValueError("hidden state size does not match embedding_size")
        return np.ascontiguousarray(hidden)

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        if np.any(token_ids < 0) or np.any(token_ids >= self.embedding_weight.shape[0]):
            raise ValueError("token ID is outside the embedding vocabulary")
        with self.perf.scope("llm_mtp.embedding"):
            ids = torch.from_numpy(token_ids).long().view(1, -1)
            embeddings = F.embedding(ids, self.embedding_weight)
            return embeddings.detach().cpu().numpy()

    def _padded_embeddings(self, token_ids: np.ndarray, length: int) -> np.ndarray:
        if token_ids.size > length:
            raise ValueError(f"token block is too long: {token_ids.size} > {length}")
        embeddings = self._embed(token_ids)
        padded = np.zeros(
            (1, length, self.embedding_size), dtype=self.embedding_weight.numpy().dtype
        )
        padded[:, : token_ids.size, :] = embeddings
        return padded

    def preprocess(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> Qwen36MtpPreparedRequest:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        with self.perf.scope("llm_mtp.tokenize"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text, return_tensors="np", add_special_tokens=False
            )
        input_ids = np.asarray(inputs["input_ids"], dtype=np.int64).reshape(-1)
        if input_ids.size == 0:
            raise ValueError("tokenizer produced an empty prompt")
        return Qwen36MtpPreparedRequest(input_ids=np.ascontiguousarray(input_ids))

    def prepare_prefill_chunk(
        self,
        request: Qwen36MtpPreparedRequest,
        state: Qwen36MtpGenerationState,
        start: int,
        prefill_length: int,
        embedding_size: int,
    ) -> StageInputs:
        if embedding_size != self.embedding_size:
            raise ValueError("prefill embedding size does not match Process embedding")
        if start < 0 or start >= request.input_ids.size:
            raise ValueError("prefill start is outside the request")
        end = min(start + prefill_length, request.input_ids.size)
        token_ids = request.input_ids[start:end]
        current_length = int(token_ids.size)
        valid_length = state.context_length + start
        positions = self._positions(valid_length, prefill_length)
        return StageInputs(
            tensors=(
                self._padded_embeddings(token_ids, prefill_length),
                np.array([valid_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
                *positions,
                self._attention_mask(prefill_length, current_length),
            ),
            metadata={"current_length": current_length},
        )

    def prepare_mtp_prefill_chunk(
        self,
        hidden_states,
        token_ids,
        state: Qwen36MtpGenerationState,
        mtp_prefill_length: int,
    ) -> StageInputs:
        ids = self._token_ids(token_ids)
        hidden = self._hidden_states(hidden_states, self.embedding_size)
        current_length = int(ids.size)
        if hidden.shape[1] != current_length:
            raise ValueError("MTP hidden and token lengths differ")
        if current_length > mtp_prefill_length:
            raise ValueError("MTP prefill chunk exceeds graph capacity")
        padded_hidden = np.zeros(
            (1, mtp_prefill_length, self.embedding_size), dtype=np.float16
        )
        padded_hidden[:, :current_length, :] = hidden
        return StageInputs(
            tensors=(
                padded_hidden,
                self._padded_embeddings(ids, mtp_prefill_length),
                *self._positions(state.mtp_context_length, mtp_prefill_length),
                np.array([state.mtp_context_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
            ),
            metadata={"current_length": current_length},
        )

    def prepare_draft(
        self,
        hidden_state,
        token: int,
        position: int,
    ) -> StageInputs:
        hidden = self._hidden_states(hidden_state, self.embedding_size)
        if hidden.shape[1] != 1:
            raise ValueError("draft hidden state must contain exactly one position")
        token_ids = np.array([int(token)], dtype=np.int64)
        return StageInputs(
            tensors=(
                hidden,
                self._embed(token_ids),
                *self._positions(position, 1),
                np.array([position], dtype=np.int32),
                np.array([1], dtype=np.int32),
            ),
            metadata={"current_length": 1},
        )

    def prepare_verify(
        self,
        token_ids,
        state: Qwen36MtpGenerationState,
        verify_length: int,
    ) -> StageInputs:
        ids = self._token_ids(token_ids)
        if ids.size != verify_length:
            raise ValueError(
                f"verify block must contain {verify_length} tokens, got {ids.size}"
            )
        current_length = int(ids.size)
        return StageInputs(
            tensors=(
                self._padded_embeddings(ids, verify_length),
                np.array([state.context_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
                *self._positions(state.context_length, verify_length),
                self._attention_mask(verify_length, current_length),
            ),
            metadata={"current_length": current_length},
        )

    def postprocess(
        self,
        state: Qwen36MtpGenerationState,
        *,
        final: bool = False,
    ) -> str:
        with self.perf.scope("llm_mtp.text.postprocess"):
            text = self.tokenizer.decode(
                state.generated_ids, skip_special_tokens=True
            )
        delta = text[len(state.emitted_text) :]
        if final:
            state.emitted_text = text
            return delta
        if delta and _is_stable_char(delta[-1]):
            state.emitted_text = text
            return delta
        return ""


__all__ = [
    "Qwen36MtpGenerationState",
    "Qwen36MtpPreparedRequest",
    "Qwen36MtpProcess",
]
