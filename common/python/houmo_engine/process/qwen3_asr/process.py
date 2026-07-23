# Copyright (c) 2026 HOUMO AI
#
# File: process.py
# Description:
#   Qwen3-ASR input and output Process implementation.
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

import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoConfig
import torchaudio
from ...core import ModelProcess
from ...core.types import GenerationState, StageInputs, StageOutputs
from ...perf import PerfTracker

AUDIO_PAD_TOKEN = "<|audio_pad|>"
ASR_TEXT_PATTERN = re.compile(r"(?<=<asr_text>)[\s\S]*")
PUNCTUATION_CHARS = set("，。！？；：" "''（）【】《》、·…—" ",.!?;:\"'()[]<>-" " ")


@dataclass
class Qwen3AsrFeatureChunk:
    input_features: torch.Tensor
    feature_length: int
    chunk_index: int


@dataclass
class Qwen3AsrPreparedRequest:
    input_ids: torch.Tensor
    feature_chunks: tuple[Qwen3AsrFeatureChunk, ...]
    audio_length_s: float


@dataclass
class Qwen3AsrPrefillRequest:
    token_embeds: np.ndarray
    current_length: int
    chunk_index: int


def is_valid_char(codepoint: int) -> bool:
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
        or 0x0030 <= codepoint <= 0x0039
        or chr(codepoint) in PUNCTUATION_CHARS
    )


def filter_valid_chars(text: str) -> str:
    return "".join(character for character in text if is_valid_char(ord(character)))


def extract_asr_text(text: str) -> str:
    match = ASR_TEXT_PATTERN.search(text)
    return match.group() if match else text


def _load_embedding(path, embedding_size: int) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    elif hasattr(value, "weight"):
        value = value.weight
    value = value.reshape(-1, embedding_size).to(torch.float16).cpu().numpy()
    return np.ascontiguousarray(value)


def _build_processor(processor_path):
    # Keep qwen_asr optional until an engine is actually constructed so that
    # importing the sample and running --help never initializes model code.
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    return Qwen3ASRProcessor.from_pretrained(processor_path, fix_mistral_regex=True)


class Qwen3AsrProcess(ModelProcess):
    """Qwen3-ASR audio preprocessing and token postprocessing."""

    def __init__(
        self,
        processor_path,
        embedding_path,
        embedding_size: int,
        encode_feature_length: int,
        prefill_length: int,
        *,
        stream_slide_len: int = 10,
        perf: PerfTracker,
    ):
        self.perf = perf
        self.embedding_size = embedding_size
        self.encode_feature_length = encode_feature_length
        self.prefill_length = prefill_length
        self.stream_slide_len = stream_slide_len
        self.processor = _build_processor(processor_path)
        self.tokenizer = self.processor.tokenizer
        config = AutoConfig.from_pretrained(processor_path, trust_remote_code=True)
        config_hidden_size = int(config.thinker_config.text_config.hidden_size)
        if config_hidden_size != embedding_size:
            raise ValueError(
                f"processor hidden size {config_hidden_size} does not match "
                f"prefill graph hidden size {embedding_size}"
            )
        vocab = self.tokenizer.get_vocab()
        self.audio_pad_id = (
            self.tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)
            if AUDIO_PAD_TOKEN in vocab
            else self.tokenizer.encode(AUDIO_PAD_TOKEN, add_special_tokens=False)[0]
        )
        self.embedding_weight = _load_embedding(embedding_path, embedding_size)
        if self.embedding_weight.shape[1] != embedding_size:
            raise ValueError("embedding hidden size does not match prefill graph")

    @staticmethod
    def _load_audio(audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return np.asarray(audio, dtype=np.float32).reshape(-1)
        waveform, sample_rate = torchaudio.load(str(audio))
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        return waveform.numpy()

    def preprocess(
        self,
        audio,
        system_prompt: str,
    ) -> Qwen3AsrPreparedRequest:
        with self.perf.scope("asr.audio.preprocess"):
            audio_array = self._load_audio(audio)
            if audio_array.size == 0:
                raise ValueError("audio must not be empty")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": "placeholder"}],
                },
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(
                text=prompt,
                audio=audio_array,
                return_tensors="pt",
                padding=True,
            )
        features = inputs["input_features"]
        chunks = []
        for chunk_index, start in enumerate(range(0, features.shape[2], self.encode_feature_length)):
            value = features[:, :, start : start + self.encode_feature_length]
            chunks.append(
                Qwen3AsrFeatureChunk(
                    input_features=value,
                    feature_length=int(value.shape[2]),
                    chunk_index=chunk_index,
                )
            )
        if not chunks:
            raise ValueError("processor produced no audio features")
        return Qwen3AsrPreparedRequest(
            input_ids=inputs["input_ids"],
            feature_chunks=tuple(chunks),
            audio_length_s=len(audio_array) / 16000,
        )

    def prepare_encode(self, chunk: Qwen3AsrFeatureChunk) -> StageInputs:
        features = np.asarray(chunk.input_features, dtype=np.float32)
        if chunk.feature_length > self.encode_feature_length:
            raise ValueError("audio feature chunk exceeds encode graph capacity")
        features = np.pad(
            features,
            ((0, 0), (0, 0), (0, self.encode_feature_length - chunk.feature_length)),
        )
        return StageInputs(
            tensors=(features, np.array([chunk.feature_length], dtype=np.int32)),
            metadata={
                "feature_length": chunk.feature_length,
                "chunk_index": chunk.chunk_index,
            },
        )

    @staticmethod
    def audio_output_length(input_length: int) -> int:
        remainder = input_length % 100
        feature_length = (remainder - 1) // 2 + 1
        return int(((feature_length - 1) // 2 + 1 - 1) // 2 + 1 + (input_length // 100) * 13)

    def merge_encode(
        self,
        request: Qwen3AsrPreparedRequest,
        chunk: Qwen3AsrFeatureChunk,
        outputs: StageOutputs,
    ) -> Qwen3AsrPrefillRequest:
        output_length = self.audio_output_length(chunk.feature_length)
        audio_embeds = np.asarray(outputs.tensors[0])[:, :output_length, :]
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[np.newaxis, :, :]
        audio_embeds = audio_embeds.astype(np.float16, copy=False)

        ids = request.input_ids.detach().cpu().numpy()
        text_embeds = self.embedding_weight[ids]
        pad_indices = np.where(ids == self.audio_pad_id)[1]
        if pad_indices.size == 0:
            raise ValueError("processor input does not contain an audio pad span")
        start = int(pad_indices[0])
        end = int(pad_indices[-1])

        fused = np.zeros((1, self.prefill_length, self.embedding_size), dtype=np.float16)
        cursor = min(start, self.prefill_length)
        fused[:, :cursor, :] = text_embeds[:, :cursor, :]
        audio_length = min(audio_embeds.shape[1], self.prefill_length - cursor)
        fused[:, cursor : cursor + audio_length, :] = audio_embeds[:, :audio_length, :]
        cursor += audio_length
        tail_start = end + 1
        tail_length = min(max(text_embeds.shape[1] - tail_start, 0), self.prefill_length - cursor)
        if tail_length:
            fused[:, cursor : cursor + tail_length, :] = text_embeds[:, tail_start : tail_start + tail_length, :]
            cursor += tail_length
        return Qwen3AsrPrefillRequest(
            token_embeds=fused,
            current_length=cursor,
            chunk_index=chunk.chunk_index,
        )

    def prepare_prefill(
        self,
        request: Qwen3AsrPrefillRequest,
        state: GenerationState,
    ) -> StageInputs:
        return StageInputs(
            tensors=(
                request.token_embeds,
                np.array([state.context_length], dtype=np.int32),
                np.array([request.current_length], dtype=np.int32),
            ),
            metadata={
                "current_length": request.current_length,
                "chunk_index": request.chunk_index,
            },
        )

    def prepare_decode(self, token: int, state: GenerationState) -> StageInputs:
        embedding = self.embedding_weight[token : token + 1].reshape(1, 1, -1)
        return StageInputs(
            tensors=(
                embedding,
                np.array([state.context_length], dtype=np.int32),
                np.array([1], dtype=np.int32),
            ),
            metadata={"token_id": token},
        )

    def postprocess(self, state: GenerationState, *, final: bool = False) -> str:
        if not final and len(state.generated_ids) <= self.stream_slide_len:
            return ""
        stable_ids = (
            state.generated_ids
            if final
            else state.generated_ids[: -self.stream_slide_len]
        )
        with self.perf.scope("asr.text.postprocess"):
            text = self.tokenizer.decode(stable_ids, skip_special_tokens=True)
            if final:
                text = filter_valid_chars(extract_asr_text(text))
            else:
                match = ASR_TEXT_PATTERN.search(text)
                if not match:
                    return ""
                text = filter_valid_chars(match.group())
        delta = text[len(state.emitted_text) :]
        if delta:
            state.emitted_text = text
        return delta


__all__ = [
    "Qwen3AsrFeatureChunk",
    "Qwen3AsrPreparedRequest",
    "Qwen3AsrPrefillRequest",
    "Qwen3AsrProcess",
    "extract_asr_text",
    "filter_valid_chars",
    "is_valid_char",
]
