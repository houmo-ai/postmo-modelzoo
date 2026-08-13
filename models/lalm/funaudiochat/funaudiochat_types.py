# Copyright (c) 2026 HOUMO AI
#
# File: funaudiochat_types.py
# Description:
#   Layer data contracts and generated events for Fun-Audio-Chat.
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

"""Dataclasses defining Fun-Audio-Chat requests, state, paths, and events."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from houmo_engine.core.types import GenerationState


@dataclass
class FunAudioChatRequest:
    """Describe an audio request and the pipeline that should process it."""

    stage: str
    audio: Any
    system_prompt: str


@dataclass
class PreparedAudioRequest:
    """Hold resampled audio and tokenized inputs for language inference."""

    waveform: np.ndarray
    processor_inputs: Any
    audio_length_s: float


@dataclass
class LanguagePrefill:
    """Hold padded language embeddings and metadata for prefill execution."""

    embeds: np.ndarray
    attention_mask: np.ndarray
    prompt_length: int
    original_text_embeds: np.ndarray


@dataclass
class VadPreparedRequest:
    """Hold normalized audio features prepared for VAD inference."""

    waveform: np.ndarray
    sample_rate: int
    features: np.ndarray


@dataclass
class FunAudioChatState(GenerationState):
    """Track CPU-side text, speech-token, and CRQ generation state."""

    speech_ids: list[int] = field(default_factory=list)
    generated_speech_tokens: list[int] = field(default_factory=list)
    crq_past_length: int = 0
    generate_speech: bool = False


@dataclass
class TextResult:
    """Event containing a generated text response and token counts."""

    text: str
    prompt_tokens: int
    generated_tokens: int


@dataclass
class SpeechResult:
    """Event containing generated text together with speech token IDs."""

    text: str
    speech_ids: list[int]
    prompt_tokens: int
    generated_tokens: int


@dataclass
class AudioResult:
    """Event containing synthesized audio and its sampling metadata."""

    waveform: Any
    sample_rate: int
    speech_ids: list[int]
    turn: int | None = None


@dataclass
class VadResult:
    """Event containing VAD segments and aggregate posterior statistics."""

    waveform: np.ndarray
    sample_rate: int
    segments: list[list[int]]
    stats: dict[str, Any]


@dataclass
class TurnResult:
    """Event containing one segmented conversation turn and its response."""

    turn: int
    start_ms: int
    end_ms: int
    input_waveform: np.ndarray
    text: str
    speech_ids: list[int]
    response_waveform: Any
    sample_rate: int


@dataclass
class PerformanceResult:
    """Event containing a performance label and collected report."""

    label: str
    report: Any


@dataclass
class FunAudioChatPaths:
    """Collect model, tokenizer, graph, and preprocessing resource paths."""

    tokenizer_dir: Path
    embedding_path: Path
    audio_embedding_path: Path
    pre_matching_path: Path
    flow_input_embedding_path: Path
    speaker_info_path: Path
    audio_encoder_path: Path
    prefill_path: Path
    decode_path: Path
    audio_tower_path: Path
    audio_decoder_prefill_path: Path
    audio_decoder_decode_path: Path
    flow_encoder_path: Path
    flow_spk_path: Path
    flow_decoder_path: Path
    hift_part1_path: Path
    hift_part2_path: Path
    vad_path: Path
    config_path: Path
    cmvn_path: Path


__all__ = [
    "AudioResult",
    "FunAudioChatPaths",
    "FunAudioChatRequest",
    "FunAudioChatState",
    "LanguagePrefill",
    "PerformanceResult",
    "PreparedAudioRequest",
    "SpeechResult",
    "TextResult",
    "TurnResult",
    "VadPreparedRequest",
    "VadResult",
]
