# Copyright (c) 2026 HOUMO AI
#
# File: tts.py
# Description:
#   Sampling parameters for Qwen3-TTS Talker and Code Predictor.
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

"""Configuration for Qwen3-TTS stochastic sampling.

Qwen3-TTS reproduces the original demo sampling pipeline built from
``transformers`` logits processors followed by multinomial sampling. This
dataclass only carries the numeric parameters so importing it never pulls in
torch or transformers; the engine turns these values into logits processors
when a model is actually constructed.

Defaults mirror ``Qwen3TTSCodecGenerator.init_logits_processors`` in the
original single-file demo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3TtsSamplingParams:
    """Talker and Code Predictor (subtalker) sampling configuration."""

    # Talker
    min_new_tokens: int = 2
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True

    # Code Predictor (subtalker)
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9
    subtalker_do_sample: bool = True


__all__ = ["Qwen3TtsSamplingParams"]
