# Copyright (c) 2026 HOUMO AI
#
# File: greedy.py
# Description:
#   Deterministic token sampling implementation.
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

"""Deterministic token selection with optional logits processing.

The sampler always selects the final token with ``argmax``. Its processing
pipeline is:

1. repetition penalty
2. presence penalty
3. top-k filtering on logits
4. temperature scaling
5. softmax
6. top-p filtering on probabilities
7. argmax

The default parameters make every optional step a no-op, so
``GreedySampler().sample(logits)`` is equivalent to ``numpy.argmax(logits)``.

Non-default parameters explicitly enable preprocessing before argmax:

- ``repetition_penalty != 1.0`` penalizes token IDs seen previously.
- ``presence_penalty != 0.0`` subtracts a fixed value from seen token IDs.
- ``top_k`` keeps only the highest-k logits.
- ``temperature != 1.0`` rescales logits before softmax.
- ``top_p < 1.0`` keeps the smallest highest-probability set whose cumulative
  probability reaches top-p.

These options do not enable random sampling. In particular, temperature,
top-k, and top-p usually do not change the selected token because the final
operation remains argmax. They are retained to match the existing deterministic
Houmo sampling pipeline and to expose the processed probability distribution.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedySamplingParams:
    """Configuration for deterministic argmax token selection.

    Defaults are intentionally neutral and produce the same token as applying
    ``argmax`` directly to the original logits.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens_to_keep: int = 1

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be greater than 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than 0 or None")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be greater than 0")
        if self.min_tokens_to_keep <= 0:
            raise ValueError("min_tokens_to_keep must be greater than 0")


class GreedySampler:
    """Apply optional processing and deterministically select with argmax."""

    def __init__(self, params: GreedySamplingParams | None = None):
        self.params = params or GreedySamplingParams()

    def sample(self, logits, previous_tokens=None) -> int:
        """Return the deterministic next token for one vocabulary vector."""
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits.size == 0:
            raise ValueError("logits must not be empty")
        previous_tokens = [] if previous_tokens is None else list(previous_tokens)

        if self.params.top_k == 1:
            processed = logits.copy()
            self._apply_penalties(processed, previous_tokens)
            return int(np.argmax(processed))

        return int(np.argmax(self.processed_probs(logits, previous_tokens)))

    def processed_probs(self, logits, previous_tokens=None) -> np.ndarray:
        """Return the normalized probabilities used by the final argmax."""
        processed = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
        if processed.size == 0:
            raise ValueError("logits must not be empty")
        previous_tokens = [] if previous_tokens is None else list(previous_tokens)

        self._apply_penalties(processed, previous_tokens)
        self._apply_top_k(processed)
        self._apply_temperature(processed)
        probabilities = self._softmax(processed)
        self._apply_top_p(probabilities)
        return probabilities

    def _apply_penalties(self, logits: np.ndarray, previous_tokens) -> None:
        if not previous_tokens:
            return

        apply_repetition = self.params.repetition_penalty != 1.0
        apply_presence = self.params.presence_penalty != 0.0
        if not apply_repetition and not apply_presence:
            return

        vocabulary_size = logits.shape[0]
        for token_id in {int(token) for token in previous_tokens}:
            if not 0 <= token_id < vocabulary_size:
                continue

            value = logits[token_id]
            if apply_repetition:
                value = (
                    value * self.params.repetition_penalty
                    if value < 0
                    else value / self.params.repetition_penalty
                )
            if apply_presence:
                value -= self.params.presence_penalty
            logits[token_id] = value

    def _apply_top_k(self, logits: np.ndarray) -> None:
        top_k = self.params.top_k
        vocabulary_size = logits.shape[0]
        if top_k is None or top_k >= vocabulary_size:
            return

        keep_indices = np.argpartition(logits, vocabulary_size - top_k)[
            vocabulary_size - top_k :
        ]
        remove = np.ones(vocabulary_size, dtype=bool)
        remove[keep_indices] = False
        logits[remove] = -np.inf

    def _apply_temperature(self, logits: np.ndarray) -> None:
        if self.params.temperature != 1.0:
            logits /= self.params.temperature

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        maximum = np.max(logits)
        probabilities = np.exp(logits - maximum)
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("logits do not produce a valid probability distribution")
        probabilities /= total
        return probabilities

    def _apply_top_p(self, probabilities: np.ndarray) -> None:
        if self.params.top_p >= 1.0:
            return

        descending = np.argsort(-probabilities)
        cumulative = np.cumsum(probabilities[descending])
        cutoff = int(np.searchsorted(cumulative, self.params.top_p, side="left"))
        cutoff = max(cutoff, self.params.min_tokens_to_keep - 1)
        cutoff = min(cutoff, probabilities.size - 1)

        keep = np.zeros(probabilities.shape[0], dtype=bool)
        keep[descending[: cutoff + 1]] = True
        probabilities[~keep] = 0.0
        probabilities /= probabilities.sum()
