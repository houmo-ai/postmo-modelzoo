# Copyright (c) 2026 HOUMO AI
#
# File: greedy.py
# Description:
#   Greedy token sampling for text generation.
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

"""Deterministic argmax sampling for Text-only v1."""

from typing import Any

import numpy as np

from postmo_engine.core import SampleResult


class GreedySampler:
    """Select the largest value from the final vocabulary vector."""

    def sample(self, logits: Any) -> SampleResult:
        values = np.asarray(logits)
        if values.size == 0:
            raise ValueError("logits must not be empty")
        if values.ndim == 0:
            raise ValueError("logits must have a vocabulary dimension")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("logits must contain numeric values")
        vocabulary = values.reshape(-1, values.shape[-1])[-1]
        if not np.isfinite(vocabulary).any():
            raise ValueError("logits must contain at least one finite value")
        return SampleResult(token_id=int(np.argmax(vocabulary)))
