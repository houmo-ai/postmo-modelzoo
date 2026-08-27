# Copyright (c) 2026 HOUMO AI
#
# File: input.py
# Description:
#   Deterministic fixed-length Token ID generation for performance tests.
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

"""Deterministic fixed Token ID generation."""

import numpy as np


def generate_token_ids(
    length: int,
    vocabulary_size: int,
    *,
    seed: int = 0,
    excluded_ids: set[int] | frozenset[int] = frozenset(),
) -> np.ndarray:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be a positive integer")
    valid = np.array([i for i in range(vocabulary_size) if i not in excluded_ids], dtype=np.int64)
    if valid.size == 0:
        raise ValueError("excluded_ids remove the entire vocabulary")
    return np.ascontiguousarray(np.random.default_rng(seed).choice(valid, size=length, replace=True))
