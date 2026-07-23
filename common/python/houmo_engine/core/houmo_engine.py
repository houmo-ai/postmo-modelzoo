# Copyright (c) 2026 HOUMO AI
#
# File: houmo_engine.py
# Description:
#   Base interface for Houmo inference engines.
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

from abc import ABC, abstractmethod


class HoumoEngine(ABC):
    """Common interface for Houmo inference engines."""

    def __init__(self, batch: int = 1):
        if batch <= 0:
            raise ValueError("batch must be greater than zero")
        self.batch = batch

    @abstractmethod
    def generate(self, request, **kwargs):
        """Yield model output chunks for one request."""
