# Copyright (c) 2026 HOUMO AI
#
# File: model_process.py
# Description:
#   Base interface for model input and output processing.
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


class ModelProcess(ABC):
    """Common interface for model input and output processing."""

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """Convert user input into a model-specific prepared request."""

    @abstractmethod
    def postprocess(self, state, *, final: bool = False):
        """Return an incremental output or the final remainder."""
