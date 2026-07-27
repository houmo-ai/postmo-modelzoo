# Copyright (c) 2026 HOUMO AI
#
# File: houmo_module.py
# Description:
#   Base interface for Houmo runtime modules.
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

from .types import Stage, StageInputs, StageOutputs


class HoumoModule(ABC):
    """Common interface for Houmo runtime graph execution."""

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """Load runtime graphs and initialize device resources."""

    @abstractmethod
    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        """Bind inputs for one model stage."""

    @abstractmethod
    def run(self, stage: Stage) -> None:
        """Execute and synchronize one model stage."""

    @abstractmethod
    def get_output(self, stage: Stage) -> StageOutputs:
        """Read outputs from one model stage."""
