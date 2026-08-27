# Copyright (c) 2026 HOUMO AI
#
# File: capabilities.py
# Description:
#   Capability declarations and access checks for PostMo engines.
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

"""Small capability contract used for request preflight checks."""

from dataclasses import dataclass, field
from enum import Enum

from .errors import UnsupportedFeatureError


class CapabilityAccess(str, Enum):
    AVAILABLE = "available"
    PLANNED = "planned"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EngineCapabilities:
    """Feature access map for one concrete Engine implementation."""

    features: dict[str, CapabilityAccess] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    def access(self, feature: str) -> CapabilityAccess:
        return self.features.get(feature, CapabilityAccess.BLOCKED)

    def reason(self, feature: str) -> str:
        return self.reasons.get(feature, "feature is not available")

    def require(self, feature: str) -> None:
        access = self.access(feature)
        if access is not CapabilityAccess.AVAILABLE:
            raise UnsupportedFeatureError(f"{feature}: {self.reason(feature)}")
