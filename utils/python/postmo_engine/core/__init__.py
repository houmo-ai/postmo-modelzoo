# Copyright (c) 2026 HOUMO AI
#
# File: __init__.py
# Description:
#   Public exports for the minimal PostMo Engine contracts.
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

"""Minimal public contracts for the PostMo Text-only implementation."""

from .capabilities import CapabilityAccess, EngineCapabilities
from .engine import PostMoEngine
from .errors import InvalidSessionError, PostMoError, UnsupportedFeatureError
from .module import ModuleSessionState, PostMoModule
from .processor import PostMoProcessor
from .types import (
    DecodeInputs,
    DecodeOutputs,
    EngineRequest,
    OutputChunk,
    PrefillInputs,
    PrefillOutputs,
    RequestResult,
    SampleResult,
    SessionStatus,
    StopReason,
)

__all__ = [
    "CapabilityAccess",
    "DecodeInputs",
    "DecodeOutputs",
    "EngineCapabilities",
    "EngineRequest",
    "InvalidSessionError",
    "ModuleSessionState",
    "OutputChunk",
    "PostMoEngine",
    "PostMoError",
    "PostMoModule",
    "PostMoProcessor",
    "PrefillInputs",
    "PrefillOutputs",
    "RequestResult",
    "SampleResult",
    "SessionStatus",
    "StopReason",
    "UnsupportedFeatureError",
]
