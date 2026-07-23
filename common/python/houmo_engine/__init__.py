# Copyright (c) 2026 HOUMO AI
#
# File: __init__.py
# Description:
#   Houmo Python Engine package exports.
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

from .core import (
    HoumoEngine,
    HoumoModule,
    ModelProcess,
)

__version__ = "0.3.0"

__all__ = [
    "HoumoEngine",
    "HoumoModule",
    "ModelProcess",
    "Qwen35Engine",
    "Qwen36MtpEngine",
    "Qwen3AsrEngine",
]


def __getattr__(name):
    if name == "Qwen35Engine":
        from .engine.qwen3_5 import Qwen35Engine

        return Qwen35Engine
    if name == "Qwen36MtpEngine":
        from .engine.qwen3_6_mtp import Qwen36MtpEngine

        return Qwen36MtpEngine
    if name == "Qwen3AsrEngine":
        from .engine.qwen3_asr import Qwen3AsrEngine

        return Qwen3AsrEngine
    raise AttributeError(name)
