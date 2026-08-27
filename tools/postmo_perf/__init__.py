# Copyright (c) 2026 HOUMO AI
#
# File: __init__.py
# Description:
#   Public package exports for the PostMo fixed-length performance tool.
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

"""Fixed-length device performance application for PostMo models."""

import sys
from pathlib import Path

_UTILS_PYTHON = Path(__file__).resolve().parents[2] / "utils" / "python"
if str(_UTILS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_UTILS_PYTHON))

from .config import PerfCase, PerfSettings, load_config
from .input import generate_token_ids
from .result import CaseResult, LoopResult

__all__ = [
    "CaseResult",
    "LoopResult",
    "PerfCase",
    "PerfSettings",
    "generate_token_ids",
    "load_config",
]
