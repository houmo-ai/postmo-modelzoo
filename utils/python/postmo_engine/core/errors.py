# Copyright (c) 2026 HOUMO AI
#
# File: errors.py
# Description:
#   Exceptions shared by the PostMo Engine public contracts.
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

"""Errors shared by the PostMo public contracts."""


class PostMoError(Exception):
    """Base error for PostMo contract failures."""


class InvalidSessionError(PostMoError, RuntimeError):
    """Raised when a Module session cannot safely continue."""


class UnsupportedFeatureError(PostMoError, ValueError):
    """Raised when a request asks for a non-available capability."""
