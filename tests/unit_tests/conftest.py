# Copyright (c) 2026 HOUMO AI
#
# File: conftest.py
# Description:
#  Local pytest fixtures for lightweight framework unit tests that avoid
#    functional-test log-file creation.
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

"""Local pytest configuration for framework unit tests."""

import pytest


@pytest.fixture(autouse=True)
def setup_logging(request):
    """Avoid creating functional-test log files for lightweight unit tests."""
    yield (None, request)
