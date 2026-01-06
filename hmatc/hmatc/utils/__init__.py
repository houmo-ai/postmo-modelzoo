# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   This file is part of the hmatc.utils Python package.
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
import sys
import logging
from .logging_format import LoggingFormatterWithColor


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(LoggingFormatterWithColor())

# root_logger = logging.getLogger()
# root_logger.addHandler(console_handler)
# root_logger.setLevel(logging.INFO)

logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
