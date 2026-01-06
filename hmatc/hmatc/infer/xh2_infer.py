# Copyright 2025 HOUMO AI
#
# File: xh2_infer.py
# Description:
#   XH2 inference script using TCIM Lite.
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
from .xh1_infer import Xh1Infer


class Xh2Infer(Xh1Infer):
    """
    Inference class for XH2 hardware models.
    Inherits from Xh1Infer and adapts for XH2 hardware inference.
    """

    def __init__(self):
        """
        Initialize the Xh2Infer instance.
        Sets up the backend to XH2 and inherits all other functionality from Xh1Infer.
        """
        super().__init__()
        self.backend = "xh2"
