# Copyright (c) 2026 HOUMO AI
#
# File: image_processing_qwen3.5.py
# Description:
#   Processing Qwen3.5 image data.
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
"""
Image processing for the dynamic Qwen3.5 vision model.
"""

from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


class Qwen3_5ImageProcessor(Qwen2VLImageProcessor):
    """Qwen2-VL patchification configured for dynamic visual input."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, *args, **kwargs):
        if kwargs.get("size") is None:
            kwargs.setdefault("min_pixels", 65536)
            kwargs.setdefault("max_pixels", 16777216)
        super().__init__(*args, **kwargs)
