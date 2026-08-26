# Copyright (c) 2026 HOUMO AI
#
# File: vision_config.py
# Description:
#  PaddleOCR-VL Model Configuration - Python script for defining model configuration
# PaddleOCR-VL models.
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

trace_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        pixel_values=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    ),
    w_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
    act_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
)

model = dict(
    type="XHPaddleOCRVLVisionModel",
    wrap_cfg=dict(
        max_sequence_length=16384,
        max_size_w=896,
        max_size_h=896,
        temporal_patch_size=1,
        patch_size=14,
    ),
    quant_config=quant_config,
)
