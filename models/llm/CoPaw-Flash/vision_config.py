# Copyright (c) 2025 HOUMO AI
#
# File: vision_config.py
# Description:
#  CoPaw-Flash Model Configuration - Python script for defining model configuration
# CoPaw-Flash models.
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
        fp_mode="ssfp",
        hidden_bit=False,
        bits=8,
    ),
    act_schema=dict(
        fp_mode="sefp",
        hidden_bit=True,
        bits=8,
    ),
)

model = dict(
    type="",
    wrap_cfg=dict(
        max_sequence_length=2048,
        max_size_w=448,
        max_size_h=448,
        max_size_t=2,
        temporal_patch_size=2,
        patch_size=16,
    ),
    quant_config=quant_config,
)