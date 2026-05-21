# Copyright (c) 2026 HOUMO AI
#
# File: llm_config.py
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

frontend_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        time_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        hight_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        width_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        past_seq_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        current_input_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
    ),
    w_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
    act_schema=dict(
        bits=16,
        fp_mode="sefp",
    ),
    nodes_cfg=dict(
        lm_head=dict(
            w_schema=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema=dict(
                bits=16,
                fp_mode="sefp",
            ),
        ),
    ),
)

model = dict(
    type="XHPaddleOCRVLLLMModel",
    wrap_cfg=dict(
        max_sequence_length=2048,
        max_pe_length=32768,
        input_sequence_length=256,
        patch_size=14,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
        ],
        output_names=[
            "logits",
        ],
    ),
)
