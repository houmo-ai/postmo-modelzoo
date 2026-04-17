# Copyright (c) 2025 HOUMO AI
#
# File: llm_config.py
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
                qspec=dict(fake_dtype="int64"),
            )
        ),
        hight_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int64"),
            )
        ),
        width_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int64"),
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
        linear_attn_mask=dict(
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

release = dict(
    xh_version="xh2",
    modelscope_name="CoPaw-Flash-9B",
)

model = dict(
    type="",
    wrap_cfg=dict(
        batch_size=1,
        max_pe_length=32768,
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        linear_attention_mode="auto",
        linear_chunk_size=64,
        support_long_context_over_fp16_limit=False,
        kv_cache=dict(
            cache_axis=2,
        ),
        enable_rope=True,
        enable_auto_offload=True,
        auto_offload_max_memory=None,
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
            "linear_attn_mask",
        ],
        output_names=["logits"],
    ),
)
