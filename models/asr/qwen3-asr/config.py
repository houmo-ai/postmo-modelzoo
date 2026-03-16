# Copyright (c) 2025 HOUMO AI
#
# File: config.py
# Description:
#   Configuration file for Qwen3-ASR quantization.
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
config_dir = 'Qwen3-ASR-0.6B'
device = 'cuda'
dtype = 'float16'
exec_device = 'cuda'
frontend_type = 'TorchFX'
hf_model_dir = 'Qwen3-ASR-0.6B'
model = dict(
    export_cfg=dict(
        input_names=[
            'inputs_embeds',
            'past_seq_length',
            'current_input_length',
            'past_key_cache',
            'past_value_cache',
        ],
        output_names=[
            'last_hidden_state',
        ]),
    frontend_type='TorchFX',
    hf_model='Qwen3-ASR-0.6B',
    quant_config=dict(),
    type='XHQwen3ASRLLMModel',
    wrap_cfg=dict(
        input_sequence_length=411,
        kv_cache=dict(cache_axis=2),
        max_sequence_length=2048,
        num_logits_to_keep=1,
        use_cache=True))
quant_config = dict()
target_device = 'XH2a'
work_dir = 'work_dirs/qwen3_asr_decode_xh2a'
