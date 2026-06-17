#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2026 HOUMO AI
#
# File: base_utils.py
# Description:
#   Shared utility functions for the Qwen3-Omni model example.
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

def parse_quant_types(model_config: dict) -> dict:
    """Flatten config.yaml ``quant_type`` into a {category: quant_type} dict.

    ``quant_type`` in config.yaml is a list of single-key dicts, e.g.::

        quant_type:
          - llm: "w4a8h0_ssfp"
          - projection: "w16a16h1_sefp"
          - other: "w8a8h0_sefp"

    Categories (see how ptq.py applies them):
      * ``other``      -> default quant type for vision/audio/talker/.../text-LLM
      * ``llm``        -> text-LLM quant type used when --gptqmodel is enabled
      * ``projection`` -> talker text_projection / hidden_projection
    """
    raw = model_config.get("quant_type", []) if model_config else []
    flattened: dict = {}
    if isinstance(raw, dict):
        flattened.update(raw)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                flattened.update(item)
    elif isinstance(raw, str):
        flattened["other"] = raw
    return {
        "other": flattened.get("other", "w8a8h0_sefp"),
        "llm": flattened.get("llm", "w4a8h0_ssfp"),
        "projection": flattened.get("projection", "w16a16h1_sefp"),
    }