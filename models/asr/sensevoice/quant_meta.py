# Copyright 2025 HOUMO AI
#
# File: sensevoice_export_meta.py
# Description:
#   Example script: audio/sensevoice_small/sensevoice_export_meta.py
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

import types
from typing import Any, Dict

import torch


def rebuild_for_onnx(
    model: torch.nn.Module, max_seq_len: int = 512, device: str = "cpu"
) -> torch.nn.Module:
    from funasr.utils.torch_function import sequence_mask

    model.device = device
    model.onnx_config = {"max_seq_len": max_seq_len}
    model.make_pad_mask = sequence_mask(max_seq_len, flip=False)
    model.forward = types.MethodType(_export_forward, model)
    model.export_inputs = types.MethodType(_export_inputs, model)
    model.export_input_names = types.MethodType(_export_input_names, model)
    model.export_output_names = types.MethodType(_export_output_names, model)
    model.export_axes = types.MethodType(_export_axes, model)
    model.export_name = types.MethodType(_export_name, model)
    return model


def _export_forward(
    self: Any,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    language: torch.Tensor,
    textnorm: torch.Tensor,
    **kwargs: Dict[str, Any],
):
    language_query = self.embed(language.to(speech.device)).unsqueeze(1)
    textnorm_query = self.embed(textnorm.to(speech.device)).unsqueeze(1)
    speech = torch.cat((textnorm_query, speech), dim=1)
    speech_lengths = speech_lengths + 1

    event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(speech.size(0), 1, 1)
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths = speech_lengths + 3

    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    ctc_logits = self.ctc.ctc_lo(encoder_out)
    return ctc_logits, encoder_out_lens


def _export_inputs(self: Any):
    length = self.onnx_config["max_seq_len"]
    speech = torch.randn(1, length, 560, dtype=torch.float32)
    speech_lengths = torch.tensor([length], dtype=torch.int32)
    language = torch.tensor([0], dtype=torch.int32)
    textnorm = torch.tensor([15], dtype=torch.int32)
    return (speech, speech_lengths, language, textnorm)


def _export_input_names(self: Any):
    return ["speech", "speech_lengths", "language", "textnorm"]

def _export_axes(self: Any):
    return None
def _export_output_names(self: Any):
    return ["ctc_logits", "encoder_out_lens"]

def _export_name(self: Any):
    return "model.onnx"

