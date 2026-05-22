# Copyright 2025 HOUMO AI
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

"""Common utilities for GLM-OCR example scripts."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if dtype == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_messages(image_path, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_inputs(processor, messages: List[Dict[str, Any]], device: Optional[torch.device] = None):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    if device is not None:
        inputs = inputs.to(device)
    return inputs


def msg_output_format(title: str) -> str:
    padding_str = "*" * 10
    return f"{padding_str} {title} {padding_str}"


def load_quarot_gptq_state_dict(native_model, state_dict_path: str, strict: bool = False) -> None:
    from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

    rotation_utils.fuse_layer_norms(native_model)
    state_dict = load_safetensors_file(state_dict_path)

    model_state_dict = native_model.state_dict()
    unexpected_keys = [k for k in state_dict if k not in model_state_dict]
    for key in unexpected_keys:
        value = state_dict.pop(key)
        paths = key.split(".")
        if paths[-1] != "quant_weight":
            continue

        submodule_name = ".".join(paths[:-1])
        submodule = native_model.get_submodule(submodule_name)
        if value.min().item() >= -(2**7) and value.max().item() <= (2**7) - 1:
            value = value.to(torch.int8)
        elif value.min().item() >= -(2**15) and value.max().item() <= (2**15) - 1:
            value = value.to(torch.int16)
        else:
            value = value.to(torch.float32)
        submodule.register_buffer("quant_weight", value, persistent=False)

    native_model.load_state_dict(state_dict, strict=strict)


def decode_next_token(tokenizer, logits: torch.Tensor):
    """Decode next token from model logits. Replacement for xh_model_zoo.api.decode_next_token."""
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str


def xhquant_llm_init(log_file=None, debug=False, file_mode="w"):
    """Initialize xhquant and xh_model_zoo logger. Replacement for xh_model_zoo.api.xhquant_llm_init."""
    from xhquant.api import xhquant_init
    from xh_model_zoo.utils.logger import xh2modelzoo_init_logger

    xhquant_log_file = None
    if log_file is not None:
        log_fname = Path(log_file).stem
        log_suffix = Path(log_file).suffix
        xhquant_log_name = f"{log_fname}_xhquant{log_suffix}"
        xhquant_log_file = str(Path(log_file).with_name(xhquant_log_name))

    xhquant_init(xhquant_log_file, debug=debug)
    xh2modelzoo_init_logger(log_file, "DEBUG" if debug else "INFO", "xhquant_llm", file_mode=file_mode)


def get_root_logger():
    """Get the root logger. Replacement for xh_model_zoo.utils.logger.get_root_logger."""
    from xh_model_zoo.utils.logger import get_root_logger as _get_root_logger
    return _get_root_logger()


def ensure_calibration_data_files(
    data_files: Optional[List[str]],
    output_file: Path,
    image_path: str,
    prompt: str,
    nsamples: int,
) -> List[str]:
    if data_files:
        return data_files

    output_file.parent.mkdir(parents=True, exist_ok=True)
    user_turn = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }
    assistant_turn = {"role": "assistant", "content": [""]}

    with output_file.open("w", encoding="utf-8") as fout:
        for _ in range(nsamples):
            fout.write(json.dumps([user_turn, assistant_turn], ensure_ascii=False) + "\n")

    return [str(output_file)]
