# Copyright (c) 2025 HOUMO AI
#
# File: common.py
# Description:
#   Shared helper utilities for Qwen3.5 scripts.
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

from pathlib import Path

import torch


def decode_next_token(tokenizer, logits: torch.Tensor):
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str


def xhquant_llm_init(log_file=None, debug=False, file_mode="w"):
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
    from xh_model_zoo.utils.logger import get_root_logger as _get_root_logger

    return _get_root_logger()
