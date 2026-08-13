# Copyright (c) 2026 HOUMO AI
#
# File: sam3_processor.py
# Description:
#   Shared local tokenizer helpers for SAM3 export and inference examples.
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

import numpy as np
from transformers import CLIPTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = CURRENT_DIR / "sam3"
TOKEN_CONTEXT_LENGTH = 32


def load_sam3_tokenizer(model_dir: Path = DEFAULT_MODEL_DIR) -> CLIPTokenizer:
	"""Load the SAM3 CLIP tokenizer from local model assets only."""
	model_dir = Path(model_dir)
	if not model_dir.is_dir():
		raise FileNotFoundError(f"Missing local SAM3 model directory: {model_dir}")
	return CLIPTokenizer.from_pretrained(
		str(model_dir),
		local_files_only=True,
	)


def tokenize_prompt(
	tokenizer: CLIPTokenizer,
	prompt: str,
	context_length: int = TOKEN_CONTEXT_LENGTH,
) -> np.ndarray:
	"""Tokenize a prompt and use zero for positions outside the valid sequence."""
	encoded = tokenizer(
		[prompt],
		padding="max_length",
		truncation=True,
		max_length=context_length,
		return_attention_mask=True,
		return_tensors="np",
	)
	token_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
	attention_mask = np.asarray(encoded["attention_mask"], dtype=bool)
	return np.where(attention_mask, token_ids, 0).astype(np.int64)
