# Copyright 2025 HOUMO AI
#
# File: gemma4_dense.py
# Description:
#   Gemma4-Dense Model for Dense Inference
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

# fmt: off
import os
import math
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from hmatc.utils.perf_infomations import PERFTYPE
from gemma4_base import Gemma4Base


class Gemma4(Gemma4Base):
    """Gemma4-Dense inference (no sliding-window attention, index-based embedding)."""

    sliding_window = 1024
    audio_enabled = False
    visual_bidirectional_attention = True

    def __init__(
        self,
        prefill_path,
        decode_path,
        vit_path=None,
        embedding_path=None,
        tokenizer_dir=None,
        devices=0,
        max_new_tokens=2048,
        enable_thinking=False,
        assistant_path=None,
    ):
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._init_common(devices)

        backend_name = "Xh2HalBackend"
        self.tokenizer, self.processor = self._load_tokenizer(tokenizer_dir)

        # Vision (with perf tracking)
        self.vit = None
        if vit_path and os.path.isfile(vit_path):
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            self._load_vision(vit_path, self.devices, backend_name)
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)

        # LLM (with perf tracking)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self._load_llm(prefill_path, decode_path, self.devices, backend_name)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        
        # Assistant model
        self.assistant = None
        if assistant_path and os.path.isfile(assistant_path):
            self._load_assistant(assistant_path, self.devices, backend_name)

        # Embedding (index-based, scale baked into weights)
        saved = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = nn.Embedding(
            saved["weight"].shape[0],
            saved["weight"].shape[1],
            padding_idx=self.pad_token_id,
            dtype=torch.float16,
        )
        self.embedding.load_state_dict(saved, strict=False)
        self.perf_tracker.reset_perf_time()
    
    def _read_prefill_info(self):
        self.prefill_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        self.embed_dim = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]

    def _read_decode_info(self):
        self.decode_len = self.decode.get_input_info(self.decode.get_input_name(0)).shape[1]
        self.decode_local_w = self.decode.get_input_info(self.decode.get_input_name(3)).shape[3]

    def _build_embeddings(self, input_ids, inputs):
        img_mask = input_ids == self.image_token_id
        llm_ids = input_ids.clone()
        llm_ids[img_mask] = self.tokenizer.pad_token_id or 0
        embeds: torch.Tensor = self.embedding(llm_ids)

        if img_mask.any() and self.vit is not None and inputs.get("pixel_values") is not None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            img_emb = self._run_vision(inputs)
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            embeds = self._scatter_features(embeds, input_ids, self.image_token_id, img_emb, "Image")

        return embeds

    def _prefill(self, embeds, input_len, mm_types=None):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        steps = math.ceil(input_len / self.prefill_len)
        cur_len = 0
        for s in range(steps):
            start, end = s * self.prefill_len, min((s + 1) * self.prefill_len, input_len)
            cur_len = end - start
            sub_emb = embeds[:, start:end]
            if sub_emb.shape[1] < self.prefill_len:
                sub_emb = torch.cat([sub_emb, torch.zeros(1, self.prefill_len - sub_emb.shape[1], sub_emb.shape[2])], dim=1)

            chunk_mm = mm_types[:, start:end][0] if mm_types is not None else None
            _, l_mask = self._build_masks(cur_len, start, self.prefill_len, chunk_mm)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(self.prefill.get_input_name(0), sub_emb.detach().numpy().astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(1), np.array([start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(2), np.array([cur_len], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(3), l_mask.astype(np.float16))
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        logits = self.prefill.get_output(self.prefill.get_output_name(0)).numpy().astype(np.float32) # [bs, seq_len, vocab_size]
        valid_len = min(logits.shape[1], cur_len)
        next_ids = logits[0:1, valid_len - 1:valid_len, :].argmax(-1)
        hidden_states = None
        if self.assistant is not None:
            hidden_states = self.prefill.get_output(self.prefill.get_output_name(1)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        if self.assistant is not None:
            self.sliding_cache_valid_length = min(self.decode_local_w, input_len)
        return next_ids, hidden_states

    def _decode_step(self, tok_ids: list, past_len, accepted_count=0):
        cur_len = len(tok_ids)
        padded_ids = list(map(int, tok_ids))
        if cur_len < self.decode_len:
            padded_ids += [self.pad_token_id] * (self.decode_len - cur_len)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        tok_ids = torch.from_numpy(np.array([padded_ids]))
        dec_emb: torch.Tensor = self.embedding(tok_ids)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        _, l_mask = self._build_masks(cur_len, past_len, self.decode_len)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.decode.set_input(self.decode.get_input_name(0), dec_emb.detach().numpy().astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(1), np.array([past_len], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(2), np.array([cur_len], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(3), l_mask.astype(np.float16))
        if self.accepted_count_name is not None:
            self.decode.set_input(self.accepted_count_name, np.array([accepted_count], dtype="int32"))
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.decode.run()
        self.decode.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        logits: np.ndarray = self.decode.get_output(self.decode.get_output_name(0)).numpy().astype(np.float32)
        hidden_states = None
        if self.assistant is not None:
            hidden_states = self.decode.get_output(self.decode.get_output_name(1)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        return logits, hidden_states

    def chat(self, question="", image_path=None, audio_path=None):
        q_text = question or ("请详细描述这张图片的内容。" if image_path else "你好，请介绍一下你自己。")
        logger.success(f"question: {q_text}")

        if image_path and self.vit is not None:
            from PIL import Image

            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            img = Image.open(image_path).convert("RGB")
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            content = [{"type": "image", "image": img}]
        else:
            content = []
        content.append({"type": "text", "text": q_text})
        messages = [{"role": "user", "content": content}]
        
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        inputs = self.processor.apply_chat_template(messages, enable_thinking=self.enable_thinking)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        input_ids = inputs["input_ids"]
        mm_types = inputs.get("mm_token_type_ids")
        input_len = input_ids.shape[-1]
        logger.info(f"Input length: {input_len}")
        if input_len >= self.context_max_length:
            import sys

            logger.error(f"Input too long: {input_len}")
            sys.exit(1)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        embeds = self._build_embeddings(input_ids, inputs)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

        next_ids, hidden_states = self._prefill(embeds, input_len, mm_types=mm_types)  # [bs, num_tokens]
        logger.info(f"Prefill done, first token: {self.tokenizer.decode(next_ids[0])}")

        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        output_len = self._decode_loop(next_ids, input_ids, input_len, hidden_states)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
        self.perf_tracker.set_basic_info(1, input_len, output_len, num_images=1 if image_path else 0)

class Gemma4MoE(Gemma4):
    """Gemma4-MoE inference.

    MoE and dense Gemma4 share the same HMM runtime input/output contract:
    embedding input, global/local masks, and decode output shape [B, 1, V].
    Keep this class as a compatibility wrapper for existing imports while
    reusing the dense runtime implementation.
    """

    pass
