# Copyright 2025 HOUMO AI
#
# File: gemma4_moe.py
# Description:
#   Gemma4-MoE Model for MoE Inference
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
from loguru import logger
from hmatc.utils.perf_infomations import PERFTYPE
from gemma4_base import Gemma4Base


class Gemma4MoE(Gemma4Base):
    """Gemma4-MoE inference (no PerLayerInputBuilder, index-based embedding)."""

    sliding_window = 1024
    audio_enabled = False

    def __init__(
        self,
        prefill_path,
        decode_path,
        vit_path=None,
        embedding_path=None,
        tokenizer_dir=None,
        devices=0,
        max_new_tokens=2048,
        max_size_w=448,
        max_size_h=448,
        enable_thinking=False,
    ):
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._init_common(devices)

        backend_name = "Xh2HalBackend"
        self.tokenizer, self.processor = self._load_tokenizer(tokenizer_dir)

        # Vision (with perf tracking)
        if vit_path and os.path.isfile(vit_path):
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
        self.target_image_size = [max_size_w, max_size_h]
        self._load_vision(vit_path, self.devices, backend_name)
        if self.vit is not None:
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)

        # LLM (with perf tracking)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self._load_llm(prefill_path, decode_path, self.devices, backend_name)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)

        # Decode: set is_decode flag
        self.decode.set_input(self.decode.get_input_name(2), np.array([1], dtype="int32"))

        # Embedding (index-based, scale already baked in)
        emb = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = emb["weight"] if isinstance(emb, dict) else emb
        self.embedding = self.embedding.reshape(-1, self.embed_dim).float()

        self.perf_tracker.reset_perf_time()

    def _read_prefill_info(self):
        self.prefill_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        self.embed_dim = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]
        self.global_mask_w = self.prefill.get_input_info(self.prefill.get_input_name(4)).shape[3]
        self.prefill_local_w = self.prefill.get_input_info(self.prefill.get_input_name(3)).shape[3]

    def _read_decode_info(self):
        self.decode_len = self.decode.get_input_info(self.decode.get_input_name(0)).shape[1]
        self.decode_local_w = self.decode.get_input_info(self.decode.get_input_name(3)).shape[3]

    # ---- Embedding ----

    def _build_embeddings(self, input_ids, inputs):
        img_mask = input_ids == self.image_token_id
        llm_ids = input_ids.clone()
        llm_ids[img_mask] = self.tokenizer.pad_token_id or 0
        embeds: torch.Tensor = self.embedding[llm_ids[0]].unsqueeze(0).to(torch.float16)

        if img_mask.any() and self.vit is not None and inputs.get("pixel_values") is not None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            img_emb = self._run_vision(inputs["pixel_values"])
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            logger.info(f"Vision output: {img_emb.shape}")
            embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), img_emb)

        return embeds

    # ---- Prefill ----

    def _prefill(self, embeds, input_len, mm_types=None):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        steps = math.ceil(input_len / self.prefill_len)
        for s in range(steps):
            start, end = s * self.prefill_len, min((s + 1) * self.prefill_len, input_len)
            sub_emb = embeds[:, start:end]
            if sub_emb.shape[1] < self.prefill_len:
                sub_emb = torch.cat([sub_emb, torch.zeros(1, self.prefill_len - sub_emb.shape[1], sub_emb.shape[2])], dim=1)

            chunk_mm = mm_types[:, start:end][0] if mm_types is not None else None
            g_mask, l_mask = self._build_masks(end - start, start, self.prefill_len, chunk_mm)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(self.prefill.get_input_name(0), sub_emb.numpy().astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(1), np.array([start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(2), np.array([end - start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(3), l_mask.astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(4), g_mask.astype(np.float16))
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        next_id = self.prefill.get_output(self.prefill.get_output_name(0)).numpy().argmax(-1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        return next_id

    # ---- Decode ----

    def _decode_step(self, tok_id, past_len):
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        tok = torch.from_numpy(np.array([[tok_id]]))
        dec_emb = self.embedding[tok].reshape(1, 1, -1).to(torch.float16)
        if self.decode_len > 1:
            dec_emb = torch.cat([dec_emb, torch.zeros(1, self.decode_len - 1, self.embed_dim, dtype=torch.float16)], dim=1)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        g_mask, l_mask = self._build_masks(1, past_len, self.decode_len)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.decode.set_input(self.decode.get_input_name(0), dec_emb.numpy().astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(1), np.array([past_len], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(3), l_mask.astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(4), g_mask.astype(np.float16))
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.decode.run()
        self.decode.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        next_id = self.decode.get_output(self.decode.get_output_name(0)).numpy().astype(np.float32).argmax(-1)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        return next_id

    # ---- Chat ----

    def chat(self, question="", image_path=None, audio_path=None):
        q_text = question or ("请详细描述这张图片的内容。" if image_path else "你好，请介绍一下你自己。")
        logger.success(f"question: {q_text}")

        if image_path and self.vit is not None:
            from PIL import Image

            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            img = Image.open(image_path).convert("RGB").resize(self.target_image_size, Image.Resampling.BICUBIC)
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            content = [{"type": "image", "image": img}]
        else:
            content = []
        content.append({"type": "text", "text": q_text})

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )
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

        next_id = self._prefill(embeds, input_len, mm_types=mm_types)
        logger.info(f"Prefill done, first token: {self.tokenizer.decode(next_id[0])}")

        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        output_len = self._decode_loop(next_id, input_ids, input_len)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
        self.perf_tracker.set_basic_info(1, input_len, output_len, num_images=1 if image_path else 0)
