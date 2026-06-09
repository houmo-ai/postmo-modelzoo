# Copyright 2025 HOUMO AI
#
# File: gemma4_e.py
# Description:
#   Gemma4-E Model for E2B/E4B
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
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import tcim_lite as tcim
from hmatc.utils.perf_infomations import PERFTYPE
from gemma4_base import Gemma4Base, MAX_SOFT_TOKENS


class Gemma4E(Gemma4Base):
    """Gemma4-E2B inference with PerLayerInputBuilder, Vision and Audio."""
    def __init__(
        self,
        prefill_path,
        decode_path,
        vit_path=None,
        audio_path=None,
        embedding_path=None,
        plib_embedding_path=None,
        tokenizer_dir=None,
        devices=0,
        max_new_tokens=2048,
        enable_thinking=False,
    ):
        if isinstance(devices, int):
            devices = [devices]
        self.devices = devices

        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.audio_token_id = 258881
        self.image_token_id = 258880
        self.pad_token_id = 0
        self.sliding_window = 512

        backend_name = "Xh2HalBackend"
        self.tokenizer, self.processor = self._load_tokenizer(tokenizer_dir)

        # vision
        if vit_path and os.path.isfile(vit_path):
            dmv = tcim.runtime.DevManager(devices, backend_name)
            wmv = tcim.runtime.WeightManager(dmv)
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            self.vit = tcim.runtime.load(vit_path, option=tcim.runtime.Option(wmv))
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
            self._log_model_io(self.vit, "vit")
            vit_in_shape = self.vit.get_input_info(self.vit.get_input_name(0)).shape
            vit_out_shape = self.vit.get_output_info(self.vit.get_output_name(0)).shape
            self.vit_num_patches = vit_in_shape[1]
            self.vit_num_tokens = vit_out_shape[1] if len(vit_out_shape) == 3 else vit_out_shape[0]
            self.vit_patch_dim = vit_in_shape[2]
            logger.info(f"Vision: patches={self.vit_num_patches}, tokens={self.vit_num_tokens}")
        else:
            self.vit = None
            logger.warning("Vision model not loaded, text-only mode")

        # audio
        if audio_path and os.path.isfile(audio_path):
            dma = tcim.runtime.DevManager(devices, backend_name)
            wma = tcim.runtime.WeightManager(dma)
            self.perf_tracker.perf_start(PERFTYPE.AUDIO_LOAD_TIME)
            self.audio = tcim.runtime.load(audio_path, option=tcim.runtime.Option(wma))
            self.perf_tracker.perf_end(PERFTYPE.AUDIO_LOAD_TIME)
            self._log_model_io(self.audio, "audio")
            self.audio_feature_length = self.audio.get_input_info(self.audio.get_input_name(1)).shape[1]
            self.audio_feature_size = self.audio.get_input_info(self.audio.get_input_name(0)).shape[2]
            logger.info(f"Audio loaded: feature_length={self.audio_feature_length}")
        else:
            self.audio = None
            logger.warning("Audio model not loaded, audio disabled")

        # llm
        dm = tcim.runtime.DevManager(devices, backend_name)
        wm = tcim.runtime.WeightManager(dm)
        opt0 = tcim.runtime.Option(wm)
        opt0.set_dummy_tensors(["per_layer_inputs"])
        logger.info(f"Loading prefill model from {prefill_path}")
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=opt0)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        self._log_model_io(self.prefill, "prefill")
        assert self.prefill.get_input_name(4) == "per_layer_inputs"
        self.prefill_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        self.embed_dim = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]
        self.prefill_local_w = self.prefill.get_input_info(self.prefill.get_input_name(3)).shape[3]
        self.num_hidden_layers = self.prefill.get_input_info(self.prefill.get_input_name(4)).shape[2]
        self.hidden_size_per_layer_input = self.prefill.get_input_info(self.prefill.get_input_name(4)).shape[3]

        cache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "cache" in self.prefill.get_input_name(i).lower()]
        
        self.context_max_length = self.prefill.get_input_info(cache_names[-1]).shape[2]
        logger.info(f"Prefill loaded: len={self.prefill_len}, embed_dim={self.embed_dim}, context_max_length={self.context_max_length}")

        opt1 = tcim.runtime.Option(wm)
        opt1.set_dummy_tensors(cache_names)
        logger.info(f"Loading decode model from {decode_path}")
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=opt1)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        self._log_model_io(self.decode, "decode")
        self.decode_len = self.decode.get_input_info(self.decode.get_input_name(0)).shape[1]
        self.decode_local_w = self.decode.get_input_info(self.decode.get_input_name(3)).shape[3]
        logger.info(f"Decode loaded: len={self.decode_len}")
        for name in cache_names:
            self.decode.set_input(name, self.prefill.get_dev_input(name))

        # pinned
        self.decode.set_input(self.decode.get_input_name(2), np.array([1], dtype="int32"))
        
        # Embedding (nn.Embedding + scale)
        saved = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = nn.Embedding(
            saved["weight"].shape[0],
            saved["weight"].shape[1],
            padding_idx=self.pad_token_id,
            dtype=torch.float16,
        )
        self.embedding.load_state_dict(saved)
        # quant_embedding.pt from xh2modelzoo export already folds Gemma's
        # sqrt(hidden_size) embedding scale into the saved weight.
        # Do not multiply by sqrt(embed_dim) again here.

        # PerLayerInputBuilder Embedding
        saved = torch.load(plib_embedding_path, map_location="cpu", weights_only=True)
        weight = saved["state_dict"]["embed_tokens_per_layer.weight"]
        self.plib_embedding = nn.Embedding(
            weight.shape[0],
            weight.shape[1],
            padding_idx=self.pad_token_id,
            dtype=torch.float16,
        )
        self.plib_embedding.weight.data.copy_(weight)
        self.perf_tracker.reset_perf_time()

    @staticmethod
    def _sscp_subsample(num_frames: int) -> int:
        tokens = num_frames
        for _ in range(2):
            tokens = (tokens + 2 - 3) // 2 + 1
        return tokens

    def _run_audio(self, inputs: dict):
        if self.audio is None:
            raise RuntimeError("Audio model not loaded")
        if inputs.get("input_features") is None or inputs.get("input_features_mask") is None:
            raise RuntimeError("input_features/input_features_mask not found in processor inputs")

        input_features_mask = inputs["input_features_mask"]
        if not isinstance(input_features_mask, torch.Tensor):
            input_features_mask = torch.as_tensor(input_features_mask)
        valid_frames = int(input_features_mask[0].sum().item()) if input_features_mask.dim() == 2 else int(input_features_mask.sum().item())
        expected_tokens = self._sscp_subsample(valid_frames)

        audio_input_aliases = {"attention_mask": "audio_attention_mask"}

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_INPUT_TIME)
        for i in range(self.audio.get_num_inputs()):
            name = self.audio.get_input_name(i)
            bare_name = name.removesuffix(".hmcc.format")
            input_key = bare_name if bare_name in inputs else audio_input_aliases.get(bare_name)
            if input_key not in inputs:
                raise KeyError(f"Audio input {name!r} is not found in processor inputs")
            value = inputs[input_key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            self.audio.set_input(name, self._fit_model_input(self.audio, value, name))
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_INFER_TIME)
        self.audio.run()
        self.audio.sync()
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_OUTPUT_TIME)
        audio_embeds = torch.from_numpy(self.audio.get_output(self.audio.get_output_name(0)).numpy())
        num_outputs = self.audio.get_num_outputs()
        if num_outputs > 1:
            audio_embeds_mask = torch.from_numpy(self.audio.get_output(self.audio.get_output_name(1)).numpy())
            mask_bool = audio_embeds_mask[0].to(torch.bool)
            audio_out = audio_embeds[0][mask_bool]
        elif audio_embeds.dim() == 3:
            audio_out = audio_embeds[0, :expected_tokens, :]
        else:
            audio_out = audio_embeds[:expected_tokens, :]
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_OUTPUT_TIME)
        logger.info(f"Audio output: {audio_out.shape} (valid_frames={valid_frames}, expected_tokens={expected_tokens})")
        return audio_out

    def _build_embeddings(self, input_ids, inputs):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        img_mask = input_ids == self.image_token_id
        audio_mask = input_ids == self.audio_token_id
        llm_ids = input_ids.clone()
        llm_ids[img_mask] = self.pad_token_id
        llm_ids[audio_mask] = self.pad_token_id
        embeds: torch.Tensor = self.embedding(llm_ids)

        if img_mask.any() and self.vit is not None and inputs.get("pixel_values") is not None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            img_emb = self._run_vision(inputs)
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            logger.info(f"Vision output: {img_emb.shape}")
            embeds = self._scatter_features(embeds, input_ids, self.image_token_id, img_emb, "Image")

        if audio_mask.any() and self.audio is not None and inputs.get("input_features") is not None:
            self.perf_tracker.perf_start(PERFTYPE.AUDIO_TOTAL_TIME)
            audio_emb = self._run_audio(inputs)
            self.perf_tracker.perf_end(PERFTYPE.AUDIO_TOTAL_TIME)
            embeds = self._scatter_features(embeds, input_ids, self.audio_token_id, audio_emb, "Audio")

        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        return embeds, llm_ids

    def _prefill(self, embeds, input_len, llm_ids=None, mm_types=None):
        import math

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        steps = math.ceil(input_len / self.prefill_len)
        for s in range(steps):
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
            start, end = s * self.prefill_len, min((s + 1) * self.prefill_len, input_len)
            cur_len = end - start

            sub_embeds = embeds[:, start:end]
            if sub_embeds.shape[1] < self.prefill_len:
                pad_embeds = torch.zeros(1, self.prefill_len - sub_embeds.shape[1], sub_embeds.shape[2], dtype=sub_embeds.dtype)
                sub_embeds = torch.cat([sub_embeds, pad_embeds], dim=1,)

            sub_llm_ids = llm_ids[:, start:end]
            if sub_llm_ids.shape[1] < self.prefill_len:
                pad_ids = torch.full((1, self.prefill_len - sub_llm_ids.shape[1]), self.pad_token_id, dtype=sub_llm_ids.dtype)
                sub_llm_ids = torch.cat([sub_llm_ids, pad_ids], dim=1)

            chunk_mm = mm_types[:, start:end] if mm_types is not None else None
            if chunk_mm is not None and chunk_mm.shape[1] < self.prefill_len:
                pad_mm = torch.zeros(1, self.prefill_len - chunk_mm.shape[1], dtype=chunk_mm.dtype)
                chunk_mm = torch.cat([chunk_mm, pad_mm], dim=1)

            pli: torch.Tensor = self.plib_embedding(sub_llm_ids).view(-1, self.prefill_len, self.num_hidden_layers, self.hidden_size_per_layer_input)
            sub_embeds = sub_embeds.contiguous().detach().cpu().numpy().astype(np.float16)
            pli = pli.contiguous().detach().cpu().numpy().astype(np.float16)
            _, l_mask = self._build_masks(cur_len, start, self.prefill_len, chunk_mm)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(self.prefill.get_input_name(0), sub_embeds)
            self.prefill.set_input(self.prefill.get_input_name(1), np.array([start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(2), np.array([cur_len], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(3), np.ascontiguousarray(l_mask.astype(np.float16)))
            self.prefill.set_input(self.prefill.get_input_name(4), pli)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        logits = self.prefill.get_output(self.prefill.get_output_name(0)).numpy().astype(np.float32)
        out_seq_len = logits.shape[1]
        if out_seq_len >= cur_len:
            vaild_len = cur_len
        else:
            vaild_len = out_seq_len
        next_id = logits[0:1, vaild_len - 1:vaild_len, :].argmax(-1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        return next_id

    def _decode_step(self, tok_id, past_len):
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        tok = torch.tensor([[tok_id]], dtype=torch.long)
        dec_emb = self.embedding(tok).reshape(1, 1, -1).to(torch.float16)
        dec_emb = dec_emb.detach().numpy().astype(np.float16)

        pli: torch.Tensor = self.plib_embedding(tok).view(-1, self.decode_len, self.num_hidden_layers, self.hidden_size_per_layer_input)
        _, l_mask = self._build_masks(1, past_len, self.decode_len)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.decode.set_input(self.decode.get_input_name(0), dec_emb)
        self.decode.set_input(self.decode.get_input_name(1), np.array([past_len], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(3), np.ascontiguousarray(l_mask.astype(np.float16)))
        self.decode.set_input(self.decode.get_input_name(4), pli.contiguous().detach().cpu().numpy().astype(np.float16))
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
        if image_path and audio_path:
            q_text = question or "请详细描述这张图片和音频的内容。"
        elif image_path and not audio_path:
            q_text = question or "请详细描述这张图片的内容。"
        elif audio_path and not image_path:
            q_text = question or "请详细描述这个音频的内容。"
        elif not audio_path and not image_path:
            q_text = question or "你好，请介绍一下你自己。"
        logger.success(f"question: {q_text}")

        has_image = image_path and self.vit is not None
        has_audio = audio_path and self.audio is not None

        content = []
        if has_image:
            from PIL import Image

            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            img = Image.open(image_path).convert("RGB")
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            content.append({"type": "image", "image": img})

        if has_audio:
            import torchaudio

            self.perf_tracker.perf_start(PERFTYPE.AUDIO_PREPROCESS_TIME)
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            self.perf_tracker.perf_end(PERFTYPE.AUDIO_PREPROCESS_TIME)
            content.append({"type": "audio", "audio": waveform.numpy(), "sampling_rate": 16000})

        content.append({"type": "text", "text": q_text})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[-1]
        logger.info(f"Input length: {input_len}")
        if input_len >= self.context_max_length:
            import sys

            logger.error(f"Input too long: {input_len}")
            sys.exit(1)

        mm_types = inputs.get("mm_token_type_ids")
        embeds, llm_ids = self._build_embeddings(input_ids, inputs)
        next_id = self._prefill(embeds, input_len, llm_ids=llm_ids, mm_types=mm_types)
        logger.info(f"Prefill done, first token: {next_id[0][0]}")

        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        output_len = self._decode_loop(next_id, input_ids, input_len)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

        self.perf_tracker.set_basic_info(1, input_len, output_len, num_images=1 if has_image else 0, num_audios=1 if has_audio else 0)
