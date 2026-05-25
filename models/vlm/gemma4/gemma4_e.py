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
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import tcim_lite as tcim
from hmatc.utils.perf_infomations import PERFTYPE
from gemma4_base import Gemma4Base


@dataclass
class TextConfig:
    vocab_size_per_layer_input = 262144
    num_hidden_layers = 35
    hidden_size = 1536
    hidden_size_per_layer_input = 256
    pad_token_id = 0
    rms_norm_eps = 1e-6


class PerLayerInputRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class PerLayerInputBuilder(nn.Module):
    def __init__(
        self,
        num_hidden_layers=35,
        hidden_size=1536,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
        pad_token_id=0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.embed_scale = hidden_size_per_layer_input**0.5
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = hidden_size**-0.5
        self.embed_tokens_per_layer = nn.Embedding(vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input, pad_token_id)
        self.per_layer_model_projection = nn.Linear(hidden_size, num_hidden_layers * hidden_size_per_layer_input, bias=False)
        self.per_layer_projection_norm = PerLayerInputRMSNorm(hidden_size_per_layer_input, eps=rms_norm_eps)

    @classmethod
    def from_artifact(cls, artifact_path: str, text_config: TextConfig = None):
        if text_config is None:
            text_config = TextConfig()
        saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
        state_dict = saved.get("state_dict", saved)
        builder = cls(
            num_hidden_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            hidden_size_per_layer_input=text_config.hidden_size_per_layer_input,
            vocab_size_per_layer_input=text_config.vocab_size_per_layer_input,
            pad_token_id=text_config.pad_token_id,
            rms_norm_eps=text_config.rms_norm_eps,
        )
        builder.load_state_dict(state_dict)
        builder.to(dtype=next(iter(state_dict.values())).dtype)
        return builder.eval()

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        dtype = self.per_layer_model_projection.weight.dtype
        pli: torch.Tensor = self.embed_tokens_per_layer(input_ids).to(dtype) * self.embed_scale
        pli = pli.reshape(b, s, self.num_hidden_layers, self.hidden_size_per_layer_input).permute(0, 2, 1, 3).contiguous()
        proj = self.per_layer_model_projection(inputs_embeds.to(dtype)) * self.per_layer_model_projection_scale
        proj = proj.reshape(b, s, self.num_hidden_layers, self.hidden_size_per_layer_input).permute(0, 2, 1, 3).contiguous()
        proj = self.per_layer_projection_norm(proj)
        return ((proj + pli) * self.per_layer_input_scale).to(inputs_embeds.dtype)


class Gemma4E(Gemma4Base):
    """Gemma4-E2B inference with PerLayerInputBuilder, Vision and Audio."""

    sliding_window = 512
    audio_enabled = True

    def __init__(
        self,
        prefill_path,
        decode_path,
        vit_path=None,
        audio_path=None,
        embedding_path=None,
        per_layer_input_builder_path=None,
        tokenizer_dir=None,
        devices=0,
        max_new_tokens=2048,
        enable_thinking=False,
    ):
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.audio_token_id = 258881
        self._init_common(devices)

        backend_name = "Xh2HalBackend"
        self.tokenizer, self.processor = self._load_tokenizer(tokenizer_dir)

        # Vision (with perf tracking)
        if vit_path and os.path.isfile(vit_path):
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
        self._load_vision(vit_path, self.devices, backend_name)
        if self.vit is not None:
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)

        # LLM (with perf tracking)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self._load_llm(prefill_path, decode_path, self.devices, backend_name)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        self.decode.set_input(self.decode.get_input_name(4), np.array([1], dtype="int32"))

        # Audio (optional, with perf tracking)
        if audio_path and os.path.isfile(audio_path):
            self.perf_tracker.perf_start(PERFTYPE.AUDIO_LOAD_TIME)
            dma = tcim.runtime.DevManager(self.devices, backend_name)
            wma = tcim.runtime.WeightManager(dma)
            self.audio_model = tcim.runtime.load(audio_path, option=tcim.runtime.Option(wma))
            self.audio_feature_length = self.audio_model.get_input_info(self.audio_model.get_input_name(1)).shape[1]
            self.audio_feature_size = self.audio_model.get_input_info(self.audio_model.get_input_name(0)).shape[2]
            self.perf_tracker.perf_end(PERFTYPE.AUDIO_LOAD_TIME)
            logger.info(f"Audio loaded: feature_length={self.audio_feature_length}")
        else:
            self.audio_model = None
            logger.warning("Audio model not loaded, audio disabled")
            
        # PerLayerInputBuilder
        text_config = TextConfig()
        text_config.num_hidden_layers = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        text_config.hidden_size = self.prefill.get_input_info(self.prefill.get_input_name(1)).shape[2]
        self.per_layer_input_builder = PerLayerInputBuilder.from_artifact(per_layer_input_builder_path, text_config=text_config)
        logger.info(f"Loaded per_layer_input_builder: layers={self.per_layer_input_builder.num_hidden_layers}")
        
        # Embedding (nn.Embedding + scale)
        saved = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = nn.Embedding(
            saved["weight"].shape[0],
            saved["weight"].shape[1],
            padding_idx=self.pad_token_id,
            dtype=torch.float16,
        )
        self.embedding.load_state_dict(saved)
        self.embed_scale = self.embed_dim**0.5

        self.perf_tracker.reset_perf_time()

    def _read_prefill_info(self):
        self.prefill_len = self.prefill.get_input_info(self.prefill.get_input_name(1)).shape[1]
        self.embed_dim = self.prefill.get_input_info(self.prefill.get_input_name(1)).shape[2]
        self.prefill_local_w = self.prefill.get_input_info(self.prefill.get_input_name(5)).shape[3]
        self.global_mask_w = self.prefill.get_input_info(self.prefill.get_input_name(6)).shape[3]

    def _read_decode_info(self):
        self.decode_len = self.decode.get_input_info(self.decode.get_input_name(1)).shape[1]
        self.decode_local_w = self.decode.get_input_info(self.decode.get_input_name(5)).shape[3]

    # ---- Audio ----

    @staticmethod
    def _sscp_subsample(num_frames: int) -> int:
        tokens = num_frames
        for _ in range(2):
            tokens = (tokens + 2 - 3) // 2 + 1
        return tokens

    def _run_audio_single(self, chunk_f, chunk_m):
        """Run audio model on a single chunk, return trimmed (tokens, embed_dim) tensor."""
        valid_frames = int(chunk_m[0].sum().item()) if chunk_m.dim() == 2 else int(chunk_m.sum().item())
        expected_tokens = self._sscp_subsample(valid_frames)

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_INPUT_TIME)
        self.audio_model.set_input(self.audio_model.get_input_name(0), chunk_f.numpy().astype(np.float32))
        self.audio_model.set_input(self.audio_model.get_input_name(1), chunk_m.long().numpy().astype(np.int32))
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_INFER_TIME)
        self.audio_model.run()
        self.audio_model.sync()
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.AUDIO_OUTPUT_TIME)
        audio_embeds = torch.from_numpy(self.audio_model.get_output(self.audio_model.get_output_name(0)).numpy())
        num_outputs = self.audio_model.get_num_outputs()
        if num_outputs > 1:
            audio_embeds_mask = torch.from_numpy(self.audio_model.get_output(self.audio_model.get_output_name(1)).numpy())
            mask_bool = audio_embeds_mask[0].to(torch.bool)
            chunk_out = audio_embeds[0][mask_bool]
        else:
            chunk_out = audio_embeds[:, :expected_tokens, :].squeeze(0)
        self.perf_tracker.perf_end(PERFTYPE.AUDIO_OUTPUT_TIME)
        return chunk_out

    def _run_audio(self, input_features, input_features_mask):
        if self.audio_model is None:
            raise RuntimeError("Audio model not loaded")

        total_len = input_features.shape[1]
        chunk_size = self.audio_feature_length  # 400
        # SSCP receptive field: 2 layers of kernel=3,stride=2 → ~7 input frames
        # Use 8-frame overlap to avoid boundary distortion
        overlap = 8
        stride = chunk_size - overlap
        # Corresponding output tokens to discard at overlap boundaries
        trim_tokens = self._sscp_subsample(overlap)  # 2

        # Single chunk: no overlap needed
        if total_len <= chunk_size:
            pad_len = chunk_size - total_len
            if pad_len > 0:
                input_features = torch.cat([input_features, torch.zeros(1, pad_len, input_features.shape[2], dtype=input_features.dtype)], dim=1)
                input_features_mask = torch.cat([input_features_mask, torch.zeros(1, pad_len, dtype=input_features_mask.dtype)], dim=1)
            chunk_out = self._run_audio_single(input_features, input_features_mask)
            logger.info(f"Audio output: {chunk_out.shape} (single chunk, total_len={total_len})")
            return chunk_out

        # Multiple chunks with overlap
        chunks_out = []
        offset = 0
        chunk_idx = 0
        while offset < total_len:
            end = min(offset + chunk_size, total_len)
            chunk_f = input_features[:, offset:end]
            chunk_m = input_features_mask[:, offset:end]
            cur_len = end - offset

            # Pad if last chunk is shorter
            if cur_len < chunk_size:
                pad_len = chunk_size - cur_len
                chunk_f = torch.cat([chunk_f, torch.zeros(1, pad_len, chunk_f.shape[2], dtype=chunk_f.dtype)], dim=1)
                chunk_m = torch.cat([chunk_m, torch.zeros(1, pad_len, dtype=chunk_m.dtype)], dim=1)

            chunk_out = self._run_audio_single(chunk_f, chunk_m)

            # Discard leading overlap tokens (already covered by previous chunk)
            if chunk_idx > 0 and chunk_out.shape[0] > trim_tokens:
                chunk_out = chunk_out[trim_tokens:]

            chunks_out.append(chunk_out)
            offset += stride
            chunk_idx += 1

        result = torch.cat(chunks_out, dim=0)
        logger.info(f"Audio output: {result.shape} (chunks={chunk_idx}, total_len={total_len}, overlap={overlap}, trim_tokens={trim_tokens})")
        return result

    # ---- Embedding ----

    def _build_embeddings(self, input_ids, inputs):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        img_mask = input_ids == self.image_token_id
        audio_mask = input_ids == self.audio_token_id
        llm_ids = input_ids.clone()
        llm_ids[img_mask] = self.pad_token_id
        llm_ids[audio_mask] = self.pad_token_id
        embeds: torch.Tensor = self.embedding(llm_ids) * self.embed_scale

        if img_mask.any() and self.vit is not None and inputs.get("pixel_values") is not None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
            pixel_values = inputs["pixel_values"][:, self.valid_mask].half()
            if pixel_values.shape[1] < self.vit_num_patches:
                pixel_values = torch.cat([pixel_values, torch.zeros(1, self.vit_num_patches - pixel_values.shape[1], pixel_values.shape[2])], dim=1)
            self.vit.set_input(self.vit.get_input_name(0), pixel_values[:, :self.vit_num_patches].numpy())
            self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
            self.vit.run()
            self.vit.sync()
            self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
            img_emb = torch.from_numpy(self.vit.get_output(self.vit.get_output_name(0)).numpy()).squeeze(0)
            self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            logger.info(f"Vision output: {img_emb.shape}")
            embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), img_emb)

        if audio_mask.any() and self.audio_model is not None and inputs.get("input_features") is not None:
            self.perf_tracker.perf_start(PERFTYPE.AUDIO_TOTAL_TIME)
            audio_emb = self._run_audio(inputs["input_features"], inputs["input_features_mask"])
            self.perf_tracker.perf_end(PERFTYPE.AUDIO_TOTAL_TIME)
            embeds = embeds.masked_scatter(audio_mask.unsqueeze(-1).expand_as(embeds), audio_emb)

        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        return embeds, llm_ids

    # ---- Prefill ----

    def _prefill(self, embeds, input_len, llm_ids=None, mm_types=None):
        import math

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        steps = math.ceil(input_len / self.prefill_len)
        for s in range(steps):
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

            position_ids = torch.arange(self.prefill_len).unsqueeze(0) + start
            per_layer_inputs = self.per_layer_input_builder(sub_llm_ids.long(), sub_embeds)
            g_mask, l_mask = self._build_masks(cur_len, start, self.prefill_len, chunk_mm)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(self.prefill.get_input_name(0), per_layer_inputs.detach().numpy().astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(1), sub_embeds.detach().numpy().astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(2), position_ids.numpy().astype(np.int32))
            self.prefill.set_input(self.prefill.get_input_name(3), np.array([start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(4), np.array([cur_len], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(5), l_mask.astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(6), g_mask.astype(np.float16))
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
        tok = torch.tensor([[tok_id]], dtype=torch.long)
        dec_emb = self.embedding(tok).reshape(1, 1, -1).to(torch.float16) * self.embed_scale
        per_layer_inputs = self.per_layer_input_builder(tok, dec_emb)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        g_mask, l_mask = self._build_masks(1, past_len, self.decode_len)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.decode.set_input(self.decode.get_input_name(0), per_layer_inputs.detach().numpy().astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(1), dec_emb.detach().numpy().astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(2), np.array([[past_len]], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(3), np.array([past_len], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(4), np.array([1], dtype="int32"))
        self.decode.set_input(self.decode.get_input_name(5), l_mask.astype(np.float16))
        self.decode.set_input(self.decode.get_input_name(6), g_mask.astype(np.float16))
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
        has_audio = audio_path and self.audio_model is not None

        content = []
        if has_image:
            from PIL import Image

            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            img = Image.open(image_path).convert("RGB").resize(self.target_image_size, Image.Resampling.BICUBIC)
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
