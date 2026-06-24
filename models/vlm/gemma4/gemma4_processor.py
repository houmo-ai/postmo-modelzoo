# Copyright 2025 HOUMO AI
#
# File: gemma4_processor.py
# Description:
#   Gemma4 processor wrapper for padded visual and audio preprocessing.
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
from __future__ import annotations

import os
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Gemma4Processor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

DEFAULT_VISUAL_MAX_SOFT_TOKENS = 280
FULL_VISUAL_POOLING_KERNEL_SIZE = 3
COMPACT_VISUAL_POOLING_KERNEL_SIZE = 1


class Gemma4ProcessorConfig:
    def __init__(self):
        self.max_size_h: int = 448
        self.max_size_w: int = 448
        self.patch_size: int = 16
        self.sampling_rate: int = 16000
        self.audio_feature_length: int | None = None
        self.audio_attention_chunk_size: int = 12
        self.audio_attention_context_left: int = 13
        self.audio_attention_context_right: int = 0
        self.export_mode: str = "full"
        self.enforce_fixed_image_size: bool = False


class XHGemma4Processor(Gemma4Processor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        feature_extractor=None,
        video_processor=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            **kwargs,
        )
        self.config = Gemma4ProcessorConfig()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
        processor = Gemma4Processor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if isinstance(processor, cls):
            return processor
        return cls(
            feature_extractor=getattr(processor, "feature_extractor", None),
            image_processor=getattr(processor, "image_processor", None),
            tokenizer=getattr(processor, "tokenizer", None),
            video_processor=getattr(processor, "video_processor", None),
            chat_template=getattr(processor, "chat_template", None),
            image_seq_length=getattr(processor, "image_seq_length", 280),
            audio_seq_length=getattr(processor, "audio_seq_length", 750),
            audio_ms_per_token=getattr(processor, "audio_ms_per_token", 40),
        )

    def _resize_image_for_export_contract(self, image: Any) -> Any:
        if not self.config.enforce_fixed_image_size or not isinstance(image, Image.Image):
            return image
        target_size = (self.config.max_size_w, self.config.max_size_h)
        if image.size == target_size:
            return image
        return image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)

    def _prepare_vision_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "images" not in kwargs or kwargs["images"] is None:
            return kwargs

        prepared = dict(kwargs)
        images = prepared["images"]

        def _resize_recursive(item: Any) -> Any:
            if isinstance(item, Image.Image):
                return self._resize_image_for_export_contract(item)
            if isinstance(item, (list, tuple)):
                return type(item)(_resize_recursive(subitem) for subitem in item)
            return item

        prepared["images"] = _resize_recursive(images)
        return prepared

    def __call__(self, *args, **kwargs):
        kwargs = self._prepare_vision_kwargs(kwargs)
        model_inputs = super().__call__(*args, **kwargs)
        model_inputs = self._trim_compact_image_patches_if_needed(model_inputs)
        model_inputs = self._add_padded_visual_inputs(model_inputs)

        feature_length = self.config.audio_feature_length
        if (
            feature_length is not None
            and feature_length > 0
            and "input_features" in model_inputs
            and "input_features_mask" in model_inputs
        ):
            input_features = model_inputs["input_features"]
            input_features_mask = model_inputs["input_features_mask"]
            current_length = input_features.shape[1]
            if current_length > feature_length:
                model_inputs["input_features"] = input_features[:, :feature_length, :]
                model_inputs["input_features_mask"] = input_features_mask[:, :feature_length]
            elif current_length < feature_length:
                pad_length = feature_length - current_length
                feature_pad = torch.zeros(
                    (input_features.shape[0], pad_length, input_features.shape[2]),
                    dtype=input_features.dtype,
                    device=input_features.device,
                )
                mask_pad = torch.zeros(
                    (input_features_mask.shape[0], pad_length),
                    dtype=input_features_mask.dtype,
                    device=input_features_mask.device,
                )
                model_inputs["input_features"] = torch.cat([input_features, feature_pad], dim=1)
                model_inputs["input_features_mask"] = torch.cat([input_features_mask, mask_pad], dim=1)
            model_inputs = self._retokenize_audio_placeholders_if_needed(model_inputs, kwargs)
        model_inputs = self._add_audio_attention_mask(model_inputs)
        return model_inputs

    @staticmethod
    def _build_visual_attention_mask(valid_mask: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=valid_mask.device)
        mask = torch.zeros((*valid_mask.shape[:1], 1, 1, valid_mask.shape[-1]), dtype=dtype, device=valid_mask.device)
        return mask.masked_fill(~valid_mask[:, None, None, :], neg)

    @staticmethod
    def _build_pooling_matrix(position_ids: torch.Tensor, valid_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)

        batch, total_patches, _ = position_ids.shape
        k2 = kernel_size * kernel_size
        output_length = total_patches // k2
        pooling_matrix = torch.zeros(
            (batch, output_length, total_patches),
            dtype=torch.float16,
            device=position_ids.device,
        )

        for batch_idx in range(batch):
            valid_rows = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue
            valid_pos = position_ids[batch_idx, valid_rows]
            max_x = int(valid_pos[:, 0].max().item()) + 1
            pooled_w = max_x // kernel_size
            kernel_xy = torch.div(valid_pos, kernel_size, rounding_mode="floor")
            kernel_idxs = kernel_xy[:, 0] + pooled_w * kernel_xy[:, 1]
            valid_output_length = min(output_length, int(valid_rows.numel()) // k2)
            for out_idx in range(valid_output_length):
                selected = valid_rows[kernel_idxs == out_idx]
                if selected.numel() != k2:
                    raise ValueError(
                        f"Gemma4 visual pooling expects {k2} patches per pooled token, "
                        f"got {selected.numel()} for output {out_idx}"
                    )
                pooling_matrix[batch_idx, out_idx, selected[:k2]] = 1.0 / k2
        return pooling_matrix

    def _add_padded_visual_inputs(self, inputs):
        if "pixel_values" not in inputs or "image_position_ids" not in inputs:
            return inputs

        image_position_ids = inputs["image_position_ids"]
        if image_position_ids.dim() == 2:
            image_position_ids = image_position_ids.unsqueeze(0)

        valid_mask = ~(image_position_ids == -1).all(dim=-1)
        pixel_position_ids = image_position_ids.clone()
        pixel_position_ids[~valid_mask] = 0

        image_processor = getattr(self, "image_processor", None)
        kernel_size = int(getattr(image_processor, "pooling_kernel_size", 3))
        pooling_matrix = self._build_pooling_matrix(image_position_ids, valid_mask, kernel_size)
        image_soft_token_count = (valid_mask.sum(dim=-1) // (kernel_size * kernel_size)).to(torch.long)

        inputs["pixel_position_ids"] = pixel_position_ids.to(torch.int32)
        inputs["pooling_matrix"] = pooling_matrix
        inputs["visual_attention_mask"] = self._build_visual_attention_mask(valid_mask)
        inputs["image_soft_token_count"] = image_soft_token_count
        return inputs

    def _trim_compact_image_patches_if_needed(self, model_inputs):
        if self.config.export_mode != "compact":
            return model_inputs
        pixel_values = model_inputs.get("pixel_values")
        image_position_ids = model_inputs.get("image_position_ids")
        if pixel_values is None or image_position_ids is None:
            return model_inputs

        if image_position_ids.dim() == 2:
            valid_positions = ~(image_position_ids == -1).all(dim=-1)
            real_patch_count = int(valid_positions.sum().item())
        else:
            valid_positions = ~(image_position_ids == -1).all(dim=-1)
            real_patch_counts = valid_positions.to(torch.int64).sum(dim=1)
            max_real_patch_count = int(real_patch_counts.max().item())
            min_real_patch_count = int(real_patch_counts.min().item())
            if min_real_patch_count != max_real_patch_count:
                raise ValueError(
                    "Compact Gemma4 vision export requires a uniform real patch count across the batch, "
                    f"got {real_patch_counts.tolist()}."
                )
            real_patch_count = max_real_patch_count

        if pixel_values.shape[1] == real_patch_count:
            return model_inputs

        model_inputs["pixel_values"] = pixel_values[:, :real_patch_count, :]
        if image_position_ids.dim() == 2:
            model_inputs["image_position_ids"] = image_position_ids[:real_patch_count, :]
        else:
            model_inputs["image_position_ids"] = image_position_ids[:, :real_patch_count, :]
        return model_inputs

    @staticmethod
    def _audio_feature_mask_after_subsampling(input_features_mask: torch.Tensor) -> torch.Tensor:
        mask = input_features_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return mask.to(torch.bool)[:, ::2][:, ::2]

    @staticmethod
    def _convert_audio_4d_mask_to_blocked_5d(
        mask_4d: torch.Tensor,
        *,
        chunk_size: int,
        context_left: int,
        context_right: int,
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device
        max_past_horizon = int(context_left) - 1
        max_future_horizon = int(context_right)
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)

        block_starts = torch.arange(num_blocks, device=device, dtype=torch.long) * chunk_size
        offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device, dtype=torch.long)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)
        return mask_5d.gather(-1, kv_indices)

    @classmethod
    def build_audio_attention_mask(
        cls,
        input_features_mask: torch.Tensor,
        *,
        chunk_size: int = 12,
        context_left: int = 13,
        context_right: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        valid_positions = cls._audio_feature_mask_after_subsampling(input_features_mask)
        seq_len = valid_positions.shape[-1]
        device = valid_positions.device
        left_window_size = int(context_left) - 1
        right_window_size = int(context_right)

        query_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, 1, 1, seq_len)
        dist = query_idx - key_idx
        window_mask = ((dist >= 0) & (dist < left_window_size)) | ((dist < 0) & (-dist < right_window_size))
        valid_mask = valid_positions[:, None, :, None] & valid_positions[:, None, None, :]
        blocked_bool_mask = cls._convert_audio_4d_mask_to_blocked_5d(
            valid_mask & window_mask,
            chunk_size=int(chunk_size),
            context_left=int(context_left),
            context_right=int(context_right),
        )
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
        additive_mask = torch.zeros(blocked_bool_mask.shape, dtype=dtype, device=device)
        return additive_mask.masked_fill(~blocked_bool_mask, neg)

    def _add_audio_attention_mask(self, model_inputs):
        input_features_mask = model_inputs.get("input_features_mask")
        if input_features_mask is None:
            return model_inputs
        feature_mask = input_features_mask.to(torch.float16)
        model_inputs["input_features_mask"] = feature_mask
        model_inputs["audio_attention_mask"] = self.build_audio_attention_mask(
            feature_mask,
            chunk_size=self.config.audio_attention_chunk_size,
            context_left=self.config.audio_attention_context_left,
            context_right=self.config.audio_attention_context_right,
            dtype=torch.float16,
        )
        return model_inputs

    def _compute_audio_soft_token_count_from_feature_frames(self, num_feature_frames: int) -> int:
        if num_feature_frames <= 0:
            return 0
        tokens = num_feature_frames
        for _ in range(2):
            tokens = (tokens + 2 - 3) // 2 + 1
        return min(tokens, self.audio_seq_length)

    def _retokenize_audio_placeholders_if_needed(self, model_inputs, kwargs):
        text = kwargs.get("text")
        input_ids = model_inputs.get("input_ids")
        input_features_mask = model_inputs.get("input_features_mask")
        audio_token_id = getattr(self.tokenizer, "audio_token_id", None)

        if (
            text is None
            or input_ids is None
            or input_features_mask is None
            or audio_token_id is None
            or self.audio_token is None
            or self.boa_token is None
            or self.eoa_token is None
        ):
            return model_inputs

        feature_masks = [input_features_mask] if input_features_mask.ndim == 1 else list(input_features_mask)
        expected_counts = [
            self._compute_audio_soft_token_count_from_feature_frames(int(feature_mask.to(torch.int64).sum().item()))
            for feature_mask in feature_masks
        ]
        actual_audio_token_count = int((input_ids == audio_token_id).sum().item())
        if actual_audio_token_count == sum(expected_counts):
            return model_inputs

        texts = [text] if isinstance(text, str) else list(text)
        placeholder_count = sum(prompt.count(self.audio_token) for prompt in texts)
        if placeholder_count != len(expected_counts):
            raise ValueError(
                f"Audio placeholder count does not match audio inputs: {placeholder_count} vs {len(expected_counts)}"
            )

        replacements = iter(
            f"{self.boa_token}{self.audio_token * token_count}{self.eoa_token}" for token_count in expected_counts
        )
        audio_pattern = re.escape(self.audio_token)
        adjusted_text = [re.sub(audio_pattern, lambda _: next(replacements), prompt) for prompt in texts]
        if isinstance(text, str):
            adjusted_text = adjusted_text[0]

        retokenize_kwargs = {}
        for key in (
            "padding",
            "truncation",
            "max_length",
            "pad_to_multiple_of",
            "return_attention_mask",
            "return_token_type_ids",
            "return_tensors",
        ):
            if key in kwargs:
                retokenize_kwargs[key] = kwargs[key]
        if "images" in kwargs and kwargs["images"] is not None:
            retokenize_kwargs["images"] = kwargs["images"]
        text_only_inputs = super().__call__(text=adjusted_text, **retokenize_kwargs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in text_only_inputs:
                model_inputs[key] = text_only_inputs[key]
        return model_inputs

    @staticmethod
    def _load_wav_as_numpy(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
        with wave.open(os.fspath(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported wav sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sample_rate

    @classmethod
    def _normalize_local_wav_audio(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                normalized_messages.append(message)
                continue
            new_content: list[Any] = []
            changed = False
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    new_content.append(item)
                    continue
                audio = item.get("audio")
                if isinstance(audio, (str, os.PathLike)):
                    audio_path = Path(audio)
                    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
                        audio_array, sample_rate = cls._load_wav_as_numpy(audio_path)
                        new_item = dict(item)
                        new_item["audio"] = audio_array
                        new_item.setdefault("sampling_rate", sample_rate)
                        new_content.append(new_item)
                        changed = True
                        continue
                new_content.append(item)
            normalized_messages.append({**message, "content": new_content} if changed else message)
        return normalized_messages

    def _normalize_role(self, role: str) -> str:
        if role == "assistant":
            return "model"
        return role

    def _render_messages_fallback(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        rendered_messages: list[str] = []
        bos_token = self.tokenizer.bos_token or ""
        if bos_token:
            rendered_messages.append(bos_token)

        for message in messages:
            role = self._normalize_role(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                body = content.strip()
            else:
                parts: list[str] = []
                for item in content:
                    item_type = item.get("type", "text")
                    if item_type in ("image", "image_url"):
                        parts.append("<|image|>")
                    elif item_type == "audio":
                        parts.append("<|audio|>")
                    elif item_type == "video":
                        parts.append("<|video|>")
                    else:
                        parts.append(item.get("text", "").strip())
                body = "".join(parts)
            rendered_messages.append(f"<|turn>{role}\n{body}<turn|>\n")

        if add_generation_prompt:
            rendered_messages.append("<|turn>model\n")
        return "".join(rendered_messages)

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> tuple[str, list, list, list, int]:
        images: list[Any] = []
        videos: list[Any] = []
        audios: list[Any] = []
        sampling_rate = self.config.sampling_rate

        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                for item in content:
                    item_type = item.get("type", "text")
                    if item_type in ("image", "image_url"):
                        images.append(item.get("image", item.get("url", item.get("image_url"))))
                    elif item_type == "video":
                        videos.append(item.get("video"))
                    elif item_type == "audio":
                        audios.append(item.get("audio"))
                        sampling_rate = int(item.get("sampling_rate", sampling_rate))
        if getattr(self, "chat_template", None):
            rendered = super().apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        else:
            rendered = self._render_messages_fallback(messages, add_generation_prompt)
        return rendered, images, videos, audios, sampling_rate

    def apply_chat_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs):
        messages = self._normalize_local_wav_audio(messages)
        enable_thinking = kwargs.pop("enable_thinking", False)
        text, images, videos, audios, sampling_rate = self._render_messages(
            messages, add_generation_prompt, enable_thinking=enable_thinking,
        )

        nested_processor_kwargs = kwargs.pop("processor_kwargs", None)
        kwargs.pop("tokenize", None)
        kwargs.pop("return_dict", None)
        processor_kwargs = {
            "text": text,
            "padding": True,
            "return_tensors": "pt",
        }
        if isinstance(nested_processor_kwargs, dict):
            processor_kwargs.update(nested_processor_kwargs)
        processor_kwargs.update(kwargs)
        if images:
            processor_kwargs["images"] = images[0] if len(images) == 1 else images
        if videos:
            processor_kwargs["videos"] = videos[0] if len(videos) == 1 else videos
        if audios:
            processor_kwargs["audio"] = audios[0] if len(audios) == 1 else audios
            processor_kwargs["sampling_rate"] = sampling_rate
        return self(**processor_kwargs)
