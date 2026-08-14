#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Run Ornith 1.0 multimodal generation with the standalone TCIM runtime.
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

import argparse
import math
import os
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

MODEL_DIR = Path(__file__).resolve().parent
IMODELZOO_ROOT = MODEL_DIR.parents[2]
HOUMO_ENGINE = IMODELZOO_ROOT / "utils" / "python"
sys.path.insert(0, str(HOUMO_ENGINE))

from houmo_engine.perf import PerfTracker
from processor import create_processor, process_visual_info

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
if HOUMO_TARGET not in (None, "xh2"):
    raise ValueError(f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}")

DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / "xh2"
DEFAULT_IMAGE_PATH = IMODELZOO_ROOT / "data" / "pic" / "beach.jpeg"
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISUAL_START_TOKEN_ID = 248053
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
E2E_STAGE = "llm.e2e"
TTFT_STAGE = "llm.ttft"


@dataclass
class _GenerationState:
    history: list
    generated: list = field(default_factory=list)
    context: int = 0
    emitted: str = ""
    next_id: int = None
    decode_tokens: int = 0


@dataclass
class _GenerationTimers:
    ttft_active: bool = False
    e2e_active: bool = True


def load_model_config(config_path, model_name=None, model_size=None):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_configs = config.get("model_configs", {})
    selected_name = model_name or config.get("default_model_name")
    if (
        selected_name not in model_configs
        and model_name is None
        and len(model_configs) == 1
    ):
        selected_name = next(iter(model_configs))
    size_configs = model_configs.get(selected_name, {})
    selected_size = model_size or config.get("default_model_size")
    if (
        selected_size not in size_configs
        and model_size is None
        and len(size_configs) == 1
    ):
        selected_size = next(iter(size_configs))
    try:
        return selected_name, selected_size, size_configs[selected_size]
    except KeyError as error:
        raise ValueError(
            f"unsupported model configuration: {selected_name}-{selected_size}"
        ) from error


def _names(runtime, kind):
    count = getattr(runtime, f"get_num_{kind}")()
    getter = getattr(runtime, f"get_{kind[:-1]}_name")
    return {getter(index) for index in range(count)}


_CacheContract = namedtuple(
    "_CacheContract",
    ("attention", "conv", "recurrent", "prefill_recurrent_outputs"),
)


class _VisualGridCursor:
    def __init__(self, images, videos):
        self.images = images
        self.videos = videos
        self.image_index = 0
        self.video_index = 0


def _inspect_cache_contract(prefill, decode):
    prefill_inputs = _names(prefill, "inputs")
    decode_inputs = _names(decode, "inputs")
    prefill_outputs = _names(prefill, "outputs")
    decode_outputs = _names(decode, "outputs")
    attention = sorted(name for name in prefill_inputs if "model_layers" in name)
    conv = sorted(
        name for name in prefill_inputs if name.startswith("past_conv_cache_")
    )
    recurrent = sorted(
        name for name in prefill_inputs if name.startswith("past_recurrent_state_")
    )
    recurrent_outputs = {
        name.replace("past_recurrent_state_", "recurrent_state_out_")
        for name in recurrent
    }
    prefill_recurrent_outputs = recurrent_outputs & prefill_outputs
    if not recurrent:
        raise RuntimeError("Ornith prefill is missing recurrent inputs")
    if prefill_recurrent_outputs not in (set(), recurrent_outputs):
        missing = sorted(recurrent_outputs - prefill_recurrent_outputs)
        raise RuntimeError(
            "Ornith prefill has an incomplete recurrent output set; "
            f"missing: {missing}"
        )
    if recurrent_outputs - decode_outputs:
        missing = sorted(recurrent_outputs - decode_outputs)
        raise RuntimeError(
            "Ornith decode is missing recurrent outputs; " f"missing: {missing}"
        )
    for name in attention + conv + recurrent:
        if name not in decode_inputs:
            raise RuntimeError(f"decode is missing cache input {name!r}")
    for name in conv:
        output = name.replace("past_conv_cache_", "conv_cache_out_")
        if output not in prefill_outputs or output not in decode_outputs:
            raise RuntimeError(f"conv cache contract is missing output {output!r}")
    for name in recurrent:
        output = name.replace("past_recurrent_state_", "recurrent_state_out_")
        if output not in decode_outputs:
            raise RuntimeError(f"decode is missing recurrent output {output!r}")
    return _CacheContract(attention, conv, recurrent, prefill_recurrent_outputs)


def _bind_cache_aliases(prefill, decode, contract):
    for name in contract.attention:
        decode.set_dev_input(name, prefill.get_dev_input(name))
    for name in contract.conv:
        output = name.replace("past_conv_cache_", "conv_cache_out_")
        cache = prefill.get_dev_input(name)
        prefill.set_dev_output(output, cache)
        decode.set_dev_input(name, cache)
        decode.set_dev_output(output, cache)
    for name in contract.recurrent:
        output = name.replace("past_recurrent_state_", "recurrent_state_out_")
        cache = prefill.get_dev_input(name)
        if output in contract.prefill_recurrent_outputs:
            prefill.set_dev_output(output, cache)
        decode.set_dev_input(name, cache)
        decode.set_dev_output(output, cache)


def bind_model_caches(prefill, decode):
    """Bind Ornith cache aliases according to prefill/decode HMM contracts."""
    contract = _inspect_cache_contract(prefill, decode)
    _bind_cache_aliases(prefill, decode, contract)


def get_visual_input_size(visual):
    """Return the fixed height and width expected by the visual HMM."""
    input_name = visual.get_input_name(0)
    shape = _shape(visual, input_name)
    if len(shape) != 5:
        raise RuntimeError(f"Unexpected visual input shape for {input_name!r}: {shape}")
    return shape[-2], shape[-1]


def build_image_content(image_paths, height, width):
    """Build image messages using the visual HMM's fixed input size."""
    return [
        {
            "type": "image",
            "image": str(path),
            "resized_height": height,
            "resized_width": width,
        }
        for path in image_paths
    ]


def _rope_text_positions(input_ids, attention_mask):
    import torch

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    deltas = torch.zeros(
        (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device
    )
    return position_ids.unsqueeze(0).expand(3, -1, -1), deltas


def _next_visual_segment(tokens, start, image_count, video_count, cursor):
    image_end = (
        tokens.index(IMAGE_TOKEN_ID, start)
        if image_count and IMAGE_TOKEN_ID in tokens
        else len(tokens) + 1
    )
    video_end = (
        tokens.index(VIDEO_TOKEN_ID, start)
        if video_count and VIDEO_TOKEN_ID in tokens
        else len(tokens) + 1
    )
    if image_end < video_end:
        t, h, w = cursor.images[cursor.image_index]
        cursor.image_index += 1
        return image_end, t, h, w, image_count - 1, video_count
    t, h, w = cursor.videos[cursor.video_index]
    cursor.video_index += 1
    return video_end, t, h, w, image_count, video_count - 1


def _visual_positions(t, h, w, text_len, start_index, device):
    import torch

    grid_h = h // SPATIAL_MERGE_SIZE
    grid_w = w // SPATIAL_MERGE_SIZE
    t_index = (
        torch.arange(t, device=device).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    )
    h_index = (
        torch.arange(grid_h, device=device)
        .view(1, -1, 1)
        .expand(t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, device=device)
        .view(1, 1, -1)
        .expand(t, grid_h, -1)
        .flatten()
    )
    return (
        torch.stack((t_index, h_index, w_index)) + text_len + start_index,
        grid_h,
        grid_w,
    )


def _text_positions(length, start_index, device):
    import torch

    return torch.arange(length, device=device).view(1, -1).expand(3, -1) + start_index


def _next_start_index(pieces):
    return pieces[-1].max().item() + 1 if pieces else 0


def _rope_batch_positions(full_ids, mask_row, cursor, device, seq_len):
    import torch

    ids = full_ids[mask_row == 1]
    tokens = ids.tolist()
    starts = torch.argwhere(ids == VISUAL_START_TOKEN_ID).flatten()
    visual_tokens = ids[starts + 1] if starts.numel() else ids[:0]
    image_count = int((visual_tokens == IMAGE_TOKEN_ID).sum().item())
    video_count = int((visual_tokens == VIDEO_TOKEN_ID).sum().item())
    pieces = []
    start = 0
    for _ in range(image_count + video_count):
        end, t, h, w, image_count, video_count = _next_visual_segment(
            tokens, start, image_count, video_count, cursor
        )
        text_len = end - start
        start_index = _next_start_index(pieces)
        pieces.append(_text_positions(text_len, start_index, device))
        visual, grid_h, grid_w = _visual_positions(
            t, h, w, text_len, start_index, device
        )
        pieces.append(visual)
        start = end + t * grid_h * grid_w
    if start < len(tokens):
        start_index = _next_start_index(pieces)
        pieces.append(_text_positions(len(tokens) - start, start_index, device))
    if not pieces:
        pieces.append(torch.empty((3, 0), dtype=torch.long, device=device))
    positions = torch.cat(pieces, dim=1)
    return positions, positions.max() + 1 - seq_len


def get_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    """Build Qwen3.5 three-axis M-RoPE positions and decode offset."""
    import torch

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if image_grid_thw is None and video_grid_thw is None:
        return _rope_text_positions(input_ids, attention_mask)

    position_ids = torch.zeros(
        (3, input_ids.shape[0], input_ids.shape[1]),
        dtype=torch.long,
        device=input_ids.device,
    )
    images = image_grid_thw.tolist() if image_grid_thw is not None else []
    videos = video_grid_thw.tolist() if video_grid_thw is not None else []
    cursor = _VisualGridCursor(images, videos)
    deltas = []
    seq_len = input_ids.shape[1]
    for batch_index, full_ids in enumerate(input_ids):
        positions, delta = _rope_batch_positions(
            full_ids, attention_mask[batch_index], cursor, input_ids.device, seq_len
        )
        position_ids[..., batch_index, attention_mask[batch_index] == 1] = positions
        deltas.append(delta)

    return position_ids, torch.stack(deltas).to(input_ids.device).view(-1, 1)


def scatter_image_embeds(input_ids, token_embeds, image_embeds, image_token_id):
    """Replace image placeholder embeddings with actual visual features."""
    import torch

    locations = input_ids == image_token_id
    token_count = int(locations.sum().item())
    if token_count != int(image_embeds.shape[0]):
        raise ValueError(
            f"Image tokens/features mismatch: tokens={token_count}, "
            f"features={image_embeds.shape[0]}"
        )
    if token_count:
        mask = locations.unsqueeze(-1).expand_as(token_embeds)
        values = image_embeds.to(token_embeds.device, token_embeds.dtype)
        return token_embeds.masked_scatter(mask, values)
    return token_embeds


def incremental_text(tokenizer, generated, emitted):
    """Decode generated tokens and return the newly completed suffix."""
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if text.startswith(emitted):
        return text, text[len(emitted) :]
    return text, ""


def collect_generated_tokens(first_token, eos_token_id, max_new_tokens, decode_token):
    """Collect tokens while honoring the first-token generation boundary."""
    if max_new_tokens <= 0 or first_token == eos_token_id:
        return []
    generated = [first_token]
    while len(generated) < max_new_tokens:
        next_token = decode_token()
        if next_token == eos_token_id:
            break
        generated.append(next_token)
    return generated


def _processor(tokenizer_dir, max_h, max_w, patch_size):
    return create_processor(
        tokenizer_dir,
        max_h=max_h,
        max_w=max_w,
        patch_size=patch_size,
        merge_size=SPATIAL_MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
    )


class Sampling:
    def __init__(
        self,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty

    def select(self, logits, history):
        values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
        for token in set(history):
            if self.presence_penalty:
                values[token] -= self.presence_penalty
            if self.repetition_penalty != 1.0:
                values[token] = (
                    values[token] / self.repetition_penalty
                    if values[token] >= 0
                    else values[token] * self.repetition_penalty
                )
        if self.temperature > 0 and self.temperature != 1.0:
            values /= self.temperature
        if self.top_k and 0 < self.top_k < values.size:
            keep = np.argpartition(values, -self.top_k)[-self.top_k :]
            mask = np.ones(values.size, dtype=bool)
            mask[keep] = False
            values[mask] = -np.inf
        if 0 < self.top_p < 1.0:
            order = np.argsort(values)[::-1]
            finite = np.isfinite(values[order])
            scores = values[order][finite]
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            remove = order[finite][np.cumsum(probs) > self.top_p]
            if remove.size > 0:
                values[remove[1:]] = -np.inf
        return int(np.argmax(values))


def _shape(runtime, name):
    return tuple(int(x) for x in runtime.get_input_info(name).shape)


def _set(runtime, name, value):
    expected = _shape(runtime, name)
    array = np.asarray(value)
    if array.shape != expected:
        if array.size != math.prod(expected):
            raise RuntimeError(
                f"Input shape mismatch for {name!r}: expected {expected}, got {array.shape}"
            )
        array = array.reshape(expected)
    runtime.set_input(name, array)


class OrnithRuntime:
    def __init__(self, args, perf):
        import torch
        import tcim_lite as tcim

        self.args = args
        self.perf = perf
        self.torch = torch
        devices = list(range(args.ndevice))
        manager = tcim.runtime.WeightManager(
            tcim.runtime.DevManager(devices, "Xh2HalBackend")
        )
        prefill_option = tcim.runtime.Option(manager)
        decode_option = tcim.runtime.Option(manager)
        self.prefill = tcim.runtime.load(args.prefill_path, option=prefill_option)
        dummy = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "model_layers" in self.prefill.get_input_name(i)
        ]
        decode_option.set_dummy_tensors(dummy)
        self.decode = tcim.runtime.load(args.decode_path, option=decode_option)
        bind_model_caches(self.prefill, self.decode)
        self.prefill_length = _shape(self.prefill, "input_1")[1]
        self.embedding_len = _shape(self.prefill, "input_1")[2]
        self.context_max_length = max(
            _shape(self.decode, name)[2]
            for name in _names(self.decode, "inputs")
            if "kcache" in name
        )
        self.embedding_weight = (
            self._load_embedding(args.embedding_path)
            .reshape(-1, self.embedding_len)
            .float()
        )
        self.processor = _processor(
            args.tokenizer_dir, args.max_size_h, args.max_size_w, args.patch_size
        )
        self.tokenizer = self.processor.tokenizer
        self.sampling = Sampling(
            args.temperature,
            args.top_k,
            args.top_p,
            args.presence_penalty,
            args.repetition_penalty,
        )
        self.clear_cache()
        self.visual = None
        self.visual_height = args.max_size_h
        self.visual_width = args.max_size_w
        if args.visual_path:
            visual_manager = tcim.runtime.WeightManager(
                tcim.runtime.DevManager([0], "Xh2HalBackend")
            )
            self.visual = tcim.runtime.load(
                args.visual_path, option=tcim.runtime.Option(visual_manager)
            )
            self.visual_height, self.visual_width = get_visual_input_size(self.visual)

    def _load_embedding(self, path):
        value = self.torch.load(path, map_location="cpu", weights_only=False)
        return value["weight"] if isinstance(value, dict) else value.weight

    def clear_cache(self):
        for name in _names(self.prefill, "inputs"):
            if name.startswith(("past_conv_cache_", "past_recurrent_state_")):
                info = self.prefill.get_dev_input(name).info
                zeros = np.zeros(info.shape, dtype=np.dtype(info.dtype))
                _set(self.prefill, name, zeros)
                _set(self.decode, name, zeros)

    def run_visual(self, pixels):
        if self.visual is None:
            raise RuntimeError("visual model not loaded")
        import torch

        output = []
        name = self.visual.get_input_name(0)
        for pixel in pixels:
            with self.perf.scope("llm.visual.set_input"):
                value = pixel.detach().cpu().numpy()
                _set(self.visual, name, value)
            with self.perf.scope("llm.visual.infer"):
                self.visual.run()
                self.visual.sync()
            with self.perf.scope("llm.visual.get_output"):
                value = torch.from_numpy(
                    self.visual.get_output(self.visual.get_output_name(0)).numpy()
                )
            output.append(value)
        return output

    def rope_text(self, valid, length):
        import torch

        p = (
            torch.arange(valid, valid + length, dtype=torch.long)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        return p, torch.tensor([[0]], dtype=torch.long)

    def _messages(self, question, image_paths, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = (
            build_image_content(
                image_paths,
                self.visual_height,
                self.visual_width,
            )
            if image_paths
            else []
        )
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        return messages

    def _text_inputs(self, text):
        import torch
        import torch.nn.functional as F

        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        embeds = F.embedding(input_ids, self.embedding_weight)
        positions, delta = get_rope_index(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
        return input_ids, embeds, positions, delta

    def _image_inputs(self, text, messages):
        import torch
        import torch.nn.functional as F

        with self.perf.scope("llm.visual"):
            with self.perf.scope("llm.visual.preprocess"):
                images, _ = process_visual_info(messages)
                model_inputs = self.processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = model_inputs["input_ids"]
                embeds = F.embedding(input_ids, self.embedding_weight)
            image_outputs = self.run_visual(model_inputs["hm_pixel_values"])
            with self.perf.scope("llm.visual.postprocess"):
                image_outputs = [
                    (
                        value.squeeze(0)
                        if value.dim() == 3 and value.shape[0] == 1
                        else value
                    )
                    for value in image_outputs
                ]
                image_embeds = torch.cat(image_outputs)
                image_id = getattr(self.processor, "image_token_id", IMAGE_TOKEN_ID)
                positions, delta = get_rope_index(
                    input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=model_inputs.get("attention_mask"),
                )
                embeds = scatter_image_embeds(input_ids, embeds, image_embeds, image_id)
        return input_ids, embeds, positions, delta

    def _prepare_inputs(self, question, image_paths, system_prompt):
        messages = self._messages(question, image_paths, system_prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if image_paths:
            return self._image_inputs(text, messages)
        return self._text_inputs(text)

    def _prefill_chunk(self, embeds, positions, start):
        chunk = embeds[:, start : start + self.prefill_length]
        length = chunk.shape[1]
        padded = np.zeros(
            (1, self.prefill_length, self.embedding_len),
            dtype=np.float16,
        )
        padded[:, :length] = chunk.numpy()
        pos = positions[:, 0, start : start + self.prefill_length].numpy()
        if pos.shape[1] < self.prefill_length:
            pos = np.pad(
                pos,
                ((0, 0), (0, self.prefill_length - pos.shape[1])),
                mode="edge",
            )
        mask = [[1.0] * length + [0.0] * (self.prefill_length - length)]
        return padded, pos, mask, length

    def _set_prefill_inputs(self, padded, pos, mask, context, length):
        _set(self.prefill, "input_1", padded)
        _set(self.prefill, "time_position_ids", pos[0])
        _set(self.prefill, "hight_position_ids", pos[1])
        _set(self.prefill, "width_position_ids", pos[2])
        _set(self.prefill, "valid_length", [context])
        _set(self.prefill, "current_length", [length])
        _set(self.prefill, "linear_attn_mask", mask)

    def _sample_first_token(self, state, logits, eos, stream, timers):
        state.next_id = self.sampling.select(logits, state.history)
        self.perf.end(TTFT_STAGE)
        timers.ttft_active = False
        if state.next_id != eos:
            state.generated.append(state.next_id)
            state.history.append(state.next_id)
            if stream:
                state.emitted, first_chunk = incremental_text(
                    self.tokenizer, state.generated, state.emitted
                )
                return first_chunk
        return ""

    def _run_prefill(self, embeds, positions, state, eos, stream, timers):
        first_chunk = ""
        with self.perf.scope("llm.prefill"):
            for start in range(0, embeds.shape[1], self.prefill_length):
                with self.perf.scope("llm.prefill.preprocess"):
                    padded, pos, mask, length = self._prefill_chunk(
                        embeds, positions, start
                    )
                    is_last = start + self.prefill_length >= embeds.shape[1]
                with self.perf.scope("llm.prefill.set_input"):
                    self._set_prefill_inputs(padded, pos, mask, state.context, length)
                with self.perf.scope("llm.prefill.infer"):
                    self.prefill.run()
                    self.prefill.sync()
                with self.perf.scope("llm.prefill.get_output"):
                    logits = self.prefill.get_output("logits").numpy()
                with self.perf.scope("llm.prefill.postprocess"):
                    state.context += length
                    if is_last and self.args.max_new_tokens > 0:
                        first_chunk = self._sample_first_token(
                            state, logits, eos, stream, timers
                        )
        return first_chunk

    def _decode_step(self, state, delta, use_image, eos, stream):
        with self.perf.scope("llm.decode.preprocess"):
            embedding = self.embedding_weight[state.next_id].numpy().reshape(1, 1, -1)
            position = state.context + int(delta.item()) if use_image else state.context
        with self.perf.scope("llm.decode.set_input"):
            _set(self.decode, "input_1", embedding)
            _set(self.decode, "time_position_ids", [position])
            _set(self.decode, "hight_position_ids", [position])
            _set(self.decode, "width_position_ids", [position])
            _set(self.decode, "valid_length", [state.context])
            _set(self.decode, "current_length", [1])
            _set(self.decode, "linear_attn_mask", [[1.0]])
        with self.perf.scope("llm.decode.infer"):
            self.decode.run()
            self.decode.sync()
        state.decode_tokens += 1
        with self.perf.scope("llm.decode.get_output"):
            logits = self.decode.get_output("logits").numpy()
        chunk = ""
        with self.perf.scope("llm.decode.postprocess"):
            state.context += 1
            state.next_id = self.sampling.select(logits, state.history)
            if state.next_id != eos:
                state.generated.append(state.next_id)
                state.history.append(state.next_id)
                if stream:
                    state.emitted, chunk = incremental_text(
                        self.tokenizer, state.generated, state.emitted
                    )
        return chunk

    def _can_decode(self, state, eos):
        return (
            len(state.generated) < self.args.max_new_tokens
            and state.next_id != eos
            and state.context < self.context_max_length
        )

    def _emit(self, chunk, timers):
        if not chunk:
            return
        self.perf.end(E2E_STAGE)
        timers.e2e_active = False
        print(chunk, end="", flush=True)
        self.perf.start(E2E_STAGE)
        timers.e2e_active = True

    def _decode(self, state, delta, use_image, eos, stream, timers):
        with self.perf.scope("llm.decode"):
            while self._can_decode(state, eos):
                self._emit(
                    self._decode_step(state, delta, use_image, eos, stream),
                    timers,
                )

    def generate(self, question, image_paths=None, system_prompt=None, stream=False):
        self.perf.reset(preserve_prefixes=("llm.init",))
        self.clear_cache()
        use_image = bool(image_paths)
        if use_image and self.visual is None:
            raise RuntimeError("visual model not loaded. Provide --visual_path.")

        timers = _GenerationTimers()
        self.perf.start(E2E_STAGE)
        if self.args.max_new_tokens > 0:
            self.perf.start(TTFT_STAGE)
            timers.ttft_active = True
        try:
            input_ids, embeds, positions, delta = self._prepare_inputs(
                question, image_paths, system_prompt
            )
            state = _GenerationState(input_ids.reshape(-1).tolist())
            eos = getattr(self.tokenizer, "eos_token_id", None)
            first_chunk = self._run_prefill(
                embeds, positions, state, eos, stream, timers
            )
            if first_chunk:
                self._emit(first_chunk, timers)
            if self._can_decode(state, eos):
                self._decode(state, delta, use_image, eos, stream, timers)
            self.perf.set_metrics(
                "llm",
                input_tokens=int(input_ids.numel()),
                output_tokens=(
                    (1 + state.decode_tokens) if self.args.max_new_tokens > 0 else 0
                ),
                decode_tokens=state.decode_tokens,
                num_images=len(image_paths) if image_paths else 0,
            )
            if timers.e2e_active:
                self.perf.end(E2E_STAGE)
                timers.e2e_active = False
            return self.tokenizer.decode(state.generated, skip_special_tokens=True)
        finally:
            if timers.ttft_active:
                self.perf.end(TTFT_STAGE)
            if timers.e2e_active:
                self.perf.end(E2E_STAGE)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Ornith 1.0 VLM demo.")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        default=str(DEFAULT_CONFIG_PATH),
    )
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--question", default="描述这张图片")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--image_path", nargs="+", default=[str(DEFAULT_IMAGE_PATH)])
    parser.add_argument("--prefill_path", default=None)
    parser.add_argument("--decode_path", default=None)
    parser.add_argument("--visual_path", default=None)
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--ndevice", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--max_size_w", type=int, default=None)
    parser.add_argument("--max_size_h", type=int, default=None)
    parser.add_argument("--max_size_t", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", "--top-k", type=int, default=None)
    parser.add_argument("--top_p", "--top-p", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--perf",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    args = parser.parse_args()
    args.model_name, args.model_size, cfg = load_model_config(
        args.config_path, args.model_name, args.model_size
    )
    args.ndevice = args.ndevice or int(cfg.get("ndevice", 1))
    args.batch = args.batch or int(cfg.get("batch", 1))
    if args.batch != 1:
        raise ValueError("Ornith demo only supports batch=1")
    args.max_size_w = args.max_size_w or int(cfg.get("max_size_w", 448))
    args.max_size_h = args.max_size_h or int(cfg.get("max_size_h", 448))
    args.max_size_t = args.max_size_t or int(cfg.get("max_size_t", 2))
    prefix = f"{args.model_name}-{args.model_size}"
    args.prefill_path = args.prefill_path or str(
        DEFAULT_OUTPUT_DIR / f"{prefix}_prefill.hmm"
    )
    args.decode_path = args.decode_path or str(
        DEFAULT_OUTPUT_DIR / f"{prefix}_decode.hmm"
    )
    args.visual_path = args.visual_path or str(
        DEFAULT_OUTPUT_DIR
        / f"{prefix}_visual_{args.max_size_w}x{args.max_size_h}x{args.max_size_t}.hmm"
    )
    args.embedding_path = args.embedding_path or str(
        DEFAULT_OUTPUT_DIR / "hmquant" / "quant_embedding.pt"
    )
    args.tokenizer_dir = args.tokenizer_dir or str(
        DEFAULT_OUTPUT_DIR / "hmquant" / "hf_config"
    )
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.removesuffix(".hmm") + ".hmms"
        args.decode_path = args.decode_path.removesuffix(".hmm") + ".hmms"
    return args


def main():
    args = parse_args()
    perf = PerfTracker.create(args.perf)
    with perf.scope("llm.init"):
        model = OrnithRuntime(args, perf)
    print(f"\033[1;95m\nQ: {args.question}\nA: ", end="", flush=True)
    model.generate(
        args.question,
        args.image_path,
        args.system_prompt,
        stream=True,
    )
    print("\033[0m")
    perf.print_summary()


if __name__ == "__main__":
    main()
