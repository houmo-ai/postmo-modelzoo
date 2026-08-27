# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_5.py
# Description:
#   Qwen3.5 runtime Module implementation.
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

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tcim_lite as tcim
from loguru import logger

from houmo_engine import HoumoModule
from houmo_engine.core.types import Stage, StageInputs, StageOutputs
from houmo_engine.perf import PerfTracker

VISION_INPUT_NAMES = (
    "pixel_values",
    "position_ids",
    "position_weights",
    "rotary_position_ids",
    "attention_mask",
)
VISION_GEARS = (96, 196, 384, 704, 1536)
SPATIAL_MERGE_SIZE = 2
NUM_POSITION_EMBEDDINGS = 2304
VISUAL_ROPE_CACHE_LENGTH = 3072


def _merge_major(tensor: torch.Tensor, t: int, h: int, w: int, merge_size: int) -> torch.Tensor:
    leading_shape = tensor.shape[:-2]
    tensor = tensor.unsqueeze(-3).expand(*leading_shape, t, h, w)
    tensor = tensor.reshape(*leading_shape, t, h // merge_size, merge_size, w // merge_size, merge_size)
    base = len(leading_shape)
    return tensor.permute(*range(base), base, base + 1, base + 3, base + 2, base + 4).flatten(len(leading_shape))


def _build_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    patch_capacity: int,
    position_dtype: torch.dtype,
    num_position_embeddings: int = NUM_POSITION_EMBEDDINGS,
    rotary_cache_length: int = VISUAL_ROPE_CACHE_LENGTH,
) -> tuple[dict[str, torch.Tensor], int]:
    t, h, w = (int(value) for value in grid_thw.reshape(-1).tolist())
    if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
        raise ValueError(f"dynamic visual grid must be (1, h, w), with even h/w; got {(t, h, w)}")
    valid_patches = t * h * w
    merge_unit = SPATIAL_MERGE_SIZE**2
    if valid_patches > patch_capacity or valid_patches % merge_unit:
        raise ValueError(f"image grid needs {valid_patches} patches, capacity is {patch_capacity}")
    if pixel_values.ndim == 2:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 1:
        raise ValueError(f"pixel_values must have shape [N, D] or [1, N, D], got {tuple(pixel_values.shape)}")
    if pixel_values.shape[1] != valid_patches:
        raise ValueError(f"pixel_values has {pixel_values.shape[1]} patches, grid declares {valid_patches}")
    if pixel_values.shape[1] < patch_capacity:
        padding = pixel_values.new_zeros((1, patch_capacity - valid_patches, pixel_values.shape[2]))
        pixel_values = torch.cat((pixel_values, padding), dim=1)

    side = math.isqrt(num_position_embeddings)
    if side * side != num_position_embeddings:
        raise ValueError("NUM_POSITION_EMBEDDINGS must be a square")
    h_coords = torch.linspace(0, side - 1, h, dtype=torch.float32)
    w_coords = torch.linspace(0, side - 1, w, dtype=torch.float32)
    h_floor, w_floor = h_coords.to(torch.int64), w_coords.to(torch.int64)
    h_ceil = (h_floor + 1).clamp(max=side - 1)
    w_ceil = (w_floor + 1).clamp(max=side - 1)
    dh, dw = h_coords - h_floor, w_coords - w_floor
    ids = torch.stack(
        (
            h_floor[:, None] * side + w_floor[None, :],
            h_floor[:, None] * side + w_ceil[None, :],
            h_ceil[:, None] * side + w_floor[None, :],
            h_ceil[:, None] * side + w_ceil[None, :],
        )
    )
    weights = torch.stack(
        (
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        )
    )
    ids = _merge_major(ids, t, h, w, 2).reshape(4, -1)
    weights = _merge_major(weights, t, h, w, 2).reshape(4, -1).to(position_dtype)
    position_ids = torch.zeros((4, patch_capacity), dtype=torch.int64)
    position_weights = torch.zeros((4, patch_capacity), dtype=position_dtype)
    position_ids[:, :valid_patches] = ids
    position_weights[:, :valid_patches] = weights
    rows = _merge_major(torch.arange(h).view(h, 1).expand(h, w), t, h, w, 2).reshape(-1)
    cols = _merge_major(torch.arange(w).view(1, w).expand(h, w), t, h, w, 2).reshape(-1)
    rotary_position_ids = torch.zeros((2, patch_capacity), dtype=torch.int64)
    rotary_position_ids[0, :valid_patches] = rows
    rotary_position_ids[1, :valid_patches] = cols
    if max(h, w) - 1 >= rotary_cache_length:
        raise ValueError(f"image grid {(h, w)} exceeds visual RoPE cache length {rotary_cache_length}")
    attention_mask = torch.zeros((1, 1, 1, patch_capacity), dtype=position_dtype)
    attention_mask[..., valid_patches:] = -torch.finfo(position_dtype).max
    return {
        "pixel_values": pixel_values,
        "position_ids": position_ids,
        "position_weights": position_weights,
        "rotary_position_ids": rotary_position_ids,
        "attention_mask": attention_mask,
    }, valid_patches // merge_unit


class Qwen35Module(HoumoModule):
    """Qwen3.5 HMM graphs, cache bindings, and stage execution."""

    @staticmethod
    def _is_lora_input(name: str) -> bool:
        return "lora_bundle_piplined_te" in name or "lora_bundle_pipelined_te" in name

    def __init__(
        self,
        prefill_path,
        decode_path,
        *,
        vision_paths,
        vision_min_pixels: int = 65536,
        num_position_embeddings: int = NUM_POSITION_EMBEDDINGS,
        visual_rope_cache_length: int = VISUAL_ROPE_CACHE_LENGTH,
        lora_path=None,
        ndevice: int = 1,
        perf: PerfTracker,
    ):
        self.perf = perf
        self._stage_metadata = {}
        self.load(
            prefill_path,
            decode_path,
            vision_paths=vision_paths,
            vision_min_pixels=vision_min_pixels,
            num_position_embeddings=num_position_embeddings,
            visual_rope_cache_length=visual_rope_cache_length,
            lora_path=lora_path,
            ndevice=ndevice,
        )

    def load(
        self,
        prefill_path,
        decode_path,
        *,
        vision_paths,
        vision_min_pixels: int = 65536,
        num_position_embeddings: int = NUM_POSITION_EMBEDDINGS,
        visual_rope_cache_length: int = VISUAL_ROPE_CACHE_LENGTH,
        lora_path=None,
        ndevice: int = 1,
    ) -> None:
        with self.perf.scope("llm.init"):
            if ndevice == 1:
                weight_manager = tcim.runtime.WeightManager(0)
            elif ndevice == 2:
                dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
                weight_manager = tcim.runtime.WeightManager(dev_manager)
            else:
                raise ValueError("unsupported device number")

            with self.perf.scope("llm.init.prefill_load"):
                self.prefill = tcim.runtime.load(str(prefill_path), option=tcim.runtime.Option(weight_manager))
            decode_option = tcim.runtime.Option(weight_manager)
            decode_option.set_dummy_tensors(
                [
                    self.prefill.get_input_name(index)
                    for index in range(self.prefill.get_num_inputs())
                    if "model_layers" in self.prefill.get_input_name(index)
                ]
            )
            with self.perf.scope("llm.init.decode_load"):
                self.decode = tcim.runtime.load(str(decode_path), option=decode_option)
            if not vision_paths:
                raise ValueError("dynamic Qwen3.5 requires vision_paths")
            self.vision_models = {}
            self.vision_gears = tuple(sorted(int(gear) for gear in vision_paths))
            if self.vision_gears != tuple(gear for gear in self.vision_gears if gear in VISION_GEARS):
                raise ValueError(f"unsupported vision gears: {self.vision_gears}")
            vision_dev_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
            vision_weight_manager = tcim.runtime.WeightManager(vision_dev_manager)
            with self.perf.scope("llm.init.vision_load"):
                for gear in self.vision_gears:
                    vision_path = Path(vision_paths[gear])
                    vision_option = tcim.runtime.Option(vision_weight_manager)
                    vision_model = tcim.runtime.load(str(vision_path), option=vision_option)
                    self._validate_vision_model_contract(vision_model, gear, str(vision_path))
                    self.vision_models[gear] = vision_model
            self.vision_patch_capacities = {gear: gear * SPATIAL_MERGE_SIZE**2 for gear in self.vision_gears}
            self.vision_patch_capacity = self.vision_patch_capacities[max(self.vision_gears)]
            self.vision_patch_dim = int(
                self.vision_models[max(self.vision_gears)].get_input_info("pixel_values").shape[2]
            )
            self.vision_min_pixels = int(vision_min_pixels)
            self.num_position_embeddings = int(num_position_embeddings)
            self.visual_rope_cache_length = int(visual_rope_cache_length)
            prefill_shape = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape
            self.prefill_length = int(prefill_shape[1])
            self.embedding_size = int(prefill_shape[2])
            self.context_max_length = int(self.decode.get_input_info(self.decode.get_input_name(7)).shape[2])
            vision_output_shape = (
                self.vision_models[max(self.vision_gears)]
                .get_output_info(self.vision_models[max(self.vision_gears)].get_output_name(0))
                .shape
            )
            if int(vision_output_shape[-1]) != self.embedding_size:
                raise RuntimeError(
                    f"visual embedding size {vision_output_shape[-1]} differs from LLM embedding size {self.embedding_size}"
                )
            self.lora_input_names = [
                self.prefill.get_input_name(index)
                for index in range(self.prefill.get_num_inputs())
                if self._is_lora_input(self.prefill.get_input_name(index))
            ]
            self.lora_path = Path(lora_path).expanduser().resolve() if lora_path is not None else None

            self._bind_caches()
            self.clear_session()
            self._set_input(
                self.decode,
                self.decode.get_input_name(5),
                np.array([1], dtype=np.int32),
            )

            self.activate_switch_lora()

    def _bind_caches(self) -> None:
        prefill_has_recurrent_state_output = any(
            "recurrent_state" in self.prefill.get_output_name(i) for i in range(self.prefill.get_num_outputs())
        )
        for index in range(self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "model_layers" in name or self._is_lora_input(name):
                self.decode.set_dev_input(name, self.prefill.get_dev_input(name))
            elif "conv_cache" in name:
                output = name.replace("past_conv_cache_", "conv_cache_out_")
                cache = self.prefill.get_dev_input(name)
                self.prefill.set_dev_output(output, cache)
                self.decode.set_dev_input(name, cache)
                self.decode.set_dev_output(output, cache)
            elif "recurrent_state" in name:
                output = name.replace("past_recurrent_state_", "recurrent_state_out_")
                cache = self.prefill.get_dev_input(name)
                if prefill_has_recurrent_state_output:
                    self.prefill.set_dev_output(output, cache)
                self.decode.set_dev_input(name, cache)
                self.decode.set_dev_output(output, cache)

    @staticmethod
    def _input_shape(model, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in model.get_input_info(name).shape)

    @staticmethod
    def _input_dtype(model, name: str) -> np.dtype:
        return np.dtype(model.get_input_info(name).dtype)

    @staticmethod
    def _validate_vision_model_contract(model, gear: int, path: str) -> None:
        names = tuple(model.get_input_name(i) for i in range(model.get_num_inputs()))
        if names != VISION_INPUT_NAMES:
            raise RuntimeError(f"vision m{gear} at {path} has inputs {names}, expected {VISION_INPUT_NAMES}")
        capacity = gear * SPATIAL_MERGE_SIZE**2
        expected = {
            "pixel_values": (1, capacity, 1536),
            "position_ids": (4, capacity),
            "position_weights": (4, capacity),
            "rotary_position_ids": (2, capacity),
            "attention_mask": (1, 1, 1, capacity),
        }
        for name, shape in expected.items():
            actual = tuple(int(value) for value in model.get_input_info(name).shape)
            if actual != shape:
                raise RuntimeError(f"vision m{gear} input {name} at {path}: expected {shape}, got {actual}")

    def _set_input(self, model, name: str, value) -> None:
        value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        value = value.astype(self._input_dtype(model, name), copy=False)
        shape = self._input_shape(model, name)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise RuntimeError(f"input {name!r} expects {shape}, got {value.shape}")
            value = value.reshape(shape)
        model.set_input(name, value)

    def activate_switch_lora(self) -> None:
        if not self.lora_input_names:
            return
        if self.lora_path is None:
            for name in self.lora_input_names:
                zeros = np.zeros(self._input_shape(self.prefill, name), dtype=self._input_dtype(self.prefill, name))
                self._set_input(self.prefill, name, zeros)
        else:
            for name in self.lora_input_names:
                lora_weight = np.load(self.lora_path / f"{name}.npy")
                if lora_weight.shape != self._input_shape(self.prefill, name) or lora_weight.dtype != self._input_dtype(
                    self.prefill, name
                ):
                    raise RuntimeError(
                        f"lora weight {name!r} expects {self._input_shape(self.prefill, name)} "
                        f"with dtype {self._input_dtype(self.prefill, name)}, got "
                        f"{lora_weight.shape} with dtype {lora_weight.dtype}"
                    )
                self._set_input(self.prefill, name, lora_weight)

    def reset_lora(self, lora_path) -> None:
        if lora_path is not None:
            lora_path = Path(lora_path).expanduser().resolve()
        if lora_path != self.lora_path:
            self.lora_path = lora_path
            self.activate_switch_lora()
            self.clear_session()

    def clear_session(self) -> None:
        for index in range(self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "conv_cache" not in name and "recurrent_state" not in name:
                continue
            info = self.prefill.get_dev_input(name).info
            zeros = np.zeros(info.shape, dtype=np.float16)
            self._set_input(self.prefill, name, zeros)
            self._set_input(self.decode, name, zeros)

    def _run_vision_item(self, item, index: int, outputs) -> None:
        _, vision_model, values, valid_image_tokens = item
        with self.perf.scope("llm.vision.set_input"):
            for name in VISION_INPUT_NAMES:
                self._set_input(vision_model, name, values[name])
        with self.perf.scope("llm.vision.infer"):
            vision_model.run()
            vision_model.sync()
        with self.perf.scope("llm.vision.get_output"):
            output = vision_model.get_output(vision_model.get_output_name(0)).numpy()
        tensor = torch.from_numpy(output)
        tensor = tensor.squeeze(0) if tensor.ndim == 3 else tensor
        if tensor.ndim != 2 or tensor.shape[0] < valid_image_tokens:
            raise RuntimeError(f"dynamic visual output shape is invalid: {tuple(tensor.shape)}")
        outputs[index] = tensor[:valid_image_tokens]

    def run_vision(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> StageOutputs:
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)
        if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        grids = torch.as_tensor(image_grid_thw).reshape(-1, 3)
        split_sizes = [int(grid.prod()) for grid in grids]
        if pixel_values.ndim != 2 or int(pixel_values.shape[0]) != sum(split_sizes):
            raise ValueError("dynamic visual pixel_values and image_grid_thw are inconsistent")
        if int(pixel_values.shape[1]) != self.vision_patch_dim:
            raise ValueError(f"dynamic visual patch width must be {self.vision_patch_dim}, got {pixel_values.shape[1]}")
        prepared = self._prepare_vision_batch(pixel_values, grids, split_sizes)
        outputs = [None] * len(prepared)
        for gear in self.vision_gears:
            for index, item in enumerate(prepared):
                if item[0] == gear:
                    self._run_vision_item(item, index, outputs)
        return StageOutputs(tensors=(torch.cat(outputs, dim=0),))

    def _prepare_vision_batch(self, pixel_values, grids, split_sizes):
        prepared = []
        for image, grid in zip(torch.split(pixel_values, split_sizes), grids, strict=True):
            valid_image_tokens = int(grid.prod().item()) // SPATIAL_MERGE_SIZE**2
            gear = next((gear for gear in self.vision_gears if valid_image_tokens <= gear), None)
            if gear is None:
                raise ValueError(f"image needs {valid_image_tokens} tokens, loaded gears are {self.vision_gears}")
            model = self.vision_models[gear]
            dtype = torch.from_numpy(np.empty((), dtype=model.get_input_info("position_weights").dtype)).dtype
            values, valid_image_tokens = _build_vision_inputs(
                image,
                grid,
                self.vision_patch_capacities[gear],
                dtype,
                self.num_position_embeddings,
                self.visual_rope_cache_length,
            )
            prepared.append((gear, model, values, valid_image_tokens))
        return prepared

    def _stage_model(self, stage: Stage):
        if stage == Stage.PREFILL:
            return self.prefill, "llm.prefill"
        elif stage == Stage.DECODE:
            return self.decode, "llm.decode"
        raise ValueError(f"unsupported stage: {stage}")

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        model, path = self._stage_model(stage)
        self._stage_metadata[stage] = dict(inputs.metadata)
        with self.perf.scope(f"{path}.set_input"):
            for index, value in enumerate(inputs.tensors):
                self._set_input(model, model.get_input_name(index), value)
            if stage == Stage.DECODE:
                self._set_input(
                    model,
                    model.get_input_name(6),
                    np.ones((1, 1), dtype=np.float16),
                )

    def run(self, stage: Stage) -> None:
        model, path = self._stage_model(stage)
        with self.perf.scope(f"{path}.infer"):
            model.run()
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        model, path = self._stage_model(stage)
        with self.perf.scope(f"{path}.get_output"):
            output = model.get_output(model.get_output_name(0)).numpy()
        return StageOutputs(
            tensors=(output,),
            metadata=self._stage_metadata.pop(stage, {}),
        )
_SUFFIX = ".hmcc.format"


def _bare(name: str) -> str:
    return name[: -len(_SUFFIX)] if name.endswith(_SUFFIX) else name


def _numpy_dtype(dtype: Any) -> np.dtype:
    return np.dtype(dtype)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "to_host"):
        value = value.to_host()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class _PrefillNames:
    """Adapt legacy Qwen prefill/verify graph tensor names."""

    _CONV_KIND_ORDER = {"q": 0, "k": 1, "v": 2, "": 3}

    def __init__(self, model):
        inputs = [model.get_input_name(index) for index in range(model.get_num_inputs())]
        outputs = [model.get_output_name(index) for index in range(model.get_num_outputs())]

        self.activation = self._pick(inputs, "input_1")
        self.valid_length = self._pick(inputs, "valid_length")
        self.current_length = self._pick(inputs, "current_length")
        self.time_position_ids = self._pick(inputs, "time_position_ids")
        self.height_position_ids = self._pick(inputs, "hight_position_ids")
        self.width_position_ids = self._pick(inputs, "width_position_ids")
        self.linear_attn_mask = self._pick_optional(inputs, "linear_attn_mask")
        self.kv_in = sorted(name for name in inputs if "_attn_kcache_input" in name or "_attn_vcache_input" in name)

        conv_in = self._conv_names(inputs, "past_conv_cache")
        conv_out = self._conv_names(outputs, "conv_cache_out")
        recurrent_in = self._indexed_names(inputs, r"past_recurrent_state_(\d+)$")
        recurrent_out = self._indexed_names(outputs, r"recurrent_state_out_(\d+)$")
        if not conv_in:
            raise RuntimeError("missing recurrent conv cache inputs")

        layer_indices = sorted({layer for layer, _, _ in conv_in})
        self._require_indices("past_recurrent_state", layer_indices, recurrent_in)
        if recurrent_out:
            self._require_indices("recurrent_state_out", layer_indices, recurrent_out)

        self.conv_cache_in = [name for _, _, name in conv_in]
        self.conv_key_by_input = {name: (layer, kind) for layer, kind, name in conv_in}
        conv_out_by_key = {(layer, kind): name for layer, kind, name in conv_out}
        self.conv_out_by_input = {
            name: conv_out_by_key[key] for name, key in self.conv_key_by_input.items() if key in conv_out_by_key
        }
        self.conv_out_by_key = conv_out_by_key

        self.recurrent_state_in = [name for _, name in recurrent_in]
        self.recurrent_layer_by_input = {name: layer for layer, name in recurrent_in}
        self.recurrent_in_by_layer = {layer: name for layer, name in recurrent_in}
        recurrent_out_by_layer = {layer: name for layer, name in recurrent_out}
        self.recurrent_out_by_input = {
            name: recurrent_out_by_layer[layer]
            for name, layer in self.recurrent_layer_by_input.items()
            if layer in recurrent_out_by_layer
        }
        self.recurrent_out_by_layer = recurrent_out_by_layer

        split_conv = self._split_conv_outputs(outputs, "conv_cache_out")
        self.split_conv_out_by_input = {
            name: split_conv[key] for name, key in self.conv_key_by_input.items() if key in split_conv
        }
        split_recurrent = self._split_outputs(outputs, r"recurrent_state_out_(\d+)_(\d+)$")
        self.split_recurrent_out_by_input = {
            name: split_recurrent[layer]
            for name, layer in self.recurrent_layer_by_input.items()
            if layer in split_recurrent
        }

        self.logits_out = self._pick(outputs, "logits")
        self.hidden_out = self._pick_any(outputs, ("hidden_states", "post_norm_hidden", "pre_norm_hidden"))

    @staticmethod
    def _pick(names: Sequence[str], exact: str) -> str:
        if exact in names:
            return exact
        raise RuntimeError(f"input/output {exact!r} not found")

    @staticmethod
    def _pick_optional(names: Sequence[str], exact: str) -> str | None:
        return exact if exact in names else None

    @staticmethod
    def _pick_any(names: Sequence[str], keywords: Sequence[str]) -> str:
        for keyword in keywords:
            for name in names:
                if keyword in _bare(name):
                    return name
        raise RuntimeError(f"failed to find tensor with keywords {list(keywords)}, " f"candidates={list(names)}")

    @staticmethod
    def _indexed_names(names: Sequence[str], pattern: str) -> list[tuple[int, str]]:
        compiled = re.compile(pattern)
        matches = []
        for name in names:
            match = compiled.match(_bare(name))
            if match:
                matches.append((int(match.group(1)), name))
        return sorted(matches)

    @classmethod
    def _conv_names(cls, names: Sequence[str], prefix: str) -> list[tuple[int, str, str]]:
        compiled = re.compile(rf"{re.escape(prefix)}(?:_([A-Za-z]+))?_(\d+)$")
        matches = []
        for name in names:
            match = compiled.match(_bare(name))
            if match:
                matches.append((int(match.group(2)), match.group(1) or "", name))
        return sorted(
            matches,
            key=lambda item: (
                item[0],
                cls._CONV_KIND_ORDER.get(item[1], 100),
                item[1],
            ),
        )

    @staticmethod
    def _require_indices(
        label: str,
        expected: Sequence[int],
        actual: Sequence[tuple[int, str]],
    ) -> None:
        actual_indices = [index for index, _ in actual]
        if list(expected) != actual_indices:
            raise RuntimeError(f"{label} layer indices mismatch: expected {list(expected)}, " f"got {actual_indices}")

    @staticmethod
    def _split_outputs(names: Sequence[str], pattern: str) -> dict[int, list[str]]:
        compiled = re.compile(pattern)
        matches: dict[int, list[tuple[int, str]]] = {}
        for name in names:
            match = compiled.match(_bare(name))
            if match:
                matches.setdefault(int(match.group(1)), []).append((int(match.group(2)), name))
        return {layer: [name for _, name in sorted(items)] for layer, items in matches.items()}

    @staticmethod
    def _split_conv_outputs(names: Sequence[str], prefix: str) -> dict[tuple[int, str], list[str]]:
        compiled = re.compile(rf"{re.escape(prefix)}(?:_([A-Za-z]+))?_(\d+)_(\d+)$")
        matches: dict[tuple[int, str], list[tuple[int, str]]] = {}
        for name in names:
            match = compiled.match(_bare(name))
            if match:
                key = (int(match.group(2)), match.group(1) or "")
                matches.setdefault(key, []).append((int(match.group(3)), name))
        return {key: [name for _, name in sorted(items)] for key, items in matches.items()}


class _MtpNames:
    """Adapt legacy Qwen MTP prefill/draft graph tensor names."""

    def __init__(self, model):
        inputs = [model.get_input_name(index) for index in range(model.get_num_inputs())]
        outputs = [model.get_output_name(index) for index in range(model.get_num_outputs())]
        self.hidden_states = self._pick_any(inputs, ("hidden_states", "post_norm_hidden", "pre_norm_hidden"))
        self.input_embedding = self._pick_any(inputs, ("input_embedding", "next_token_embedding"))
        self.position_ids = tuple(name for name in inputs if "position_ids" in _bare(name))
        self.past_seq_length = self._pick_scalar(model, inputs, ("past_seq", "valid_length"))
        self.current_input_length = self._pick_scalar(model, inputs, ("current_",))
        self.past_key_cache = self._pick_any(inputs, ("past_key_cache",))
        self.past_value_cache = self._pick_any(inputs, ("past_value_cache",))
        self.logits_out = self._pick_any(outputs, ("mtp_logits", "logits"))
        self.hidden_out = self._pick_any(
            outputs,
            (
                "mtp_hidden_states",
                "post_norm_out",
                "hidden_states",
                "post_norm_hidden",
            ),
        )

    @staticmethod
    def _pick_any(names: Sequence[str], keywords: Sequence[str]) -> str:
        for keyword in keywords:
            for name in names:
                if keyword in _bare(name):
                    return name
        raise RuntimeError(f"failed to find tensor with keywords {list(keywords)}, " f"candidates={list(names)}")

    @staticmethod
    def _pick_scalar(model, names: Sequence[str], keywords: Sequence[str]) -> str:
        matches = []
        for name in names:
            if int(np.prod(model.get_dev_input(name).info.shape)) != 1:
                continue
            if any(keyword in _bare(name) for keyword in keywords):
                matches.append(name)
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"failed to find unique scalar tensor with keywords " f"{list(keywords)}, candidates={matches}"
        )


class Qwen36MtpModule(HoumoModule):
    """Qwen3.6 MTP HMM graphs, device caches, and stage execution."""

    def __init__(
        self,
        prefill_path,
        prefill_mtp_path,
        decode_mtp_path,
        decode_verify_path,
        *,
        ndevice: int = 1,
        perf: PerfTracker,
        debug: bool = False,
    ):
        self.perf = perf
        self.debug = debug
        self._stage_metadata = {}
        self.load(
            prefill_path,
            prefill_mtp_path,
            decode_mtp_path,
            decode_verify_path,
            ndevice=ndevice,
        )

    def load(
        self,
        prefill_path,
        prefill_mtp_path,
        decode_mtp_path,
        decode_verify_path,
        *,
        ndevice: int = 1,
    ) -> None:
        if ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif ndevice == 2:
            dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("unsupported device number")

        with self.perf.scope("llm_mtp.init.prefill_load"):
            self.prefill = tcim.runtime.load(str(prefill_path), option=tcim.runtime.Option(weight_manager))
        self._prefill_names = _PrefillNames(self.prefill)

        with self.perf.scope("llm_mtp.init.mtp_prefill_load"):
            self.mtp_prefill = tcim.runtime.load(str(prefill_mtp_path), option=tcim.runtime.Option(weight_manager))
        self._mtp_prefill_names = _MtpNames(self.mtp_prefill)

        verify_option = tcim.runtime.Option(weight_manager)
        verify_option.set_dummy_tensors(list(self._prefill_names.kv_in))
        with self.perf.scope("llm_mtp.init.verify_load"):
            self.verify = tcim.runtime.load(str(decode_verify_path), option=verify_option)
        self._verify_names = _PrefillNames(self.verify)

        with self.perf.scope("llm_mtp.init.draft_load"):
            self.draft = tcim.runtime.load(str(decode_mtp_path), option=tcim.runtime.Option(weight_manager))
        self._draft_names = _MtpNames(self.draft)

        prefill_shape = self._input_shape(self.prefill, self._prefill_names.activation)
        mtp_prefill_shape = self._input_shape(self.mtp_prefill, self._mtp_prefill_names.hidden_states)
        verify_shape = self._input_shape(self.verify, self._verify_names.activation)
        self.embedding_size = int(prefill_shape[2])
        self.prefill_length = int(prefill_shape[1])
        self.mtp_prefill_length = int(mtp_prefill_shape[1])
        self.verify_length = int(verify_shape[1])
        self.draft_block_size = self.verify_length - 1
        if not self._prefill_names.kv_in:
            raise RuntimeError("missing prefill KV cache inputs")
        self.context_max_length = max(
            int(dim) for dim in self.prefill.get_dev_input(self._prefill_names.kv_in[0]).info.shape
        )

        self._validate_and_bind_verify_kv()
        self._validate_and_bind_mtp_cache()
        self.clear_session()
        if self.debug:
            logger.info(
                "Qwen3.6 MTP graphs loaded: prefill={}, mtp_prefill={}, " "verify={}, draft_block={}, context={}",
                self.prefill_length,
                self.mtp_prefill_length,
                self.verify_length,
                self.draft_block_size,
                self.context_max_length,
            )

    @staticmethod
    def _input_shape(model, name: str) -> tuple[int, ...]:
        return tuple(int(dim) for dim in model.get_dev_input(name).info.shape)

    @staticmethod
    def _tensor_info(model, name: str):
        return model.get_dev_input(name).info

    def _set_input(self, model, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        expected = self._input_shape(model, name)
        if value.shape != expected:
            if value.size != int(np.prod(expected)):
                raise RuntimeError(f"input {name!r} expects {expected}, got {value.shape}")
            if self.debug:
                logger.info(
                    "reshaping input {!r} from {} to {}",
                    name,
                    value.shape,
                    expected,
                )
            value = value.reshape(expected)
        model.set_input(name, value)

    @staticmethod
    def _assert_cache_compatible(
        left_model,
        left_name: str,
        right_model,
        right_name: str,
        label: str,
    ) -> None:
        left = left_model.get_dev_input(left_name).info
        right = right_model.get_dev_input(right_name).info
        if tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(
                f"{label} cache shape differs: {left_name} {tuple(left.shape)} " f"vs {right_name} {tuple(right.shape)}"
            )
        if _numpy_dtype(left.dtype) != _numpy_dtype(right.dtype):
            raise RuntimeError(
                f"{label} cache dtype differs: {left_name} {left.dtype} " f"vs {right_name} {right.dtype}"
            )

    def _validate_and_bind_verify_kv(self) -> None:
        prefill_names = self._prefill_names.kv_in
        verify_names = self._verify_names.kv_in
        if len(prefill_names) != len(verify_names):
            raise RuntimeError(
                "prefill and verify KV cache counts differ: " f"{len(prefill_names)} vs {len(verify_names)}"
            )
        for prefill_name, verify_name in zip(prefill_names, verify_names):
            self._assert_cache_compatible(
                self.prefill,
                prefill_name,
                self.verify,
                verify_name,
                "prefill/verify KV",
            )
            self.verify.set_input(verify_name, self.prefill.get_dev_input(prefill_name))
        if self.debug:
            logger.info("bound {} verify KV caches", len(prefill_names))

    def _validate_and_bind_mtp_cache(self) -> None:
        for prefill_name, draft_name in (
            (
                self._mtp_prefill_names.past_key_cache,
                self._draft_names.past_key_cache,
            ),
            (
                self._mtp_prefill_names.past_value_cache,
                self._draft_names.past_value_cache,
            ),
        ):
            self._assert_cache_compatible(
                self.mtp_prefill,
                prefill_name,
                self.draft,
                draft_name,
                "MTP prefill/draft",
            )
            self.draft.set_input(draft_name, self.mtp_prefill.get_dev_input(prefill_name))
        if self.debug:
            logger.info("bound MTP draft caches to MTP prefill caches")

    def clear_session(self) -> None:
        for name in self._prefill_names.conv_cache_in + self._prefill_names.recurrent_state_in:
            info = self._tensor_info(self.prefill, name)
            self._set_input(
                self.prefill,
                name,
                np.zeros(info.shape, dtype=_numpy_dtype(info.dtype)),
            )
        for name in (
            self._mtp_prefill_names.past_key_cache,
            self._mtp_prefill_names.past_value_cache,
        ):
            info = self._tensor_info(self.mtp_prefill, name)
            self._set_input(
                self.mtp_prefill,
                name,
                np.zeros(info.shape, dtype=_numpy_dtype(info.dtype)),
            )
        self._validate_and_bind_mtp_cache()

    def _propagate_prefill_cache(self) -> None:
        for input_name in self._prefill_names.conv_cache_in:
            output_name = self._prefill_names.conv_out_by_input.get(input_name)
            if output_name is None:
                raise RuntimeError(f"missing prefill conv output for input {input_name}")
            self.prefill.set_input(input_name, self.prefill.get_dev_output(output_name))

        if not self._prefill_names.recurrent_out_by_layer:
            return

        for input_name in self._prefill_names.recurrent_state_in:
            output_name = self._prefill_names.recurrent_out_by_input.get(input_name)
            if output_name is None:
                raise RuntimeError(f"missing prefill recurrent output for input {input_name}")
            self.prefill.set_input(input_name, self.prefill.get_dev_output(output_name))

    def prepare_verify_from_prefill(self) -> None:
        """Point verify recurrent inputs at the latest prefill cache tensors."""
        for input_name in self._verify_names.conv_cache_in:
            key = self._verify_names.conv_key_by_input[input_name]
            output_name = self._prefill_names.conv_out_by_key.get(key)
            if output_name is None:
                raise RuntimeError(f"missing prefill conv output for verify input {input_name}")
            self._assert_output_input_compatible(self.prefill, output_name, self.verify, input_name)
            self.verify.set_input(input_name, self.prefill.get_dev_output(output_name))

        if self._prefill_names.recurrent_out_by_layer:
            recurrent_names = self._prefill_names.recurrent_out_by_layer
            get_recurrent_state = self.prefill.get_dev_output
            assert_compatible = self._assert_output_input_compatible
            source_type = "output"
        else:
            recurrent_names = self._prefill_names.recurrent_in_by_layer
            get_recurrent_state = self.prefill.get_dev_input
            assert_compatible = self._assert_input_input_compatible
            source_type = "input"

        for input_name in self._verify_names.recurrent_state_in:
            layer = self._verify_names.recurrent_layer_by_input[input_name]
            recurrent_name = recurrent_names.get(layer)
            if recurrent_name is None:
                raise RuntimeError(f"missing prefill recurrent {source_type} for verify input {input_name}")
            assert_compatible(self.prefill, recurrent_name, self.verify, input_name)
            self.verify.set_input(input_name, get_recurrent_state(recurrent_name))

    @staticmethod
    def _assert_input_input_compatible(
        source_model,
        source_name: str,
        input_model,
        input_name: str,
    ) -> None:
        source = source_model.get_dev_input(source_name).info
        target = input_model.get_dev_input(input_name).info
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"cache shape differs: input {source_name} {tuple(source.shape)} "
                f"vs input {input_name} {tuple(target.shape)}"
            )
        if _numpy_dtype(source.dtype) != _numpy_dtype(target.dtype):
            raise RuntimeError(
                f"cache dtype differs: input {source_name} {source.dtype} " f"vs input {input_name} {target.dtype}"
            )

    @staticmethod
    def _assert_output_input_compatible(
        output_model,
        output_name: str,
        input_model,
        input_name: str,
    ) -> None:
        output = output_model.get_dev_output(output_name).info
        target = input_model.get_dev_input(input_name).info
        if tuple(output.shape) != tuple(target.shape):
            raise RuntimeError(
                f"cache shape differs: output {output_name} {tuple(output.shape)} "
                f"vs input {input_name} {tuple(target.shape)}"
            )
        if _numpy_dtype(output.dtype) != _numpy_dtype(target.dtype):
            raise RuntimeError(
                f"cache dtype differs: output {output_name} {output.dtype} " f"vs input {input_name} {target.dtype}"
            )

    def _commit_verify_conv_cache(self, accepted_position: int) -> None:
        names = self._verify_names
        for input_name in names.conv_cache_in:
            split_outputs = names.split_conv_out_by_input.get(input_name, [])
            if split_outputs:
                output_name = split_outputs[min(accepted_position, len(split_outputs) - 1)]
                self.verify.set_input(input_name, self.verify.get_dev_output(output_name))
                continue

            output_name = names.conv_out_by_input.get(input_name)
            if output_name is None:
                raise RuntimeError(f"missing verify conv output for input {input_name}")
            output = _to_numpy(self.verify.get_output(output_name))
            kernel = int(self.verify.get_dev_input(input_name).info.shape[-1])
            if output.shape[-1] > kernel:
                start = min(accepted_position, output.shape[-1] - kernel)
                output = output[..., start : start + kernel].copy()
            else:
                output = output.copy()
            self._set_input(self.verify, input_name, output)

    def _commit_verify_recurrent_cache(self, accepted_position: int) -> None:
        names = self._verify_names
        for input_name in names.recurrent_state_in:
            split_outputs = names.split_recurrent_out_by_input.get(input_name, [])
            if split_outputs:
                output_name = split_outputs[min(accepted_position, len(split_outputs) - 1)]
            else:
                output_name = names.recurrent_out_by_input.get(input_name)
                if output_name is None:
                    raise RuntimeError(f"missing verify recurrent output for input {input_name}")
            self.verify.set_input(input_name, self.verify.get_dev_output(output_name))

    def commit_verify_cache(self, accepted_steps: int) -> None:
        """Commit verify recurrent state at the last accepted verify step."""
        if accepted_steps <= 0 or accepted_steps > self.verify_length:
            raise ValueError(f"accepted_steps must be in [1, {self.verify_length}], " f"got {accepted_steps}")
        accepted_position = accepted_steps - 1
        self._commit_verify_conv_cache(accepted_position)
        self._commit_verify_recurrent_cache(accepted_position)

    @staticmethod
    def _stage_value(stage: Stage) -> str:
        return str(getattr(stage, "value", stage))

    @staticmethod
    def _input_bindings(stage: str, names, tensors) -> tuple[tuple[str, Any], ...]:
        if len(tensors) != 7:
            raise ValueError(f"{stage} expects 7 semantic tensors, got {len(tensors)}")
        if stage in ("prefill", "verify"):
            bindings = (
                (names.activation, tensors[0]),
                (names.valid_length, tensors[1]),
                (names.current_length, tensors[2]),
                (names.time_position_ids, tensors[3]),
                (names.height_position_ids, tensors[4]),
                (names.width_position_ids, tensors[5]),
            )
            if names.linear_attn_mask is not None:
                bindings += ((names.linear_attn_mask, tensors[6]),)
            return bindings
        position_values = tensors[2:5]
        if len(names.position_ids) > len(position_values):
            raise ValueError(
                f"{stage} graph exposes {len(names.position_ids)} position inputs, "
                f"but only {len(position_values)} semantic position tensors exist"
            )
        return (
            (names.hidden_states, tensors[0]),
            (names.input_embedding, tensors[1]),
            *zip(names.position_ids, position_values),
            (names.past_seq_length, tensors[5]),
            (names.current_input_length, tensors[6]),
        )

    def _stage_runtime(self, stage: Stage):
        stage_value = self._stage_value(stage)
        if stage_value == "prefill":
            return stage_value, self.prefill, self._prefill_names, "llm_mtp.prefill"
        elif stage_value == "mtp_prefill":
            return (
                stage_value,
                self.mtp_prefill,
                self._mtp_prefill_names,
                "llm_mtp.mtp_prefill",
            )
        elif stage_value == "draft":
            return stage_value, self.draft, self._draft_names, "llm_mtp.draft"
        elif stage_value == "verify":
            return stage_value, self.verify, self._verify_names, "llm_mtp.verify"
        raise ValueError(f"unsupported stage: {stage}")

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        stage_value, model, names, path = self._stage_runtime(stage)
        self._stage_metadata[stage] = dict(inputs.metadata)
        bindings = self._input_bindings(stage_value, names, inputs.tensors)
        with self.perf.scope(f"{path}.set_input"):
            for name, value in bindings:
                self._set_input(model, name, value)

    def run(self, stage: Stage) -> None:
        _, model, _, path = self._stage_runtime(stage)
        with self.perf.scope(f"{path}.infer"):
            model.run(sync=False)
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        stage_value, model, names, path = self._stage_runtime(stage)
        metadata = self._stage_metadata.pop(stage, {})
        if stage_value == "mtp_prefill":
            return StageOutputs(tensors=(), metadata=metadata)

        with self.perf.scope(f"{path}.get_output"):
            logits = _to_numpy(model.get_output(names.logits_out)).copy()
            hidden = _to_numpy(model.get_output(names.hidden_out)).copy()
        if stage_value == "prefill":
            current_length = int(metadata["current_length"])
            if current_length <= 0 or current_length > self.prefill_length:
                raise ValueError(f"current_length must be in [1, {self.prefill_length}], " f"got {current_length}")
            logits = logits[:, :current_length, :].copy()
            hidden = hidden[:, :current_length, :].copy()
            self._propagate_prefill_cache()
        return StageOutputs(tensors=(logits, hidden), metadata=metadata)


__all__ = ["Qwen35Module", "Qwen36MtpModule"]
