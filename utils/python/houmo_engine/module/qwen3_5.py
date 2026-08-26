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
from pathlib import Path

import numpy as np
import torch
import tcim_lite as tcim

from ..core import HoumoModule
from ..core.types import Stage, StageInputs, StageOutputs
from ..perf import PerfTracker

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
