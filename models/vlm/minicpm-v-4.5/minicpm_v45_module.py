# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v45_module.py
# Description:
#   Runtime graphs and device cache ownership for MiniCPM-V 4.5.
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

from __future__ import annotations

from pathlib import Path

import numpy as np
import tcim_lite as tcim
import torch

from hmatc.python.get_hm_devices import get_hm_devices
from houmo_engine import HoumoModule
from houmo_engine.core.types import Stage, StageInputs, StageOutputs
from houmo_engine.perf import PerfTracker
from minicpm_v45_types import MiniCPMV45Paths


def _names(model, kind):
    count = model.get_num_inputs() if kind == "input" else model.get_num_outputs()
    getter = model.get_input_name if kind == "input" else model.get_output_name
    return [getter(i) for i in range(count)]


class MiniCPMV45Module(HoumoModule):
    def __init__(self, paths: MiniCPMV45Paths, *, ndevice: int, perf: PerfTracker):
        self.perf = perf
        self.models = {}
        self.metadata = {}
        self.load(paths, ndevice=ndevice)

    def load(self, paths: MiniCPMV45Paths, *, ndevice: int) -> None:
        if ndevice <= 0:
            raise ValueError("ndevice must be greater than zero")
        manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weights = tcim.runtime.WeightManager(manager)
        option = tcim.runtime.Option(weights)
        with self.perf.scope("llm.init.prefill_load"):
            self.prefill = tcim.runtime.load(str(paths.prefill_path), option=option)
        input_names = _names(self.prefill, "input")
        cache_names = [name for name in input_names if "model_layers" in name]
        decode_option = tcim.runtime.Option(weights)
        decode_option.set_dummy_tensors(cache_names)
        with self.perf.scope("llm.init.decode_load"):
            self.decode = tcim.runtime.load(str(paths.decode_path), option=decode_option)
        self.vision = None
        self.vision_input_names = []
        self.vision_group_capacity = 1
        self.supports_video = False
        self.video_vision = None
        self.video_vision_input_names = []
        if paths.vision_path is not None:
            vision_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
            vision_weights = tcim.runtime.WeightManager(vision_manager)
            vision_option = tcim.runtime.Option(vision_weights)
            with self.perf.scope("llm.init.vision_load"):
                self.vision = tcim.runtime.load(str(paths.vision_path), option=vision_option)
            self.vision_input_names = _names(self.vision, "input")
            pixel_shape = self.vision.get_input_info("pixel_values").shape
            self.vision_group_capacity = int(pixel_shape[0])
            self.supports_video = "resampler_temporal_pos_embed" in self.vision_input_names
        if paths.video_vision_path is not None and paths.video_vision_path != paths.vision_path:
            video_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
            video_option = tcim.runtime.Option(tcim.runtime.WeightManager(video_manager))
            with self.perf.scope("llm.init.video_vision_load"):
                self.video_vision = tcim.runtime.load(str(paths.video_vision_path), option=video_option)
            self.video_vision_input_names = _names(self.video_vision, "input")
            self.vision_group_capacity = int(self.video_vision.get_input_info("pixel_values").shape[0])
            self.supports_video = "resampler_temporal_pos_embed" in self.video_vision_input_names
        self.cache_names = [name for name in input_names if
                            "model_layers" in name or "conv_cache" in name or "recurrent_state" in name]
        self._bind_caches()
        self.prefill_length = int(self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1])
        self.embedding_dim = int(self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2])
        self.context_max_length = int(self.decode.get_input_info(self.decode.get_input_name(3)).shape[2])
        self.models = {Stage.PREFILL: self.prefill, Stage.DECODE: self.decode}
        if self.vision is not None:
            self.models[Stage.VISION] = self.vision
        self._clear_state_caches()
        self._set(self.decode, self.decode.get_input_name(2), np.array([1], np.int32))

    @staticmethod
    def _set(model, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        info = model.get_input_info(name)
        shape = tuple(int(x) for x in info.shape)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise ValueError(f"input {name} expects {shape}, got {value.shape}")
            value = value.reshape(shape)
        model.set_input(name, value.astype(np.dtype(info.dtype), copy=False))

    def _clear_state_caches(self):
        for name in self.cache_names:
            if "conv_cache" in name or "recurrent_state" in name:
                info = self.prefill.get_dev_input(name).info
                zeros = np.zeros(info.shape, dtype=np.float16)
                self._set(self.prefill, name, zeros)
                self._set(self.decode, name, zeros)

    def _bind_caches(self) -> None:
        for name in self.cache_names:
            cache = self.prefill.get_dev_input(name)
            self.decode.set_dev_input(name, cache)
            if "conv_cache" in name:
                output = name.replace("past_conv_cache_", "conv_cache_out_")
                self.prefill.set_dev_output(output, cache)
                self.decode.set_dev_output(output, cache)
            elif "recurrent_state" in name:
                output = name.replace("past_recurrent_state_", "recurrent_state_out_")
                self.prefill.set_dev_output(output, cache)
                self.decode.set_dev_output(output, cache)

    def clear_session(self) -> None:
        self._clear_state_caches()

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        profile = inputs.metadata.get("profile", "vision_1x")
        if stage == Stage.VISION and profile == "vision_6x":
            model = self.video_vision or (self.vision if self.supports_video else None)
            names = self.video_vision_input_names or self.vision_input_names
        else:
            model = self.models.get(stage)
            names = _names(model, "input") if model is not None else []
        if model is None:
            raise ValueError(f"unsupported stage: {stage}")
        self.metadata[stage] = dict(inputs.metadata)
        values = inputs.tensors
        with self.perf.scope(f"llm.{stage.value}.set_input"):
            for name, value in zip(names[:len(values)], values, strict=True):
                self._set(model, name, value)

    def run(self, stage: Stage) -> None:
        metadata = self.metadata.get(stage, {})
        model = (
            self.video_vision or self.vision
            if stage == Stage.VISION and metadata.get("profile") == "vision_6x"
            else self.models[stage]
        )
        with self.perf.scope(f"llm.{stage.value}.infer"):
            model.run()
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        metadata = self.metadata.get(stage, {})
        model = (
            self.video_vision or self.vision
            if stage == Stage.VISION and metadata.get("profile") == "vision_6x"
            else self.models[stage]
        )
        with self.perf.scope(f"llm.{stage.value}.get_output"):
            values = [model.get_output(model.get_output_name(0)).numpy()]
            if stage == Stage.VISION:
                effective = self.metadata[stage]["effective_tokens"]
                values[0] = torch.from_numpy(values[0][:, :effective, :])
        return StageOutputs(tuple(values), self.metadata.pop(stage, {}))
