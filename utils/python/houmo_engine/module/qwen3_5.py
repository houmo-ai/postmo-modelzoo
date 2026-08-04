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

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import tcim_lite as tcim

from ..core import HoumoModule
from ..core.types import Stage, StageInputs, StageOutputs
from ..perf import PerfTracker


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
        vision_path=None,
        lora_path=None,
        ndevice: int = 1,
        perf: PerfTracker,
    ):
        self.perf = perf
        self._stage_metadata = {}
        self.load(
            prefill_path,
            decode_path,
            vision_path=vision_path,
            lora_path=lora_path,
            ndevice=ndevice,
        )

    def load(
        self,
        prefill_path,
        decode_path,
        *,
        vision_path=None,
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
                self.prefill = tcim.runtime.load(
                    str(prefill_path), option=tcim.runtime.Option(weight_manager)
                )
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
            self.vision = None
            if vision_path is not None:
                vision_path = Path(vision_path)
                vision_devices = [0, 1] if vision_path.suffix == ".hmms" else [0]
                vision_manager = tcim.runtime.DevManager(
                    vision_devices, "Xh2HalBackend"
                )
                vision_option = tcim.runtime.Option(
                    tcim.runtime.WeightManager(vision_manager)
                )
                with self.perf.scope("llm.init.vision_load"):
                    self.vision = tcim.runtime.load(
                        str(vision_path), option=vision_option
                    )
            prefill_shape = self.prefill.get_input_info(
                self.prefill.get_input_name(0)
            ).shape
            self.prefill_length = int(prefill_shape[1])
            self.embedding_size = int(prefill_shape[2])
            self.context_max_length = int(
                self.decode.get_input_info(self.decode.get_input_name(7)).shape[2]
            )
            self.lora_input_names = [self.prefill.get_input_name(index) for index in range(self.prefill.get_num_inputs())
                if self._is_lora_input(self.prefill.get_input_name(index))]
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
                output = name.replace(
                    "past_recurrent_state_", "recurrent_state_out_"
                )
                cache = self.prefill.get_dev_input(name)
                self.prefill.set_dev_output(output, cache)
                self.decode.set_dev_input(name, cache)
                self.decode.set_dev_output(output, cache)

    @staticmethod
    def _input_shape(model, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in model.get_input_info(name).shape)

    @staticmethod
    def _input_dtype(model, name: str) -> np.dtype:
        return np.dtype(model.get_input_info(name).dtype)
    
    def _set_input(self, model, name: str, value) -> None:
        value = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        shape = self._input_shape(model, name)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise RuntimeError(
                    f"input {name!r} expects {shape}, got {value.shape}"
                )
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
                if lora_weight.shape != self._input_shape(self.prefill, name) or lora_weight.dtype != self._input_dtype(self.prefill, name):
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

    def run_vision(self, pixel_values: Sequence[torch.Tensor]) -> StageOutputs:
        if self.vision is None:
            raise RuntimeError("vision input requires a vision model")
        outputs = []
        name = self.vision.get_input_name(0)
        for value in pixel_values:
            with self.perf.scope("llm.vision.set_input"):
                self._set_input(self.vision, name, value)
            with self.perf.scope("llm.vision.infer"):
                self.vision.run()
                self.vision.sync()
            with self.perf.scope("llm.vision.get_output"):
                output = self.vision.get_output(
                    self.vision.get_output_name(0)
                ).numpy()
            tensor = torch.from_numpy(output)
            outputs.append(tensor.squeeze(0) if tensor.ndim == 3 else tensor)
        return StageOutputs(tensors=(torch.cat(outputs, dim=0),))

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
