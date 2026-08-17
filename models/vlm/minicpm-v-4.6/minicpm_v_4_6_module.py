# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v_4_6_module.py
# Description:
#   MiniCPM-V 4.6 runtime Module implementation.
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

import numpy as np
import torch
import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices

from houmo_engine import HoumoModule
from houmo_engine.core.types import Stage, StageInputs, StageOutputs
from houmo_engine.perf import PerfTracker


class MiniCPMV46Module(HoumoModule):
    """MiniCPM-V 4.6 HMM graphs, cache bindings, and execution."""

    def __init__(
        self,
        prefill_path,
        decode_path,
        *,
        vision_path=None,
        ndevice: int = 1,
        perf: PerfTracker,
    ):
        self.perf = perf
        self._stage_metadata = {}
        self.load(
            prefill_path,
            decode_path,
            vision_path=vision_path,
            ndevice=ndevice,
        )

    def load(
        self,
        prefill_path,
        decode_path,
        *,
        vision_path=None,
        ndevice: int = 1,
    ) -> None:
        with self.perf.scope("llm.init"):
            dev_manager = tcim.runtime.DevManager(
                get_hm_devices(ndevice), "Xh2HalBackend"
            )
            weight_manager = tcim.runtime.WeightManager(dev_manager)
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
                self.decode = tcim.runtime.load(
                    str(decode_path), option=decode_option
                )

            self.vision = None
            if vision_path is not None:
                vision_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
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
        self._bind_caches()
        self.clear_session()
        self._set_input(
            self.decode,
            self.decode.get_input_name(5),
            np.array([1], dtype=np.int32),
        )

    @staticmethod
    def _input_shape(model, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in model.get_input_info(name).shape)

    def _set_input(self, model, name: str, value) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        expected = self._input_shape(model, name)
        if value.shape != expected:
            if value.size != int(np.prod(expected)):
                raise ValueError(f"input {name} expects {expected}, got {value.shape}")
            value = value.reshape(expected)
        dtype = np.dtype(model.get_input_info(name).dtype)
        model.set_input(name, value.astype(dtype, copy=False))

    def _bind_caches(self) -> None:
        for index in range(self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "model_layers" in name:
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

    def clear_session(self) -> None:
        for index in range(self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "conv_cache" not in name and "recurrent_state" not in name:
                continue
            info = self.prefill.get_dev_input(name).info
            zeros = np.zeros(info.shape, dtype=np.float16)
            self._set_input(self.prefill, name, zeros)
            self._set_input(self.decode, name, zeros)

    def _stage_model(self, stage: Stage):
        if stage == Stage.VISION:
            if self.vision is None:
                raise RuntimeError("vision model is not loaded")
            return self.vision, "llm.vision"
        if stage == Stage.PREFILL:
            return self.prefill, "llm.prefill"
        if stage == Stage.DECODE:
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


__all__ = ["MiniCPMV46Module"]
